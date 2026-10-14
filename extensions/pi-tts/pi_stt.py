#!/usr/bin/env python3
"""Long-lived speech-to-text helper for the pi-tts extension.

Loads the Qwen3-ASR model once, then reads one JSON object per line from stdin
and writes one JSON response per line to stdout. A "record" request captures
the default input device, closes itself on trailing silence (or on request) and
answers with the transcript, so dictating a prompt costs about a second once
the model is resident.

  -> {"id": "1", "cmd": "record"}
  <- {"id": "1", "cmd": "record_start", "device": "MacBook Pro Microphone", "threshold": 0.012}
  <- {"id": "1", "cmd": "level", "seconds": 0.4, "rms": 0.031, "speaking": true}
  <- {"id": "1", "cmd": "transcript", "text": "...", "seconds": 3.2, "reason": "silence"}

Requests: {"cmd": "record"|"stop"|"cancel"|"transcribe"|"ping"|"quit"}
  record      start capturing; auto-stops on trailing silence or at max_seconds
  stop        end the capture now and transcribe whatever was heard
  cancel      end the capture and throw it away
  transcribe  {"path": "audio.wav"} transcribe a file instead of the microphone

Env:
  QWEN_ASR_MODEL         model dir or HF repo id            (default $PI_TTS_CACHE/models/...)
  QWEN_ASR_LANG          language hint, empty = auto-detect (default empty)
  QWEN_ASR_DEVICE        input device index or name         (default system default)
  QWEN_ASR_MAX_SECONDS   hard cap per capture               (default 60)
  QWEN_ASR_SILENCE_SECONDS trailing silence that closes it  (default 1.0)
  QWEN_ASR_MIN_SECONDS   never cut before this              (default 0.5)
  QWEN_ASR_START_TIMEOUT give up on a room that never spoke (default 15)
  QWEN_ASR_THRESHOLD     absolute RMS floor                 (default 0.01)
  QWEN_ASR_BEEP          "0" mutes the start/end blips      (default on)
  QWEN_ASR_WARMUP        "0" skips the first-call warmup    (default on)
"""

import json
import os
import queue
import sys
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
from mlx_audio.stt.utils import load_model

CACHE = os.environ.get(
    "PI_TTS_CACHE", os.path.join(os.path.expanduser("~"), ".cache/pi-tts")
)
MODEL = os.environ.get(
    "QWEN_ASR_MODEL", os.path.join(CACHE, "models/Qwen3-ASR-0.6B-8bit")
)
LANG = os.environ.get("QWEN_ASR_LANG") or None
DEVICE = os.environ.get("QWEN_ASR_DEVICE") or None
OUT_DIR = os.environ.get("QWEN_ASR_OUT_DIR", os.path.join(tempfile.gettempdir(), "pi-tts"))
SAMPLE_RATE = int(os.environ.get("QWEN_ASR_SAMPLE_RATE", "16000"))
MAX_SECONDS = float(os.environ.get("QWEN_ASR_MAX_SECONDS", "60"))
SILENCE_SECONDS = float(os.environ.get("QWEN_ASR_SILENCE_SECONDS", "1.0"))
MIN_SECONDS = float(os.environ.get("QWEN_ASR_MIN_SECONDS", "0.5"))
START_TIMEOUT = float(os.environ.get("QWEN_ASR_START_TIMEOUT", "15"))
FLOOR = float(os.environ.get("QWEN_ASR_THRESHOLD", "0.01"))
MAX_TOKENS = int(os.environ.get("QWEN_ASR_MAX_TOKENS", "4096"))
BEEP = os.environ.get("QWEN_ASR_BEEP", "1") != "0"
WARMUP = os.environ.get("QWEN_ASR_WARMUP", "1") != "0"
POLL = 0.1  # seconds between level reports and end-of-speech checks
SETTLE = 0.4  # opening window ignored: device wake-up and the start blip

os.makedirs(OUT_DIR, exist_ok=True)

requests: "queue.Queue[dict]" = queue.Queue()
lock = threading.Lock()
active = None  # the Recorder currently being filled, if any
# A stop/cancel pressed in the moment between sending "record" and the input
# stream opening used to fall on the floor, and the capture ran on regardless.
pending_signal = None


def log(message):
    print(f"[pi_stt] {message}", file=sys.stderr, flush=True)


def emit(payload):
    print(json.dumps(payload), flush=True)


def beep(freq, seconds=0.06, volume=0.12):
    """Short blip so it is obvious when the microphone is open."""
    if not BEEP:
        return
    try:
        t = np.linspace(0, seconds, int(seconds * SAMPLE_RATE), endpoint=False, dtype=np.float32)
        tone = np.sin(2 * np.pi * freq * t) * volume
        fade = max(1, int(SAMPLE_RATE * 0.008))
        tone[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        tone[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        sd.play(tone.astype(np.float32), SAMPLE_RATE, blocking=False)
        time.sleep(seconds + 0.03)  # let it finish before the mic opens
    except Exception as error:
        log(f"beep failed: {error}")


class Recorder:
    """Input-device callback that only copies audio and tracks the level.

    Deciding when a capture is over happens in the worker thread, next to the
    stop/cancel events, so the silence timer and an explicit stop are handled in
    one place. The callback keeps the loudest level seen since the last poll so
    a short word between polls is not mistaken for silence.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.blocks = []
        self.samples = 0
        self.since_poll = 0.0
        self.loudest = 0.0
        self.threshold = None  # calibrated from the opening stretch of audio
        self.last_voice = 0.0
        self.finish = threading.Event()
        self.cancel = threading.Event()

    def callback(self, indata, frames, time_info, status):
        block = np.array(indata[:, 0], dtype=np.float32, copy=True)
        rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
        with self.lock:
            self.blocks.append(block)
            self.samples += block.size
            self.since_poll = max(self.since_poll, rms)
            self.loudest = max(self.loudest, rms)

    def audio(self):
        with self.lock:
            if not self.blocks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self.blocks)


def calibrate(audio):
    """RMS floor of the opening stretch, so a noisy room is not read as speech.

    The first fraction of a second is the moment right after the keypress, when
    the speaker has not started yet; its quietest fifth is the room, and speech
    has to clear four times that (and the absolute floor) to count. The blip is
    cut out of the window because it is the loudest thing in it.
    """
    window = audio[int(SETTLE * SAMPLE_RATE) : int(0.8 * SAMPLE_RATE)]
    if window.size < SAMPLE_RATE // 10:
        return FLOOR
    frames = window[: window.size - (window.size % 160)].reshape(-1, 160)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    # Capped: someone who starts talking inside the window must still be heard.
    return min(max(FLOOR, float(np.percentile(rms, 20)) * 4.0), FLOOR * 8)


def resample(audio, rate):
    """Linear resample to the model rate, so any wav can be transcribed."""
    if rate == SAMPLE_RATE or audio.size == 0:
        return audio
    target = int(round(audio.size * SAMPLE_RATE / rate))
    x = np.linspace(0, audio.size - 1, target, dtype=np.float64)
    return np.interp(x, np.arange(audio.size, dtype=np.float64), audio).astype(np.float32)


def device_name(device):
    try:
        return sd.query_devices(device or sd.default.device[0])["name"]
    except Exception:
        return str(device)


def transcribe(request_id, index, audio, seconds, reason, peak=None, threshold=None):
    peak = round(peak, 5) if peak is not None else None
    if audio.size == 0 or seconds < MIN_SECONDS:
        emit(
            {
                "id": request_id,
                "cmd": "skip",
                "reason": "too short",
                "seconds": round(seconds, 2),
                "peak": peak,
            }
        )
        return
    path = os.path.join(OUT_DIR, f"pi-stt-{os.getpid()}-{index}.wav")
    sf.write(path, audio, SAMPLE_RATE)
    began = time.time()
    try:
        result = model.generate(audio, language=LANG, temperature=0.0, max_tokens=MAX_TOKENS)
    except Exception as error:  # keep the loop alive across failures
        emit({"id": request_id, "cmd": "error", "error": str(error), "path": path})
        return
    text = (getattr(result, "text", "") or "").strip()
    language = getattr(result, "language", None)
    if isinstance(language, (list, tuple)):
        language = language[0] if language else None
    if not language or str(language).lower() == "none":
        language = None
    emit(
        {
            "id": request_id,
            "cmd": "transcript" if text else "skip",
            "text": text,
            "language": language,
            "seconds": round(seconds, 2),
            "reason": reason,
            "path": path,
            "peak": peak,
            "threshold": round(threshold, 5) if threshold is not None else None,
            "gen_seconds": round(time.time() - began, 2),
        }
    )


def capture(request_id, index, request):
    """Record, watch the level, then transcribe. Runs on the worker thread."""
    global active, pending_signal
    rec = Recorder()
    max_seconds = float(request.get("max_seconds") or MAX_SECONDS)
    silence = float(request.get("silence_seconds") or SILENCE_SECONDS)
    beep(880.0)
    try:
        stream = sd.InputStream(
            device=DEVICE,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=rec.callback,
        )
    except Exception as error:
        emit({"id": request_id, "cmd": "error", "error": f"could not open input: {error}"})
        return

    began = time.monotonic()
    with lock:
        active = rec
        if pending_signal is not None:
            cmd, pending_signal = pending_signal, None
            (rec.finish if cmd == "stop" else rec.cancel).set()
    emit(
        {
            "id": request_id,
            "cmd": "record_start",
            "device": device_name(DEVICE),
            "max_seconds": max_seconds,
        }
    )
    reason = None
    try:
        with stream:
            while True:
                time.sleep(POLL)
                elapsed = time.monotonic() - began
                now = time.monotonic()
                with rec.lock:
                    rms = rec.since_poll
                    rec.since_poll = 0.0
                if elapsed < SETTLE:
                    # The blip that announces the capture comes back through the
                    # microphone; counting it as speech would close the capture
                    # a second later, before the user has said anything.
                    rms = 0.0
                if rec.threshold is None and elapsed >= 0.8:
                    rec.threshold = calibrate(rec.audio())
                    emit({"id": request_id, "cmd": "calibrated", "threshold": round(rec.threshold, 5)})
                if rms >= (rec.threshold or FLOOR):
                    rec.last_voice = now
                spoke = rec.last_voice > 0
                emit(
                    {
                        "id": request_id,
                        "cmd": "level",
                        "seconds": round(elapsed, 1),
                        "rms": round(rms, 5),
                        "speaking": spoke,
                    }
                )
                if rec.cancel.is_set():
                    reason = "cancelled"
                    break
                if rec.finish.is_set():
                    reason = "manual"
                    break
                if elapsed >= max_seconds:
                    reason = "max"
                    break
                if spoke and elapsed > MIN_SECONDS and now - rec.last_voice > silence:
                    reason = "silence"
                    break
                if not spoke and elapsed > START_TIMEOUT:
                    reason = "quiet"
                    break
    finally:
        with lock:
            active = None

    beep(392.0)
    audio = rec.audio()
    seconds = audio.size / SAMPLE_RATE
    if reason == "cancelled":
        # Thrown away, not transcribed: in conversation mode a transcript would
        # otherwise be sent even though the speaker gave up on it.
        emit(
            {
                "id": request_id,
                "cmd": "skip",
                "reason": "cancelled",
                "seconds": round(seconds, 2),
                "peak": round(rec.loudest, 5),
            }
        )
        return
    transcribe(
        request_id,
        index,
        audio,
        seconds,
        reason or "unknown",
        peak=rec.loudest,
        threshold=rec.threshold or FLOOR,
    )


def read_requests():
    """Stdin reader thread, so "stop" is heard even while capturing."""
    global pending_signal
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            emit({"cmd": "error", "error": f"bad json: {error}"})
            continue

        cmd = request.get("cmd", "record")
        if cmd == "ping":
            emit({"cmd": "pong", "id": request.get("id")})
        elif cmd in ("stop", "cancel"):
            with lock:
                rec = active
                if rec is None:
                    pending_signal = cmd
                    continue
            (rec.finish if cmd == "stop" else rec.cancel).set()
        elif cmd == "quit":
            requests.put(request)
            return
        else:
            with lock:
                # A signal left over from the previous capture is stale.
                pending_signal = None
            requests.put(request)

    requests.put({"cmd": "quit"})


started = time.time()
model = load_model(MODEL)
rate = int(getattr(model, "sample_rate", SAMPLE_RATE))
if rate != SAMPLE_RATE:
    log(f"model wants {rate}Hz, capturing at {SAMPLE_RATE}Hz instead")
    SAMPLE_RATE = rate
if WARMUP:
    # First generate pays for kernel compilation; pay it before "ready".
    model.generate(np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32), temperature=0.0, max_tokens=8)
log(f"model loaded in {time.time() - started:.1f}s, input: {device_name(DEVICE)}")
emit(
    {
        "cmd": "ready",
        "seconds": round(time.time() - started, 1),
        "sample_rate": SAMPLE_RATE,
        "device": device_name(DEVICE),
        "language": LANG or "auto",
    }
)

threading.Thread(target=read_requests, daemon=True).start()

counter = 0
while True:
    request = requests.get()
    if request.get("cmd") == "quit":
        break

    counter += 1
    request_id = str(request.get("id") or counter)
    try:
        if request.get("cmd") == "transcribe":
            audio, rate = sf.read(request["path"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = np.ascontiguousarray(resample(np.ascontiguousarray(audio), rate))
            transcribe(request_id, counter, audio, audio.size / SAMPLE_RATE, "file")
        else:
            capture(request_id, counter, request)
    except Exception as error:
        log(f"request failed: {error}")
        emit({"id": request_id, "cmd": "error", "error": str(error)})

sd.stop()
