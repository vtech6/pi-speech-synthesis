#!/usr/bin/env python3
"""Long-lived TTS helper for the pi-tts extension.

Loads the Qwen3-TTS model once, then reads one JSON object per line from stdin
and writes one JSON response per line to stdout. This process plays the audio
itself, so speech starts as soon as the first streamed chunk is decoded instead
of after the whole utterance is generated.

  -> {"id": "1", "cmd": "say", "text": "..."}
  <- {"id": "1", "cmd": "audio_start", "first_chunk_seconds": 1.2}
  <- {"id": "1", "cmd": "audio_done", "path": "...", "seconds": 4.2}

Requests: {"cmd": "say"|"stop"|"clear"|"ref"|"ping"|"extract"|"quit"}
  stop   cut the current utterance off and discard anything queued
  clear  let the current utterance finish, discard anything queued
  ref    change the cloned voice (persona) without reloading the model:
         {"cmd":"ref","ref_audio":path,"ref_text":caption,"lang":"english"}
  extract {"cmd":"extract","path":file} answers {"cmd":"text","text":...} — the
         text of a file, for reading it aloud (.pdf needs pypdf)
  tune   change the audio knobs at runtime:
         {"cmd":"tune","volume":0.2,"streaming_interval":1.0}

Batch mode writes a file instead of talking, and prints progress instead:

  pi_tts.py --render --job JOB.json [--out FILE]

  JOB.json  {"out":path,"pieces":["text",...],"ref_audio":path,"ref_text":caption,
             "lang":"english","volume":1.0}
             The extension of "out" picks the encoder: .wav, .mp3, or whatever
             libsndfile makes of the name it is given.
  stdout    {"cmd":"start",...} {"cmd":"progress",...} ... then "done", "cancelled"
            or "error". Nothing is played and no output device is opened, so a
            render runs beside a live voice without making a sound of its own.
            SIGTERM closes the file where it got to and reports "cancelled".

Env:
  QWEN_TTS_MODEL            model dir or HF repo id       (default $PI_TTS_CACHE/models/...)
  QWEN_TTS_REF_AUDIO        reference clip to clone         (default $PI_TTS_CACHE/ref.wav)
  QWEN_TTS_REF_TEXT         transcript of that clip         (default empty)
  QWEN_TTS_VOLUME           output gain 0.0-1.0                 (default 0.1)
  QWEN_TTS_TEMPERATURE      sampling temperature                (default 0.5)
  QWEN_TTS_TOP_K            candidate pool per token           (default 20)
  QWEN_TTS_STREAMING        "0" waits for the full utterance    (default on)
  QWEN_TTS_STREAM_INTERVAL  seconds of speech per chunk         (default 1.0)
  QWEN_TTS_RENDER_INTERVAL  seconds of speech per chunk when rendering to a file
                            (default 2.0: coarser than live speech, less work)
  QWEN_TTS_PREROLL          seconds buffered before playback    (default 0.25)
  QWEN_TTS_LATENCY          requested output latency, seconds   (default 0.25)
"""

import collections
import json
import os
import queue
import signal
import sys
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
from mlx_audio.tts.utils import load_model

# The extension names the model and the reference clip in the environment, both
# taken from the personas file; these defaults are for running the helper by hand.
CACHE = os.environ.get(
    "PI_TTS_CACHE", os.path.join(os.path.expanduser("~"), ".cache/pi-tts")
)
MODEL = os.environ.get(
    "QWEN_TTS_MODEL", os.path.join(CACHE, "models/Qwen3-TTS-12Hz-0.6B-Base-8bit")
)
REF_AUDIO = os.environ.get("QWEN_TTS_REF_AUDIO", os.path.join(CACHE, "ref.wav"))
REF_TEXT = os.environ.get("QWEN_TTS_REF_TEXT", "")
LANG = os.environ.get("QWEN_TTS_LANG", "english")
OUT_DIR = os.environ.get("QWEN_TTS_OUT_DIR", os.path.join(tempfile.gettempdir(), "pi-tts"))
TEMPERATURE = float(os.environ.get("QWEN_TTS_TEMPERATURE", "0.5"))
TOP_K = int(os.environ.get("QWEN_TTS_TOP_K", "20"))
MAX_TOKENS = int(os.environ.get("QWEN_TTS_MAX_TOKENS", "2048"))
VOLUME = min(1.0, max(0.0, float(os.environ.get("QWEN_TTS_VOLUME", "0.1"))))
STREAMING = os.environ.get("QWEN_TTS_STREAMING", "1") != "0"
STREAM_INTERVAL = float(os.environ.get("QWEN_TTS_STREAM_INTERVAL", "1.0"))
RENDER_INTERVAL = float(os.environ.get("QWEN_TTS_RENDER_INTERVAL", "2.0"))
PREROLL = float(os.environ.get("QWEN_TTS_PREROLL", "0.25"))
LATENCY = float(os.environ.get("QWEN_TTS_LATENCY", "0.25"))
CAPTURE = os.environ.get("QWEN_TTS_CAPTURE") == "1"

os.makedirs(OUT_DIR, exist_ok=True)

requests: "queue.Queue[dict]" = queue.Queue()
abort = threading.Event()
# Set by SIGTERM/SIGINT while rendering, so the file is closed whole rather
# than truncated mid-frame.
stopped = threading.Event()
# Bumped by every stop. A queued sentence carrying an older generation is
# stale, whether it is still in the queue or already handed to the worker.
generation = 0


def log(message):
    print(f"[pi_tts] {message}", file=sys.stderr, flush=True)


def emit(payload):
    print(json.dumps(payload), flush=True)


def to_numpy(audio):
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    return np.asarray(audio).reshape(-1).astype(np.float32)


class Player:
    """Output device fed by a callback that pulls from our own ring buffer.

    Blocking writes (stream.write) pace the caller to the device: the writer
    thread sits inside write() for as long as the audio it just handed over,
    so generation and playback take turns instead of overlapping. The PortAudio
    FIFO is only as deep as the device latency (~28 ms on built-in output), so
    the next chunk, which takes ~350 ms to decode, always arrives after the
    device has run dry: an underrun, and a click, on every chunk boundary.

    With a callback the device pulls from a buffer we size ourselves, so a
    decode slower than the audio it produced is just a shorter buffer, not a
    glitch. Preroll keeps the first chunk from starting the same way.
    """

    def __init__(self):
        self.cond = threading.Condition()
        self.chunks: "collections.deque" = collections.deque()
        self.pending = 0  # frames queued but not yet handed to the device
        self.started = False  # preroll gate
        self.preroll = 0
        self.stream = None
        self.sample_rate = None
        self.xruns = 0
        self.gaps = 0  # callback found the buffer empty mid-utterance
        self.ended = False
        self.capture = None  # debug: exactly what the device was handed
        self.multi = 0  # blocks that had to span more than one queued chunk
        self.late = 0  # callbacks that missed their deadline (dropouts)
        self._last_cb = 0.0

    def _callback(self, outdata, frames, time_info, status):
        if status:
            self.xruns += 1
        now = time.monotonic()
        last, self._last_cb = self._last_cb, now
        if last and now - last > frames / max(self.sample_rate or 1, 1) * 1.5:
            self.late += 1
        outdata.fill(0)
        with self.cond:
            if not self.started:
                if self.pending < self.preroll:
                    return
                self.started = True
            filled = 0
            blocks = 0
            while filled < frames and self.chunks:
                head = self.chunks[0]
                take = min(head.shape[0], frames - filled)
                outdata[filled : filled + take, 0] = head[:take]
                if take == head.shape[0]:
                    self.chunks.popleft()
                else:
                    self.chunks[0] = head[take:]
                filled += take
                blocks += 1
            if blocks > 1:
                self.multi += 1
            if filled < frames and not self.ended:
                self.gaps += 1
            self.pending -= filled
            if self.capture is not None:
                self.capture.append(outdata[:, 0].copy())
            if self.pending <= 0:
                self.pending = 0
                self.cond.notify_all()

    def ensure(self, sample_rate):
        """Open (or reopen) the output stream. Only ever called by the speaker."""
        if (
            self.stream is not None
            and int(self.sample_rate) == int(sample_rate)
            and self.stream.active
        ):
            return True
        self.close()
        self.preroll = int(sample_rate * PREROLL)
        try:
            self.stream = sd.OutputStream(
                samplerate=int(sample_rate),
                channels=1,
                dtype="float32",
                callback=self._callback,
                latency=LATENCY,
            )
            self.stream.start()
        except Exception as error:
            log(f"could not open output stream at {sample_rate}Hz: {error}")
            self.stream = None
            return False
        self.sample_rate = int(sample_rate)
        # The requested latency decides the callback's block size, and that is
        # what makes this survive generation: at the device default PortAudio
        # calls back every 15 frames (1600 Python calls a second) and starts
        # missing deadlines as soon as the model is working. A larger latency
        # buys 4096-frame blocks, about 6 calls a second, which stay on time
        # even with every core busy. The device clamps to its own maximum.
        log(
            f"output: {self.sample_rate}Hz latency={self.stream.latency:.3f}s "
            f"block={self.stream.blocksize or 'device'} "
            f"(~{1 / max(self.stream.latency, 1e-6):.0f} callbacks/s)"
        )
        with self.cond:
            self.started = False
        return True

    def push(self, chunk):
        with self.cond:
            self.chunks.append(chunk)
            self.pending += chunk.shape[0]

    def mark_end(self):
        """No more chunks coming: play out whatever is left, preroll or not."""
        with self.cond:
            self.started = True
            self.ended = True
            self.cond.notify_all()

    def wait_drained(self, timeout):
        deadline = time.time() + timeout
        with self.cond:
            while self.pending > 0:
                if abort.is_set() or time.time() > deadline:
                    return False
                if self.stream is None or not self.stream.active:
                    return False
                self.cond.wait(0.05)
            return True

    def clear(self):
        """Cut the sound now by emptying what the device would pull next."""
        with self.cond:
            self.chunks.clear()
            self.pending = 0
            self.started = False
            self.ended = False
            self.cond.notify_all()

    def close(self):
        stream, self.stream = self.stream, None
        self.sample_rate = None
        if stream is not None:
            for name in ("abort", "close"):
                try:
                    getattr(stream, name)()
                except Exception:
                    pass


player = Player()


def speak_utterance(request_id, index, text):
    began = time.time()
    sample_rate = int(model.sample_rate)
    player.clear()
    player.xruns = 0
    player.gaps = 0
    player.multi = 0
    player.late = 0
    player._last_cb = 0.0
    if CAPTURE:
        player.capture = []
    first_chunk = None
    played = []
    total = 0

    results = model.generate(
        text=text,
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        lang_code=LANG,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        stream=STREAMING,
        streaming_interval=STREAM_INTERVAL,
    )
    try:
        for result in results:
            if abort.is_set():
                break
            chunk = np.clip(to_numpy(result.audio) * VOLUME, -1.0, 1.0).astype(np.float32)
            if chunk.size == 0:
                continue
            if first_chunk is None:
                first_chunk = time.time() - began
                emit({"id": request_id, "cmd": "audio_start"})
            if not player.ensure(sample_rate):
                break
            player.push(chunk)
            played.append(chunk)
            total += chunk.shape[0]
    finally:
        if hasattr(results, "close"):
            results.close()

    stopped = abort.is_set()
    if stopped:
        player.clear()
    elif total:
        player.mark_end()
        player.wait_drained(timeout=total / sample_rate + 5.0)

    audio = np.concatenate(played) if played else np.zeros(0, dtype=np.float32)
    path = None
    if audio.size:
        path = os.path.join(OUT_DIR, f"pi-{os.getpid()}-{index}.wav")
        sf.write(path, audio, sample_rate)
    if CAPTURE and player.capture:
        device = np.concatenate(player.capture)
        sf.write(os.path.join(OUT_DIR, f"pi-{os.getpid()}-{index}-device.wav"), device, sample_rate)
        log(f"{request_id}: device capture {device.shape[0] / sample_rate:.2f}s, "
            f"{player.multi} multi-chunk callbacks, "
            f"max diff vs generated {np.abs(device[: audio.shape[0]] - audio).max():.6f}")
        player.capture = None

    emit(
        {
            "id": request_id,
            "cmd": "audio_done" if total else "skip",
            "path": path,
            "seconds": round(total / sample_rate, 2),
            "first_chunk_seconds": round(first_chunk, 2) if first_chunk else None,
            "gen_seconds": round(time.time() - began, 2),
            "stopped": stopped,
            "reason": "stopped" if stopped else None,
            "xruns": player.xruns,
            "buffer_gaps": player.gaps,
            "late_callbacks": player.late,
        }
    )
    if player.xruns or player.gaps or player.late:
        log(
            f"{request_id}: {player.xruns} over/underruns, {player.gaps} empty "
            f"buffer callbacks, {player.late} late callbacks"
        )


def extract_text(path):
    """Text of a file, to be read aloud.

    Only PDFs need a parser here; everything else is read as text and left to
    the caller to clean up. pypdf is imported on demand, so a session that never
    reads a document never pays for it.
    """
    if not path.lower().endswith(".pdf"):
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def render_chunks(text):
    """Samples of one piece, generated but never played."""
    results = model.generate(
        text=text,
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        lang_code=LANG,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        stream=True,
        streaming_interval=RENDER_INTERVAL,
    )
    try:
        for result in results:
            if stopped.is_set():
                break
            chunk = np.clip(to_numpy(result.audio) * VOLUME, -1.0, 1.0).astype(np.float32)
            if chunk.size:
                yield chunk
    finally:
        if hasattr(results, "close"):
            results.close()


def open_audio(path, sample_rate):
    """The file being written.

    wav is the raw thing at 48 KB a minute; mp3 is the same speech through
    libsndfile's encoder at a sixth of the size, which is what a document you
    take for a walk wants. Anything else is left to libsndfile's own reading of
    the extension, so .flac and .ogg work where they are built.
    """
    lower = path.lower()
    if lower.endswith(".wav"):
        return sf.SoundFile(
            path, mode="w", samplerate=sample_rate, channels=1, format="WAV", subtype="PCM_16"
        )
    if lower.endswith(".mp3"):
        return sf.SoundFile(
            path, mode="w", samplerate=sample_rate, channels=1, format="MP3", subtype="MPEG_LAYER_III"
        )
    return sf.SoundFile(path, mode="w", samplerate=sample_rate, channels=1)


def render_job(job_path, out_override):
    """Synthesize a whole document into one audio file, reporting as it goes.

    The pieces come from the caller, already cut at sentence ends, because that
    is where the live voice cuts too; the estimate of the finished length is
    corrected by every piece that completes, so the bar it feeds is honest by
    the second piece.
    """
    global REF_AUDIO, REF_TEXT, LANG, VOLUME
    try:
        with open(job_path, encoding="utf-8") as handle:
            job = json.load(handle)
    except Exception as error:
        emit({"cmd": "error", "error": f"could not read the job {job_path}: {error}"})
        return 2

    pieces = [piece.strip() for piece in (job.get("pieces") or []) if piece and piece.strip()]
    out = out_override or job.get("out") or ""
    if not pieces or not out:
        emit({"cmd": "error", "error": "the job needs pieces and out"})
        return 2
    if job.get("ref_audio"):
        REF_AUDIO = job["ref_audio"]
    if job.get("ref_text") is not None:
        REF_TEXT = job["ref_text"]
    if job.get("lang"):
        LANG = job["lang"]
    if job.get("volume") is not None:
        VOLUME = min(1.0, max(0.0, float(job["volume"])))

    sample_rate = int(model.sample_rate)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    except OSError as error:
        emit({"cmd": "error", "error": f"could not create {out}: {error}"})
        return 2

    total_chars = sum(len(piece) for piece in pieces)
    # Characters per second, guessed once and then measured: it only ever feeds
    # the progress bar, never the audio.
    rate = 16.0
    began = time.time()
    last_report = 0.0
    frames = 0
    done_chars = 0

    def report(piece_index, piece_seconds):
        piece = pieces[piece_index]
        inside = min(len(piece), int(piece_seconds * rate)) if piece_seconds else 0
        emit(
            {
                "cmd": "progress",
                "piece": piece_index + 1,
                "pieces": len(pieces),
                "chars": total_chars,
                "chars_done": min(total_chars, done_chars + inside),
                "seconds": round(frames / sample_rate, 1),
                "est_seconds": round(total_chars / rate, 1),
                "elapsed": round(time.time() - began, 1),
            }
        )

    emit(
        {
            "cmd": "start",
            "out": out,
            "pieces": len(pieces),
            "chars": total_chars,
            "sample_rate": sample_rate,
            "est_seconds": round(total_chars / rate, 1),
        }
    )

    wav = None
    try:
        wav = open_audio(out, sample_rate)
        for index, piece in enumerate(pieces):
            if stopped.is_set():
                break
            piece_seconds = 0.0
            for chunk in render_chunks(piece):
                wav.write(chunk)
                frames += chunk.shape[0]
                piece_seconds += chunk.shape[0] / sample_rate
                now = time.time()
                if now - last_report >= 0.5:
                    last_report = now
                    report(index, piece_seconds)
            if stopped.is_set():
                break
            done_chars += len(piece)
            if piece_seconds:
                rate = 0.7 * rate + 0.3 * (len(piece) / piece_seconds)
            report(index, 0.0)
    except Exception as error:
        emit({"cmd": "error", "error": f"could not render {out}: {error}"})
        return 1
    finally:
        if wav is not None:
            wav.close()

    elapsed = round(time.time() - began, 1)
    seconds = round(frames / sample_rate, 1)
    if stopped.is_set():
        emit({"cmd": "cancelled", "path": out, "seconds": seconds, "elapsed": elapsed})
        return 130
    emit(
        {
            "cmd": "done",
            "path": out,
            "seconds": seconds,
            "pieces": len(pieces),
            "elapsed": elapsed,
        }
    )
    log(f"rendered {seconds}s of audio to {out} in {elapsed}s")
    return 0


def drop_pending():
    """Discard queued requests; the utterance being spoken is left alone."""
    while True:
        try:
            requests.get_nowait()
        except queue.Empty:
            return


def read_requests():
    """Stdin reader thread, so "stop" is heard even while generating."""
    global generation, REF_AUDIO, REF_TEXT, LANG, VOLUME, STREAM_INTERVAL
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            emit({"cmd": "error", "error": f"bad json: {error}"})
            continue

        cmd = request.get("cmd", "say")
        log(f"recv {cmd} {request.get('id') or ''}")
        if cmd == "ping":
            emit({"cmd": "pong", "id": request.get("id")})
        elif cmd == "stop":
            generation += 1
            drop_pending()
            abort.set()
            player.clear()
        elif cmd == "clear":
            generation += 1
            drop_pending()
        elif cmd == "ref":
            # Persona switch: the next utterance is synthesized in the new voice.
            path = request.get("ref_audio") or REF_AUDIO
            if not os.path.exists(path):
                emit(
                    {
                        "cmd": "error",
                        "id": request.get("id"),
                        "error": f"reference audio not found: {path}",
                    }
                )
                continue
            REF_AUDIO = path
            if request.get("ref_text") is not None:
                REF_TEXT = request["ref_text"]
            if request.get("lang"):
                LANG = request["lang"]
            emit(
                {
                    "cmd": "ref",
                    "id": request.get("id"),
                    "ref_audio": REF_AUDIO,
                    "lang": LANG,
                }
            )
        elif cmd == "tune":
            volume = request.get("volume")
            if volume is not None:
                VOLUME = min(1.0, max(0.0, float(volume)))
            interval = request.get("streaming_interval")
            if interval is not None:
                STREAM_INTERVAL = max(0.1, float(interval))
            emit(
                {
                    "cmd": "tune",
                    "id": request.get("id"),
                    "volume": VOLUME,
                    "streaming_interval": STREAM_INTERVAL,
                }
            )
        elif cmd == "extract":
            path = request.get("path") or ""
            if not os.path.exists(path):
                emit(
                    {
                        "cmd": "error",
                        "id": request.get("id"),
                        "error": f"file not found: {path}",
                    }
                )
                continue
            try:
                emit(
                    {
                        "cmd": "text",
                        "id": request.get("id"),
                        "text": extract_text(path),
                    }
                )
            except Exception as error:
                emit(
                    {
                        "cmd": "error",
                        "id": request.get("id"),
                        "error": f"could not read {path}: {error}",
                    }
                )
        elif cmd == "quit":
            requests.put(request)
            return
        else:
            request["_gen"] = generation
            requests.put(request)

    generation += 1
    abort.set()
    player.clear()
    requests.put({"cmd": "quit"})


started = time.time()
model = load_model(MODEL)
log(f"model loaded in {time.time() - started:.1f}s")

BATCH = "--render" in sys.argv[1:]
if BATCH:
    # A file, not the room: no output stream, so no device wake and no sound.
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    argv = sys.argv[1:]
    job = argv[argv.index("--job") + 1] if "--job" in argv else None
    extra = argv[argv.index("--out") + 1] if "--out" in argv else None
    sys.exit(render_job(job, extra))

# Opening an output stream wakes the CoreAudio device, and that thumps. Do it
# here, once, instead of under the first utterance, and leave it running: the
# callback keeps feeding it silence, so the device stays awake and speech never
# begins on a stream-start transient.
player.ensure(int(model.sample_rate))
if CAPTURE:
    player.capture = []
    log("device capture on")
if not os.path.exists(REF_AUDIO):
    log(
        f"warning: no reference audio at {REF_AUDIO} — voice cloning has nothing to "
        "clone; set QWEN_TTS_REF_AUDIO or write a personas file"
    )
emit({"cmd": "ready", "seconds": round(time.time() - started, 1), "volume": VOLUME})

threading.Thread(target=read_requests, daemon=True).start()

counter = 0
while True:
    request = requests.get()
    if request.get("cmd") == "quit":
        break

    counter += 1
    request_id = str(request.get("id") or counter)
    text = (request.get("text") or "").strip()
    if not text:
        emit({"id": request_id, "cmd": "skip", "reason": "empty text"})
        continue

    if request.get("_gen") != generation:
        # Superseded by a stop or a clear while it waited in the queue.
        emit({"id": request_id, "cmd": "skip", "reason": "stopped"})
        continue

    abort.clear()
    try:
        speak_utterance(request_id, counter, text)
    except Exception as error:  # keep the loop alive across failures
        log(f"generation failed: {error}")
        emit({"id": request_id, "cmd": "error", "error": str(error)})

player.clear()
player.close()
