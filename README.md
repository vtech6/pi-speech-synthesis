# Pi Speech Synthesis

Speak [pi](https://github.com/badlogic/pi-mono) out loud, and dictate prompts back, with local
MLX voice models. No cloud, no API key, nothing leaves the machine.

The reply is cut into sentences as it streams and each finished sentence goes straight to the
synthesizer, so you hear the first words long before the model stops writing. A new prompt cuts
the voice off mid-word rather than queueing up behind it. Dictation is the other half: press a
key, talk, and the transcript lands in the input — or is sent immediately, if you would rather
have a conversation than a typing session.

Two models are in the box, both local:

| | model | size |
|---|---|---|
| speaking | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` | ~1.3 GB |
| listening | `mlx-community/Qwen3-ASR-0.6B-8bit` | ~1.0 GB |

Both stay resident in one long-lived process each, so the model load is paid once at session
start instead of per utterance.

**Requires macOS on Apple Silicon** (MLX, and the audio timing work below is CoreAudio-specific).

## Quick start

```bash
pi install git:github.com/vtech6/pi-speech-synthesis
~/.pi/agent/git/github.com/vtech6/pi-speech-synthesis/scripts/setup.sh   # ~2.3 GB of models
```

Start pi, and give it something to sound like. **No voices ship with the extension** — a voice here
is a few seconds of someone talking plus the exact transcript of those seconds, and those clips are
not ours to redistribute:

```
/persona add
```

It asks for the clip, what the clip says, a name, and an optional personality. Or skip the
questions: put a clip at `~/.cache/pi-tts/ref.wav` with its transcript in `ref.txt`, and it is the
`default` persona with no configuration at all. [PERSONAS.md](PERSONAS.md) is about what makes a
clip that works.

Then hear it:

```
/persona say testing one two three
```

`ctrl+i` dictates, `ctrl+i` again stops. `/persona` opens the settings window; `/persona help`
lists every command with the state the session is actually in.

Until a clip exists there is nothing to clone. The `default` persona is still listed and still
selectable, and the first thing you ask it to say comes back `reference audio not found` — that is
what a first run looks like, not a broken install. Add a clip, or `/persona add`, and it speaks.

## Install

```bash
pi install git:github.com/vtech6/pi-speech-synthesis
~/.pi/agent/git/github.com/vtech6/pi-speech-synthesis/scripts/setup.sh
```

`pi install` clones the package under `~/.pi/agent/git/`; the setup script is run from that
checkout. If you cloned the repo somewhere yourself, run `./scripts/setup.sh` from the clone —
the two are the same script, and pi's checkout is replaced on `pi update`.

`setup.sh` creates `~/.cache/pi-tts/venv`, installs `requirements.txt` into it, downloads both
models to `~/.cache/pi-tts/models`, and drops a `personas.json` you can edit. Re-run it any time;
it skips what is already there. Then add a reference clip (below) and start pi — `/persona help`
lists every command with the state the session is in.

Already have a python with mlx-audio — conda, uv, whatever?

```bash
$PI_TTS_PYTHON -m pip install -r requirements.txt
scripts/setup.sh --skip-venv
export PI_TTS_PYTHON=~/miniconda3/envs/voice/bin/python
```

To take it back out: `pi remove git:github.com/vtech6/pi-speech-synthesis`. `~/.cache/pi-tts` — the
venv, the models, your clips — and `~/.pi/agent/pi-tts/personas.json` are left where they are, so
removing the package keeps the downloads; delete those yourself if you want the ~2.3 GB back.

## Add a voice

The extension ships no voices: a voice is a few seconds of someone talking, plus the exact
transcript of those seconds. Put a clip at `~/.cache/pi-tts/ref.wav` (and its transcript at
`ref.txt`) and it is available as the `default` persona with no configuration at all.

For more than one voice, edit `~/.pi/agent/pi-tts/personas.json`:

```json
{
  "my-voice": {
    "label": "My voice",
    "refAudio": "~/.cache/pi-tts/ref.wav",
    "refText": "This is the reference clip, and this sentence is exactly what it says, word for word.",
    "sample": "Testing one two three. The voice is working."
  }
}
```

Each entry is a key you type after `/persona`, the clip to clone, and what that clip says.
`character` adds a personality to the system prompt, `thinking` sets how much the persona is
allowed to think before answering. `/persona add` writes these for you, one question at a time;
`personas.example.json` has two worked examples, and **[PERSONAS.md](PERSONAS.md)** covers what
makes a usable clip and how to write the personality.

## Using it

One command, `/persona`: it drives the voice, the voices file, dictation and the settings. No
argument opens the settings window. `/persona help` prints all of this in the terminal, with the
state the session is actually in.

| | |
|---|---|
| `/persona` | the settings window: voice, mode, persona, thinking, volume, dictation — `s` saves |
| `/persona help` | every row below, in the terminal |
| **the voice** | |
| `/persona on\|off` | speak or stay quiet |
| `/persona stop` | cut the current utterance off |
| `/persona say <text>` | say this instead of a reply |
| `/persona read <file>` | read a `.txt`, `.md` or `.pdf` aloud (`.pdf` needs `pypdf`) |
| `/persona render [stop\|status\|measure\|<file> [out]]` | turn a file into an audio file in the background, with a progress bar |
| `/persona measure <file>` | how long and how big that render would be, without doing it |
| `/persona test` | say this persona's `sample` line, in that voice — the audio check |
| `/persona mode coding\|talking` | coding leaves the system prompt alone; talking injects the persona |
| **the voices** — `~/.pi/agent/pi-tts/personas.json` | |
| `/persona <name>` | switch voice (and personality) mid-session |
| `/persona add` | guided: path to the clip, what it says, a name, a personality |
| `/persona manage` | the voices submenu: use a voice, change its details, remove it |
| `/persona edit <name>` | change one entry, each question showing what is filed now |
| `/persona remove <name>` | drop it from the file; the clip stays on disk |
| `/persona list` | the voices this session loaded, and the file they came from |
| **dictation** | |
| `ctrl+i` | press to talk, press again to stop |
| `ctrl+shift+i` | throw the capture away |
| `/persona listen` | start dictating — the same thing `ctrl+i` does |
| `/persona listen stop\|cancel\|on\|off\|consecutive\|conversation\|auto\|file <path>` | dictation, by command |

A voice is keyed by the name you type after `/persona`, so those subcommand names are reserved and
cannot be used as a voice: `on off stop say read render measure test mode help listen add list
manage edit remove`.

Dictation has two modes. `consecutive` leaves the transcript in the input so you can fix it up and
send it by hand. `conversation` sends it the moment you stop talking, and reopens the microphone
once the reply has been spoken, so the exchange keeps running without touching the keyboard.

`ctrl+i` is TAB in terminals that do not speak the Kitty keyboard protocol — `PI_TTS_STT_SHORTCUT`
picks another key.

`/persona read <file>` speaks a document instead of a reply, cut into sentences the same way a
streamed answer is, so a new prompt or `/persona stop` cuts it off rather than queueing behind page
nine. Text and markdown are read directly (code blocks and markup are dropped on the way to the
synthesizer); a PDF is unpacked by the voice helper, which needs `pypdf` — `requirements.txt` has
it, and it is the only reason that package is there.

`/persona render <file> [out]` makes the same text into an **audio file** instead of speaking it: a
second helper process, holding its own copy of the model, writing while you keep working. A bar
above the editor counts the pieces and the minutes of audio, `status` repeats it and `stop` closes
the file with whatever has been written so far. It runs with the voice off, it is not cut to
`PI_TTS_READ_MAX_CHARS`, and it says the whole document. Output is mp3 (~0.5 MB a minute, which is
about 28 MB an hour; the same hour as wav is 173 MB) — pass a `.wav` name for the uncompressed
version. Expect roughly half real time: an hour of speech takes about half an hour to render, and
shares the machine with the live voice while it does.

`/persona measure <file>` (or `/persona render measure <file>`) prices that job before you pay for it:
the same cut the renderer would make, the minutes of audio it comes to, the minutes the render will
take, and the size of the file. It starts from an assumed 16 characters a second of speech and 2.4×
realtime throughput, and moves those numbers towards the truth with every render over 20 seconds of
audio that session makes — which is why the answer gets better, and says so, after the first job.

## Configuration

Everything has a default that `setup.sh` satisfies; the environment wins over `personas.json`,
which wins over the defaults saved by `s` in the settings window. The names are the original
`pi-tts` ones — the cache directory, the personas file, the extension directory and every `PI_TTS_*`
variable still carry that prefix; only the project was renamed.

| variable | default | |
|---|---|---|
| `PI_TTS_CACHE` | `~/.cache/pi-tts` | venv, models, `ref.wav` |
| `PI_TTS_PYTHON` | `$PI_TTS_CACHE/venv/bin/python`, else `python3` | needs mlx-audio |
| `PI_TTS_MODEL` | `$PI_TTS_CACHE/models/Qwen3-TTS-12Hz-0.6B-Base-8bit` | directory or HF repo id |
| `PI_TTS_ASR_MODEL` | `$PI_TTS_CACHE/models/Qwen3-ASR-0.6B-8bit` | directory or HF repo id |
| `PI_TTS_PERSONAS` | `~/.pi/agent/pi-tts/personas.json` | the voices |
| `PI_TTS_PERSONA` | `default` | starting persona |
| `PI_TTS_REF_AUDIO`<br>`PI_TTS_REF_TEXT`<br>`PI_TTS_LANG` | from the persona | clip, its transcript and language, for the persona the session starts with |
| `PI_TTS_THINKING` | from the persona | `minimal` \| `medium` \| `high` |
| `PI_TTS_MODE` | `coding` | `coding` \| `talking` |
| `PI_TTS_VOICE_ENABLED` | `1` | |
| `PI_TTS_VOLUME` | `0.1` | output gain, 0.0–1.0 |
| `PI_TTS_MAX_CHARS` | `700` | spoken text cap |
| `PI_TTS_READ_MAX_CHARS` | `12000` | longest `/persona read` will speak |
| `PI_TTS_RENDER_DIR` | `~/.cache/pi-tts/rendered` | where `/persona render` writes |
| `PI_TTS_RENDER_FORMAT` | `mp3` | `mp3` \| `wav` for an unnamed output; a name you pass wins |
| `PI_TTS_RENDER_PIECE_CHARS` | `1200` | characters per synthesis request when rendering |
| `PI_TTS_RENDER_MAX_CHARS` | `200000` | longest document `/persona render` will take |
| `PI_TTS_RENDER_VOLUME` | `1.0` | gain of the written file, independent of the spoken volume |
| `PI_TTS_RENDER_SPEECH_RATE` | `16` | characters per second of speech, assumed by `/persona measure` |
| `PI_TTS_RENDER_SPEED` | `2.4` | audio seconds per second of wall clock, assumed by `/persona measure` |
| `PI_TTS_STREAM_INTERVAL` | `1.0` | seconds of speech per chunk |
| `PI_TTS_STT_SHORTCUT` | `ctrl+i` | |
| `PI_TTS_STT_CANCEL_SHORTCUT` | `ctrl+shift+i` | |
| `PI_TTS_STT_MAX_SECONDS` | `60` | longest single capture |
| `PI_TTS_STT_MODE` | `consecutive` | `consecutive` \| `conversation` |
| `PI_TTS_STT_AUTO` | `1` | conversation mode reopens the mic after each reply |
| `PI_TTS_ASR_LANG` | auto-detect | language hint for dictation |
| `PI_TTS_ASR_DEVICE` | system default | input device index or name |
| `PI_TTS_SETTINGS_FILE` | `~/.pi/agent/pi-tts.json` | where `s` writes |
| `PI_TTS_DEBUG` | `1` | `0` stops the dictation trace |

The helpers take their own knobs (`QWEN_TTS_TEMPERATURE`, `QWEN_TTS_TOP_K`, `QWEN_TTS_PREROLL`,
`QWEN_TTS_LATENCY`, `QWEN_TTS_RENDER_INTERVAL`, `QWEN_ASR_THRESHOLD`, `QWEN_ASR_SILENCE_SECONDS`,
…); they are documented in the docstring at the top of each file.

## How it is built

```
extensions/pi-tts/
├── index.ts        the pi extension: sentence splitting, queue, commands, settings window
├── pi_tts.py   long-lived Qwen3-TTS helper, streams audio to the output device
└── pi_stt.py   long-lived Qwen3-ASR helper, captures the mic, closes on silence
```

`index.ts` talks newline-delimited JSON to both helpers over stdin/stdout
(`say` `stop` `clear` `ref` `tune` `ping` / `record` `stop` `cancel` `transcribe`), so either half
can be run and tested on its own:

```bash
~/.cache/pi-tts/venv/bin/python extensions/pi-tts/pi_tts.py   # then: {"cmd":"say","text":"hello"}
~/.cache/pi-tts/venv/bin/python extensions/pi-tts/pi_stt.py   # then: {"cmd":"record"}
```

Two things in there are load-bearing and were paid for in clicks: the output device is fed by a
PortAudio callback reading a buffer we size ourselves, because blocking writes pace the generator
to the device and underrun on every chunk boundary; and the stream asks for a large latency so
PortAudio calls back ~6 times a second instead of 1600, which keeps up while the model is working.
Details in the `Player` docstring.

Nothing here is anybody's voice: no clips and no models are in the repo, and the two personas in
`personas.example.json` are placeholders pointing at clips you have to put there yourself.

## Troubleshooting

- **nothing is said** — `PI_TTS_DEBUG=0 pi`, then look at
  `${TMPDIR:-/tmp}/pi-tts/dictation.log`; the helpers print model load and every request
  to stderr, which is inherited, so it shows up in the terminal.
- **`reference audio not found`** — the persona's `refAudio` does not exist. `/persona` names the
  voices it could not load and the file it read them from.
- **the model downloaded but will not load** — check `mlx-audio --version` in the venv against
  `requirements.txt`; the checkpoints are converted with a specific mlx-audio.
- **no microphone** — System Settings → Privacy & Security → Microphone, allow your terminal.
  `/persona listen file <path>` transcribes without any permission.
- **first words are slow** — the model loads at session start in the background; the first reply
  of a session waits for it. `QWEN_TTS_PREROLL` and `PI_TTS_STREAM_INTERVAL` trade latency for
  smoothness.

## License

MIT — see [LICENSE](LICENSE). The code is MIT and that is the whole of what this repo ships. The
two models are downloaded at setup time under Apache-2.0 (`mlx-community` conversions of
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` and `Qwen/Qwen3-ASR-0.6B`), and the reference clips are yours:
whatever you download from a game, a podcast or your own recordings stays your problem, which is
why none are in this repo.
