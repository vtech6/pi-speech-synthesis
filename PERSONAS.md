# Adding a voice

A voice is three things: a clip of someone speaking, the exact transcript of that clip, and —
optionally — a personality. All of it is JSON; nothing needs recompiling.

## The guided way

```
/persona add
```

It asks for the path to the clip, what the clip says, a name, the language, the line to speak when
you switch to it, the thinking level, and the personality — then writes the entry to the personas
file and offers to start speaking as it. A path that does not exist is asked for again rather than
written down to fail later, and Esc at any question gives up without writing anything.

`/persona list` shows what the session loaded, with the active one marked and the file it came
from; `/persona <name>` switches, `/persona test` says that persona's `sample` line so you can hear
the clone before you commit to it, `/persona` opens the settings window.

## Changing a voice

The `manage personas` row at the bottom of `/persona` — or `/persona manage` — lists the voices and
offers, for any one of them:

- **speak as it**, which also says its `sample` line, so it is the quickest way to hear a voice;
- **change its details**, which asks the same questions as adding and shows what is filed for each,
  so an empty answer keeps it: fix one field without retyping the rest. A different name in the
  first question renames the entry;
- **remove it**, which takes the entry out of the file and leaves the clip on disk.

Esc leaves the submenu; the settings window reopens when it closes, over whatever the submenu
changed. The same three things are `/persona edit <name>` and `/persona remove <name>` when you
would rather not go through the window — and `edit` works on an entry that failed to load, which is
how you fix a clip that moved or a file the synthesizer could not read.

Only the file is written. A session already running picks the change up for the voice it is
speaking with; other sessions need a restart to see new entries.

## By hand

```bash
mkdir -p ~/.pi/agent/pi-tts
cp personas.example.json ~/.pi/agent/pi-tts/personas.json
$EDITOR ~/.pi/agent/pi-tts/personas.json
```

`PI_TTS_PERSONAS` points somewhere else if you would rather keep the file in a dotfiles repo.
Restart pi (or `/persona <name>` if the helper is already running), then `/persona test`.

## The clip

Qwen3-TTS clones from a few seconds of reference audio and keeps the timbre, the accent and the
prosody of whatever it hears. What it needs:

- **5–15 seconds**, one speaker, continuous. Shorter than that and the voice is unstable; longer
  than ~20 s mostly costs memory.
- **Clean**: no music, no reverb, no other voice, no laughter. A studio interview is fine; a film
  scene with a string section under it is not.
- **A WAV** (or FLAC). `afconvert -f WAVE -d LEI16 in.aiff out.wav` on macOS, or
  `ffmpeg -i in.mp3 -ar 24000 -ac 1 out.wav`.
- **A sentence that ends.** A clip cut off mid-thought teaches the model to end mid-thought.

Find the loudest clean stretch rather than trusting your ear:

```bash
ffmpeg -i ref.wav -af astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level -f null -
```

Then cut it: `ffmpeg -ss 12 -t 9 -i full.mp3 -ar 24000 -ac 1 ref.wav`.

## The transcript

`refText` is what the clip actually says, word for word, punctuation included — the model reads it
to learn the prosody, so a transcript that does not match the audio teaches it to mis-speak.
Include the commas and full stops where the speaker pauses. If you cannot transcribe it exactly,
transcribe it approximately and expect a flatter voice.

`lang` is the language name in lower case (`"english"`, `"german"`); it steers the tokenizer, not a
translation.

## The personality

`character` is the text injected into the system prompt as a `<persona>` section, and only while
`/persona mode talking` is on. In `coding` mode the system prompt stays exactly as pi built it and the
voice speaks pi's own prose — which is what you want while working.

Write it as rules about *how* to speak, not as a biography. The one in the example file, trimmed:

```
Speak as a patient narrator, reading to someone who is listening while working with their hands.
Do the engineering work precisely and completely: the persona colors the prose, never the facts,
the commands or the code.
- Plain sentences, level tone, no exclamation marks, no emoji.
- Finish the thought before adding the caveat.
- When something failed, say that first and what to do next second.
```

The line that matters most is the second one. Without it a persona starts hedging about the code.
Rules that name a specific habit ("prefer `we` to `I`", "say the failure before the fix") survive
much better than adjectives ("funny", "elegant").

Two things the extension handles for you:

- The persona is never announced. A rule is appended telling the model not to write "Persona: …" or
  name the voice, in the answer or in its thinking.
- `thinking: "minimal"` appends a rule to think very little and answer at once, because a persona
  that deliberates out loud sounds nothing like the clip you cloned. Pick `minimal` for a voice that
  is fast and conversational, `high` for one that is fine to pause and reason.

## Reference

| key | required | default | |
|---|---|---|---|
| `refAudio` | yes | | clip to clone; `~` and paths relative to the file work |
| `refText` | practically | `""` | exact transcript of the clip — without it the clone babbles, and `/persona` warns |
| `label` | | the key | shown in `/persona` |
| `lang` | | `english` | language name, lower case |
| `sample` | | a testing line | spoken when you switch to this voice |
| `thinking` | | `high` | `minimal` \| `medium` \| `high` |
| `character` | | `""` | personality for the `<persona>` prompt section |

Keys starting with `_` are ignored, so the file can carry its own notes. A persona whose `refAudio`
does not exist is skipped and named once by `/persona` rather than failing the whole file.

## Zero-config

With no personas file at all the extension uses `default`: `~/.cache/pi-tts/ref.wav`, plus
`~/.cache/pi-tts/ref.txt` as its transcript if present. Handy for trying it, too thin for keeping.
