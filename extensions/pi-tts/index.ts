/**
 * Voice extension for pi.
 *
 * Speaks the assistant's reply out loud, in a cloned voice, as it streams: the
 * text is cut into sentences and each finished sentence is handed to the
 * synthesizer while the model is still writing, so the first words are heard
 * long before the message ends. Whatever is left is spoken when the message
 * ends; a message that streamed no deltas is spoken whole at settle time.
 *
 * Uses a long-lived Python helper (pi_tts.py, next to this file) that keeps
 * the MLX model loaded and streams audio straight to the output device, so the
 * model-load cost is paid once, at session start, and speech starts after the
 * first decoded chunk. Talking stops as soon as a new user prompt arrives, and
 * prose that was queued behind it is dropped rather than caught up on.
 *
 * Dictation uses a second helper (pi_stt.py) holding the Qwen3-ASR model:
 * press the dictation shortcut (or /persona listen), talk, and the transcript
 * lands in the editor. The capture closes itself on trailing silence, and the voice is
 * muted the moment it opens so the voice never talks over the user.
 *
 * Two modes: "consecutive" leaves the transcript in the input, so it can be
 * fixed up and sent by hand (or dictated over again), and "conversation" sends
 * it the moment the capture closes, which is the round trip you want when
 * talking rather than typing. Conversation mode also reopens the microphone on
 * its own once the reply has been spoken, so the exchange keeps running; either
 * way the cancel key throws away what was heard instead of sending it.
 *
 * Commands: `/persona` opens the settings window, `/persona help` prints what it
 * can do. The voice is driven from the same command — on, off, stop, say, read,
 * render, measure, test, mode — the voices file from a bare persona name, add,
 * list, manage, edit and remove, and dictation from `/persona listen`. The
 * dictation shortcut does the same as that last one.
 *
 * Config (environment):
 *   PI_TTS_VOICE_ENABLED   "1" (default) | "0"
 *   PI_TTS_PYTHON          python with mlx-audio (default: $PI_TTS_CACHE/venv/bin/python,
 *                          falling back to python3 on PATH)
 *   PI_TTS_CACHE           where setup.sh installs the venv and the models
 *                          (default: ~/.cache/pi-tts)
 *   PI_TTS_MAX_CHARS       spoken text cap         (default: 700)
 *   PI_TTS_READ_MAX_CHARS  longest file read       (default: 12000)
 *   PI_TTS_RENDER_DIR      where /persona render writes       (default: $PI_TTS_CACHE/rendered)
 *   PI_TTS_RENDER_FORMAT   extension of an unnamed output: mp3 | wav (default: mp3)
 *   PI_TTS_RENDER_PIECE_CHARS  characters per synthesis request when rendering
 *                          (default: 1200 — longer pieces are fewer seams, and a
 *                          request is bounded by QWEN_TTS_MAX_TOKENS)
 *   PI_TTS_RENDER_MAX_CHARS  longest document rendered (default: 200000)
 *   PI_TTS_RENDER_VOLUME   gain of the written file 0.0-1.0  (default: 1.0)
 *   PI_TTS_RENDER_SPEECH_RATE  characters per second of speech, used to price a
 *                          render before doing it (default: 16, corrected by the
 *                          renders this session actually made)
 *   PI_TTS_RENDER_SPEED    audio seconds per second of wall clock (default: 2.4)
 *   PI_TTS_VOLUME          output gain 0.0-1.0     (default: 0.1)
 *   PI_TTS_STREAM_INTERVAL seconds of speech per chunk (default: 1.0)
 *   PI_TTS_MODE            coding (default, prompt untouched) | talking (the
 *                          persona is injected as a <persona> section)
 *   PI_TTS_PERSONA         starting persona: any key of the personas file
 *                          (default: "default")
 *   PI_TTS_THINKING        starting thinking level: minimal | medium | high
 *   PI_TTS_SETTINGS_FILE   defaults file  (default: ~/.pi/agent/pi-tts.json)
 *   PI_TTS_PERSONAS        the voices file  (default: ~/.pi/agent/pi-tts/personas.json,
 *                          see personas.example.json and PERSONAS.md)
 *   PI_TTS_VOICE_ENABLED, PI_TTS_VOLUME, PI_TTS_MAX_CHARS, PI_TTS_STREAM_INTERVAL,
 *   PI_TTS_STT_MODE, PI_TTS_STT_AUTO
 *                          every one of these is also a row in the /persona
 *                          window, where `s` saves them to the defaults file;
 *                          the environment still wins over the saved defaults
 *   PI_TTS_MODEL           TTS model directory or HF repo id (default:
 *                          $PI_TTS_CACHE/models/Qwen3-TTS-12Hz-0.6B-Base-8bit)
 *   PI_TTS_REF_AUDIO / PI_TTS_REF_TEXT / PI_TTS_LANG
 *                          reference audio, its caption and the language, for
 *                          the persona the session starts with
 *                          (forwarded to pi_tts.py)
 *   PI_TTS_STT_SHORTCUT    dictation shortcut        (default: ctrl+i)
 *   PI_TTS_STT_MAX_SECONDS longest single capture    (default: 60)
 *   PI_TTS_STT_MODE        consecutive (default: transcript waits in the input)
 *                          | conversation (sent as soon as the capture closes)
 *   PI_TTS_STT_AUTO        "0" stops conversation mode from reopening the mic
 *   PI_TTS_STT_CANCEL_SHORTCUT  discard the capture in progress (ctrl+shift+i)
 *   PI_TTS_ASR_MODEL / PI_TTS_ASR_LANG / PI_TTS_ASR_DEVICE
 *                          forwarded to pi_stt.py
 */

import { spawn, type ChildProcessByStdio } from "node:child_process";
import {
	appendFileSync,
	existsSync,
	mkdirSync,
	readFileSync,
	statSync,
	unlinkSync,
	writeFileSync,
} from "node:fs";
import type { Readable, Writable } from "node:stream";
import { homedir, tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { getSettingsListTheme } from "@earendil-works/pi-coding-agent";
import {
	SettingsList,
	type KeyId,
	type SettingItem,
	truncateToWidth,
	wrapTextWithAnsi,
} from "@earendil-works/pi-tui";

const HERE = dirname(fileURLToPath(import.meta.url));

/** A helper process: every stream piped, so nothing it prints lands in the TUI. */
type Helper = ChildProcessByStdio<Writable, Readable, Readable>;

/**
 * The three things a checkout cannot know: the python that has mlx-audio, and
 * the two model directories. scripts/setup.sh puts all of them under CACHE_DIR,
 * so a default install needs no environment at all; PI_TTS_PYTHON, PI_TTS_MODEL
 * and PI_TTS_ASR_MODEL override for conda and uv users.
 */
const CACHE_DIR = process.env.PI_TTS_CACHE ?? join(homedir(), ".cache/pi-tts");
const VENV_PYTHON = join(CACHE_DIR, "venv/bin/python");

const CONFIG = {
	python:
		process.env.PI_TTS_PYTHON ?? (existsSync(VENV_PYTHON) ? VENV_PYTHON : "python3"),
	helper: join(HERE, "pi_tts.py"),
};

const MODEL_DIR =
	process.env.PI_TTS_MODEL ?? join(CACHE_DIR, "models/Qwen3-TTS-12Hz-0.6B-Base-8bit");
const ASR_MODEL_DIR =
	process.env.PI_TTS_ASR_MODEL ?? join(CACHE_DIR, "models/Qwen3-ASR-0.6B-8bit");

/**
 * A cloned voice: reference audio plus the transcript of that audio, which the
 * base model uses for prosody. Switching persona swaps the reference in the
 * running helper, so it costs no model reload.
 */
type Persona = {
	label: string;
	refAudio: string;
	refText: string;
	lang: string;
	/** Short line spoken back to confirm the switch, in the new voice. */
	sample: string;
	/** How much the persona is allowed to think before it answers. */
	thinking: PersonaThinking;
	/** Personality added to the system prompt while the persona is active. */
	character: string;
};

/** The three levels the settings window offers; pi accepts any of its own. */
type PersonaThinking = "minimal" | "medium" | "high";
const THINKING_LEVELS: PersonaThinking[] = ["minimal", "medium", "high"];

/** The persona belongs to the system prompt; it is not news in the reply. */
const NO_ANNOUNCE =
	"The persona is your system prompt, not something to report: never write \"Persona: ...\", never name the voice or the person you speak as, and never explain the style — not in the answer, and not in your thinking or reasoning either.";

/** What a persona with the minimum thinking level has to accept in its prompt. */
const MINIMAL_THINKING = `Thinking is at its minimum, so think very little and respond immediately.
Do not deliberate in the answer and do not disclose your thoughts: no "let me think", no "I considered", no summary of reasoning, no thinking block. Say the result, at once, in the persona.`;

/** The voice used when nothing is installed yet: a clip dropped in CACHE_DIR. */
const DEFAULT_PERSONA = "default";

/** Where the voices live. One JSON object per voice, keyed by its name. */
const PERSONAS_FILE =
	process.env.PI_TTS_PERSONAS ?? join(homedir(), ".pi/agent/pi-tts/personas.json");

/** A persona as written in the file: only refAudio matters, the rest defaults. */
type PersonaSpec = {
	label?: string;
	refAudio?: string;
	refText?: string;
	lang?: string;
	sample?: string;
	thinking?: string;
	character?: string;
};

/** Personas that were skipped while loading; /persona reports them once. */
const PERSONA_WARNINGS: string[] = [];

/** `~` is the home directory; a bare path is relative to the file that wrote it. */
function expandPath(path: string, base: string): string {
	if (path === "~") return homedir();
	if (path.startsWith("~/")) return join(homedir(), path.slice(2));
	return isAbsolute(path) ? path : resolve(base, path);
}

function normalizePersona(key: string, spec: PersonaSpec, base: string): Persona | null {
	const refAudio = spec.refAudio ? expandPath(spec.refAudio, base) : "";
	if (!refAudio) {
		PERSONA_WARNINGS.push(`persona "${key}" has no refAudio`);
		return null;
	}
	if (!existsSync(refAudio)) {
		PERSONA_WARNINGS.push(`persona "${key}": reference audio not found: ${refAudio}`);
		return null;
	}
	const level = (spec.thinking ?? "").toLowerCase().trim();
	if (!spec.refText?.trim())
		PERSONA_WARNINGS.push(
			`persona "${key}" has no refText: the clone will babble without the transcript of the clip`,
		);
	return {
		label: spec.label ?? key,
		refAudio,
		refText: spec.refText ?? "",
		lang: spec.lang ?? "english",
		sample: spec.sample ?? "Testing complete. The voice is working.",
		thinking: THINKING_LEVELS.includes(level as PersonaThinking)
			? (level as PersonaThinking)
			: "high",
		character: spec.character ?? "",
	};
}

/**
 * The voices this install offers, read from PERSONAS_FILE. Each one is a
 * reference clip plus the transcript of that clip; the personality is optional,
 * so a voice can speak with the system prompt left exactly as pi built it.
 * With no usable voice in it there is still DEFAULT_PERSONA, pointing at
 * CACHE_DIR/ref.wav, so the commands have something to name and the helper can
 * report the missing clip rather than the settings window having no rows.
 */
function loadPersonas(): Record<string, Persona> {
	const personas: Record<string, Persona> = {};
	let raw: unknown;
	try {
		raw = JSON.parse(readFileSync(PERSONAS_FILE, "utf8"));
	} catch {
		PERSONA_WARNINGS.push(`no personas file at ${PERSONAS_FILE}, using "${DEFAULT_PERSONA}"`);
	}
	if (raw && typeof raw === "object") {
		const base = dirname(PERSONAS_FILE);
		for (const [key, spec] of Object.entries(raw as Record<string, unknown>)) {
			if (key.startsWith("_") || !spec || typeof spec !== "object") continue;
			const entry = normalizePersona(key, spec as PersonaSpec, base);
			if (entry) personas[key] = entry;
		}
	}
	// Only when nothing else loaded: a real voice and a ref.wav are two entries
	// for the same job, and the placeholder cannot speak until the clip appears.
	if (!personas[DEFAULT_PERSONA] && !Object.keys(personas).length)
		personas[DEFAULT_PERSONA] = placeholderPersona();
	return personas;
}

/**
 * The voice used when nothing is installed. Kept even when the clip is missing,
 * so the commands and the settings window still have a voice to name; the helper
 * reports the missing clip the first time it is asked to speak. Installed again
 * when `/persona` removes the last voice, so there is always one to name.
 */
function placeholderPersona(): Persona {
	let refText = "";
	try {
		refText = readFileSync(join(CACHE_DIR, "ref.txt"), "utf8").trim();
	} catch {
		/* no caption: the clip speaks for itself */
	}
	return {
		label: "Default",
		refAudio: join(CACHE_DIR, "ref.wav"),
		refText,
		lang: "english",
		sample: "Testing complete. The voice is working.",
		thinking: "high",
		character: "",
	};
}

const PERSONAS: Record<string, Persona> = loadPersonas();

/** Where `d` in the settings window writes the chosen defaults. */
const SETTINGS_FILE =
	process.env.PI_TTS_SETTINGS_FILE ?? join(homedir(), ".pi/agent/pi-tts.json");

type PersonaSettings = {
	voice?: string;
	mode?: string;
	persona?: string;
	thinking?: string;
	volume?: string;
	maxChars?: string;
	streamInterval?: string;
	dictation?: string;
	dictationMode?: string;
	autoListen?: string;
};

const onOff = (value: string | undefined, fallback: boolean): boolean =>
	value === "on" ? true : value === "off" ? false : fallback;

/** Knobs are offered as steps; a saved or env value snaps to the nearest one. */
function snap(value: unknown, steps: number[], fallback: number): number {
	const parsed = Number(value ?? fallback);
	if (!Number.isFinite(parsed)) return fallback;
	return steps.reduce(
		(best, step) => (Math.abs(step - parsed) < Math.abs(best - parsed) ? step : best),
		steps[0]!,
	);
}

function loadSettings(): PersonaSettings {
	try {
		const parsed = JSON.parse(readFileSync(SETTINGS_FILE, "utf8")) as PersonaSettings;
		return parsed && typeof parsed === "object" ? parsed : {};
	} catch {
		return {}; // no defaults saved yet
	}
}

function saveSettings(settings: PersonaSettings): void {
	try {
		writeFileSync(SETTINGS_FILE, `${JSON.stringify(settings, null, "\t")}\n`);
	} catch {
		/* a read-only home is not worth an error dialog */
	}
}

const saved = loadSettings();
const initialPersona = (process.env.PI_TTS_PERSONA ?? saved.persona ?? DEFAULT_PERSONA)
	.toLowerCase()
	.trim();
// A saved persona this install does not have (a fresh machine, a renamed
// entry) falls back to the placeholder if there is one, and otherwise to the
// first voice that did load, so PERSONAS[persona] always resolves.
let persona = PERSONAS[initialPersona]
	? initialPersona
	: PERSONAS[DEFAULT_PERSONA]
		? DEFAULT_PERSONA
		: (Object.keys(PERSONAS)[0] ?? DEFAULT_PERSONA);

/** Tag name of the system prompt section carrying the active personality. */
const PERSONA_SECTION = "persona";

/** How much the model thinks: the saved default, or what the window picked. */
const initialThinking = (process.env.PI_TTS_THINKING ?? saved.thinking ?? PERSONAS[persona].thinking)
	.toLowerCase()
	.trim();
let thinking: PersonaThinking = THINKING_LEVELS.includes(initialThinking as PersonaThinking)
	? (initialThinking as PersonaThinking)
	: PERSONAS[persona].thinking;

function setThinking(level: PersonaThinking, pi: ExtensionAPI) {
	thinking = level;
	pi.setThinkingLevel(level);
}

/** Coding leaves the system prompt as pi built it; talking injects the persona. */
type VoiceMode = "coding" | "talking";
const VOICE_MODES: VoiceMode[] = ["coding", "talking"];
let mode: VoiceMode =
	VOICE_MODES.includes((process.env.PI_TTS_MODE ?? saved.mode ?? "coding") as VoiceMode)
		? ((process.env.PI_TTS_MODE ?? saved.mode ?? "coding") as VoiceMode)
		: "coding";

const VOLUME_STEPS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0];
const MAX_CHARS_STEPS = [300, 700, 1200, 2500];
const STREAM_INTERVAL_STEPS = [0.5, 1.0, 2.0];
const VOLUME_VALUES = VOLUME_STEPS.map((step) => step.toFixed(2));
const MAX_CHARS_VALUES = MAX_CHARS_STEPS.map(String);
const STREAM_INTERVAL_VALUES = STREAM_INTERVAL_STEPS.map((step) => step.toFixed(1));

let volume = snap(process.env.PI_TTS_VOLUME ?? saved.volume, VOLUME_STEPS, 0.1);
let maxChars = snap(process.env.PI_TTS_MAX_CHARS ?? saved.maxChars, MAX_CHARS_STEPS, 700);
let streamInterval = snap(
	process.env.PI_TTS_STREAM_INTERVAL ?? saved.streamInterval,
	STREAM_INTERVAL_STEPS,
	1.0,
);

/** What goes into <persona>: the personality, then the thinking rule.
 *  Coding mode injects nothing at all; a persona with no personality
 *  (no personality) stays quiet unless the thinking level needs saying. */
function personaSection(): string {
	if (mode === "coding") return "";
	const target = PERSONAS[persona];
	if (!target) return "";
	const parts: string[] = [];
	if (target.character) parts.push(target.character, NO_ANNOUNCE);
	if (thinking === "minimal") parts.push(MINIMAL_THINKING);
	return parts.join("\n\n");
}

/** Reference of a persona, with the explicit env vars pinning the starting one. */
function personaRef(name: string): Persona {
	const base = PERSONAS[name] ?? PERSONAS[persona]!;
	return {
		...base,
		refAudio: process.env.PI_TTS_REF_AUDIO ?? base.refAudio,
		refText: process.env.PI_TTS_REF_TEXT ?? base.refText,
		lang: process.env.PI_TTS_LANG ?? base.lang,
	};
}

const STT_CONFIG = {
	helper: join(HERE, "pi_stt.py"),
	shortcut: (process.env.PI_TTS_STT_SHORTCUT ?? "ctrl+i") as KeyId,
	cancelShortcut: (process.env.PI_TTS_STT_CANCEL_SHORTCUT ?? "ctrl+shift+i") as KeyId,
	maxSeconds: Number(process.env.PI_TTS_STT_MAX_SECONDS ?? 60),
};

const STT_STATUS = "pi-tts-dictation";

/** Trace file for the helpers: dictation, the shortcut, the renderer, and the two processes. */
const DEBUG_LOG = join(tmpdir(), "pi-tts", "helper.log");
const DEBUG = process.env.PI_TTS_DEBUG !== "0";

function debug(...parts: unknown[]): void {
	if (!DEBUG) return;
	const line = parts
		.map((part) => (typeof part === "string" ? part : JSON.stringify(part)))
		.join(" ");
	try {
		appendFileSync(DEBUG_LOG, `${new Date().toISOString().slice(11, 23)} ${line}\n`);
	} catch {
		/* logging must never break the voice */
	}
}

/** How many lines the settings window keeps for the description of a row. */
const SETTINGS_HELP_LINES = 3;

/**
 * The helpers log every request to stderr. Inherited, that prints under the TUI
 * and piles up at the bottom of the screen, so it is piped to the trace file
 * instead, keeping the tail for the notice when a helper dies.
 */
function drainStderr(
	child: ChildProcessByStdio<Writable | null, Readable, Readable>,
	tag: string,
	tail: string[],
): void {
	let partial = "";
	child.stderr.on("data", (chunk: Buffer) => {
		partial += chunk.toString("utf8");
		const lines = partial.split("\n");
		partial = lines.pop() ?? "";
		for (const line of lines) {
			if (!line.trim()) continue;
			tail.push(line);
			if (tail.length > 12) tail.shift();
			debug(tag, line);
		}
	});
	child.stderr.resume();
}

/** The last words of a helper, for the message about it dying. */
function lastError(tail: string[]): string {
	return tail.length ? `: ${tail[tail.length - 1]}` : "";
}

// Shortest stretch of prose worth a synthesis request on its own; shorter
// fragments would turn one sentence into several clipped utterances.
const MIN_CHUNK = 24;

let enabled = onOff(saved.voice, process.env.PI_TTS_VOICE_ENABLED !== "0");
let lastAssistantText = "";
let ctxRef: ExtensionContext | undefined;
let raw = ""; // streamed text not yet handed to the queue
let streamedAny = false; // current assistant message delivered deltas
let spokeThisRun = false; // something was spoken live, so settle need not repeat it
let helper: Helper | null = null;
let starting: Promise<Helper> | null = null;
let queue: Promise<void> = Promise.resolve();
let requestId = 0;
let speechToken = 0;
const pending = new Map<string, (result: SpeechResult | null) => void>();
let stdoutBuffer = "";

type SpeechResult = {
	cmd: string;
	path?: string;
	seconds?: number;
	text?: string;
	stopped?: boolean;
	error?: string;
};

/** Keep only speakable prose: drop code, tables, markup and links. */
function stripMarkup(raw: string): string {
	return raw
		.replace(/```[\s\S]*?```/g, " code block omitted. ")
		.replace(/`([^`]*)`/g, "$1")
		.replace(/^\s*([-*+]|\d+\.)\s+/gm, "")
		.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
		.replace(/https?:\/\/\S+/g, "link omitted")
		.replace(/[*_#>|]/g, " ")
		.replace(/\s+/g, " ")
		.trim();
}

/** What the synthesizer is asked to say: prose, cut to one utterance. */
function toSpeech(raw: string): string {
	let text = stripMarkup(raw);
	if (text.length > maxChars) {
		const cut = text.slice(0, maxChars);
		const stop = Math.max(cut.lastIndexOf(". "), cut.lastIndexOf("! "), cut.lastIndexOf("? "));
		text = (stop > maxChars * 0.3 ? cut.slice(0, stop + 1) : `${cut}…`).trim();
	}
	return text;
}

function messageText(message: { role: string; content?: unknown }): string {
	if (message.role !== "assistant" || !Array.isArray(message.content)) return "";
	return message.content
		.filter((part): part is { type: "text"; text: string } => {
			return !!part && typeof part === "object" && (part as { type?: string }).type === "text";
		})
		.map((part) => part.text)
		.join(" ")
		.trim();
}

function extractText(messages: Array<{ role: string; content?: unknown }>): string {
	for (let i = messages.length - 1; i >= 0; i--) {
		const text = messageText(messages[i]);
		if (text) return text;
	}
	return "";
}

/** A cut is unsafe if it lands inside inline code, a link label or a target. */
function isCleanCut(text: string): boolean {
	const count = (char: string) => text.split(char).length - 1;
	if (count("`") % 2) return false;
	if (count("[") !== count("]")) return false;
	return count("(") === count(")");
}

/** End of the longest speakable sentence, or -1 to keep waiting for more. */
function findCut(text: string): number {
	// Never speak into an unterminated code fence.
	const limit = text.split("```").length % 2 === 0 ? text.lastIndexOf("```") : text.length;
	const sentence = /[.!?]+(?=\s|$)/g;
	let cut = -1;
	let match: RegExpExecArray | null;
	while ((match = sentence.exec(text)) !== null) {
		const end = match.index + match[0].length;
		if (end > limit) break;
		if (end < MIN_CHUNK || !isCleanCut(text.slice(0, end))) continue;
		cut = end;
	}
	if (cut === -1 && text.length > maxChars) {
		const space = text.lastIndexOf(" ", maxChars);
		if (space > MIN_CHUNK) cut = space + 1;
	}
	return cut;
}

/** Hand every complete sentence in the buffer to the synthesizer. */
function pump(flush = false): void {
	while (raw) {
		const cut = flush ? raw.length : findCut(raw);
		if (cut <= 0) return;
		const piece = raw.slice(0, cut);
		raw = raw.slice(cut);
		const speech = toSpeech(piece);
		if (speech) {
			spokeThisRun = true;
			enqueue(speech);
		}
	}
}

/** Level meter for the status line; rms is a 0..1 amplitude. */
function meter(rms: number): string {
	const db = 20 * Math.log10(Math.max(rms, 1e-7));
	const filled = Math.max(0, Math.min(12, Math.round(((db + 60) / 54) * 12)));
	return "█".repeat(filled) + "░".repeat(12 - filled);
}

/** "●" waits in the input, "▶" goes straight to the agent, "◌" is listening on its own. */
function listeningText(seconds: number | undefined, rms: number, auto = false): string {
	const glyph = auto ? "◌" : dictationMode === "conversation" ? "▶" : "●";
	return `${glyph} ${seconds?.toFixed(1) ?? "0.0"}s ${meter(rms)} ${STT_CONFIG.shortcut} stop, ${STT_CONFIG.cancelShortcut} cancel`;
}

function readStdout(chunk: Buffer) {
	stdoutBuffer += chunk.toString("utf8");
	const lines = stdoutBuffer.split("\n");
	stdoutBuffer = lines.pop() ?? "";
	for (const line of lines) {
		if (!line.trim()) continue;
		let message: { id?: string; cmd?: string; path?: string; error?: string };
		try {
			message = JSON.parse(line);
		} catch {
			debug("tts < raw", line.slice(0, 120));
			continue; // model loader chatter on stdout
		}
		debug("tts <", message.cmd, message.id ?? "");
		// audio_start is progress only; the helper plays the audio itself.
		if (message.id && pending.has(message.id) && message.cmd !== "audio_start") {
			const resolve = pending.get(message.id)!;
			pending.delete(message.id);
			resolve(message as SpeechResult);
		}
	}
}

function startHelper(): Promise<Helper> {
	if (helper) return Promise.resolve(helper);
	if (starting) return starting;

	starting = new Promise<Helper>((resolve, reject) => {
		const ref = personaRef(persona);
		const ttsErrors: string[] = [];
		const child = spawn(CONFIG.python, [CONFIG.helper], {
			stdio: ["pipe", "pipe", "pipe"],
			env: {
				...process.env,
				QWEN_TTS_MODEL: MODEL_DIR,
				QWEN_TTS_REF_AUDIO: ref.refAudio,
				QWEN_TTS_REF_TEXT: ref.refText,
				QWEN_TTS_LANG: ref.lang,
				QWEN_TTS_VOLUME: String(volume),
				QWEN_TTS_STREAM_INTERVAL: String(streamInterval),
			},
		});
		child.stdout.on("data", readStdout);
		drainStderr(child, "tts stderr", ttsErrors);
		child.on("exit", (code) => {
			helper = null;
			starting = null;
			for (const resolve of pending.values()) resolve(null);
			pending.clear();
			if (code && ctxRef)
				ctxRef.ui.notify(`voice helper stopped (${code})${lastError(ttsErrors)}`, "error");
		});
		child.on("error", (error) => {
			helper = null;
			starting = null;
			reject(error);
		});
		// First stdout line is {"cmd":"ready"}; the model load takes a few seconds.
		const onReady = (chunk: Buffer) => {
			if (chunk.toString("utf8").includes('"ready"')) {
				child.stdout.off("data", onReady);
				helper = child;
				starting = null;
				resolve(child);
			}
		};
		child.stdout.on("data", onReady);
	}).catch((error) => {
		starting = null;
		throw error;
	});

	return starting;
}

/** Let the current utterance finish, but drop everything queued behind it. */
function dropQueued() {
	speechToken++;
	if (helper) {
		try {
			helper.stdin.write(`${JSON.stringify({ cmd: "clear" })}\n`);
		} catch {
			/* ignore */
		}
	}
}

/** Cut the current utterance off and drop anything still queued behind it. */
function interrupt() {
	speechToken++;
	if (helper) {
		try {
			helper.stdin.write(`${JSON.stringify({ cmd: "stop" })}\n`);
		} catch {
			/* ignore */
		}
	}
}

function stopAll() {
	speechToken++;
	if (helper) {
		try {
			helper.stdin.write(`${JSON.stringify({ cmd: "quit" })}\n`);
		} catch {
			/* ignore */
		}
		helper.kill();
		helper = null;
		starting = null;
	}
}

/** Turn the voice on or off; on again means loading the model again. */
function setVoiceEnabled(on: boolean) {
	enabled = on;
	if (!on) stopAll();
	else void startHelper().catch(() => {});
}

/** Coding keeps pi's prompt; talking injects the persona. */
function setMode(next: VoiceMode, ctx: ExtensionContext) {
	mode = next;
	ctx.ui.notify(
		mode === "talking"
			? "mode: talking — the persona is injected in the system prompt"
			: "mode: coding — the system prompt is left as pi builds it",
		"info",
	);
}

/** Push the audio knobs to the running helper; no model reload. */
function tuneHelper() {
	if (!helper) return;
	try {
		helper.stdin.write(
			`${JSON.stringify({ cmd: "tune", volume, streaming_interval: streamInterval })}\n`,
		);
	} catch {
		/* the helper is going away */
	}
}

/** Everything the settings window controls, written out as the saved defaults. */
function currentSettings(): PersonaSettings {
	return {
		voice: enabled ? "on" : "off",
		mode,
		persona,
		thinking,
		volume: volume.toFixed(2),
		maxChars: String(maxChars),
		streamInterval: streamInterval.toFixed(1),
		dictation: sttEnabled ? "on" : "off",
		dictationMode,
		autoListen: autoListen ? "on" : "off",
	};
}

async function speak(text: string, ctx?: ExtensionContext, token = speechToken): Promise<void> {
	const speech = toSpeech(text);
	if (!speech || token !== speechToken) return;

	const child = await startHelper().catch(async (error) => {
		ctx?.ui.notify(`pi-tts: could not start ${CONFIG.python} ${CONFIG.helper}: ${error}`, "error");
		return null;
	});
	if (!child) return;
	if (token !== speechToken) return;

	const id = String(++requestId);
	// The helper streams decoded chunks to the output device, so this resolves
	// only once the utterance has finished playing (or was stopped).
	await new Promise<SpeechResult | null>((resolve) => {
		pending.set(id, resolve);
		child.stdin.write(`${JSON.stringify({ id, cmd: "say", text: speech })}\n`);
	});
}

function enqueue(text: string, ctx?: ExtensionContext) {
	if (!enabled) return;
	if (ctx) ctxRef = ctx;
	const token = speechToken;
	queue = queue.then(() => speak(text, ctx, token)).catch(() => {});
}

/** Longest stretch a file is allowed to speak; a paper is an afternoon. */
const READ_MAX_CHARS = Number(process.env.PI_TTS_READ_MAX_CHARS ?? 12000);

/**
 * Cut a document into utterances: at sentence ends, packed up to maxChars each.
 * findCut is the streaming cut and deliberately takes the longest complete
 * sentence it can see, which on a whole file would be the whole file.
 */
function splitForSpeech(text: string, limit = maxChars): string[] {
	const pieces: string[] = [];
	let rest = text;
	while (rest.trim()) {
		const sentence = /[.!?]+(?=\s|$)/g;
		let cut = -1;
		let match: RegExpExecArray | null;
		while ((match = sentence.exec(rest)) !== null) {
			const end = match.index + match[0].length;
			if (end > limit) break;
			if (end < MIN_CHUNK || !isCleanCut(rest.slice(0, end))) continue;
			cut = end;
		}
		if (cut === -1) {
			// No sentence end inside the limit: cut at a space, or say what is left.
			const space = rest.lastIndexOf(" ", limit);
			cut = space > MIN_CHUNK ? space + 1 : rest.length;
		}
		pieces.push(rest.slice(0, cut));
		rest = rest.slice(cut);
	}
	return pieces;
}

/**
 * Text of a file. Plain text and markdown are read here, since the synthesizer
 * strips the markup anyway; a PDF needs a parser, which the voice helper already
 * has in the process, so it is asked for it over stdin.
 */
async function fileText(path: string, ctx: ExtensionContext): Promise<string | null> {
	if (!/\.pdf$/i.test(path)) {
		try {
			return readFileSync(path, "utf8");
		} catch (error) {
			ctx.ui.notify(`pi-tts: could not read ${displayPath(path)}: ${error}`, "error");
			return null;
		}
	}
	const child = await startHelper().catch(() => null);
	if (!child) {
		ctx.ui.notify("pi-tts: the voice is not running, so the pdf cannot be unpacked", "error");
		return null;
	}
	const id = String(++requestId);
	const result = await new Promise<SpeechResult | null>((resolve) => {
		pending.set(id, resolve);
		child.stdin.write(`${JSON.stringify({ id, cmd: "extract", path })}\n`);
	});
	if (!result || result.cmd === "error") {
		ctx.ui.notify(`pi-tts: ${result?.error ?? "no text came back from the pdf"}`, "error");
		return null;
	}
	return result.text ?? "";
}

/**
 * Read a file aloud, cut into sentences the way a streamed reply is, so a new
 * prompt cuts it off rather than queueing up behind page nine. `/persona stop`
 * works, and so does simply typing over it.
 */
async function readAloud(args: string, ctx: ExtensionContext) {
	if (!enabled) return ctx.ui.notify("the voice is off — /persona on first", "info");
	const asked = (args ?? "").trim() || (await ctx.ui.input("read which file?", "~/notes.md"));
	if (!asked) return;
	const path = expandPath(asked, process.cwd());
	if (!existsSync(path)) return ctx.ui.notify(`no file at ${path}`, "error");

	const text = (await fileText(path, ctx))?.trim();
	if (!text) {
		if (text === null) return; // fileText already said why
		return ctx.ui.notify(`${displayPath(path)} has no text in it`, "warning");
	}

	const spoken = text.length > READ_MAX_CHARS ? text.slice(0, READ_MAX_CHARS) : text;
	ctx.ui.notify(
		text.length > spoken.length
			? `reading the first ${spoken.length} of ${text.length} characters of ${displayPath(path)} — PI_TTS_READ_MAX_CHARS raises it`
			: `reading ${displayPath(path)} — ${spoken.length} characters, /persona stop to stop`,
		"info",
	);
	for (const piece of splitForSpeech(spoken)) {
		if (toSpeech(piece)) enqueue(piece, ctx);
	}
}

/* ---------------------------------------------------------------- render to a file */

/**
 * Where rendered audio lands, and how much of a document it takes. A render is
 * not held to the live voice's limits: the whole text is the point, the pieces
 * are longer because nothing has to start quickly, and the file is played
 * somewhere else, so it is written at full gain rather than whispered.
 */
const RENDER_DIR = process.env.PI_TTS_RENDER_DIR ?? join(CACHE_DIR, "rendered");
/** mp3 is a sixth of the wav and every player takes it; a named output still wins. */
const RENDER_FORMAT = (process.env.PI_TTS_RENDER_FORMAT ?? "mp3").toLowerCase() === "wav" ? "wav" : "mp3";
const RENDER_PIECE_CHARS = Number(process.env.PI_TTS_RENDER_PIECE_CHARS ?? 1200);
const RENDER_MAX_CHARS = Number(process.env.PI_TTS_RENDER_MAX_CHARS ?? 200000);
const RENDER_VOLUME = Math.min(1, Math.max(0, Number(process.env.PI_TTS_RENDER_VOLUME ?? 1)));
const RENDER_STATUS = "pi-tts-render";
/** Width of the bar in cells. */
const RENDER_BAR = 28;

/**
 * What a document costs, until this session has rendered one and can answer for
 * itself: how fast the voice speaks, and how much faster than realtime the
 * machine synthesizes. Both are guesses until a render has landed, and both are
 * blended towards the truth by every render that does.
 */
const SPEECH_RATE = Number(process.env.PI_TTS_RENDER_SPEECH_RATE ?? 16);
const RENDER_SPEED = Number(process.env.PI_TTS_RENDER_SPEED ?? 2.4);
let speechRate = SPEECH_RATE > 0 ? SPEECH_RATE : 16;
let renderSpeed = RENDER_SPEED > 0 ? RENDER_SPEED : 2.4;
let ratesAreMeasured = false;
/** What a minute of audio costs, per encoder. */
const KB_PER_MINUTE: Record<string, number> = { mp3: 480, wav: 2880 };

/** What the renderer prints on stdout, one object per line. */
type RenderMessage = {
	cmd: string;
	out?: string;
	path?: string;
	pieces?: number;
	piece?: number;
	chars?: number;
	chars_done?: number;
	seconds?: number;
	est_seconds?: number;
	elapsed?: number;
	error?: string;
};

type RenderJob = {
	child: ChildProcessByStdio<null, Readable, Readable>;
	/** What is being read, as the user asked for it. */
	label: string;
	out: string;
	pieces: number;
	piece: number;
	chars: number;
	charsDone: number;
	/** Characters the renderer was given, which is what the next estimate wants. */
	speechChars: number;
	/** Seconds of audio written so far, and what the whole thing is expected to be. */
	seconds: number;
	estSeconds: number;
	began: number;
	stopping: boolean;
	finished: boolean;
};

/** The one render in flight; a second would only slow the first down. */
let rendering: RenderJob | null = null;

/** What the live voice says, in the persona, when a render lands. */
const RENDER_DONE_LINE = "File processing complete.";

/** 4:07, or 1:02:11 past an hour. */
function duration(seconds: number): string {
	const total = Math.max(0, Math.round(seconds));
	const hours = Math.floor(total / 3600);
	const minutes = Math.floor((total % 3600) / 60);
	const secs = String(total % 60).padStart(2, "0");
	return hours
		? `${hours}:${String(minutes).padStart(2, "0")}:${secs}`
		: `${minutes}:${secs}`;
}

/** `2025-10-11-1104`, for a file name that does not collide with the last run. */
function stamp(): string {
	const now = new Date().toISOString();
	return `${now.slice(0, 10)}-${now.slice(11, 16).replace(":", "")}`;
}

/** "2 pieces", "1 piece". */
function count(n: number, noun: string): string {
	return `${n} ${noun}${n === 1 ? "" : "s"}`;
}

/** 412 KB, 19.4 MB. */
function formatBytes(bytes: number): string {
	return bytes >= 1024 * 1024
		? `${(bytes / 1024 / 1024).toFixed(1)} MB`
		: `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

/** 412 KB, 19.4 MB — what the rendered file cost, or "" when it is not there. */
function fileSize(path: string): string {
	try {
		return formatBytes(statSync(path).size);
	} catch {
		return "";
	}
}

/**
 * The bar, above the editor: the render is a process of its own, so the session
 * keeps talking and typing underneath it, and the bar is the only thing that has
 * to say where it got to.
 */
function paintRender(ctx: ExtensionContext): void {
	const job = rendering;
	if (!job || ctx.mode !== "tui") return;
	const theme = ctx.ui.theme;
	const share = job.chars ? Math.min(1, job.charsDone / job.chars) : 0;
	const filled = Math.round(share * RENDER_BAR);
	const elapsed = (Date.now() - job.began) / 1000;
	ctx.ui.setWidget(RENDER_STATUS, [
		` ${theme.fg("toolTitle", theme.bold(`rendering ${job.label}`))} ${theme.fg("muted", `→ ${displayPath(job.out)}`)}`,
		` ${theme.fg(share >= 1 ? "success" : "accent", "█".repeat(filled) + "░".repeat(RENDER_BAR - filled))} ` +
			`${theme.bold(`${Math.round(share * 100)}%`)} ${theme.fg(
				"muted",
				`${job.piece}/${job.pieces} pieces · ${duration(job.seconds)} of ~${duration(
					job.estSeconds,
				)} audio · ${duration(elapsed)} elapsed · /persona render stop`,
			)}`,
	]);
}

function applyRenderMessage(message: RenderMessage, ctx: ExtensionContext): void {
	const job = rendering;
	if (!job) return;
	if (message.cmd === "start") {
		job.pieces = message.pieces ?? job.pieces;
		job.chars = message.chars ?? job.chars;
		job.estSeconds = message.est_seconds ?? job.estSeconds;
	} else if (message.cmd === "progress") {
		job.piece = message.piece ?? job.piece;
		job.charsDone = message.chars_done ?? job.charsDone;
		job.seconds = message.seconds ?? job.seconds;
		job.estSeconds = message.est_seconds ?? job.estSeconds;
	}
	paintRender(ctx);
}

/** Kill the renderer; SIGTERM lets it close the wav with what it has written. */
function stopRender(ctx: ExtensionContext) {
	if (!rendering) return ctx.ui.notify("nothing is rendering", "info");
	rendering.stopping = true;
	rendering.child.kill("SIGTERM");
	ctx.ui.notify("stopping the render — the file keeps the audio it has so far", "info");
}

function renderStatus(ctx: ExtensionContext) {
	const job = rendering;
	if (!job) return ctx.ui.notify("nothing is rendering", "info");
	ctx.ui.notify(
		`rendering ${job.label} → ${displayPath(job.out)}: ${job.piece}/${job.pieces} pieces, ` +
			`${duration(job.seconds)} of ~${duration(job.estSeconds)} audio after ${duration(
				(Date.now() - job.began) / 1000,
			)}`,
		"info",
	);
}

/**
 * Fold a render into the rates the estimates are made of, so the next `measure`
 * answers with this machine's numbers instead of a guess. Short jobs are not
 * evidence of anything: startup and the first pieces are the slow ones.
 */
function learnRate(job: RenderJob, elapsed: number): void {
	if (job.seconds < 20 || elapsed <= 0 || !job.speechChars) return;
	const blend = (was: number, now: number) => 0.7 * was + 0.3 * now;
	renderSpeed = blend(renderSpeed, job.seconds / elapsed);
	speechRate = blend(speechRate, job.speechChars / job.seconds);
	ratesAreMeasured = true;
}

/**
 * `/persona render measure <file>`: what the job would cost, without loading a
 * model to find out. The text is cut exactly as the renderer would cut it, then
 * priced at the speaking rate and throughput on hand.
 */
async function measureRender(args: string, ctx: ExtensionContext) {
	const asked = (args ?? "").trim() || (await ctx.ui.input("measure which file?", "~/notes.md"));
	if (!asked) return;
	const path = expandPath(asked.split(/\s+/)[0]!, process.cwd());
	if (!existsSync(path)) return ctx.ui.notify(`no file at ${path}`, "error");

	const text = (await fileText(path, ctx))?.trim();
	if (!text) {
		if (text === null) return; // fileText already said why
		return ctx.ui.notify(`${displayPath(path)} has no text in it`, "warning");
	}
	const source = text.length > RENDER_MAX_CHARS ? text.slice(0, RENDER_MAX_CHARS) : text;
	const pieces = splitForSpeech(source, RENDER_PIECE_CHARS)
		.map(stripMarkup)
		.filter(Boolean);
	if (!pieces.length) return ctx.ui.notify(`nothing speakable in ${displayPath(path)}`, "warning");

	const chars = pieces.reduce((sum, piece) => sum + piece.length, 0);
	const audio = chars / speechRate;
	const bytes = (audio / 60) * (KB_PER_MINUTE[RENDER_FORMAT] ?? KB_PER_MINUTE.mp3!) * 1024;
	ctx.ui.notify(
		`${displayPath(path)}: ${chars} speakable characters in ${count(pieces.length, "piece")} — about ` +
			`${duration(audio)} of audio, ${duration(audio / renderSpeed)} to render at ` +
			`${renderSpeed.toFixed(1)}× realtime, ${formatBytes(bytes)} as ${RENDER_FORMAT}` +
			(ratesAreMeasured ? " (priced on this machine's own renders)" : "") +
			(text.length > source.length
				? `; only the first ${RENDER_MAX_CHARS} characters are taken, PI_TTS_RENDER_MAX_CHARS raises it`
				: ""),
		"info",
	);
}

/**
 * Turn a file into an audio file: a second helper process, holding its own copy
 * of the model, synthesizing the whole document while this session carries on.
 * It is not the live voice, so it runs with the voice off, it is not cut to
 * PI_TTS_READ_MAX_CHARS, and nothing it does makes a sound here.
 */
async function renderFile(path: string, out: string, ctx: ExtensionContext) {
	const text = (await fileText(path, ctx))?.trim();
	if (!text) {
		if (text === null) return; // fileText already said why
		return ctx.ui.notify(`${displayPath(path)} has no text in it`, "warning");
	}
	const source = text.length > RENDER_MAX_CHARS ? text.slice(0, RENDER_MAX_CHARS) : text;
	const pieces = splitForSpeech(source, RENDER_PIECE_CHARS)
		.map(stripMarkup)
		.filter(Boolean);
	if (!pieces.length) return ctx.ui.notify(`nothing speakable in ${displayPath(path)}`, "warning");
	const speechChars = pieces.reduce((sum, piece) => sum + piece.length, 0);
	const target =
		out ||
		join(RENDER_DIR, `${basename(path).replace(/\.[^.]+$/, "")}-${stamp()}.${RENDER_FORMAT}`);
	const jobFile = join(tmpdir(), "pi-tts", `render-${process.pid}-${Date.now()}.json`);
	const voice = personaRef(persona);
	try {
		mkdirSync(dirname(jobFile), { recursive: true });
		writeFileSync(
			jobFile,
			JSON.stringify({
				out: target,
				pieces,
				ref_audio: voice.refAudio,
				ref_text: voice.refText,
				lang: voice.lang,
				volume: RENDER_VOLUME,
			}),
		);
	} catch (error) {
		return ctx.ui.notify(`pi-tts: could not write the render job: ${error}`, "error");
	}

	const errors: string[] = [];
	const child = spawn(CONFIG.python, [CONFIG.helper, "--render", "--job", jobFile], {
		stdio: ["ignore", "pipe", "pipe"],
		env: { ...process.env, QWEN_TTS_MODEL: MODEL_DIR },
	});
	const job: RenderJob = {
		child,
		label: displayPath(path),
		out: target,
		pieces: pieces.length,
		piece: 0,
		chars: source.length,
		charsDone: 0,
		speechChars,
		seconds: 0,
		estSeconds: 0,
		began: Date.now(),
		stopping: false,
		finished: false,
	};
	rendering = job;

	let buffer = "";
	child.stdout.on("data", (chunk: Buffer) => {
		buffer += chunk.toString("utf8");
		const lines = buffer.split("\n");
		buffer = lines.pop() ?? "";
		for (const line of lines) {
			if (!line.trim()) continue;
			let message: RenderMessage;
			try {
				message = JSON.parse(line) as RenderMessage;
			} catch {
				debug("render < raw", line.slice(0, 120));
				continue; // model loader chatter on stdout
			}
			debug("render <", message.cmd, message.piece ?? "");
			if (message.cmd === "error") ctx.ui.notify(`pi-tts: ${message.error}`, "error");
			else applyRenderMessage(message, ctx);
		}
	});
	drainStderr(child, "render stderr", errors);

	const finish = (line: string, kind: "info" | "error" | "warning") => {
		if (job.finished) return;
		job.finished = true;
		if (rendering === job) rendering = null;
		if (ctx.mode === "tui") ctx.ui.setWidget(RENDER_STATUS, undefined);
		try {
			unlinkSync(jobFile);
		} catch {
			/* the job file is only the scratch it read from */
		}
		ctx.ui.notify(line, kind);
	};

	child.on("exit", (code) => {
		const audio = duration(job.seconds);
		const elapsed = (Date.now() - job.began) / 1000;
		learnRate(job, elapsed);
		if (job.stopping)
			return finish(`render stopped: ${audio} of audio in ${displayPath(target)}`, "info");
		if (code) return finish(`pi-tts: render failed (${code})${lastError(errors)}`, "error");
		const parts = [
			`${audio} of audio`,
			fileSize(target),
			`written in ${duration(elapsed)}`,
			`voice: ${voice.label}`,
		].filter(Boolean);
		finish(`rendered ${displayPath(target)} — ${parts.join(", ")}`, "info");
		// The file is finished somewhere out of sight, so the voice is what says so.
		enqueue(RENDER_DONE_LINE, ctx);
	});
	child.on("error", (error) =>
		finish(`pi-tts: could not start the renderer: ${error}`, "error"),
	);

	ctx.ui.notify(
		`rendering ${count(pieces.length, "piece")} of ${displayPath(path)} (~${duration(
			speechChars / speechRate,
		)} of audio, ~${duration(speechChars / speechRate / renderSpeed)} to go) → ${displayPath(
			target,
		)}; the voice keeps working while it runs`,
		"info",
	);
	paintRender(ctx);
}

/** `/persona render [stop|status|measure <file>|<file> [out]]`. */
async function renderAloud(args: string, ctx: ExtensionContext) {
	const [first, ...rest] = (args ?? "").trim().split(/\s+/);
	const sub = (first ?? "").toLowerCase();
	if (sub === "measure") return measureRender(rest.join(" "), ctx);
	if (sub === "stop") return stopRender(ctx);
	if (sub === "status") return renderStatus(ctx);

	const asked = (args ?? "").trim() || (await ctx.ui.input("render which file to audio?", "~/notes.md"));
	if (!asked) return;
	if (rendering)
		return ctx.ui.notify(
			`already rendering ${rendering.label} — /persona render stop first`,
			"warning",
		);
	const [file, ...outParts] = asked.split(/\s+/);
	const path = expandPath(file, process.cwd());
	if (!existsSync(path)) return ctx.ui.notify(`no file at ${path}`, "error");
	const outArg = outParts.join(" ").trim();
	const out = outArg ? expandPath(outArg, process.cwd()) : "";
	await renderFile(path, out, ctx);
}

/** Home-relative where possible, so the file stays readable on another machine. */
function displayPath(path: string): string {
	const home = homedir();
	return path === home || path.startsWith(`${home}/`) ? `~/${relative(home, path)}` : path;
}

/** The personas file as an object, so one entry can be written without losing the rest. */
function readPersonasFile(): Record<string, unknown> {
	try {
		const raw: unknown = JSON.parse(readFileSync(PERSONAS_FILE, "utf8"));
		return raw && typeof raw === "object" && !Array.isArray(raw)
			? (raw as Record<string, unknown>)
			: {};
	} catch {
		return {}; // no file yet, or one that is not JSON
	}
}

/** A loaded voice back out as a file entry, so it can be rewritten whole. */
function personaSpec(key: string, target: Persona): PersonaSpec {
	return {
		label: target.label,
		refAudio: displayPath(target.refAudio),
		refText: target.refText,
		lang: target.lang,
		sample: target.sample,
		thinking: target.thinking,
		character: target.character,
	};
}

/** What is filed for one voice, as the file has it: an edit starts from here. */
function filedPersona(key: string): PersonaSpec {
	const raw = readPersonasFile()[key];
	if (raw && typeof raw === "object" && !Array.isArray(raw)) return { ...(raw as PersonaSpec) };
	const loaded = PERSONAS[key];
	return loaded ? personaSpec(key, loaded) : {};
}

/** Write the file back whole; returns the error text, or null when it landed. */
function writePersonasFile(file: Record<string, unknown>): string | null {
	try {
		mkdirSync(dirname(PERSONAS_FILE), { recursive: true });
		writeFileSync(PERSONAS_FILE, `${JSON.stringify(file, null, "\t")}\n`);
		return null;
	} catch (error) {
		return String(error);
	}
}

/**
 * Make an entry that was just written live in this session, so /persona <name>
 * works without a restart, and forget what this key complained about at startup:
 * the edit was probably about exactly that.
 */
function registerPersona(key: string, spec: PersonaSpec): void {
	const entry = normalizePersona(key, spec, dirname(PERSONAS_FILE));
	if (entry) PERSONAS[key] = entry;
	for (let i = PERSONA_WARNINGS.length - 1; i >= 0; i--) {
		if (PERSONA_WARNINGS[i]!.includes(`"${key}"`) || PERSONA_WARNINGS[i]!.includes("no personas file"))
			PERSONA_WARNINGS.splice(i, 1);
	}
}

/**
 * Ask for a voice one question at a time and hand back the entry: the clip, what
 * the clip says, a name, and the personality to speak with. Only the clip is
 * required, and a path that does not exist is asked for again rather than written
 * down to fail later. Esc at any question gives up without writing.
 *
 * `current` is the entry being edited. Each question shows what is filed now and
 * an empty answer keeps it, so one field can be fixed without retyping the rest,
 * and a different name in the first answer renames the voice.
 */
async function askPersona(
	ctx: ExtensionContext,
	current?: { key: string; spec: PersonaSpec },
): Promise<{ key: string; spec: PersonaSpec } | null> {
	const spec = current?.spec ?? {};
	const cancel = () => {
		ctx.ui.notify(
			current
				? `edit "${current.key}": nothing written`
				: "add persona: nothing written",
			"info",
		);
		return null;
	};
	// The placeholder is what an empty answer gets, which is how an edit keeps the
	// fields it did not touch; `hint` says what to type when nothing is filed yet,
	// and `fallback` is what an empty answer means for a new entry.
	const orDefault = async (title: string, value: string | undefined, hint: string, fallback = "") => {
		const answer = await ctx.ui.input(title, value || hint);
		return answer === undefined ? undefined : answer.trim() || value || fallback;
	};

	const asked = await ctx.ui.input(
		current ? `name this voice — "${current.key}" unless you mean to rename it` : "name this voice",
		current ? current.key : "what you type after /persona, e.g. narrator",
	);
	if (asked === undefined) return cancel();
	const key = asked
		.toLowerCase()
		.trim()
		.replace(/[^a-z0-9._-]+/g, "-")
		.replace(/^[-.]+|[-.]+$/g, "");
	if (!key) {
		ctx.ui.notify(`"${asked}" has no letters or digits in it`, "error");
		return null;
	}
	if (key !== current?.key && PERSONAS[key] && !(await ctx.ui.confirm("persona exists", `replace "${key}"?`)))
		return cancel();

	const filed = spec.refAudio ?? "";
	let refAudio = "";
	for (let attempt = 0; attempt < 3 && !refAudio; attempt++) {
		const answer = await ctx.ui.input(
			attempt ? "reference audio, again" : "path to the reference audio",
			filed || "a few clean seconds of one voice, e.g. ~/.cache/pi-tts/ref.wav",
		);
		if (answer === undefined) return cancel();
		// An empty answer keeps the clip already on the entry; a new one has to
		// exist before it is written down.
		const path = answer.trim() ? expandPath(answer.trim(), dirname(PERSONAS_FILE)) : filed;
		if (!path) continue;
		if (!existsSync(path)) {
			ctx.ui.notify(`no audio at ${path}`, "error");
			continue;
		}
		refAudio = path;
	}
	if (!refAudio) {
		ctx.ui.notify("add persona: no reference audio to clone", "error");
		return null;
	}

	const refText = await orDefault(
		"what does the clip say?",
		spec.refText,
		"word for word, with the pauses punctuated",
	);
	if (refText === undefined) return cancel();
	if (!refText) ctx.ui.notify("no transcript: the voice will come out flatter", "warning");

	const label = await orDefault("label shown in /persona", spec.label, key, key);
	if (label === undefined) return cancel();
	const lang = await orDefault("language", spec.lang, "english", "english");
	if (lang === undefined) return cancel();
	const sample = await orDefault(
		"line to speak when you switch to it",
		spec.sample,
		"Testing complete. The voice is working.",
		"Testing complete. The voice is working.",
	);
	if (sample === undefined) return cancel();
	const level = (await ctx.ui.select(
		`how much this voice may think before it answers${
			spec.thinking ? ` (now: ${spec.thinking})` : ""
		}`,
		[...THINKING_LEVELS],
	)) as PersonaThinking | undefined;
	if (!level) return cancel();
	const written = await ctx.ui.editor(
		"personality for talking mode — empty leaves the system prompt alone",
		spec.character ?? "",
	);
	if (written === undefined) return cancel();
	const character = written.trimEnd();

	return {
		key,
		spec: { label, refAudio: displayPath(refAudio), refText, lang, sample, thinking: level, character },
	};
}

/** `/persona add`: ask for a voice and write it. */
async function addPersona(ctx: ExtensionContext, pi: ExtensionAPI) {
	const asked = await askPersona(ctx);
	if (!asked) return;
	const file = readPersonasFile();
	file[asked.key] = asked.spec;
	const refused = writePersonasFile(file);
	if (refused) return ctx.ui.notify(`could not write ${PERSONAS_FILE}: ${refused}`, "error");
	registerPersona(asked.key, asked.spec);
	ctx.ui.notify(`persona "${asked.key}" saved to ${PERSONAS_FILE}`, "info");
	if (await ctx.ui.confirm("speak as it now?", `${asked.spec.label} (${asked.spec.thinking} thinking)`))
		await setPersona(asked.key, ctx, pi);
}

/**
 * `/persona edit <name>`: the same questions as adding, each showing what is
 * filed, so one field can be changed without retyping the others. A new name
 * moves the entry. When the voice being edited is the one speaking, the helper
 * is told, since the clip may be a different file now.
 */
async function editPersona(key: string, ctx: ExtensionContext, pi: ExtensionAPI) {
	const asked = await askPersona(ctx, { key, spec: filedPersona(key) });
	if (!asked) return;
	const file = readPersonasFile();
	if (asked.key !== key) delete file[key];
	file[asked.key] = asked.spec;
	const refused = writePersonasFile(file);
	if (refused) return ctx.ui.notify(`could not write ${PERSONAS_FILE}: ${refused}`, "error");
	if (asked.key !== key) delete PERSONAS[key];
	registerPersona(asked.key, asked.spec);
	ctx.ui.notify(
		asked.key === key
			? `persona "${key}" saved to ${PERSONAS_FILE}`
			: `persona "${key}" is now "${asked.key}"`,
		"info",
	);
	if (key !== persona && asked.key !== persona) return;
	const live = PERSONAS[asked.key];
	if (!live) return;
	persona = asked.key;
	const failed = await applyRef(live);
	if (failed) ctx.ui.notify(`pi-tts: could not apply ${live.label}: ${failed}`, "error");
	if (thinking !== live.thinking) setThinking(live.thinking, pi);
}

/**
 * `/persona remove <name>`: drop a voice from the file. The clip stays on disk —
 * it is the user's recording, not ours. Removing the voice that is speaking falls
 * back the way startup does, and removing the last one reinstalls the placeholder,
 * so there is always a persona for the rows and the commands to name.
 */
async function removePersona(key: string, ctx: ExtensionContext, pi: ExtensionAPI) {
	const target = PERSONAS[key];
	const sure = await ctx.ui.confirm(
		"remove persona",
		`${target?.label ?? key}?` +
			(target ? `\n${displayPath(target.refAudio)} stays where it is.` : "") +
			`\n${PERSONAS_FILE}`,
	);
	if (!sure) return;
	const file = readPersonasFile();
	delete file[key];
	const refused = writePersonasFile(file);
	if (refused) return ctx.ui.notify(`could not write ${PERSONAS_FILE}: ${refused}`, "error");
	delete PERSONAS[key];
	ctx.ui.notify(`persona "${key}" removed from ${PERSONAS_FILE}`, "info");
	if (key !== persona) return;
	if (!Object.keys(PERSONAS).length) {
		PERSONAS[DEFAULT_PERSONA] = placeholderPersona();
		return ctx.ui.notify(
			`no voices left — /persona add, or put a clip at ${join(CACHE_DIR, "ref.wav")}`,
			"warning",
		);
	}
	const fallback = PERSONAS[DEFAULT_PERSONA] ? DEFAULT_PERSONA : Object.keys(PERSONAS)[0]!;
	persona = fallback;
	const live = PERSONAS[fallback]!;
	const failed = await applyRef(live);
	if (failed) ctx.ui.notify(`pi-tts: could not switch to ${live.label}: ${failed}`, "error");
	setThinking(live.thinking, pi);
	ctx.ui.notify(`voice persona: ${live.label} (${thinking} thinking)`, "info");
}

/**
 * The voices submenu: use a voice, change what is filed for it, or drop it, with
 * the guided add alongside so the whole life of one is in the same place. Esc at
 * the first prompt leaves the submenu, Esc at the second goes back to the list.
 */
async function managePersonas(ctx: ExtensionContext, pi: ExtensionAPI) {
	for (;;) {
		const entries = Object.entries(PERSONAS);
		const width = Math.max(...entries.map(([key]) => key.length), 4);
		const rows = entries.map(
			([key, target]) =>
				`${key === persona ? "*" : " "} ${key.padEnd(width)}  ${target.label} — ${target.thinking} thinking, ` +
				`${target.character ? "has a personality" : "voice only"}`,
		);
		const byRow = new Map(rows.map((row, i) => [row.trim(), entries[i]![0]]));
		const picked = await ctx.ui.select(`voices — ${PERSONAS_FILE}`, [...rows, "  add a voice"]);
		if (picked === undefined) return;
		const key = byRow.get(picked.trim());
		if (key === undefined) {
			await addPersona(ctx, pi);
			continue;
		}
		const target = PERSONAS[key]!;
		const what = await ctx.ui.select(`${key} — ${target.label}`, [
			"speak as it (and hear its line)",
			"change its details",
			"remove it",
		]);
		if (!what) continue;
		if (what.startsWith("speak")) await setPersona(key, ctx, pi);
		else if (what.startsWith("change")) await editPersona(key, ctx, pi);
		else await removePersona(key, ctx, pi);
	}
}

/** Every voice the session loaded, with the active one marked. */
function listPersonas(ctx: ExtensionContext) {
	const entries = Object.entries(PERSONAS);
	const width = Math.max(...entries.map(([key]) => key.length), 4);
	const lines = entries.map(
		([key, target]) =>
			`${key === persona ? "*" : " "} ${key.padEnd(width)}  ${target.label} — ${target.thinking} thinking, ` +
			`${target.character ? "has a personality" : "voice only"}\n      ${displayPath(target.refAudio)}`,
	);
	ctx.ui.notify(
		`voices (${PERSONAS_FILE}):\n${lines.join("\n")}` +
			(PERSONA_WARNINGS.length ? `\nnot loaded: ${PERSONA_WARNINGS.join("; ")}` : ""),
		"info",
	);
}

/** Say the persona's own line from the personas file, in that persona's voice. */
function testPersona(ctx: ExtensionContext) {
	if (!enabled) return ctx.ui.notify("the voice is off — /persona on first", "info");
	const target = PERSONAS[persona];
	const line = target.sample || "Testing the voice. One, two, three.";
	ctx.ui.notify(`${target.label}: ${line}`, "info");
	enqueue(line, ctx);
}

/**
 * `/persona listen [stop|cancel|on|off|consecutive|conversation|auto|file <path>]`.
 * No argument opens the microphone, which is what the shortcut does too.
 */
async function listenCommand(args: string, ctx: ExtensionContext, pi: ExtensionAPI) {
	const arg = (args ?? "").trim();
	const [command, ...rest] = arg.split(" ");
	const tail = rest.join(" ").trim();
	const sub = command.toLowerCase();

	if (!arg) return listen(ctx, pi);
	if (sub === "stop") {
		if (!capturing) return ctx.ui.notify("not dictating", "info");
		sttSend({ cmd: "stop" });
		return;
	}
	if (sub === "cancel") return cancelCapture(ctx);
	if (sub === "auto") {
		const state = (rest[0] ?? "").toLowerCase();
		if (state === "on" || state === "off") autoListen = state === "on";
		return ctx.ui.notify(
			`auto-listen ${autoListen ? "on" : "off"} — conversation mode ${
				autoListen ? "reopens the mic after each reply" : "waits for the key"
			}`,
			"info",
		);
	}
	if (sub === "on" || sub === "off") {
		sttEnabled = sub === "on";
		if (!sttEnabled) stopSttHelper();
		else void startSttHelper().catch(() => {});
		return ctx.ui.notify(`dictation ${sttEnabled ? "on" : "off"}`, "info");
	}
	if (sub === "consecutive" || sub === "conversation") {
		dictationMode = sub;
		return ctx.ui.notify(
			dictationMode === "conversation"
				? "dictation mode: conversation — speech is sent as soon as you stop"
				: "dictation mode: consecutive — the transcript waits in the input",
			"info",
		);
	}
	if (sub === "mode") return ctx.ui.notify(`dictation mode: ${dictationMode}`, "info");
	if (sub === "file" && tail) {
		const child = await startSttHelper().catch(() => null);
		if (!child) return ctx.ui.notify("pi-tts: dictation helper did not start", "error");
		const result = await sttRequest({ cmd: "transcribe", path: tail });
		if (result?.cmd === "transcript" && result.text) return deliverTranscript(result.text, ctx, pi);
		return ctx.ui.notify(`pi-tts: ${result?.error ?? "no transcript"}`, "error");
	}
	ctx.ui.notify(
		`dictation ${sttEnabled ? "on" : "off"}, mode ${dictationMode}${
			autoListen && dictationMode === "conversation" ? ", auto-listen on" : ""
		} — ${STT_CONFIG.shortcut} talk/stop, ${STT_CONFIG.cancelShortcut} cancel, /persona listen stop|cancel|on|off|consecutive|conversation|auto|file <path>`,
		"info",
	);
}

/** Cut this utterance off and drop whatever is queued behind it. */
function stopVoice(ctx: ExtensionContext) {
	stopAll();
	sttSend({ cmd: "cancel" });
	ctx.ui.notify("voice stopped", "info");
}

/**
 * Everything the extension does from the keyboard, in one screen. `/persona` on
 * its own opens the settings window, which is the same state as a form; this is
 * for when you would rather read it than tab through it.
 */
function personaHelp(ctx: ExtensionContext) {
	ctx.ui.notify(
		[
			`pi-tts — voice ${enabled ? "on" : "off"}, mode ${mode}, persona ${PERSONAS[persona].label}`,
			"",
			"the voice",
			"  on | off                speak, or stay quiet",
			"  stop                    cut this utterance off and drop the queue",
			"  say <text>              say this instead of a reply",
			"  read <file>             read a .txt, .md or .pdf aloud (PI_TTS_READ_MAX_CHARS caps it)",
			"  test                    say this persona's sample line — the audio check",
			"  mode coding | talking   coding leaves pi's prompt alone, talking injects the persona",
			"  measure <file>          what rendering it would cost, before paying for it",
			"  render [stop | status | measure <file> | <file> [out]]",
			"                          make the file into an audio file in the background",
			"",
			`the voices — ${PERSONAS_FILE}`,
			"  <name>                  switch voice and personality mid-session",
			"  add                     guided: the clip, what it says, a name, a personality",
			"  list                    the voices this session loaded, and the file they came from",
			"  manage                  submenu: use a voice, change its details, remove one",
			"  edit <name>             change one entry, each question showing what is filed now",
			"  remove <name>           drop it from the file; the clip stays on disk",
			"",
			"  /persona                the settings window: voice, mode, persona, thinking, volume,",
			"                          max chars, stream interval, dictation, mode, auto-listen.",
			"                          Enter cycles a row, s saves these as the defaults, Esc closes —",
			"                          what you changed stays in effect either way, only saving is scoped.",
			"",
			`dictation — ${STT_CONFIG.shortcut} to talk and to stop, ${STT_CONFIG.cancelShortcut} to throw it away`,
			`  /persona listen [stop | cancel | on | off | consecutive | conversation | auto | file <path>]`,
			"  consecutive leaves the transcript in the input to fix up and send; conversation",
			"  sends it the moment you stop and reopens the mic after the reply is spoken.",
			`  dictation ${sttEnabled ? "on" : "off"}, mode ${dictationMode}, auto-listen ${autoListen ? "on" : "off"}`,
			"",
			"A voice is keyed by the name you type, so these subcommand names are reserved:",
			"  on off stop say read render measure test mode help listen add list manage edit remove",
			"Everything has a PI_TTS_* environment variable; README.md and PERSONAS.md have the detail.",
		].join("\n"),
		"info",
	);
}

/**
 * The one command: the voice, the voices file, the settings window, and help.
 * A bare argument that is not a subcommand is taken for a persona name, which is
 * why the subcommand names above are reserved against the voices file.
 */
async function personaCommand(args: string, ctx: ExtensionContext, pi: ExtensionAPI) {
	const arg = (args ?? "").trim();
	const [word, ...rest] = arg.split(/\s+/);
	const sub = (word ?? "").toLowerCase();
	const tail = rest.join(" ").trim();

	// The voice itself.
	if (sub === "on" || sub === "off") {
		enabled = sub === "on";
		if (!enabled) stopAll();
		return ctx.ui.notify(`voice ${enabled ? "on" : "off"}`, "info");
	}
	if (sub === "stop") return stopVoice(ctx);
	if (sub === "test") return testPersona(ctx);
	if (sub === "mode") {
		const next = tail.toLowerCase();
		if (next === "coding" || next === "talking") setMode(next, ctx);
		else ctx.ui.notify(`mode is ${mode} — /persona mode coding|talking`, "info");
		return;
	}
	if (sub === "say") {
		if (!tail) return ctx.ui.notify("say what? — /persona say <text>", "info");
		return enqueue(tail, ctx);
	}
	// A txt/md/pdf read aloud; no argument asks for the path.
	if (sub === "read") return readAloud(tail, ctx);
	// The same text as an audio file, made by a process of its own.
	if (sub === "render") return renderAloud(tail, ctx);
	// What that job would cost, before paying for it.
	if (sub === "measure") return measureRender(tail, ctx);

	// Dictation.
	if (sub === "listen") return listenCommand(tail, ctx, pi);

	// The voices file.
	if (sub === "help") return personaHelp(ctx);
	if (sub === "add" || sub === "new") return addPersona(ctx, pi);
	if (sub === "list" || sub === "ls") return listPersonas(ctx);
	if (sub === "manage") return managePersonas(ctx, pi);
	if (sub === "edit" || sub === "remove" || sub === "rm") {
		const name = tail.toLowerCase();
		// Editing is allowed for an entry that failed to load — a clip that moved
		// or a format the synthesizer cannot read is exactly what needs fixing.
		if (!PERSONAS[name] && !(name in readPersonasFile()))
			return ctx.ui.notify(
				`unknown persona "${name || tail}" — ${Object.keys(PERSONAS).join(", ")}`,
				"error",
			);
		return sub === "edit" ? editPersona(name, ctx, pi) : removePersona(name, ctx, pi);
	}
	return arg ? setPersona(arg, ctx, pi) : personaSettings(ctx, pi);
}

/**
 * Adopt another cloned voice; the helper swaps references, no model reload.
 * `announce` speaks the persona's sample line back, which is what you want when
 * the switch was asked for by name and not when the settings window is cycling
 * through the voices: each switch cuts the previous utterance off, so cycling
 * would leave nothing but the last syllable of each sample. `/persona test`
 * says the line on purpose.
 */
async function setPersona(
	name: string,
	ctx: ExtensionContext,
	pi: ExtensionAPI,
	announce = true,
) {
	const key = name.toLowerCase().trim();
	const current = () =>
		ctx.ui.notify(
			`voice persona: ${PERSONAS[persona].label}, ${thinking} thinking — /persona opens the settings`,
			"info",
		);
	if (!key) return current();

	const target = PERSONAS[key];
	if (!target) {
		ctx.ui.notify(`unknown persona "${name}" — ${Object.keys(PERSONAS).join(", ")}`, "error");
		return;
	}
	if (key === persona) return current();

	// Nothing said in the old voice should keep playing after the switch, and
	// anything already queued was written for the old one.
	interrupt();
	persona = key;
	// The persona brings a thinking level of its own; the settings window can
	// still change it afterwards.
	setThinking(target.thinking, pi);

	const refused = await applyRef(target);
	if (refused) {
		ctx.ui.notify(`pi-tts: could not switch to ${target.label}: ${refused}`, "error");
		return;
	}
	// Not running yet: it picks the persona up from the environment at spawn.
	ctx.ui.notify(`voice persona: ${target.label} (${thinking} thinking)`, "info");
	if (announce) enqueue(target.sample, ctx);
}

/**
 * Point a running helper at another persona's clip; no model reload. Returns the
 * error text when it refused the clip. Null means it took it, or that no helper
 * is running yet and it will pick the persona up from the environment at spawn.
 */
async function applyRef(target: Persona): Promise<string | null> {
	const child = helper;
	if (!child) return null;
	const id = String(++requestId);
	const result = await new Promise<SpeechResult | null>((resolve) => {
		pending.set(id, resolve);
		try {
			child.stdin.write(
				`${JSON.stringify({
					id,
					cmd: "ref",
					ref_audio: target.refAudio,
					ref_text: target.refText,
					lang: target.lang,
				})}\n`,
			);
			debug("tts >", "ref", id);
		} catch {
			resolve(null);
		}
	});
	if (!result) return "helper died";
	return result.cmd === "error" ? (result.error ?? "it refused the clip") : null;
}

/**
 * The settings window: every knob the extension has, applied as it changes.
 * Enter or Space on a row cycles its value, `s` (or the save row) writes them
 * all to SETTINGS_FILE for the next session, Esc closes the window — the
 * changes made inside it stay in effect either way, only the saving is scoped.
 */
async function personaSettings(ctx: ExtensionContext, pi: ExtensionAPI) {
	// Voices that were named in the personas file but could not be loaded are
	// worth one complaint per session, not one per /persona.
	if (PERSONA_WARNINGS.length) {
		ctx.ui.notify(
			`pi-tts: ${PERSONA_WARNINGS.join("; ")} (${PERSONAS_FILE})`,
			"warning",
		);
		PERSONA_WARNINGS.length = 0;
	}
	if (ctx.mode !== "tui") {
		ctx.ui.notify(
			`voice ${enabled ? "on" : "off"}, mode ${mode}, persona ${PERSONAS[persona].label}, ${thinking} thinking, ` +
				`volume ${volume.toFixed(2)}, dictation ${sttEnabled ? "on" : "off"} (${dictationMode}, auto-listen ${
					autoListen ? "on" : "off"
				})`,
			"info",
		);
		return;
	}

	// The submenu asks questions of its own, which cannot be asked while this
	// window holds the keyboard: its row closes the window with the flag set, the
	// submenu runs, and the window reopens over the state it left.
	let manage = false;
	await ctx.ui.custom((tui, theme, _keybindings, done) => {
		const items: SettingItem[] = [
			{
				id: "voice",
				label: "voice",
				currentValue: enabled ? "on" : "off",
				values: ["on", "off"],
				description: "Speak the reply as it streams.",
			},
			{
				id: "mode",
				label: "mode",
				currentValue: mode,
				values: [...VOICE_MODES],
				description: "Coding leaves the system prompt as pi built it; talking injects the persona.",
			},
			{
				id: "persona",
				label: "voice persona",
				currentValue: persona,
				values: Object.keys(PERSONAS),
				description: "Cloned voice, and the personality talking mode injects with it.",
			},
			{
				id: "thinking",
				label: "thinking",
				currentValue: thinking,
				values: [...THINKING_LEVELS],
				description: "Minimal also says: think very little, answer at once, do not disclose the thoughts.",
			},
			{
				id: "volume",
				label: "volume",
				currentValue: volume.toFixed(2),
				values: [...VOLUME_VALUES],
				description: "Output gain, sent to the voice that is already running.",
			},
			{
				id: "maxChars",
				label: "max spoken chars",
				currentValue: String(maxChars),
				values: [...MAX_CHARS_VALUES],
				description: "Speech is cut at a sentence near this length; the reply on screen is not shortened.",
			},
			{
				id: "streamInterval",
				label: "stream interval",
				currentValue: streamInterval.toFixed(1),
				values: [...STREAM_INTERVAL_VALUES],
				description: "Seconds of speech decoded per chunk: lower starts sooner, higher is steadier.",
			},
			{
				id: "dictation",
				label: "dictation",
				currentValue: sttEnabled ? "on" : "off",
				values: ["on", "off"],
				description: `Qwen3-ASR capture on ${STT_CONFIG.shortcut}.`,
			},
			{
				id: "dictationMode",
				label: "dictation mode",
				currentValue: dictationMode,
				values: ["consecutive", "conversation"],
				description: "Consecutive leaves the transcript in the input; conversation sends it at once.",
			},
			{
				id: "autoListen",
				label: "auto-listen",
				currentValue: autoListen ? "on" : "off",
				values: ["on", "off"],
				description: "Conversation mode reopens the microphone after each reply.",
			},
			{
				id: "manage",
				label: "manage personas",
				currentValue: "open",
				values: ["open"],
				description: `Add, change or remove a voice in ${PERSONAS_FILE}. Closes this window while it asks.`,
			},
			{
				id: "save",
				label: "save as default",
				currentValue: "press enter",
				values: ["press enter"],
				description: `Write all of the above to ${SETTINGS_FILE} for the next session.`,
			},
		];
		const listTheme = getSettingsListTheme();

		const save = () => {
			saveSettings(currentSettings());
			ctx.ui.notify(`defaults saved — ${SETTINGS_FILE}`, "info");
			done(undefined);
		};

		// SettingsList prints the selected row's description itself, which grows and
		// shrinks the block by a line or two as the cursor moves and leaves the chat
		// above it half repainted. The rows go over without descriptions, and the
		// panel below is always the same height.
		const descriptions = new Map(items.map((item) => [item.id, item.description ?? ""]));
		const rows: SettingItem[] = items.map((item) => ({ ...item, description: undefined }));
		const settings = new SettingsList(
			rows,
			rows.length + 1,
			listTheme,
			(id, value) => {
				switch (id) {
					case "voice":
						setVoiceEnabled(value === "on");
						break;
					case "mode":
						setMode(value === "talking" ? "talking" : "coding", ctx);
						break;
					case "persona":
						// Cycling the row must not speak: /persona test does that.
						void setPersona(value, ctx, pi, false);
						// The persona carries a thinking level of its own; show it.
						settings.updateValue("thinking", thinking);
						break;
					case "thinking":
						setThinking(value as PersonaThinking, pi);
						ctx.ui.notify(`thinking level: ${value}`, "info");
						break;
					case "volume":
						volume = Number(value);
						tuneHelper();
						ctx.ui.notify(`volume: ${volume.toFixed(2)}`, "info");
						break;
					case "maxChars":
						maxChars = Number(value);
						ctx.ui.notify(`max spoken chars: ${maxChars}`, "info");
						break;
					case "streamInterval":
						streamInterval = Number(value);
						tuneHelper();
						ctx.ui.notify(`stream interval: ${streamInterval}s`, "info");
						break;
					case "dictation":
						sttEnabled = value === "on";
						if (sttEnabled) void startSttHelper().catch(() => {});
						else stopSttHelper();
						ctx.ui.notify(`dictation ${sttEnabled ? "on" : "off"}`, "info");
						break;
					case "dictationMode":
						dictationMode = value === "conversation" ? "conversation" : "consecutive";
						ctx.ui.notify(`dictation mode: ${dictationMode}`, "info");
						break;
					case "autoListen":
						autoListen = value === "on";
						ctx.ui.notify(`auto-listen ${autoListen ? "on" : "off"}`, "info");
						break;
					case "manage":
						manage = true;
						done(undefined);
						return;
					case "save":
						save();
						return;
				}
				tui.requestRender();
			},
			() => done(undefined),
		);

		return {
			render(width: number) {
				const index = Math.max(
					0,
					Math.min(
						rows.length - 1,
						(settings as unknown as { selectedIndex: number }).selectedIndex ?? 0,
					),
				);
				const wrapped = wrapTextWithAnsi(descriptions.get(rows[index]!.id) ?? "", width - 4);
				const panel = wrapped.slice(0, SETTINGS_HELP_LINES);
				if (wrapped.length > panel.length && panel.length)
					panel[panel.length - 1] = truncateToWidth(panel[panel.length - 1]!, width - 4, "…");
				while (panel.length < SETTINGS_HELP_LINES) panel.push("");
				const inner = settings.render(width);
				// SettingsList closes with a blank and its own "Enter/Space to change";
				// dropped, since the hint below says that and adds the save key.
				if (inner.length > 2 && inner[inner.length - 1]?.includes("Esc to cancel"))
					inner.splice(-2, 2);
				return [
					theme.fg("accent", theme.bold("Voice settings")),
					"",
					...inner,
					...panel.map((line) => (line ? listTheme.description(`  ${line}`) : "")),
					listTheme.hint(" s — save as the defaults for new sessions · esc closes, changes stay applied"),
				];
			},
			invalidate() {
				settings.invalidate();
			},
			handleInput(data: string) {
				if (data === "s" || data === "S") return save();
				settings.handleInput(data);
				tui.requestRender();
			},
		};
	});
	if (manage) {
		await managePersonas(ctx, pi);
		return personaSettings(ctx, pi);
	}
}

/* ------------------------------------------------------------------ dictation */

type DictationMode = "consecutive" | "conversation";

type Capture = {
	id: string;
	ctx: ExtensionContext;
	/** Opened by the extension at the end of a turn, not by a keypress. */
	auto: boolean;
	/** Set when the cancel key lands during transcription. */
	cancelled: boolean;
};

let sttEnabled = onOff(saved.dictation, true);
let dictationMode: DictationMode =
	saved.dictationMode === "conversation" || saved.dictationMode === "consecutive"
		? saved.dictationMode
		: process.env.PI_TTS_STT_MODE === "conversation"
			? "conversation"
			: "consecutive";
let autoListen = onOff(saved.autoListen, process.env.PI_TTS_STT_AUTO !== "0");
let sttHelper: Helper | null = null;
let sttStarting: Promise<Helper> | null = null;
let sttBuffer = "";
let sttId = 0;
let capturing: Capture | null = null;
let startingCapture = false;
const sttPending = new Map<string, (result: SttResult | null) => void>();

type SttResult = {
	cmd: string;
	id?: string;
	text?: string;
	seconds?: number;
	rms?: number;
	peak?: number;
	threshold?: number;
	reason?: string;
	error?: string;
};

function sttSend(payload: Record<string, unknown>) {
	if (!sttHelper) return;
	try {
		sttHelper.stdin.write(`${JSON.stringify(payload)}\n`);
	} catch {
		/* the helper is going away */
	}
}

function readSttStdout(chunk: Buffer) {
	sttBuffer += chunk.toString("utf8");
	const lines = sttBuffer.split("\n");
	sttBuffer = lines.pop() ?? "";
	for (const line of lines) {
		if (!line.trim()) continue;
		let message: SttResult;
		try {
			message = JSON.parse(line);
		} catch {
			continue; // model loader chatter on stdout
		}
		if (message.cmd === "level") {
			const capture = capturing;
			if (capture && capture.id === message.id) {
				if (Math.round((message.seconds ?? 0) * 10) % 20 === 0) {
					debug("stt level", message.seconds, message.rms);
				}
				capture.ctx.ui.setStatus(
					STT_STATUS,
					listeningText(message.seconds, message.rms ?? 0, capture.auto),
				);
			}
			continue;
		}
		// Progress only: the capture is over when transcript/skip/error comes back.
		if (message.cmd === "record_start" || message.cmd === "calibrated") {
			debug("stt <", message.cmd, message.id ?? "");
			continue;
		}
		if (message.id && sttPending.has(message.id)) {
			debug("stt <", message.cmd, message.id, message.reason ?? "", message.error ?? "");
			const resolve = sttPending.get(message.id)!;
			sttPending.delete(message.id);
			resolve(message);
		}
	}
}

function startSttHelper(): Promise<Helper> {
	if (sttHelper) return Promise.resolve(sttHelper);
	if (sttStarting) return sttStarting;

	sttStarting = new Promise<Helper>((resolve, reject) => {
		debug("stt helper spawn", CONFIG.python, STT_CONFIG.helper);
		const sttErrors: string[] = [];
		const child = spawn(CONFIG.python, [STT_CONFIG.helper], {
			stdio: ["pipe", "pipe", "pipe"],
			env: {
				...process.env,
				QWEN_ASR_MODEL: ASR_MODEL_DIR,
				QWEN_ASR_LANG: process.env.PI_TTS_ASR_LANG ?? "",
				QWEN_ASR_DEVICE: process.env.PI_TTS_ASR_DEVICE ?? "",
				QWEN_ASR_MAX_SECONDS: String(STT_CONFIG.maxSeconds),
			},
		});
		child.stdout.on("data", readSttStdout);
		drainStderr(child, "stt stderr", sttErrors);
		child.on("exit", (code, signal) => {
			debug("stt helper exit", code, signal);
			if (code && ctxRef)
				ctxRef.ui.notify(`dictation helper stopped (${code})${lastError(sttErrors)}`, "error");
			sttHelper = null;
			sttStarting = null;
			for (const resolve of sttPending.values()) resolve(null);
			sttPending.clear();
		});
		child.on("error", (error) => {
			sttHelper = null;
			sttStarting = null;
			reject(error);
		});
		const onReady = (chunk: Buffer) => {
			if (chunk.toString("utf8").includes('"ready"')) {
				child.stdout.off("data", onReady);
				debug("stt helper ready");
				sttHelper = child;
				sttStarting = null;
				resolve(child);
			}
		};
		child.stdout.on("data", onReady);
	}).catch((error) => {
		sttStarting = null;
		throw error;
	});

	return sttStarting;
}

function stopSttHelper() {
	capturing = null;
	if (sttHelper) {
		sttSend({ cmd: "quit" });
		sttHelper.kill();
		sttHelper = null;
		sttStarting = null;
	}
}

function sttRequest(payload: Record<string, unknown>): Promise<SttResult | null> {
	const id = String(payload.id ?? ++sttId);
	return new Promise((resolve) => {
		sttPending.set(id, resolve);
		sttSend({ ...payload, id });
	});
}

/** Put the transcript where the user can use it: input in consecutive, prompt in conversation. */
function deliverTranscript(text: string, ctx: ExtensionContext, pi: ExtensionAPI) {
	debug("deliver", dictationMode, JSON.stringify(text.slice(0, 80)));
	if (dictationMode === "conversation") {
		pi.sendUserMessage(text);
		return;
	}
	if (ctx.mode === "tui") {
		const existing = ctx.ui.getEditorText().trim();
		ctx.ui.setEditorText(existing ? `${existing} ${text}` : text);
		debug("editor now", JSON.stringify(ctx.ui.getEditorText().slice(0, 80)));
		return;
	}
	ctx.ui.notify(text, "info");
}

/** Start a capture, or close the one in progress and transcribe it. */
async function listen(ctx: ExtensionContext, pi: ExtensionAPI) {
	debug("listen press", capturing ? "stop" : startingCapture ? "busy" : "start");
	if (capturing) {
		sttSend({ cmd: "stop" });
		return;
	}
	if (startingCapture) return; // pressed again while the model was still loading
	startingCapture = true;
	try {
		await recordOnce(ctx, pi);
	} finally {
		startingCapture = false;
	}
}

async function recordOnce(ctx: ExtensionContext, pi: ExtensionAPI, auto = false) {
	if (!sttEnabled) {
		ctx.ui.notify("dictation is off — /persona listen on", "info");
		return;
	}
	// The voice must not answer while the user is talking.
	interrupt();

	const child = await startSttHelper().catch((error) => {
		ctx.ui.notify(`pi-tts: could not start ${CONFIG.python} ${STT_CONFIG.helper}: ${error}`, "error");
		return null;
	});
	if (!child || capturing) return;

	const id = String(++sttId);
	const capture: Capture = { id, ctx, auto, cancelled: false };
	capturing = capture;
	debug("stt record", id, auto ? "auto" : "key");
	ctx.ui.setStatus(STT_STATUS, listeningText(0, 0, auto));
	const result = await sttRequest({ id, cmd: "record" }).finally(() => {
		capturing = null;
		ctx.ui.setStatus(STT_STATUS, undefined);
	});

	if (!result) return;
	if (capture.cancelled) {
		// Cancelled after the capture ended, while it was being transcribed.
		ctx.ui.notify("dictation cancelled", "info");
		return;
	}
	if (result.cmd === "transcript" && result.text) {
		deliverTranscript(result.text, ctx, pi);
	} else if (result.cmd === "skip") {
		if (result.reason === "cancelled") {
			ctx.ui.notify("dictation cancelled", "info");
		} else if (auto) {
			// Nobody took the turn; go quiet and wait for the key instead.
			debug("auto capture heard nothing", result.reason ?? "");
		} else {
			// The loudest level of the capture tells a quiet microphone apart
			// from a silent room: raise the mic gain or lower QWEN_ASR_THRESHOLD.
			const peak = result.peak === undefined ? "" : `, peak ${result.peak}`;
			ctx.ui.notify(`heard nothing (${result.reason}${peak})`, "warning");
		}
	} else if (result.cmd === "error") {
		ctx.ui.notify(`dictation failed: ${result.error}`, "error");
	}
}

/** Throw the capture away: nothing is sent, not even in conversation mode. */
function cancelCapture(ctx: ExtensionContext) {
	if (!capturing) {
		ctx.ui.notify("not dictating", "info");
		return;
	}
	debug("listen cancel", capturing.id);
	capturing.cancelled = true;
	sttSend({ cmd: "cancel" });
}

export default function (pi: ExtensionAPI) {
	// agent_end fires after each low-level run (retries, compaction and queued
	// follow-ups can continue afterwards), so it is the right place to capture
	// the assistant text. agent_settled is the "no more work" boundary.
	// Prewarm: the helper keeps the model resident, so load it in the background
	// while the first prompt is being typed instead of on the first utterance.
	pi.on("session_start", () => {
		if (enabled) void startHelper().catch(() => {});
		pi.setThinkingLevel(thinking);
	});

	// The user is talking now: go quiet and drop whatever was queued to be said.
	// The persona is inserted in the system prompt here, as a <persona> section
	// replaced on the next run, so /persona <name> changes how it thinks as well
	// as how it sounds. A persona with no personality leaves the prompt alone.
	pi.on("before_agent_start", (event) => {
		interrupt();
		const section = personaSection();
		if (section) event.systemPromptOptions.sections[PERSONA_SECTION] = section;
		else delete event.systemPromptOptions.sections[PERSONA_SECTION];
	});

	pi.on("agent_start", () => {
		raw = "";
		streamedAny = false;
		spokeThisRun = false;
	});

	// Streaming: speak each sentence as soon as it is complete.
	pi.on("message_start", (event) => {
		if (event.message.role !== "assistant") return;
		raw = "";
		streamedAny = false;
		// Prose from earlier messages is stale now; do not queue up behind it.
		dropQueued();
	});

	pi.on("message_update", (event, ctx) => {
		const update = event.assistantMessageEvent;
		if (update.type !== "text_delta" || !enabled) return;
		ctxRef = ctx;
		raw += update.delta;
		streamedAny = true;
		pump();
	});

	pi.on("message_end", (event, ctx) => {
		if (event.message.role !== "assistant") return;
		ctxRef = ctx;
		const full = messageText(event.message);
		if (streamedAny) {
			raw = full.endsWith(raw) ? raw : full; // resync if the final text differs
			pump(true);
		} else if (full) {
			spokeThisRun = true;
			enqueue(full, ctx);
		}
		raw = "";
	});

	pi.on("agent_end", (event) => {
		lastAssistantText = extractText(event.messages as Array<{ role: string; content?: unknown }>);
	});

	// Fallback for replies that never streamed (voice switched on mid-run, or a
	// resumed session): streamed prose was already said as it arrived.
	pi.on("agent_settled", async (_event, ctx) => {
		ctxRef = ctx;
		const text = lastAssistantText;
		lastAssistantText = "";
		if (!enabled || spokeThisRun || !text) return;
		enqueue(text, ctx);
	});

	// Conversation mode: hand the turn back to the microphone once the reply has
	// been spoken, so the exchange runs without touching the keyboard. If nobody
	// takes the turn the capture times out quietly and the key takes over again.
	pi.on("agent_settled", (_event, ctx) => {
		if (!autoListen || !sttEnabled || dictationMode !== "conversation") return;
		const token = speechToken;
		void queue.then(() => {
			// A new prompt, an interrupt or `/persona listen off` while the reply was still
			// being spoken: the turn is no longer ours to close.
			if (token !== speechToken || capturing || startingCapture) return;
			if (!autoListen || !sttEnabled || dictationMode !== "conversation") return;
			debug("auto listen after reply");
			void recordOnce(ctx, pi, true);
		});
	});

	pi.on("session_shutdown", () => {
		stopAll();
		stopSttHelper();
		rendering?.child.kill("SIGTERM");
	});

	// One key to talk: press to open the microphone, press again to close it.
	// (In terminals without the Kitty keyboard protocol ctrl+i is TAB.)
	pi.registerShortcut(STT_CONFIG.shortcut, {
		description: "Dictate a prompt (Qwen3-ASR)",
		handler: (ctx) => listen(ctx, pi),
	});

	pi.registerShortcut(STT_CONFIG.cancelShortcut, {
		description: "Discard the dictation in progress",
		handler: (ctx) => cancelCapture(ctx),
	});

	pi.registerCommand("persona", {
		description:
			"The voice, the voices and dictation (on | off | stop | say <text> | read <file> | render | measure <file> | test | mode coding|talking | listen [stop|cancel|on|off|consecutive|conversation|auto|file <path>] | <name> | add | list | manage | edit | remove | help) — no argument opens the settings",
		handler: (args, ctx) => personaCommand(args ?? "", ctx, pi),
	});
}
