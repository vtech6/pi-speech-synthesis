#!/usr/bin/env bash
# One-time setup for pi-tts: a python environment with mlx-audio, the two Qwen3
# voice models, and a personas file to edit. Everything lands under
# ~/.cache/pi-tts, which is where the extension looks by default, so a normal
# install needs no environment variables. Safe to re-run.
set -euo pipefail

CACHE="${PI_TTS_CACHE:-$HOME/.cache/pi-tts}"
VENV="$CACHE/venv"
MODELS="$CACHE/models"
PERSONAS="${PI_TTS_PERSONAS:-$HOME/.pi/agent/pi-tts/personas.json}"
TTS_REPO="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
ASR_REPO="mlx-community/Qwen3-ASR-0.6B-8bit"
PYTHON="${PI_TTS_PYTHON:-python3}"
SKIP_MODELS=0
SKIP_VENV=0

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '    warning: %s\n' "$1" >&2; }

usage() {
	cat <<EOF
usage: scripts/setup.sh [options]

  --venv DIR      python virtualenv to create/use (default \$PI_TTS_CACHE/venv)
  --models DIR    where the models are downloaded (default \$PI_TTS_CACHE/models)
  --personas FILE personas file to create (default ~/.pi/agent/pi-tts/personas.json)
  --python BIN    interpreter to build the venv with (default: python3 on PATH)
  --skip-models   environment only, no downloads
  --skip-venv     downloads only, using the venv as it is
  -h, --help      this message

Use --venv to point at an existing conda or uv environment instead:
  PI_TTS_PYTHON=~/miniconda3/envs/voice/bin/python   # then --skip-venv, and
  \$PI_TTS_PYTHON -m pip install -r requirements.txt
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--venv) VENV="$2"; shift 2 ;;
	--models) MODELS="$2"; shift 2 ;;
	--personas) PERSONAS="$2"; shift 2 ;;
	--python) PYTHON="$2"; shift 2 ;;
	--skip-models) SKIP_MODELS=1; shift ;;
	--skip-venv) SKIP_VENV=1; shift ;;
	-h | --help) usage; exit 0 ;;
	*) echo "unknown option: $1" >&2; usage; exit 2 ;;
	esac
done

[[ "$(uname -s)" == "Darwin" ]] || warn "MLX is Apple-only; the helpers will not load on $(uname -s)"
[[ "$(uname -m)" == "arm64" ]] || warn "MLX wants Apple Silicon; this is $(uname -m)"
command -v "$PYTHON" >/dev/null 2>&1 || {
	echo "no python at '$PYTHON'. Install one (brew install python@3.12) or pass --python BIN." >&2
	exit 1
}

if [[ $SKIP_VENV -eq 0 ]]; then
	step "python environment: $VENV"
	[[ -x "$VENV/bin/python" ]] || "$PYTHON" -m venv "$VENV"
	"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
	"$VENV/bin/pip" install -r "$HERE/requirements.txt"
fi
PY="$VENV/bin/python"
[[ -x "$PY" ]] || { echo "no venv at $VENV (drop --skip-venv, or run without it first)" >&2; exit 1; }

if [[ $SKIP_MODELS -eq 0 ]]; then
	step "models: ~2.3 GB into $MODELS"
	download() {
		local repo="$1" dir="$MODELS/$2"
		if [[ -e "$dir/model.safetensors" ]]; then
			echo "    already there: $dir"
			return
		fi
		echo "    $repo"
		mkdir -p "$dir"
		"$VENV/bin/hf" download "$repo" --local-dir "$dir"
	}
	download "$TTS_REPO" "Qwen3-TTS-12Hz-0.6B-Base-8bit"
	download "$ASR_REPO" "Qwen3-ASR-0.6B-8bit"
fi

step "personas: $PERSONAS"
if [[ -f "$PERSONAS" ]]; then
	echo "    kept the file you already have"
else
	mkdir -p "$(dirname "$PERSONAS")"
	cp "$HERE/personas.example.json" "$PERSONAS"
	echo "    installed the example — the refAudio paths in it are yours to fill in"
fi

step "next"
cat <<EOF
  1. Put a reference clip where your personas file points, or at
     $CACHE/ref.wav (plus an optional $CACHE/ref.txt holding its transcript).
     PERSONAS.md explains what makes a good clip.
  2. Start pi and try it:  /persona say testing one two three
     /persona help lists every command; /persona opens the settings.
     Dictation is ctrl+i (talk, then again to stop) — the terminal must be
     allowed to use the microphone, and /persona listen file <wav> works without that.
  3. Nothing running? PI_TTS_DEBUG=0 turns the trace off; the log is at
     \${TMPDIR:-/tmp}/pi-tts/dictation.log and the helpers print to stderr.

  Overrides, if you would rather not use the defaults:
     PI_TTS_PYTHON, PI_TTS_MODEL, PI_TTS_ASR_MODEL, PI_TTS_PERSONAS, PI_TTS_CACHE
EOF
