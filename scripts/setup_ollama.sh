#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-qwen2.5-coder:7b}"
OLLAMA_BIN="${OLLAMA_BIN:-ollama}"

if ! command -v "$OLLAMA_BIN" >/dev/null 2>&1; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "error: ollama is not installed and Homebrew was not found." >&2
    echo "Install Ollama from https://ollama.com/download, then rerun this script." >&2
    exit 1
  fi
  brew install ollama
fi

if ! "$OLLAMA_BIN" list >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew services start ollama >/dev/null || true
  fi
fi

if ! "$OLLAMA_BIN" list >/dev/null 2>&1; then
  echo "Starting Ollama for this shell..."
  OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}" \
    OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}" \
    "$OLLAMA_BIN" serve >/tmp/pfit-ollama.log 2>&1 &
  sleep 3
fi

"$OLLAMA_BIN" list >/dev/null
"$OLLAMA_BIN" pull "$MODEL"

cat <<EOF
Ollama is ready.

Use this pfit config:

llm:
  model: $MODEL
  base_url: http://localhost:11434
EOF
