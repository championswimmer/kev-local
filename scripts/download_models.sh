#!/usr/bin/env bash
# Downloads the Qwen3.5 base weights + Kev LoRA adapters into models/hf-cache.
# Safe to re-run: hf download resumes/skips already-fetched files.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$SCRIPT_DIR/../models/hf-cache"
mkdir -p "$HF_HOME"
# The xet CDN backend stalls/retries heavily on this network; plain HTTP is reliable.
export HF_HUB_DISABLE_XET=1

echo "=== Downloading base models ==="
hf download Qwen/Qwen3.5-4B-Base --cache-dir "$HF_HOME"
hf download Qwen/Qwen3.5-9B-Base --cache-dir "$HF_HOME"

echo "=== Downloading Kev adapters ==="
hf download jaredpalmer/kev-4b --cache-dir "$HF_HOME"
hf download jaredpalmer/kev-9b --cache-dir "$HF_HOME"

echo "=== DONE ==="
