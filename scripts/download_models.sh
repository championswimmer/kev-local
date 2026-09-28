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

# Deliberately NOT passing --cache-dir: that puts files directly under $HF_HOME
# (models--*/ with no "hub" level), while every HF_HOME-only consumer (transformers,
# kev.serve, huggingface_hub itself) looks under $HF_HOME/hub/models--*. Mixing the two
# once produced two divergent, half-populated caches and a very confusing hang - let
# HF_HOME alone pick the standard layout everywhere.
echo "=== Downloading base models ==="
hf download Qwen/Qwen3.5-4B-Base
hf download Qwen/Qwen3.5-9B-Base

echo "=== Downloading Kev adapters ==="
hf download jaredpalmer/kev-4b
hf download jaredpalmer/kev-9b

echo "=== DONE ==="
