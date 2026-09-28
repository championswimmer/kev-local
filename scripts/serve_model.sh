#!/usr/bin/env bash
# Serve a Kev model locally: ./serve_model.sh 4b|9b [port]
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEV_LOCAL="$(cd "$SCRIPT_DIR/.." && pwd)"

SIZE="${1:?usage: serve_model.sh 4b|9b [port]}"
PORT="${2:-8008}"
case "$SIZE" in
  4b) MODEL="jaredpalmer/kev-4b" ;;
  9b) MODEL="jaredpalmer/kev-9b" ;;
  *) echo "unknown size: $SIZE (expected 4b or 9b)" >&2; exit 1 ;;
esac

export HF_HOME="$KEV_LOCAL/models/hf-cache"
# Eager PyTorch path: fused Triton/fla kernels + CUDA graphs are CUDA-tuned and not
# validated on ROCm/gfx1151 yet, so keep them off for a correct first run.
export KEV_FUSED=0
export KEV_CUDA_GRAPHS=0
export KEV_DTYPE=bf16

echo "Serving $MODEL on :$PORT (HF_HOME=$HF_HOME)"
exec "$SCRIPT_DIR/run_gpu.sh" \
  "$KEV_LOCAL/.venv/bin/python" -m kev.serve --run "$MODEL" --port "$PORT"
