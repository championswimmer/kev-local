#!/usr/bin/env bash
# One-time environment setup for running Kev locally on Apple Silicon (MPS/MLX).
#
# Unlike the AMD/ROCm path (setup_env.sh), nothing here needs patching: kev's own
# pyproject.toml already pulls in mlx-lm automatically on sys_platform == 'darwin' and
# platform_machine == 'arm64' (the [serve] extra), and stock PyPI torch wheels support
# MPS out of the box - no special index like the ROCm one is needed. kev.serve's "auto"
# backend then picks kev.mlx_model (Metal kernels via mlx-lm) whenever the device is mps
# and the checkpoint's base is a hybrid Qwen3.5 backbone, which is the case for both
# kev-4b and kev-9b - see kev/checkpoint.py's LoadOptions docstring.
#
# The one real constraint: repo/pyproject.toml caps requires-python at "<3.14", so a
# bare Homebrew python3.14 won't satisfy it. Rather than patch the submodule (as
# patch_repo.sh does for the ROCm case), this just asks uv to provision a 3.13
# it manages itself - no vendored-file edits needed on this platform.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEV_LOCAL="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$KEV_LOCAL/repo"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "this script is for macOS - use scripts/setup_env.sh on Linux/ROCm" >&2
  exit 1
fi

echo "=== Provisioning a uv-managed Python 3.13 (repo/pyproject.toml caps at <3.14) ==="
uv python install 3.13

# uv's default project-venv location is <pyproject dir>/.venv; point it at kev-local/.venv
# instead so the venv lives next to our own scripts, not inside the submodule.
export UV_PROJECT_ENVIRONMENT="$KEV_LOCAL/.venv"

cd "$REPO"

echo "=== Installing kev + deps (torch MPS wheel + mlx-lm, both plain PyPI on darwin/arm64) ==="
uv sync --extra serve --python 3.13

echo "=== Verifying torch sees MPS and mlx-lm imports ==="
"$KEV_LOCAL/.venv/bin/python" - <<'PYEOF'
import torch
print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
x = torch.randn(4096, 4096, device="mps", dtype=torch.bfloat16)
y = x @ x
torch.mps.synchronize()
print("matmul ok, mean:", y.float().mean().item())

import mlx.core as mx
print("mlx:", mx.__version__ if hasattr(mx, "__version__") else "ok")
PYEOF

echo "=== Setup complete ==="
