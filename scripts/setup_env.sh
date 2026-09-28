#!/usr/bin/env bash
# One-time environment setup for running Kev locally on the AMD Strix Halo iGPU (gfx1151) via ROCm.
#
# Prereq (run once, needs sudo, NOT done by this script) - GPU device access:
#   sudo usermod -aG render "$USER"
#   # then log out/in (or reboot) so the group membership takes effect -
#   # or just keep using scripts/run_gpu.sh, which wraps commands in `sg render -c ...`,
#   # so you don't have to log out for a first test.
#
# Note on ROCm version: this uses the official PyTorch ROCm 7.2 wheel from
# download.pytorch.org (pinned in repo/pyproject.toml by patch_repo.sh), NOT Ubuntu's
# own apt python3-torch-rocm package. Ubuntu 26.04 ships ROCm 7.1, which segfaults
# inside libamdhip64 on any GPU memory allocation on this exact chip (Strix Halo /
# gfx1151) - a documented, widely-reported issue (pytorch/pytorch#173367,
# ROCm/ROCm#5824), fixed in ROCm 7.2. Verified locally: 7.1 crashes, 7.2 doesn't.
# This also means any reasonably current Python 3.14 works here (system or Homebrew) -
# unlike the apt package, this wheel isn't tied to one specific interpreter build.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEV_LOCAL="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$KEV_LOCAL/repo"

SYS_PY="/usr/bin/python3.14"
echo "Using interpreter: $SYS_PY ($("$SYS_PY" --version))"

echo "=== Patching vendored repo for this hardware ==="
"$SCRIPT_DIR/patch_repo.sh" "$REPO"

# uv's default project-venv location is <pyproject dir>/.venv; point it at kev-local/.venv
# instead so the venv lives next to our own scripts, not inside the submodule.
export UV_PROJECT_ENVIRONMENT="$KEV_LOCAL/.venv"

cd "$REPO"

echo "=== Installing kev + deps (torch from the ROCm 7.2 index, see patch_repo.sh) ==="
uv sync --extra serve --python "$SYS_PY"

echo "=== Verifying torch sees the GPU and can actually compute (needs the render group) ==="
"$SCRIPT_DIR/run_gpu.sh" "$KEV_LOCAL/.venv/bin/python" - <<'PYEOF'
import torch
print("torch:", torch.__version__, "| rocm:", torch.version.hip)
print("cuda (ROCm) available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print("matmul ok, mean:", y.float().mean().item())
PYEOF

echo "=== Setup complete ==="
