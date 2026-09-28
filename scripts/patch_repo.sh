#!/usr/bin/env bash
# Idempotently patches the vendored kev submodule's pyproject.toml for this hardware.
# Kept as a script (not a committed diff) so `repo/` stays an unmodified vendor checkout -
# only this repo (kev-local) carries "the things we did to run it".
#
# What/why:
#   - requires-python <3.14 -> <3.15: the repo's cap is about upstream torch wheel
#     availability for CUDA/MLX at the time it was written, not a real incompatibility.
#   - pin torch to the official ROCm 7.2 wheel from download.pytorch.org, via a
#     dedicated `[tool.uv.sources]` index. ROCm 7.1 (what Ubuntu 26.04 ships, and what
#     PyTorch's own rocm7.1 wheel builds against) segfaults inside libamdhip64 on any
#     GPU memory allocation on this exact chip (Strix Halo / gfx1151) - a documented,
#     widely-reported issue (e.g. pytorch/pytorch#173367, ROCm/ROCm#5824). ROCm 7.2
#     fixes it; verified locally (matmul + bf16 on the actual GPU, no crash).
set -euo pipefail
REPO="${1:?usage: patch_repo.sh <path-to-kev-checkout>}"
cd "$REPO"

sed -i 's/^requires-python = ">=3.12,<3.14"$/requires-python = ">=3.12,<3.15"/' pyproject.toml

if grep -q '^    "torch>=2.6,<2.9",$' pyproject.toml; then
  sed -i 's/^    "torch>=2.6,<2.9",$/    "torch==2.14.0",/' pyproject.toml
fi

if ! grep -q '^\[tool.uv.sources\]$' pyproject.toml; then
  cat >>pyproject.toml <<'EOF'

[tool.uv.sources]
torch = { index = "pytorch-rocm" }

[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm7.2"
explicit = true
EOF
fi

# Always regenerate from the patched pyproject rather than trust a stale/vendored lock.
rm -f uv.lock

echo "patched $REPO/pyproject.toml:"
grep -n "requires-python\|torch\|tool.uv" pyproject.toml || true
