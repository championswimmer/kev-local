# Kev, running locally on an AMD Ryzen AI Max+ 395 (Strix Halo)

[Kev](https://github.com/jaredpalmer/kev) is a small open family of *decision models* by Jared Palmer —
an open reconstruction of TypeSafe's "Jev": you send it one document (a *state*) plus a set of typed
yes/no, multiple-choice or score questions, and it returns a calibrated probability per question in a
single forward pass. No text generation. It's built as LoRA adapters + a pointer head on top of Qwen3.5
(0.8B / 4B / 9B) and Qwen3.8 (27B).

This directory documents/automates running the **4B** and **9B** checkpoints locally on this machine:

- **CPU:** AMD Ryzen AI Max+ 395 (Strix Halo, "gfx1151" iGPU, Radeon 8060S, 40 CUs)
- **Memory:** 128GB total, split by the BIOS into 32GB host RAM (what `/proc/meminfo` sees) + **96GB
  dedicated GPU VRAM** (a fixed UMA carve-out, confirmed via `rocminfo`'s per-agent pool sizes -
  `free -h` alone only shows the 32GB half and undersells this board). Plenty of headroom for both models.
- **OS:** Ubuntu 26.04, kernel 7.0

`repo/` is a **git submodule** pointing at the real [jaredpalmer/kev](https://github.com/jaredpalmer/kev)
- this directory only carries what we did to get it running here, not a copy of the project itself.

## The one real gotcha: ROCm 7.1 segfaults on this chip

Ubuntu 26.04 conveniently ships ROCm 7.1 + a matching PyTorch build directly via apt
(`python3-torch-rocm`) - no separate AMD repo needed. **Don't use it.** It reliably segfaults inside
`libamdhip64.so` on the very first GPU memory allocation on Strix Halo (gfx1151) - confirmed here via
`journalctl -k` (`segfault ... in libamdhip64.so.7.1.52801`, same offset every time) and matches widely
reported upstream issues ([pytorch/pytorch#173367](https://github.com/pytorch/pytorch/issues/173367),
[ROCm/ROCm#5824](https://github.com/ROCm/ROCm/issues/5824)). Raw `hipMalloc` via ctypes worked fine in
isolation, so this is specifically a bug in that ROCm 7.1 build's interaction with torch's allocator
path on this hardware, not a driver/kernel/firmware problem (`amdgpu`/`kfd` init is clean).

**The fix:** use the official PyTorch **ROCm 7.2** wheel from `download.pytorch.org` instead of any
apt package. `scripts/patch_repo.sh` pins this (`torch==2.14.0` via a `[tool.uv.sources]` /
`[[tool.uv.index]]` override in the submodule's `pyproject.toml`, applied at setup time - see below).
Verified locally: bf16 matmul on the actual GPU, no crash, `torch.cuda.get_device_name(0)` correctly
reports "Radeon 8060S Graphics".

One upside of this path: since it's a self-contained wheel (bundles its own ROCm runtime libs) rather
than something tied to Ubuntu's specific apt-packaged Python build, it isn't picky about *which*
Python 3.14 you use - Homebrew's works just as well as the system one (see below). You don't need
`python3-torch-rocm`/`rocm-smi` installed via apt at all for this path; the only real OS-level prereq is
GPU device access.

## One-time setup

### 1. sudo prerequisite (you run this once)

```bash
sudo usermod -aG render "$USER"
```

("video" guards the display device `/dev/dri/cardN`, not needed for headless compute - only "render"
is needed, for `/dev/kfd` + `/dev/dri/renderD128`.) You don't need to log out/in afterwards -
`scripts/run_gpu.sh` wraps GPU commands in `sg render -c '...'`, which picks up new group membership
immediately in any shell.

### 2. Environment + models

```bash
cd kev-local
./scripts/setup_env.sh          # patches the submodule's pyproject (see patch_repo.sh), creates .venv,
                                 # installs kev + deps incl. the ROCm 7.2 torch wheel, runs a GPU smoke test
./scripts/download_models.sh    # pulls Qwen3.5-4B-Base, Qwen3.5-9B-Base, kev-4b, kev-9b into models/hf-cache
                                 # (safe to re-run - resumes/skips what's already there)
```

Export `HF_TOKEN` beforehand (e.g. `export HF_TOKEN=...` or have it in your shell profile) for faster,
rate-limit-free downloads - the `hf` CLI picks it up automatically, no script changes needed.

`setup_env.sh` also points `uv`'s project-venv at `kev-local/.venv` explicitly
(`UV_PROJECT_ENVIRONMENT`), rather than trusting its default (which is `<pyproject dir>/.venv` -
i.e. it would otherwise create a stray venv *inside the submodule* at `repo/.venv`, which happened once
during setup and wasted 20+ minutes installing an unused CUDA build of torch there).

## Running a model

```bash
./scripts/serve_model.sh 4b        # serves jaredpalmer/kev-4b on :8008
./scripts/serve_model.sh 9b 8009   # or 9b on a different port
```

In another terminal:

```bash
./scripts/test_model.sh 8008       # sends a sample /v1/systemone request, pretty-prints the response
```

Or with the TypeSafe-compatible Python SDK:

```python
from typesafe_sdk import Noul, Choice, Score, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")
response = client.system_one(
    state="I was charged twice. Please fix this ASAP.",
    questions={
        "billing": Noul(instructions="Is this about billing?"),
        "tone": Choice(instructions="Customer tone?", criteria={"calm": None, "frustrated": None, "angry": None}),
    },
)
```

Fused Triton/`flash-linear-attention` kernels and CUDA graphs (`KEV_FUSED`, `KEV_CUDA_GRAPHS`) are
CUDA-tuned and untested on ROCm/gfx1151, so `serve_model.sh` disables both and runs the plain eager
PyTorch path. Slower than a tuned CUDA server, fine for local use; revisit if you want to chase
throughput.

## Layout

```
kev-local/
  repo/                git submodule -> jaredpalmer/kev (pyproject.toml patched in-place by
                        patch_repo.sh at setup time - see "the one real gotcha" above; the submodule
                        itself is never committed with local edits, only its own upstream commits)
  models/hf-cache/     HF_HOME - all downloaded weights live here (gitignored)
  .venv/               project venv (gitignored)
  scripts/
    setup_env.sh       one-time: patches repo, creates venv, installs deps, GPU smoke test
    patch_repo.sh       the actual pyproject.toml patch (python cap, torch source/version)
    download_models.sh pulls the 4 HF repos needed (2 bases + 2 adapters)
    run_gpu.sh         runs a command with the render group active (no relogin needed)
    serve_model.sh     starts kev.serve for 4b or 9b
    test_model.sh      sample /v1/systemone request against a running server
  Dockerfile           optional fallback (rocm/pytorch base image) if you ever need this
                        containerized or on a different distro - not needed on this machine
```

## Using a different Python (e.g. Homebrew's) instead of the system one

Because the fix above uses PyTorch's official ROCm 7.2 *wheel* (self-contained, not tied to one
distro's Python build the way apt's `python3-torch-rocm` was), this is no longer the landmine it used
to be - `setup_env.sh` hardcodes `/usr/bin/python3.14`, but you can point it at Homebrew's
(`/home/linuxbrew/.linuxbrew/bin/python3.14`) or any other cp314 interpreter and it should work the same
way, since `download.pytorch.org/whl/rocm7.2` publishes a `cp314` wheel independent of how that
interpreter was built.

## Troubleshooting

- **`Unable to open /dev/kfd read-write: Permission denied` / rocminfo complains about the "render"
  group** - you haven't run the `usermod` command above yet, or this shell predates it and you're not
  going through `scripts/run_gpu.sh`.
- **Segfault in `libamdhip64.so` on any GPU tensor op** - you're on Ubuntu's apt ROCm 7.1 torch instead
  of the pinned ROCm 7.2 wheel. Re-run `scripts/setup_env.sh` (it re-patches + re-syncs) rather than
  installing torch by hand.
- **Downloads stall at a fixed byte count for minutes** - the `hf` CLI's default "xet" CDN backend
  stalled repeatedly on this network. `download_models.sh` sets `HF_HUB_DISABLE_XET=1` to force plain
  HTTP, which was reliable here.
- **`uv sync` seems to hang or take a very long time, or you find an unused `repo/.venv`** - check
  `UV_PROJECT_ENVIRONMENT` is actually set (`setup_env.sh` does this) - without it, `uv sync` creates
  its own venv inside the submodule instead of using `kev-local/.venv`.
