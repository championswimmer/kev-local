# Kev, running locally on AMD Strix Halo and Apple Silicon

[Kev](https://github.com/jaredpalmer/kev) is a small open family of *decision models* by Jared Palmer —
an open reconstruction of TypeSafe's "Jev": you send it one document (a *state*) plus a set of typed
yes/no, multiple-choice or score questions, and it returns a calibrated probability per question in a
single forward pass. No text generation. It's built as LoRA adapters + a pointer head on top of Qwen3.5
(0.8B / 4B / 9B) and Qwen3.8 (27B).

This directory documents/automates running the **4B** and **9B** checkpoints locally on two machines:

- **AMD Ryzen AI Max+ 395** (Strix Halo, "gfx1151" iGPU, Radeon 8060S, 40 CUs) - 128GB total, split by
  the BIOS into 32GB host RAM (what `/proc/meminfo` sees) + **96GB dedicated GPU VRAM** (a fixed UMA
  carve-out, confirmed via `rocminfo`'s per-agent pool sizes - `free -h` alone only shows the 32GB half
  and undersells this board). Ubuntu 26.04, kernel 7.0. Runs the models on **ROCm 7.2 / torch's CUDA-style
  `cuda` device** (eager PyTorch path).
- **Apple M5 Pro** (18 cores: 6 Super + 12 Performance) - 48GB unified memory, macOS 26.6.2. Runs the
  models on **MPS + the MLX backend** (kev's own `kev.mlx_model`, Metal kernels via `mlx-lm` - picked
  automatically by `kev.serve`'s default `backend="auto"` for these hybrid Qwen3.5 checkpoints, no config
  needed).

`repo/` is a **git submodule** pointing at the real [jaredpalmer/kev](https://github.com/jaredpalmer/kev)
- this directory only carries what we did to get it running on each machine, not a copy of the project
itself. `git submodule update --init --recursive` before anything else if `repo/` is empty.

**Verified working end-to-end on both machines**: `kev-4b` and `kev-9b` load, run on the GPU, and answer
`/v1/systemone` requests correctly (`./scripts/test_model.sh` output for a sample angry-billing-complaint
state: `billing` ~0.92-0.94, `tone`="angry", `urgency`~2.9 - sensible on all four combinations of
model x machine). Per-request latency: ~194-366ms on the AMD box (eager ROCm), ~114-201ms on the Apple
M5 Pro (MLX) - see [Benchmarks](#benchmarks), which also times the real, hosted Jev (TypeSafe's original,
proprietary model that kev reconstructs) via OpenRouter for reference, at ~288ms/req over the network.

## AMD Strix Halo: the one real gotcha - ROCm 7.1 segfaults on this chip

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
reports "Radeon 8060S Graphics". One wrinkle: `[tool.uv.sources]` only redirects packages *your project
itself* depends on, not arbitrary transitive dependencies pulled in by another package's wheel metadata
- so `triton-rocm` (torch's own dependency) also has to be declared as a direct dependency here, or it
silently resolves against an unrelated ancient package of the same name on PyPI instead of the one that
ships alongside the real ROCm torch build.

One upside of this path: since it's a self-contained wheel (bundles its own ROCm runtime libs) rather
than something tied to Ubuntu's specific apt-packaged Python build, it isn't picky about *which*
Python 3.14 you use - Homebrew's works just as well as the system one (see below). You don't need
`python3-torch-rocm`/`rocm-smi` installed via apt at all for this path; the only real OS-level prereq is
GPU device access.

## Apple Silicon: no gotcha, it just works

Unlike the ROCm case above, nothing needs patching on macOS. `repo/pyproject.toml` already declares
`mlx-lm` as a dependency conditioned on `sys_platform == 'darwin' and platform_machine == 'arm64'` (kev's
own `[serve]` extra), and stock PyPI `torch` wheels support Apple's `mps` device out of the box - no
special package index like ROCm needed. `kev.serve` then defaults `LoadOptions.backend` to `"auto"`,
which picks the **MLX backend** (`kev.mlx_model`, Metal kernels via `mlx-lm`) automatically whenever the
device is `mps` and the checkpoint's base is a hybrid Qwen3.5 backbone - true for both `kev-4b` and
`kev-9b`. Confirmed via `/v1/models` and the server's own startup line on this M5 Pro:

```
serving jaredpalmer/kev-4b (...) on mps via mlx (bfloat16) 127.0.0.1:8008
```

The only real constraint is `repo/pyproject.toml`'s `requires-python = ">=3.12,<3.14"` cap, which excludes
a bare Homebrew Python 3.14. Rather than patch the submodule (as `patch_repo.sh` does for ROCm),
`scripts/setup_env_mac.sh` just asks `uv` to provision its own Python 3.13 (`uv python install 3.13`) -
no vendored-file edits needed on this platform at all.

## One-time setup

### AMD Strix Halo (Linux / ROCm)

**1. sudo prerequisite (you run this once):**

```bash
sudo usermod -aG render "$USER"
```

("video" guards the display device `/dev/dri/cardN`, not needed for headless compute - only "render"
is needed, for `/dev/kfd` + `/dev/dri/renderD128`.) You don't need to log out/in afterwards -
`scripts/run_gpu.sh` wraps GPU commands in `sg render -c '...'`, which picks up new group membership
immediately in any shell.

**2. Environment + models:**

```bash
cd kev-local
git submodule update --init --recursive
./scripts/setup_env.sh          # patches the submodule's pyproject (see patch_repo.sh), creates .venv,
                                 # installs kev + deps incl. the ROCm 7.2 torch wheel, runs a GPU smoke test
./scripts/download_models.sh    # pulls Qwen3.5-4B-Base, Qwen3.5-9B-Base, kev-4b, kev-9b into models/hf-cache
                                 # (safe to re-run - resumes/skips what's already there)
```

`setup_env.sh` also points `uv`'s project-venv at `kev-local/.venv` explicitly
(`UV_PROJECT_ENVIRONMENT`), rather than trusting its default (which is `<pyproject dir>/.venv` -
i.e. it would otherwise create a stray venv *inside the submodule* at `repo/.venv`, which happened once
during setup and wasted 20+ minutes installing an unused CUDA build of torch there).

### Apple Silicon (macOS)

No sudo/group prerequisite - Metal/MPS needs no special device permissions.

```bash
cd kev-local
git submodule update --init --recursive
./scripts/setup_env_mac.sh      # uv python install 3.13, uv sync --extra serve (torch MPS wheel +
                                 # mlx-lm, both plain PyPI on darwin/arm64), MPS + mlx smoke test
./scripts/download_models.sh    # same script as the AMD path - platform-agnostic
```

The `hf` CLI (used by `download_models.sh`) isn't installed by `setup_env_mac.sh` - if it's missing,
`uv tool install "huggingface_hub[cli]"` puts it on `~/.local/bin`, which the scripts already add to
`PATH`.

### Both platforms

Export `HF_TOKEN` beforehand (e.g. `export HF_TOKEN=...` or have it in your shell profile) for faster,
rate-limit-free downloads - the `hf` CLI picks it up automatically, no script changes needed. (On macOS,
downloads were reliably fast even without a token and without the xet-stall issue noted below - that
appears to be network/Linux-specific, not a general HF Hub issue.)

## Running a model

Same scripts on both platforms - `serve_model.sh` detects Linux vs. Darwin itself (only Linux needs the
`run_gpu.sh` render-group wrapper; macOS runs `kev.serve` directly) and `kev.serve`'s device/backend
autodetection (torch `cuda` on ROCm, `mps`+MLX on Apple Silicon) handles the rest.

**Foreground** (blocks the terminal, logs straight to stdout - good for a first run/debugging):

```bash
./scripts/serve_model.sh 4b        # serves jaredpalmer/kev-4b on :8008
./scripts/serve_model.sh 9b 8009   # or 9b on a different port
```

**Background**, with start/stop/status and the port reported on stdout:

```bash
port=$(./scripts/kev_ctl.sh 4b start)   # backgrounds it, waits for readiness, prints the port
./scripts/kev_ctl.sh 4b status          # prints the port if running (exit 1, stderr message if not)
./scripts/kev_ctl.sh 4b stop
```

(`start` is a no-op - just re-prints the port - if that size is already running; pid/port/log per size
live under `run/`, gitignored.)

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
PyTorch path there. Slower than a tuned CUDA server, fine for local use; revisit if you want to chase
throughput. These two env vars are no-ops on the MLX backend (Apple Silicon) - the MLX path always runs
the backbone as stored (bf16), ignoring both.

## Benchmarks

`scripts/benchmark.py <port> -n 100` fires sequential `/v1/systemone` requests (3 questions each:
noul/choice/score) at a running server, cycling through 10 distinct sample states so the prefix cache
doesn't flatter the numbers, and reports latency + throughput. Same script, same request shape on both
machines - only the backend differs.

### AMD Ryzen AI Max+ 395 (Strix Halo, ROCm 7.2, eager PyTorch path)

5 warmup requests, then 100 timed, ambient system otherwise idle:

| Model  | req/s | mean ms/req | p50 ms | p90 ms | p99 ms | output tok/s | mean in/out tokens |
|--------|------:|------------:|-------:|-------:|-------:|-------------:|-------------------:|
| kev-4b | 5.15  | 193.9       | 193.3  | 195.0  | 217.1  | 1116         | 82.9 / 216.6       |
| kev-9b | 2.73  | 366.1       | 365.9  | 369.7  | 370.8  | 591          | 82.9 / 216.5       |

Latency is very tight (p50 ≈ mean ≈ p90 for both) - no GPU thermal throttling or contention observed
over a 100-request run. 9B is ~1.9x the latency and ~0.53x the throughput of 4B, roughly tracking its
~2.25x parameter count. "output tokens" here is kev's own usage accounting for the pointer-head forward
pass (no text is generated), not a text-generation token count - useful as a relative throughput number
between the two models, not directly comparable to an LLM's decode tok/s.

### Apple M5 Pro (18 cores, 48GB unified memory, MLX backend via mlx-lm)

Same procedure, same sample states, 5 warmup + 100 timed:

| Model  | req/s | mean ms/req | p50 ms | p90 ms | p99 ms | output tok/s | mean in/out tokens |
|--------|------:|------------:|-------:|-------:|-------:|-------------:|-------------------:|
| kev-4b | 8.73  | 114.5       | 113.9  | 117.9  | 124.9  | 1893.5       | 82.9 / 216.9       |
| kev-9b | 4.98  | 200.7       | 200.1  | 204.7  | 208.8  | 1077.7       | 82.9 / 216.3       |

Also very tight (p50 ≈ mean ≈ p90) - the 5-request warmup absorbs MLX's one-time compile/graph-build cost
(a lone cold-start `test_model.sh` call took ~3s on kev-4b vs. the 114.5ms warmed mean; that gap didn't
recur once warm). 9B is ~1.76x the latency and ~0.57x the throughput of 4B here - a shallower gap than
ROCm's 1.9x/0.53x, plausibly because MLX's unified-memory/Metal path scales differently than eager
PyTorch on ROCm across model sizes. Faster than the AMD box on both models (~1.7-1.8x req/s) - not a
head-to-head chip comparison (different backend, dtype path and framework, not just different silicon),
but the practical number if you're choosing which machine to point a workload at.

Reproduce (either machine): `port=$(./scripts/kev_ctl.sh 4b start); python3 scripts/benchmark.py "$port" -n 100 --json`.

### For reference: the real, hosted Jev (TypeSafe's System One model) via OpenRouter

kev is an open reconstruction of TypeSafe's proprietary "Jev" - not a copy, so it's worth also timing the
real thing for a speed comparison. OpenRouter's usual chat endpoint does **not** serve it: the only
chat-completions listing (`typesafe/jev-router`) is a general-purpose model *router* that dispatches to
arbitrary underlying LLMs (three test calls with a billing/tone/urgency prompt all silently routed to
`openai/gpt-6-luna` on Azure) - not Jev, despite the name. The actual Jev decision model,
`typesafe/jev-1.13`, lives only behind OpenRouter's dedicated **`/api/alpha/decisions`** endpoint
(`chat/completions` rejects it: *"is a decisions model, use /api/alpha/decisions instead"*). That
endpoint takes the same `{model, state, questions}` shape as kev's own `/v1/systemone` and returns the
same per-question calibrated probabilities - `scripts/benchmark_openrouter.py` fires the identical sample
states/questions at it over the network with `OPENROUTER_API_KEY` and reports the same metrics.

5 warmup requests, then 100 timed, sequential, from the Apple M5 Pro's network connection:

| Model                       | req/s | mean ms/req | p50 ms | p90 ms | p99 ms | output tok/s | mean in/out tokens |
|------------------------------|------:|------------:|-------:|-------:|-------:|-------------:|-------------------:|
| jev-1.13 (`typesafe/jev-1.13-20260917`) | 3.47  | 288.1       | 264.3  | 370.9  | 1061.2 | 258.5        | 384.9 / 74.5        |

This is **not a fair head-to-head with the two boxes above** - it's a network round trip to a hosted,
shared, metered API (TypeSafe's own infra, whatever machine and load it happens to be under), not a
local process on hardware we control; the p99 tail (1061ms vs. a ~371ms p90) shows exactly that kind of
external variance neither local benchmark exhibits. Input/output token accounting also isn't directly
comparable - TypeSafe's own tokenizer counts ~4.6x the input tokens kev's does for the identical
state+questions JSON, and far fewer output tokens (74.5 vs. kev's ~217; kev-4b/9b returns 3 richer
per-question fields, Jev's response is more compact). Treat this purely as "what does calling the real,
hosted product cost in wall-clock time," not as a hardware or model-quality comparison - both local kev
boxes are meaningfully faster in this measurement, but they're also not paying for network egress,
queueing, or someone else's multi-tenant load. Each request costs a small metered amount (~$0.0000165
seen here; 100 requests ≈ $0.0017) - real spend against `OPENROUTER_API_KEY`, unlike the free local runs.

Reproduce: `OPENROUTER_API_KEY=... python3 scripts/benchmark_openrouter.py -n 100 --json`.

### Quantization: investigated, not worth it here

Both serving paths run bf16 today (no quantization). Looked into whether a 4-bit/8-bit MLX quantized
checkpoint could speed up serving on the Apple box - it can't, without upstream changes to kev itself:

- `kev/mlx_model.py`'s `merge_lora` folds the LoRA adapter into the base with plain float arithmetic
  (`base.astype(mx.float32) + delta`) directly on the loaded weight tensors. A quantized mlx-lm layer's
  "weight" isn't a plain tensor - it's a packed `uint32` array plus separate `scales`/`biases` - so this
  merge can't run on it as-is.
- `kev/checkpoint.py` also refuses the other order (quantize first, merge later): serving a pre-merged
  full-weight checkpoint on `backend="mlx"` raises outright (`"the MLX backend merges an adapter into the
  base; full-weight checkpoints run on backend=torch"`) - the MLX path only ever accepts a LoRA adapter +
  unquantized base, by design.

Even setting that aside, quantization's usual win (relieving memory-bandwidth pressure during
autoregressive decode) doesn't apply to kev's serving path: it's a single prefill-style forward pass per
request, not token-by-token decode - the "output tokens" in the benchmarks above are pointer-head usage
accounting, not generated tokens. A short batched prefill is compute-bound on the GEMM kernels instead,
where 4-bit weights mostly add dequant overhead rather than remove a bottleneck. The real payoff of
quantization here would be ~4x smaller memory footprint (useful for fitting Kev-27B), not faster
requests for 4B/9B - not pursued for that reason.

## Layout

```
kev-local/
  repo/                git submodule -> jaredpalmer/kev (on Linux/ROCm, pyproject.toml patched in-place
                        by patch_repo.sh at setup time - see "the one real gotcha" above; untouched on
                        macOS. The submodule itself is never committed with local edits either way, only
                        its own upstream commits)
  models/hf-cache/     HF_HOME - all downloaded weights live here (gitignored)
  .venv/               project venv (gitignored)
  scripts/
    setup_env.sh       Linux/ROCm one-time setup: patches repo, creates venv, installs deps incl. the
                        ROCm 7.2 torch wheel, GPU smoke test
    setup_env_mac.sh   macOS one-time setup: uv-provisions Python 3.13, creates venv, installs deps
                        (plain PyPI torch MPS wheel + mlx-lm, no submodule patch needed), MPS+mlx smoke test
    patch_repo.sh       the actual pyproject.toml patch for ROCm (python cap, torch source/version)
    download_models.sh pulls the 4 HF repos needed (2 bases + 2 adapters) - same on both platforms
    run_gpu.sh         Linux only: runs a command with the render group active (no relogin needed)
    serve_model.sh     starts kev.serve for 4b or 9b (foreground); detects Linux vs. Darwin itself to
                        decide whether to go through run_gpu.sh
    kev_ctl.sh         start|stop|status a model in the background; prints its port - same on both
    test_model.sh      sample /v1/systemone request against a running server - same on both
    benchmark.py       N sequential requests against a running server; latency + throughput - same on both
    benchmark_openrouter.py  same benchmark methodology against the real, hosted Jev via OpenRouter's
                        /api/alpha/decisions endpoint (needs OPENROUTER_API_KEY) - for reference only,
                        see "the real, hosted Jev" section above
  Dockerfile           optional fallback (rocm/pytorch base image) for the AMD box if you ever need this
                        containerized or on a different Linux distro - not needed for local runs on
                        either machine documented here
```

## Using a different Python (e.g. Homebrew's) instead of the system one

**AMD/Linux:** because the ROCm fix above uses PyTorch's official ROCm 7.2 *wheel* (self-contained, not
tied to one distro's Python build the way apt's `python3-torch-rocm` was), this is no longer the landmine
it used to be - `setup_env.sh` hardcodes `/usr/bin/python3.14`, but you can point it at Homebrew's
(`/home/linuxbrew/.linuxbrew/bin/python3.14`) or any other cp314 interpreter and it should work the same
way, since `download.pytorch.org/whl/rocm7.2` publishes a `cp314` wheel independent of how that
interpreter was built.

**Apple Silicon:** `setup_env_mac.sh` doesn't hardcode a system Python at all - it has `uv` provision and
manage its own Python 3.13 (`uv python install 3.13`), sidestepping the submodule's `requires-python
<3.14` cap without touching Homebrew's own interpreter or the vendored `pyproject.toml`.

## Troubleshooting

- **`Unable to open /dev/kfd read-write: Permission denied` / rocminfo complains about the "render"
  group** (Linux) - you haven't run the `usermod` command above yet, or this shell predates it and
  you're not going through `scripts/run_gpu.sh`.
- **Segfault in `libamdhip64.so` on any GPU tensor op** (Linux) - you're on Ubuntu's apt ROCm 7.1 torch
  instead of the pinned ROCm 7.2 wheel. Re-run `scripts/setup_env.sh` (it re-patches + re-syncs) rather
  than installing torch by hand.
- **Downloads stall at a fixed byte count for minutes** (seen on Linux, not on macOS) - the `hf` CLI's
  default "xet" CDN backend stalled repeatedly on the AMD box's network. `download_models.sh` sets
  `HF_HUB_DISABLE_XET=1` to force plain HTTP, which was reliable there; downloads on the Apple M5 Pro
  were reliable without hitting this at all.
- **`hf: command not found` when running `download_models.sh` on macOS** - `setup_env_mac.sh` doesn't
  install the `hf` CLI itself (it's a general-purpose tool, not a project dependency). Install it once
  with `uv tool install "huggingface_hub[cli]"`, which puts it on `~/.local/bin` - the same PATH the
  scripts already prepend.
- **`uv sync` seems to hang or take a very long time, or you find an unused `repo/.venv`** - check
  `UV_PROJECT_ENVIRONMENT` is actually set (`setup_env.sh` does this) - without it, `uv sync` creates
  its own venv inside the submodule instead of using `kev-local/.venv`.
- **`kev.serve` hangs forever at `Fetching N files: 0%` while loading the base model, and killing +
  retrying doesn't help** - this happened once from mixing cache conventions: `hf download --cache-dir
  X` puts files directly under `X/models--*`, while everything that only sets the `HF_HOME` *env var*
  (transformers, `kev.serve`, huggingface_hub itself) looks under `X/hub/models--*` instead. The two
  silently produced separate, differently-populated caches, and `kev.serve` ended up retrying network
  fetches against a mostly-empty second copy while the real, complete one sat unused right next to it.
  `download_models.sh` no longer passes `--cache-dir` for exactly this reason - if you scripted your own
  download some other way, make sure it didn't either (`find models/hf-cache -maxdepth 1` should show
  only a `hub/` directory, no stray top-level `models--*` dirs next to it).
