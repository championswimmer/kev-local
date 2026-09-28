# Optional/fallback path for running Kev in a container with ROCm GPU passthrough.
#
# Not needed on this machine (Ubuntu 26.04 already ships ROCm 7.1 + a matching PyTorch
# build directly via apt - see scripts/setup_env.sh, which is the primary, faster path:
# no ~15GB base-image pull, no duplicate driver stack). Use this Dockerfile if you want
# to run Kev on a different distro/host, or want the environment fully isolated.
#
# Build:
#   docker build -t kev-local .
# Run (needs the host's amdgpu/kfd devices passed through):
#   docker run --rm -it \
#     --device=/dev/kfd --device=/dev/dri \
#     --group-add video --group-add render \
#     --security-opt seccomp=unconfined \
#     --shm-size 8g \
#     -p 8008:8008 \
#     -v "$(pwd)/models/hf-cache:/root/.cache/huggingface" \
#     kev-local \
#     python -m kev.serve --run jaredpalmer/kev-4b --port 8008

FROM rocm/pytorch:latest

WORKDIR /opt/kev
COPY repo/ /opt/kev/

# Same override as scripts applied locally to repo/pyproject.toml: the base image ships a
# newer torch/python than the repo's original pins, and the pins are about wheel
# availability upstream, not a real functional requirement.
RUN pip install --no-cache-dir --no-deps -e ".[serve]" || \
    (pip install --no-cache-dir -e ".[serve]")

ENV KEV_FUSED=0
ENV KEV_CUDA_GRAPHS=0
ENV KEV_DTYPE=bf16
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8008
CMD ["python", "-m", "kev.serve", "--run", "jaredpalmer/kev-4b", "--port", "8008"]
