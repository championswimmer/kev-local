#!/usr/bin/env bash
# Downloads the Qwen3.5 base weights + Kev LoRA adapters into models/hf-cache.
# Safe to re-run: hf download resumes/skips already-fetched files.
#
# Usage: download_models.sh [4b|9b] [--remove]
#   (no size)   both 4b and 9b
#   4b | 9b     only that size (base + adapter)
#   --remove    delete the selected size(s) from the cache instead (asks for confirmation)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$SCRIPT_DIR/../models/hf-cache"
# The xet CDN backend stalls/retries heavily on this network; plain HTTP is reliable.
export HF_HUB_DISABLE_XET=1

usage() { echo "Usage: $0 [4b|9b] [--remove]" >&2; exit 1; }

SIZES=()
REMOVE=0
for arg in "$@"; do
  case "$arg" in
    4b|9b) SIZES+=("$arg") ;;
    --remove) REMOVE=1 ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $arg" >&2; usage ;;
  esac
done
[ ${#SIZES[@]} -eq 0 ] && SIZES=(4b 9b)

repos_for() {
  case "$1" in
    4b) echo "Qwen/Qwen3.5-4B-Base jaredpalmer/kev-4b" ;;
    9b) echo "Qwen/Qwen3.5-9B-Base jaredpalmer/kev-9b" ;;
  esac
}

REPOS=()
for size in "${SIZES[@]}"; do
  for repo in $(repos_for "$size"); do REPOS+=("$repo"); done
done

if [ "$REMOVE" -eq 1 ]; then
  # Cache layout is $HF_HOME/hub/models--<org>--<name> (see note below on --cache-dir).
  DIRS=()
  for repo in "${REPOS[@]}"; do
    dir="$HF_HOME/hub/models--${repo//\//--}"
    [ -d "$dir" ] && DIRS+=("$dir")
  done
  if [ ${#DIRS[@]} -eq 0 ]; then
    echo "Nothing to remove for: ${SIZES[*]}"
    exit 0
  fi
  echo "The following will be deleted:"
  du -sh "${DIRS[@]}"
  read -r -p "Proceed? [y/N] " answer
  case "$answer" in
    y|Y|yes|YES) rm -rf "${DIRS[@]}"; echo "=== Removed ===" ;;
    *) echo "Aborted." ;;
  esac
  exit 0
fi

mkdir -p "$HF_HOME"
# Deliberately NOT passing --cache-dir: that puts files directly under $HF_HOME
# (models--*/ with no "hub" level), while every HF_HOME-only consumer (transformers,
# kev.serve, huggingface_hub itself) looks under $HF_HOME/hub/models--*. Mixing the two
# once produced two divergent, half-populated caches and a very confusing hang - let
# HF_HOME alone pick the standard layout everywhere.
for repo in "${REPOS[@]}"; do
  echo "=== Downloading $repo ==="
  hf download "$repo"
done

echo "=== DONE ==="
