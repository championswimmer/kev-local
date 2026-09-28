#!/usr/bin/env bash
# Runs a command with render group access (needed for /dev/kfd + /dev/dri/renderD128,
# i.e. ROCm compute), without needing a fresh login. `sg` re-reads /etc/group at exec
# time, so this works immediately after `sudo usermod -aG render,video "$USER"`, even
# in a shell that predates it. (Only "render" is needed for headless compute - "video"
# guards /dev/dri/cardN, the display device, which we don't touch. Nesting two `sg`
# calls doesn't compose - the inner one replaces the outer's group instead of adding to
# it - so this only ever requests one.)
set -euo pipefail
exec sg render -c "$(printf '%q ' "$@")"
