#!/usr/bin/env bash
# Start/stop a Kev model server in the background and report which port it's on.
#
# Usage:
#   kev_ctl.sh <4b|9b> start [port]   # default port: 4b->8008, 9b->8009
#   kev_ctl.sh <4b|9b> stop
#   kev_ctl.sh <4b|9b> status
#
# On success, `start` and `status` print ONLY the port number to stdout (everything
# else goes to stderr), so callers can do: port=$(./scripts/kev_ctl.sh 4b start)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEV_LOCAL="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$KEV_LOCAL/run"
mkdir -p "$RUN_DIR"

SIZE="${1:?usage: kev_ctl.sh <4b|9b> <start|stop|status> [port]}"
ACTION="${2:?usage: kev_ctl.sh <4b|9b> <start|stop|status> [port]}"
case "$SIZE" in
  4b) DEFAULT_PORT=8008 ;;
  9b) DEFAULT_PORT=8009 ;;
  *) echo "unknown size: $SIZE (expected 4b or 9b)" >&2; exit 1 ;;
esac

PIDFILE="$RUN_DIR/kev-$SIZE.pid"
PORTFILE="$RUN_DIR/kev-$SIZE.port"
LOGFILE="$RUN_DIR/kev-$SIZE.log"

is_running() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null
}

case "$ACTION" in
  start)
    if is_running; then
      echo "kev-$SIZE already running (pid $(cat "$PIDFILE"))" >&2
      cat "$PORTFILE"
      exit 0
    fi
    PORT="${3:-$DEFAULT_PORT}"
    echo "Starting kev-$SIZE on :$PORT (log: $LOGFILE) ..." >&2
    nohup "$SCRIPT_DIR/serve_model.sh" "$SIZE" "$PORT" >"$LOGFILE" 2>&1 &
    PID=$!
    disown
    echo "$PID" >"$PIDFILE"
    echo "$PORT" >"$PORTFILE"

    # Wait for it to actually come up (or fail) before printing the port - serve_model.sh
    # execs through run_gpu.sh/sg, so $PID stays valid for the real server process throughout.
    for _ in $(seq 1 120); do
      if grep -q "Uvicorn running" "$LOGFILE" 2>/dev/null; then
        echo "kev-$SIZE ready on :$PORT (pid $PID)" >&2
        echo "$PORT"
        exit 0
      fi
      if ! kill -0 "$PID" 2>/dev/null; then
        echo "kev-$SIZE failed to start - see $LOGFILE" >&2
        rm -f "$PIDFILE" "$PORTFILE"
        exit 1
      fi
      sleep 1
    done
    echo "kev-$SIZE did not report ready within 120s - see $LOGFILE" >&2
    exit 1
    ;;

  stop)
    if ! is_running; then
      echo "kev-$SIZE is not running" >&2
      rm -f "$PIDFILE" "$PORTFILE"
      exit 0
    fi
    PID="$(cat "$PIDFILE")"
    kill "$PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$PID" 2>/dev/null || true
    rm -f "$PIDFILE" "$PORTFILE"
    echo "stopped kev-$SIZE (pid $PID)" >&2
    ;;

  status)
    if is_running; then
      echo "kev-$SIZE running (pid $(cat "$PIDFILE"))" >&2
      cat "$PORTFILE"
    else
      echo "kev-$SIZE not running" >&2
      exit 1
    fi
    ;;

  *)
    echo "unknown action: $ACTION (expected start, stop, or status)" >&2
    exit 1
    ;;
esac
