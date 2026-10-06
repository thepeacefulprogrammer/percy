#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/runtime-env.sh"
cd "$PERCY_ROOT_DIR"
if [[ ! -f "$PERCY_RUN_DIR/server.pid" ]]; then
  echo "Not running"
  exit 0
fi

pid="$(cat "$PERCY_RUN_DIR/server.pid")"
child_pid="$(cat "$PERCY_RUN_DIR/server.child.pid" 2>/dev/null || true)"

if kill -0 "$pid" 2>/dev/null; then
  kill "$pid" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  echo "Stopped supervisor PID $pid"
else
  echo "Stale supervisor PID file ($pid)"
fi

if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
  kill "$child_pid" 2>/dev/null || true
fi

rm -f "$PERCY_RUN_DIR/server.pid" "$PERCY_RUN_DIR/server.child.pid"
