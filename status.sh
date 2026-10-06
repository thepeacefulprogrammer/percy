#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/runtime-env.sh"
cd "$PERCY_ROOT_DIR"
if [[ -f "$PERCY_RUN_DIR/server.pid" ]] && kill -0 "$(cat "$PERCY_RUN_DIR/server.pid")" 2>/dev/null; then
  echo "Running supervisor PID $(cat "$PERCY_RUN_DIR/server.pid")"
  if [[ -f "$PERCY_RUN_DIR/server.child.pid" ]] && kill -0 "$(cat "$PERCY_RUN_DIR/server.child.pid")" 2>/dev/null; then
    echo "Child PID $(cat "$PERCY_RUN_DIR/server.child.pid")"
  fi
  if [[ -f "$PERCY_RUN_DIR/server.watchdog.pid" ]] && kill -0 "$(cat "$PERCY_RUN_DIR/server.watchdog.pid")" 2>/dev/null; then
    echo "Watchdog PID $(cat "$PERCY_RUN_DIR/server.watchdog.pid")"
  fi
  echo "URL: $PERCY_PUBLIC_URL"
else
  echo "Not running"
fi
