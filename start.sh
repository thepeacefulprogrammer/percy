#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/runtime-env.sh"
cd "$PERCY_ROOT_DIR"
mkdir -p "$PERCY_RUN_DIR"

if [[ -f "$PERCY_RUN_DIR/server.pid" ]]; then
  pid="$(cat "$PERCY_RUN_DIR/server.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "Already running on supervisor PID $pid"
    exit 0
  fi
  rm -f "$PERCY_RUN_DIR/server.pid"
fi

rm -f "$PERCY_RUN_DIR/server.child.pid"
nohup ./supervisor.sh > "$PERCY_RUN_DIR/supervisor.log" 2>&1 < /dev/null &
supervisor_pid=$!
echo "$supervisor_pid" > "$PERCY_RUN_DIR/server.pid"

for _ in {1..20}; do
  if curl -fsS --max-time 2 "$PERCY_READYZ_URL" >/dev/null 2>&1; then
    echo "Started Pi Web Chat: $PERCY_PUBLIC_URL"
    exit 0
  fi
  sleep 0.5
done

echo "Supervisor started on PID $supervisor_pid, but readiness check did not pass yet ($PERCY_READYZ_URL)"
exit 1
