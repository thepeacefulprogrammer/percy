#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .run

if [[ -f .run/server.pid ]]; then
  pid="$(cat .run/server.pid)"
  if kill -0 "$pid" 2>/dev/null; then
    echo "Already running on supervisor PID $pid"
    exit 0
  fi
  rm -f .run/server.pid
fi

rm -f .run/server.child.pid
nohup ./supervisor.sh > .run/supervisor.log 2>&1 < /dev/null &
supervisor_pid=$!
echo "$supervisor_pid" > .run/server.pid

for _ in {1..20}; do
  if curl -fsS --max-time 2 http://127.0.0.1:8787/api/healthz >/dev/null 2>&1; then
    echo "Started Pi Web Chat: http://127.0.0.1:8787"
    exit 0
  fi
  sleep 0.5
done

echo "Supervisor started on PID $supervisor_pid, but health check did not pass yet"
exit 1
