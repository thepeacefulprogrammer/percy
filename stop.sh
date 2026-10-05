#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f .run/server.pid ]]; then
  echo "Not running"
  exit 0
fi
pid="$(cat .run/server.pid)"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "Stopped PID $pid"
else
  echo "Stale PID file ($pid)"
fi
rm -f .run/server.pid
