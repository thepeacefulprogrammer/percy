#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .run/server.pid ]] && kill -0 "$(cat .run/server.pid)" 2>/dev/null; then
  echo "Running PID $(cat .run/server.pid)"
  echo "URL: http://127.0.0.1:8787"
else
  echo "Not running"
fi
