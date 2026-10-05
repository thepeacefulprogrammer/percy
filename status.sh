#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .run/server.pid ]] && kill -0 "$(cat .run/server.pid)" 2>/dev/null; then
  echo "Running supervisor PID $(cat .run/server.pid)"
  if [[ -f .run/server.child.pid ]] && kill -0 "$(cat .run/server.child.pid)" 2>/dev/null; then
    echo "Child PID $(cat .run/server.child.pid)"
  fi
  echo "URL: http://127.0.0.1:8787"
else
  echo "Not running"
fi
