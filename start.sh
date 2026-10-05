#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .run
if [[ -f .run/server.pid ]] && kill -0 "$(cat .run/server.pid)" 2>/dev/null; then
  echo "Already running on PID $(cat .run/server.pid)"
  exit 0
fi
nohup node server.js > .run/server.log 2>&1 < /dev/null &
echo $! > .run/server.pid
sleep 1
echo "Started Pi Web Chat: http://127.0.0.1:8787"
