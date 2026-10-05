#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .run

child_pid=""
stopping=0

cleanup() {
  stopping=1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  rm -f .run/server.child.pid .run/server.pid
  exit 0
}

trap cleanup INT TERM

echo $$ > .run/server.pid

while true; do
  /data/data/com.termux/files/usr/bin/node server.js >> .run/server.log 2>&1 &
  child_pid=$!
  echo "$child_pid" > .run/server.child.pid

  set +e
  wait "$child_pid"
  exit_code=$?
  set -e

  rm -f .run/server.child.pid

  if [[ "$stopping" -eq 1 ]]; then
    break
  fi

  printf '%s server.js exited with code %s, restarting in 1s\n' "$(date -Is)" "$exit_code" >> .run/server.log
  sleep 1

done

rm -f .run/server.pid
