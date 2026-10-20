#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${ROOT_DIR}/.run/local-rp-model.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Local RP model is not running"
  exit 0
fi

pid="$(cat "$PID_FILE")"
rm -f "$PID_FILE"

if ! kill -0 "$pid" 2>/dev/null; then
  echo "Local RP model is not running"
  exit 0
fi

kill "$pid" 2>/dev/null || true
for _ in {1..20}; do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Stopped local RP model (PID $pid)"
    exit 0
  fi
  sleep 0.5
done

kill -9 "$pid" 2>/dev/null || true
echo "Force-stopped local RP model (PID $pid)"
