#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
sleep 1
./stop.sh >/dev/null 2>&1 || true
sleep 1
./start.sh >/dev/null 2>&1 || true
