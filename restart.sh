#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
./stop.sh || true
sleep 1
./start.sh
