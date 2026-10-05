#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
./start.sh >/dev/null || true
termux-open-url http://127.0.0.1:8787
