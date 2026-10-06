#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/runtime-env.sh"
cd "$PERCY_ROOT_DIR"
./start.sh >/dev/null || true
termux-open-url "$PERCY_PUBLIC_URL"
