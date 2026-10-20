#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${ROOT_DIR}/.run/local-qwen-coder-model.pid"
MODEL_PORT="${LOCAL_QWEN_MODEL_PORT:-8092}"
MODEL_HOST="${LOCAL_QWEN_MODEL_HOST:-127.0.0.1}"
HEALTH_URL="http://${MODEL_HOST}:${MODEL_PORT}/health"
MODELS_URL="http://${MODEL_HOST}:${MODEL_PORT}/v1/models"
LOG_FILE="${ROOT_DIR}/.run/local-qwen-coder-model.log"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "PID: $pid"
    if curl -fsS --max-time 3 "$HEALTH_URL" >/dev/null 2>&1; then
      echo "Health: ready"
      curl -fsS --max-time 3 "$MODELS_URL" || true
      exit 0
    fi
    echo "Health: starting"
    echo "Log: $LOG_FILE"
    exit 0
  fi
fi

echo "Local Qwen coder model is not running"
exit 1
