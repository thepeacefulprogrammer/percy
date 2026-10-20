#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${ROOT_DIR}/.run"
mkdir -p "$RUN_DIR"

MODEL_PORT="${LOCAL_RP_MODEL_PORT:-8091}"
MODEL_HOST="${LOCAL_RP_MODEL_HOST:-127.0.0.1}"
MODEL_ALIAS="${LOCAL_RP_MODEL_ALIAS:-emo-sex-v3-rp-1b-local}"
MODEL_REF="${LOCAL_RP_MODEL_REF:-Novaciano/EMO_SEX_V3-RP-3.2-1B-Q4_K_M-iMatrix-GGUF:Q4_K_M}"
# Pi web chat's system prompt and tool list can exceed 4k tokens,
# so default to a larger context window for local model testing.
MODEL_CTX="${LOCAL_RP_MODEL_CTX:-32768}"
MODEL_THREADS="${LOCAL_RP_MODEL_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
PID_FILE="${RUN_DIR}/local-rp-model.pid"
LOG_FILE="${RUN_DIR}/local-rp-model.log"
HEALTH_URL="http://${MODEL_HOST}:${MODEL_PORT}/health"
MODELS_URL="http://${MODEL_HOST}:${MODEL_PORT}/v1/models"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    if curl -fsS --max-time 3 "$HEALTH_URL" >/dev/null 2>&1; then
      echo "Local RP model already running on PID $pid"
      echo "OpenAI endpoint: http://${MODEL_HOST}:${MODEL_PORT}/v1"
      exit 0
    fi
  fi
  rm -f "$PID_FILE"
fi

echo "Starting local RP model server on http://${MODEL_HOST}:${MODEL_PORT}"
echo "Model: ${MODEL_REF}"
echo "Log: ${LOG_FILE}"
echo "First start may take several minutes while the GGUF downloads."

nohup llama-server \
  -hf "$MODEL_REF" \
  --jinja \
  --host "$MODEL_HOST" \
  --port "$MODEL_PORT" \
  --alias "$MODEL_ALIAS" \
  -ngl 0 \
  -c "$MODEL_CTX" \
  -t "$MODEL_THREADS" \
  > "$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"

for _ in {1..240}; do
  if curl -fsS --max-time 3 "$HEALTH_URL" >/dev/null 2>&1 && curl -fsS --max-time 3 "$MODELS_URL" >/dev/null 2>&1; then
    echo "Local RP model is ready."
    echo "OpenAI endpoint: http://${MODEL_HOST}:${MODEL_PORT}/v1"
    exit 0
  fi
  sleep 2
done

echo "Model server process started, but it did not become ready in time."
echo "Check log: ${LOG_FILE}"
exit 1
