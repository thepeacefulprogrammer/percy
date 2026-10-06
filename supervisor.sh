#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/runtime-env.sh"
cd "$PERCY_ROOT_DIR"
mkdir -p "$PERCY_RUN_DIR"

child_pid=""
watchdog_pid=""
stopping=0
log_file="$PERCY_RUN_DIR/server.log"
watchdog_interval="${PERCY_WATCHDOG_INTERVAL:-3}"
watchdog_timeout="${PERCY_WATCHDOG_TIMEOUT:-2}"
watchdog_max_failures="${PERCY_WATCHDOG_MAX_FAILURES:-3}"
watchdog_startup_grace="${PERCY_WATCHDOG_STARTUP_GRACE:-3}"

stop_watchdog() {
  if [[ -n "$watchdog_pid" ]] && kill -0 "$watchdog_pid" 2>/dev/null; then
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
  fi
  watchdog_pid=""
  rm -f "$PERCY_RUN_DIR/server.watchdog.pid"
}

cleanup() {
  stopping=1
  stop_watchdog
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  rm -f "$PERCY_RUN_DIR/server.child.pid" "$PERCY_RUN_DIR/server.watchdog.pid" "$PERCY_RUN_DIR/server.pid"
  exit 0
}

start_watchdog() {
  local monitored_pid="$1"

  (
    local startup_failures=0
    local consecutive_failures=0

    while kill -0 "$monitored_pid" 2>/dev/null; do
      sleep "$watchdog_interval"

      if ! kill -0 "$monitored_pid" 2>/dev/null; then
        exit 0
      fi

      if curl -fsS --max-time "$watchdog_timeout" "$PERCY_READYZ_URL" >/dev/null 2>&1; then
        startup_failures=0
        consecutive_failures=0
        continue
      fi

      if (( startup_failures < watchdog_startup_grace )); then
        startup_failures=$((startup_failures + 1))
        printf '%s watchdog: health check failed during startup grace (%s/%s) for pid %s\n' "$(date -Is)" "$startup_failures" "$watchdog_startup_grace" "$monitored_pid" >> "$log_file"
        continue
      fi

      consecutive_failures=$((consecutive_failures + 1))
      printf '%s watchdog: health check failed (%s/%s) for pid %s\n' "$(date -Is)" "$consecutive_failures" "$watchdog_max_failures" "$monitored_pid" >> "$log_file"

      if (( consecutive_failures >= watchdog_max_failures )); then
        printf '%s watchdog: server unhealthy, restarting pid %s\n' "$(date -Is)" "$monitored_pid" >> "$log_file"
        kill "$monitored_pid" 2>/dev/null || true
        sleep 2
        if kill -0 "$monitored_pid" 2>/dev/null; then
          kill -9 "$monitored_pid" 2>/dev/null || true
        fi
        exit 0
      fi
    done
  ) &

  watchdog_pid=$!
  echo "$watchdog_pid" > "$PERCY_RUN_DIR/server.watchdog.pid"
}

trap cleanup INT TERM

echo $$ > "$PERCY_RUN_DIR/server.pid"

while true; do
  /data/data/com.termux/files/usr/bin/node server.js >> "$log_file" 2>&1 &
  child_pid=$!
  echo "$child_pid" > "$PERCY_RUN_DIR/server.child.pid"
  start_watchdog "$child_pid"

  set +e
  wait "$child_pid"
  exit_code=$?
  set -e

  stop_watchdog
  rm -f "$PERCY_RUN_DIR/server.child.pid"

  if [[ "$stopping" -eq 1 ]]; then
    break
  fi

  printf '%s server.js exited with code %s, restarting in 1s\n' "$(date -Is)" "$exit_code" >> "$log_file"
  sleep 1

done

rm -f "$PERCY_RUN_DIR/server.pid"
