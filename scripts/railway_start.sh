#!/usr/bin/env bash
# Railway: Streamlit on $PORT (public) + daily pipeline loop in the background.
# Requires a persistent volume mounted at /data when DB_PATH=/data/etf_bot.db.
set -u

DB_PATH="${DB_PATH:-/data/etf_bot.db}"
export DB_PATH
mkdir -p "$(dirname "$DB_PATH")"

cleanup() {
  if [[ -n "${ST_PID:-}" ]] && kill -0 "$ST_PID" 2>/dev/null; then
    kill -TERM "$ST_PID" 2>/dev/null || true
  fi
  if [[ -n "${SCHED_PID:-}" ]] && kill -0 "$SCHED_PID" 2>/dev/null; then
    kill -TERM "$SCHED_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}

trap cleanup EXIT SIGTERM SIGINT

if [[ "${ENABLE_SCHEDULER:-true}" == "true" ]]; then
  python main.py --schedule --time "${SCHEDULE_UTC_TIME:-13:35}" &
  SCHED_PID=$!
fi

PORT="${PORT:-8501}"
streamlit run dashboard.py \
  --server.port="$PORT" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false \
  &
ST_PID=$!

wait "$ST_PID"
