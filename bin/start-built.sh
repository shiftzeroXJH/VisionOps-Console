#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
PID_FILE="$LOG_DIR/platform.pid"
FRONTEND_INDEX="$ROOT/frontend/dist/index.html"
PYTHON_BIN="${YOLO_PYTHON:-python3}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$FRONTEND_INDEX" ]]; then
  echo "Frontend build not found: $FRONTEND_INDEX"
  echo "Run: cd frontend && npm ci && npm run build"
  exit 1
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "YOLO Platform is already running. PID: $(cat "$PID_FILE")"
  echo "URL: http://127.0.0.1:8765/"
  exit 0
fi

cd "$ROOT"
YOLO_DB_PATH="${YOLO_DB_PATH:-$ROOT/yolo_state.sqlite}" \
YOLO_HOST="${YOLO_HOST:-0.0.0.0}" \
YOLO_PORT="${YOLO_PORT:-8765}" \
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
nohup "$PYTHON_BIN" -m backend.api > "$LOG_DIR/platform.stdout.log" 2> "$LOG_DIR/platform.stderr.log" &
echo $! > "$PID_FILE"

echo "YOLO Platform started. PID: $(cat "$PID_FILE")"
echo "URL: http://127.0.0.1:8765/"
