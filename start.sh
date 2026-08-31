#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
PYTHON_BIN="${YOLO_PYTHON:-python3}"
mkdir -p "$LOG_DIR"

cd "$ROOT"
YOLO_DB_PATH="${YOLO_DB_PATH:-$ROOT/yolo_state.sqlite}" \
YOLO_HOST="${YOLO_HOST:-127.0.0.1}" \
YOLO_PORT="${YOLO_PORT:-8765}" \
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
nohup "$PYTHON_BIN" -m backend.api > "$LOG_DIR/dev-backend.stdout.log" 2> "$LOG_DIR/dev-backend.stderr.log" &
echo $! > "$LOG_DIR/dev-backend.pid"

cd "$ROOT/frontend"
nohup npm run dev -- --host 127.0.0.1 > "$LOG_DIR/dev-frontend.stdout.log" 2> "$LOG_DIR/dev-frontend.stderr.log" &
echo $! > "$LOG_DIR/dev-frontend.pid"

echo "Development servers started."
echo "Frontend: http://127.0.0.1:5173/"
echo "Backend:  http://127.0.0.1:8765/"
