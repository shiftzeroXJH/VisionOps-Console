#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT/logs/platform.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "platform.pid not found. YOLO Platform may already be stopped."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "YOLO Platform process $PID stopped."
else
  echo "Process $PID is not running. Removed stale pid file."
fi
rm -f "$PID_FILE"
