#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for name in dev-frontend dev-backend; do
  PID_FILE="$ROOT/logs/$name.pid"
  [[ -f "$PID_FILE" ]] || continue
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Stopped $name process $PID."
  fi
  rm -f "$PID_FILE"
done
