#!/bin/sh
# Start Audio Clean on http://127.0.0.1:8765 (this Mac only).
cd "$(dirname "$0")"
PORT="${PORT:-8765}"
( sleep 2 && open "http://127.0.0.1:$PORT" ) &
exec uv run uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
