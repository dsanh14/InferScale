#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

echo "Starting disaggregated services (prefill + decode)..."

# Start prefill service in background
echo "  Prefill on port ${PREFILL_PORT:-8004}"
python -m uvicorn app.prefill_service.main:app \
    --host 0.0.0.0 \
    --port "${PREFILL_PORT:-8004}" &
PREFILL_PID=$!

# Start decode service in background
echo "  Decode on port ${DECODE_PORT:-8005}"
python -m uvicorn app.decode_service.main:app \
    --host 0.0.0.0 \
    --port "${DECODE_PORT:-8005}" &
DECODE_PID=$!

echo "  PIDs: prefill=$PREFILL_PID decode=$DECODE_PID"
echo "  Press Ctrl+C to stop both services."

trap "kill $PREFILL_PID $DECODE_PID 2>/dev/null" EXIT
wait
