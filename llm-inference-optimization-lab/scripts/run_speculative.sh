#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

echo "Starting speculative decoding service on port ${SPECULATIVE_PORT:-8003}..."
python -m uvicorn app.speculative_service.main:app \
    --host 0.0.0.0 \
    --port "${SPECULATIVE_PORT:-8003}" \
    --reload
