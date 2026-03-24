#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

echo "Starting quantized service on port ${QUANTIZED_PORT:-8002}..."
python -m uvicorn app.quantized_service.main:app \
    --host 0.0.0.0 \
    --port "${QUANTIZED_PORT:-8002}" \
    --reload
