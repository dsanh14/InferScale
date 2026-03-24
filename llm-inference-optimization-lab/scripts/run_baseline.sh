#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

echo "Starting baseline service on port ${BASELINE_PORT:-8001}..."
python -m uvicorn app.baseline_service.main:app \
    --host 0.0.0.0 \
    --port "${BASELINE_PORT:-8001}" \
    --reload
