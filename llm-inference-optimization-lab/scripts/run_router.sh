#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

echo "Starting router on port ${ROUTER_PORT:-8000}..."
python -m uvicorn app.router.main:app \
    --host 0.0.0.0 \
    --port "${ROUTER_PORT:-8000}" \
    --reload
