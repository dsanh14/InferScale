#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

ROUTER_URL="${ROUTER_URL:-http://localhost:8000}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"

echo "=== Running Full Benchmark Sweep ==="
echo "Router: $ROUTER_URL"
echo "Output: $OUTPUT_DIR"
echo ""

python -m app.benchmark.run_benchmarks \
    --router-url "$ROUTER_URL" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Benchmark Complete ==="
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
