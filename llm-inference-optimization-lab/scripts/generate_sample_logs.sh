#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

echo "Generating sample event logs for replay validator testing..."

# Generate two identical log files
cat > logs/sample_trace_a.jsonl << 'JSONL'
{"timestamp_ns": 1000000000, "request_id": "req001", "service_name": "baseline", "phase": "inference", "event_type": "generate_start", "sequence_no": 0, "payload_hash": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4", "metadata": {"prompt_len": 10}}
{"timestamp_ns": 1000500000, "request_id": "req001", "service_name": "baseline", "phase": "inference", "event_type": "generate_done", "sequence_no": 1, "payload_hash": "d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1", "metadata": {"output_tokens": 32, "latency_ms": 500.0}}
{"timestamp_ns": 1001000000, "request_id": "req002", "service_name": "baseline", "phase": "inference", "event_type": "generate_start", "sequence_no": 0, "payload_hash": "1a2b3c4d5e6f1a2b3c4d5e6f1a2b3c4d", "metadata": {"prompt_len": 25}}
{"timestamp_ns": 1001800000, "request_id": "req002", "service_name": "baseline", "phase": "inference", "event_type": "generate_done", "sequence_no": 1, "payload_hash": "4d3c2b1a6f5e4d3c2b1a6f5e4d3c2b1a", "metadata": {"output_tokens": 64, "latency_ms": 800.0}}
JSONL

# Copy as identical trace for matching test
cp logs/sample_trace_a.jsonl logs/sample_trace_b.jsonl

# Create a divergent trace
cat > logs/sample_trace_divergent.jsonl << 'JSONL'
{"timestamp_ns": 1000000000, "request_id": "req001", "service_name": "baseline", "phase": "inference", "event_type": "generate_start", "sequence_no": 0, "payload_hash": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4", "metadata": {"prompt_len": 10}}
{"timestamp_ns": 1000500000, "request_id": "req001", "service_name": "baseline", "phase": "inference", "event_type": "generate_done", "sequence_no": 1, "payload_hash": "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", "metadata": {"output_tokens": 31, "latency_ms": 499.0}}
{"timestamp_ns": 1001000000, "request_id": "req002", "service_name": "baseline", "phase": "inference", "event_type": "generate_start", "sequence_no": 0, "payload_hash": "1a2b3c4d5e6f1a2b3c4d5e6f1a2b3c4d", "metadata": {"prompt_len": 25}}
{"timestamp_ns": 1001800000, "request_id": "req002", "service_name": "baseline", "phase": "inference", "event_type": "generate_done", "sequence_no": 1, "payload_hash": "4d3c2b1a6f5e4d3c2b1a6f5e4d3c2b1a", "metadata": {"output_tokens": 64, "latency_ms": 800.0}}
JSONL

echo "Created:"
echo "  logs/sample_trace_a.jsonl (reference trace)"
echo "  logs/sample_trace_b.jsonl (identical copy)"
echo "  logs/sample_trace_divergent.jsonl (diverges at event 1)"
echo ""
echo "Test with:"
echo "  ./cpp/replay_validator/build/replay_validator --log_a logs/sample_trace_a.jsonl --log_b logs/sample_trace_b.jsonl"
echo "  ./cpp/replay_validator/build/replay_validator --log_a logs/sample_trace_a.jsonl --log_b logs/sample_trace_divergent.jsonl"
