.PHONY: install test lint run-router run-baseline run-quantized run-speculative \
       run-disagg benchmark cpp-build cpp-test replay-demo clean

# ── Setup ──────────────────────────────────────────────────────────
install:
	bash scripts/setup.sh

# ── Testing ────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v --tb=short

lint:
	python -m ruff check app/ tests/

# ── Service Runners ────────────────────────────────────────────────
run-router:
	bash scripts/run_router.sh

run-baseline:
	bash scripts/run_baseline.sh

run-quantized:
	bash scripts/run_quantized.sh

run-speculative:
	bash scripts/run_speculative.sh

run-disagg:
	bash scripts/run_disagg.sh

# ── Benchmarking ───────────────────────────────────────────────────
benchmark:
	bash scripts/run_all_benchmarks.sh

# ── C++ Replay Validator ──────────────────────────────────────────
cpp-build:
	@echo "Building C++ replay validator..."
	mkdir -p cpp/replay_validator/build
	cd cpp/replay_validator/build && cmake .. && make

cpp-test: cpp-build
	@echo "Running C++ replay validator tests..."
	cd cpp/replay_validator/build && ./replay_validator_test

replay-demo: cpp-build
	@echo "Generating sample logs..."
	bash scripts/generate_sample_logs.sh
	@echo ""
	@echo "=== Test 1: Matching traces (should PASS) ==="
	cd cpp/replay_validator/build && ./replay_validator \
		--log_a ../../../logs/sample_trace_a.jsonl \
		--log_b ../../../logs/sample_trace_b.jsonl || true
	@echo ""
	@echo "=== Test 2: Divergent traces (should FAIL) ==="
	cd cpp/replay_validator/build && ./replay_validator \
		--log_a ../../../logs/sample_trace_a.jsonl \
		--log_b ../../../logs/sample_trace_divergent.jsonl || true

# ── Docker ─────────────────────────────────────────────────────────
docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

# ── Cleanup ────────────────────────────────────────────────────────
clean:
	rm -rf cpp/replay_validator/build
	rm -rf logs/*.jsonl
	rm -rf results/*.csv results/*.json results/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
