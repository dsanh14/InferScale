# LLM Inference Optimization Lab

A systems-heavy testbed for exploring, benchmarking, and comparing modern LLM inference optimization techniques — **baseline serving, quantized inference, speculative decoding, and disaggregated prefill/decode** — with structured observability and a C++ deterministic replay validator.

## Why This Project Matters

Production LLM inference is a systems engineering challenge. Reducing latency, maximizing throughput, and maintaining quality at scale requires deep understanding of autoregressive generation bottlenecks, hardware-aware optimization, and distributed systems design. This project demonstrates that understanding by implementing the core components of an inference optimization platform in a modular, measurable, and extensible way.

## Features

| Component | Description |
|-----------|-------------|
| **Router** | FastAPI service that dispatches requests to different serving backends based on mode |
| **Baseline serving** | Standard HuggingFace `model.generate()` with full metrics collection |
| **Quantized serving** | Pluggable backend adapter: portable INT8 fallback + TensorRT Model Optimizer integration seam |
| **Speculative decoding** | Real draft-target verification loop with acceptance rate tracking |
| **Disaggregated serving** | Explicit prefill → cache artifact → decode flow with transfer timing |
| **Benchmark harness** | Full workload sweeps over prompt lengths, token counts, concurrency, and modes |
| **Metrics / observability** | p50/p95/p99 latency, TTFT, throughput, Prometheus-compatible metrics endpoint |
| **C++ replay validator** | Deterministic trace comparison tool with SHA-256 state hashing |
| **Structured event logging** | JSONL events consumed by both Python analysis and C++ replay validation |

## Architecture

```
                    ┌──────────────┐
                    │   Client /   │
                    │  Benchmark   │
                    └──────┬───────┘
                           │ POST /generate
                           ▼
                    ┌──────────────┐
                    │    Router    │
                    │   :8000     │
                    └──┬──┬──┬──┬─┘
         ┌─────────────┘  │  │  └─────────────┐
         ▼                ▼  ▼                ▼
  ┌──────────┐   ┌──────────┐ ┌──────────┐ ┌────────────────────┐
  │ Baseline │   │Quantized │ │Speculative│ │  Disaggregated     │
  │  :8001   │   │  :8002   │ │  :8003   │ │  Prefill → Decode  │
  └──────────┘   └──────────┘ └──────────┘ │  :8004     :8005   │
                                            └────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for the full component breakdown.

## Quickstart

```bash
# Clone and enter the project
cd llm-inference-optimization-lab

# Set up the environment
make install
source .venv/bin/activate

# Start services (in separate terminals)
make run-baseline        # Terminal 1
make run-router          # Terminal 2

# Send a test request
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain what a GPU is", "max_new_tokens": 32, "mode": "baseline"}'
```

### Running All Services

```bash
# Terminal 1: Baseline
make run-baseline

# Terminal 2: Quantized
make run-quantized

# Terminal 3: Speculative (loads two models — needs more memory)
make run-speculative

# Terminal 4: Disaggregated (prefill + decode)
make run-disagg

# Terminal 5: Router
make run-router

# Terminal 6: Run benchmarks
make benchmark
```

### Docker

```bash
# Launch all services
docker compose up --build -d

# Check health
curl http://localhost:8000/health

# Run benchmarks against the containerized stack
python -m app.benchmark.run_benchmarks --router-url http://localhost:8000
```

## Example Commands

```bash
# Baseline inference
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_new_tokens": 64, "mode": "baseline"}' | python -m json.tool

# Quantized inference
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantization", "max_new_tokens": 32, "mode": "quantized"}' | python -m json.tool

# Speculative decoding
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a hello world program", "max_new_tokens": 64, "mode": "speculative"}' | python -m json.tool

# Disaggregated prefill/decode
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is a transformer?", "max_new_tokens": 32, "mode": "disaggregated"}' | python -m json.tool

# Check live metrics
curl http://localhost:8000/metrics | python -m json.tool

# Prometheus metrics
curl http://localhost:8000/metrics/prometheus
```

## Benchmark Methodology

The benchmark runner sweeps over a matrix of:
- **Prompt lengths**: short (~10 tokens), medium (~40), long (~100+)
- **Output lengths**: 32, 64, 128 tokens
- **Concurrency**: 1, 2, 4, 8 concurrent requests
- **Modes**: baseline, quantized, speculative, disaggregated

This produces 144 configurations per sweep. Each configuration is measured with configurable repetitions. Results are saved to CSV/JSON and four plots are generated:

| Plot | Purpose |
|------|---------|
| `latency_by_mode.png` | Box plot comparing end-to-end latency distributions |
| `throughput_by_mode.png` | Mean tokens/second per serving mode |
| `ttft_by_prompt_length.png` | Time-to-first-token across prompt lengths |
| `speculative_acceptance_rate.png` | Distribution of draft token acceptance rates |

See [docs/benchmark_methodology.md](docs/benchmark_methodology.md) for interpretation guidance and caveats.

### Interpreting Results

With the default GPT-2 / DistilGPT-2 models on CPU:
- **Baseline** provides the reference latency for unoptimized HuggingFace generation
- **Quantized** (dynamic INT8) may show modest latency differences on CPU; the real gains come with TensorRT INT8/FP8 on GPU
- **Speculative** demonstrates correct acceptance-rate mechanics; latency may be higher due to running two models
- **Disaggregated** shows baseline latency + simulated transfer overhead, demonstrating the prefill/decode separation architecture

The purpose is to demonstrate the **infrastructure and measurement methodology**, not to claim optimization gains on a developer laptop with a 124M parameter model.

## Deterministic Replay Validator

The C++ replay validator is a systems reliability tool that detects nondeterminism in inference traces.

```bash
# Build the validator
make cpp-build

# Run the full demo (matching + divergent traces)
make replay-demo
```

**Why deterministic replay matters in distributed inference:**

In multi-GPU, multi-node inference systems, nondeterminism can arise from floating-point reduction order, asynchronous execution, load balancing changes, or software bugs. Being able to replay a request trace and verify that the same sequence of events occurs is essential for debugging, performance analysis, and correctness validation. The replay validator hashes each event's stable fields and computes a cumulative trace hash, reporting the first point of divergence when traces differ.

## NVIDIA Relevance

This project maps directly to NVIDIA inference infrastructure concepts:

| This Project | NVIDIA Equivalent |
|---|---|
| Baseline serving | TensorRT-LLM runtime |
| Quantized serving | Model Optimizer PTQ (INT8/FP8) |
| `TensorRTModelOptAdapter` | Clean integration seam for `modelopt` |
| Speculative decoding | TensorRT-LLM speculative decoding |
| Disaggregated serving | Dynamo prefill/decode separation |
| Replay validator | Systems reliability tooling |
| Metrics layer | Triton Inference Server metrics |

See [docs/nvidia_mapping.md](docs/nvidia_mapping.md) for the full mapping and integration guide.

**Design philosophy**: The adapter interfaces are real — `QuantizedBackend`, routing policies, cache artifacts — so that swapping in TensorRT-LLM, Dynamo, or Triton backends requires implementing the interface, not rewriting the architecture.

## Resume Bullet Suggestions

**Software Engineering:**
> Built a modular LLM inference testbed with 6 microservices, a benchmark harness covering 144 workload configurations, structured JSONL observability, and a C++ deterministic replay validator — demonstrating systems design for high-throughput ML serving.

**AI Infrastructure:**
> Designed and implemented a multi-mode inference optimization platform supporting baseline, quantized (INT8), speculative decoding, and disaggregated prefill/decode serving, with pluggable backend adapters for TensorRT Model Optimizer integration and comprehensive latency/throughput benchmarking.

**NVIDIA-Targeted:**
> Architected an inference optimization testbed mirroring NVIDIA Dynamo's prefill/decode disaggregation and TensorRT-LLM's speculative decoding, with adapter-based backend interfaces ready for Model Optimizer INT8/FP8 quantization — including p50/p95/p99 latency measurement, acceptance-rate tracking, and a C++ replay validator for trace determinism verification.

## Honest Limitations

- **Small models**: GPT-2 (124M) and DistilGPT-2 (82M) are used for local runnability. Optimization gains scale with model size.
- **CPU-first**: Most benchmarks run on CPU. Real quantization and speculative decoding gains require GPU execution.
- **Simulated KV cache transfer**: The disaggregated path simulates network transfer. Real implementations use RDMA or NVLink.
- **No continuous batching**: Requests are processed individually. Production systems use iteration-level scheduling.
- **TensorRT adapter is a stub**: The `TensorRTModelOptAdapter` defines the interface but raises `NotImplementedError` without the NVIDIA stack installed.
- **Speculative decoding is simplified**: The verification loop is correct but not optimized for throughput (no tree-based speculation, no batched verification).

## Future Work

- True TensorRT-LLM backend integration
- NVIDIA Triton Inference Server deployment
- Dynamo-style router with GPU-aware scheduling
- Real KV cache serialization and transfer
- GPU profiling with Nsight Systems
- Continuous batching scheduler
- Custom CUDA attention kernels

See [docs/future_work.md](docs/future_work.md) for the full roadmap.

## Project Structure

```
llm-inference-optimization-lab/
├── app/
│   ├── common/           # Shared schemas, config, events, metrics, hashing
│   ├── router/           # Request routing and dispatch
│   ├── baseline_service/ # Standard HF inference
│   ├── quantized_service/# Quantized inference with adapter interface
│   ├── speculative_service/ # Draft-target speculative decoding
│   ├── prefill_service/  # Disaggregated prefill phase
│   ├── decode_service/   # Disaggregated decode phase
│   ├── benchmark/        # Workload generation, execution, analysis, plots
│   └── ui/               # Dashboard stub
├── cpp/
│   └── replay_validator/ # C++ deterministic replay tool
├── configs/              # YAML configuration files
├── docs/                 # Architecture, methodology, NVIDIA mapping
├── scripts/              # Shell scripts for running services
├── tests/                # pytest test suite
├── sample_data/          # Benchmark prompts
└── results/              # Output directory for benchmarks
```

## Tests

```bash
make test
```

Covers routing policy, schema validation, event logging, speculative verification, disaggregated handoff, benchmark workloads, and hashing consistency.

## License

MIT
