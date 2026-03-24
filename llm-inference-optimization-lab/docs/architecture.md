# Architecture

## System Overview

```
                    ┌──────────────┐
                    │   Client /   │
                    │  Benchmark   │
                    └──────┬───────┘
                           │ HTTP POST /generate
                           ▼
                    ┌──────────────┐
                    │    Router    │
                    │   :8000     │
                    └──┬──┬──┬──┬─┘
         ┌─────────────┘  │  │  └─────────────┐
         ▼                ▼  ▼                ▼
  ┌──────────┐   ┌──────────┐ ┌──────────┐ ┌────────────────────┐
  │ Baseline │   │Quantized │ │Speculative│ │  Disaggregated     │
  │  :8001   │   │  :8002   │ │  :8003   │ │  ┌───────┐         │
  └──────────┘   └──────────┘ └──────────┘ │  │Prefill │         │
                                            │  │ :8004  │         │
                                            │  └───┬───┘         │
                                            │      │ cache       │
                                            │      │ artifact    │
                                            │  ┌───▼───┐         │
                                            │  │Decode │         │
                                            │  │ :8005 │         │
                                            └──┴───────┴─────────┘
```

## Component Responsibilities

### Router (port 8000)
- Accepts all inference requests via `POST /generate`
- Resolves the serving mode from `request.mode`
- For `baseline`, `quantized`, `speculative`: forwards to the corresponding service
- For `disaggregated`: orchestrates the prefill → decode two-phase flow
- Records structured event logs for replay validation
- Exposes `/metrics` and `/metrics/prometheus` endpoints

### Baseline Service (port 8001)
- Loads a Hugging Face causal LM (default: `gpt2`)
- Runs standard autoregressive `model.generate()`
- Returns output text + InferenceMetrics (latency, TTFT, throughput)
- Works on CPU or GPU transparently

### Quantized Service (port 8002)
- Same API as baseline
- Uses a pluggable backend adapter interface (`QuantizedBackend`)
- Default: `PortableQuantizedBackend` — applies torch dynamic INT8 quantization on CPU
- Adapter stub: `TensorRTModelOptAdapter` — clean seam for NVIDIA Model Optimizer
- Includes calibration workflow abstraction

### Speculative Decoding Service (port 8003)
- Loads a draft model (distilgpt2) and a target model (gpt2)
- Implements the draft-propose / target-verify loop:
  1. Draft model greedily proposes K tokens
  2. Target model verifies in a single forward pass
  3. Accepted prefix is committed; bonus token from target on mismatch
- Tracks acceptance rate, proposed/accepted counts, TTFT

### Prefill Service (port 8004)
- Runs the prompt through the model to produce a KV cache
- Outputs a `CacheArtifact` with cache fingerprint, size, and timing
- Models the "prefill phase" of disaggregated serving

### Decode Service (port 8005)
- Receives a `CacheArtifact` from the prefill service
- Simulates network transfer latency and queue wait
- Continues autoregressive generation
- Reports combined prefill + transfer + decode latency

## Request Lifecycle

### Baseline / Quantized
```
Client → Router → Backend Service → model.generate() → Response
```

### Speculative
```
Client → Router → Speculative Service → [draft propose → target verify]×N → Response
```

### Disaggregated
```
Client → Router → Prefill Service → CacheArtifact
                → (transfer) → Decode Service → model.generate() → Response
```

## Event Logging

Every service emits structured JSONL events with:
- `timestamp_ns`, `request_id`, `service_name`, `phase`, `event_type`
- `sequence_no` (per-request monotonic counter)
- `payload_hash` (SHA-256 of metadata)

These logs feed the C++ deterministic replay validator.

## Metrics Layer

- In-memory `MetricsCollector` aggregates p50/p95/p99 latency, throughput, TTFT
- Optional Prometheus client integration for live scraping
- Each service reports its own metrics; the router aggregates across modes
