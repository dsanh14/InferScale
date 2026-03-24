# NVIDIA Relevance Mapping

This document maps each project feature to NVIDIA AI platform concepts,
demonstrating how the testbed's architecture mirrors production inference
infrastructure.

## Feature → NVIDIA Concept

| Project Feature | NVIDIA Equivalent | Notes |
|----------------|-------------------|-------|
| **Baseline serving** | TensorRT-LLM runtime / Triton Inference Server | Standard autoregressive generation as the performance baseline |
| **Quantized serving** | TensorRT Model Optimizer (PTQ / INT8 / FP8) | The `QuantizedBackend` interface maps directly to modelopt's calibration + quantization pipeline |
| **`TensorRTModelOptAdapter`** | `modelopt.torch.quantization` | Stub adapter with TODO markers for real integration |
| **Speculative decoding** | TensorRT-LLM speculative decoding | Draft-target verification loop reducing per-token latency |
| **Disaggregated serving** | NVIDIA Dynamo prefill/decode disaggregation | Separate prefill and decode workers with explicit cache handoff |
| **Cache artifact** | KV cache transfer (RDMA / NVLink) | Simulated in Python; real implementation would use tensorrt_llm KV cache APIs |
| **Router / policy** | Dynamo request router / scheduler | Mode-based routing; extensible to cost-aware or load-balanced policies |
| **Structured event logs** | Infra observability / distributed tracing | Foundation for debugging nondeterminism in multi-GPU systems |
| **Replay validator** | Deterministic replay / testing infra | Systems reliability tool for validating trace consistency across runs |
| **Benchmark harness** | Performance characterization / MLPerf-style benchmarking | Systematic workload sweeps with statistical aggregation |
| **Metrics / Prometheus** | Production monitoring / Triton metrics | p50/p95/p99 latency, throughput, TTFT, queue depth |

## Integration Guide

### Swapping in TensorRT Model Optimizer

1. Install `nvidia-modelopt` and `tensorrt-llm`
2. Implement `TensorRTModelOptAdapter.prepare_model()`:
   ```python
   import modelopt.torch.quantization as mtq
   model = AutoModelForCausalLM.from_pretrained(model_name)
   mtq.quantize(model, config=mtq.INT8_DEFAULT_CFG, forward_loop=calib_loop)
   ```
3. Implement `quantize_model()` to build a TensorRT engine
4. Implement `run_inference()` using `tensorrt_llm.runtime.GenerationSession`

### Swapping in Dynamo-style Routing

1. Replace `ExplicitModePolicy` with a Dynamo-aware router that considers:
   - GPU utilization per worker
   - KV cache memory pressure
   - Request priority / SLA targets
2. Use the existing `/metrics` endpoint to feed routing decisions

### Real KV Cache Transfer

1. In `prefill_service/engine.py`, serialize the actual `past_key_values` tensors
2. Transfer via gRPC / RDMA instead of passing through the router
3. In `decode_service/engine.py`, deserialize and use as `past_key_values` input

## Skills Demonstrated

This project demonstrates competency in:

- **Inference systems design**: Understanding the full request lifecycle from routing to token generation
- **Optimization techniques**: Quantization, speculative decoding, disaggregated serving
- **Systems engineering**: Structured logging, deterministic replay, metrics collection
- **API design**: Clean adapter interfaces for swappable backends
- **Benchmarking methodology**: Systematic workload sweeps with honest interpretation
