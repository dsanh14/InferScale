# Future Work

## High-Priority Extensions

### True TensorRT-LLM Backend
- Implement `TensorRTModelOptAdapter` with real `modelopt` + `tensorrt_llm` calls
- Build INT8/FP8 engines from calibrated checkpoints
- Compare TRT throughput against HuggingFace baseline on A100/H100

### Triton Inference Server Integration
- Deploy each service as a Triton backend
- Use Triton's dynamic batching and model ensemble features
- Compare Triton-managed vs. standalone FastAPI serving

### NVIDIA Dynamo Integration
- Implement the Dynamo disaggregated prefill/decode protocol
- Replace simulated KV cache transfer with real tensor serialization
- Test with Dynamo's router and worker orchestration

### Real KV Cache Transfer
- Serialize `past_key_values` to shared memory or RDMA buffer
- Implement zero-copy transfer between prefill and decode GPUs
- Measure actual transfer bandwidth vs. simulated estimates

## Medium-Priority Extensions

### GPU Profiling
- Integrate NVIDIA Nsight Systems for kernel-level profiling
- Add `torch.cuda.Event` timing for precise GPU measurement
- Profile attention kernel performance across sequence lengths

### Smart Router Policies
- Load-aware routing based on real-time GPU utilization
- SLA-based routing (latency targets per request class)
- A/B testing support for comparing serving modes in production

### Continuous Batching
- Implement an iteration-level scheduler (like vLLM's approach)
- Add preemption support for long-running requests
- Measure throughput scaling with batch size

### CUDA Kernels
- Custom fused attention kernels for prefill
- Flash Attention integration for memory-efficient attention
- Custom quantization kernels for mixed-precision compute

## Lower-Priority Extensions

### Multi-Model Serving
- Serve multiple models from a shared GPU memory pool
- Implement model swapping and warm-up strategies
- LoRA adapter hot-swapping

### Distributed Serving
- Tensor parallelism across multiple GPUs
- Pipeline parallelism for very large models
- Inter-node communication optimization

### Advanced Speculative Decoding
- Tree-based speculative decoding (multiple candidates per step)
- Self-speculative decoding (early exit from the same model)
- Adaptive draft length based on acceptance rate history

### Production Hardening
- Request queuing with backpressure
- Graceful degradation under load
- Health check cascading and circuit breakers
- Rate limiting and authentication
