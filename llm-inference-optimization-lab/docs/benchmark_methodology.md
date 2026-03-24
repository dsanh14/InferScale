# Benchmark Methodology

## Workload Design

The benchmark runner executes a full cross-product sweep over:

| Parameter | Values |
|-----------|--------|
| Prompt length bucket | short, medium, long |
| Max new tokens | 32, 64, 128 |
| Concurrency level | 1, 2, 4, 8 |
| Serving mode | baseline, quantized, speculative, disaggregated |

This produces **144 distinct workload configurations** per sweep.

### Prompt Buckets

- **Short** (~10 tokens): Single-sentence factual question
- **Medium** (~40 tokens): Multi-topic technical explanation request
- **Long** (~100+ tokens): Detailed technical writing prompt

Prompts are stored in `sample_data/prompts/` and loaded at runtime.

## Metrics Collected

For every request:

| Metric | Description |
|--------|-------------|
| `total_latency_ms` | End-to-end time from request sent to response received |
| `client_latency_ms` | Client-side round-trip time including network overhead |
| `ttft_ms` | Time to first token (estimated) |
| `tokens_per_second` | Output tokens / total generation time |
| `output_tokens` | Number of tokens generated |
| `prompt_tokens` | Number of tokens in the input |
| `queue_wait_ms` | Time waiting in queue (disaggregated mode) |
| `transfer_latency_ms` | Cache artifact transfer time (disaggregated mode) |
| `speculative_proposed` | Tokens proposed by draft model |
| `speculative_accepted` | Tokens accepted by target model |
| `speculative_acceptance_rate` | accepted / proposed |
| `success` | Whether the request completed without error |

## Aggregation

Results are grouped by `(mode, prompt_bucket, max_new_tokens, concurrency)` and aggregated:

- **mean**, **median**, **min**, **max**, **std** for latency and throughput
- **success_rate** per group

## Plots Generated

1. **latency_by_mode.png** — Box plot of latency distribution per mode
2. **throughput_by_mode.png** — Bar chart of mean tokens/second per mode
3. **ttft_by_prompt_length.png** — Grouped bars: TTFT across prompt lengths and modes
4. **speculative_acceptance_rate.png** — Histogram of acceptance rates

## How to Interpret Results

- **Baseline** establishes the reference point for unoptimized HF inference
- **Quantized** should show similar or slightly lower latency on CPU (dynamic quantization reduces FP32→INT8 but HF overhead dominates for small models)
- **Speculative** may show higher latency per request (two models loaded) but demonstrates the acceptance-rate mechanics correctly
- **Disaggregated** includes simulated transfer overhead, so latency is baseline + handoff cost

## Caveats

1. **Small models**: GPT-2 / DistilGPT-2 are too small for meaningful optimization gains. The benchmarks demonstrate infrastructure mechanics, not production speedups.
2. **CPU execution**: Most optimizations (quantization, KV cache transfer) show their benefits on GPU.  CPU benchmarks measure framework overhead.
3. **Simulated transfer**: The disaggregated mode simulates network transfer with `time.sleep()`.  Real KV cache transfer uses RDMA or gRPC.
4. **No batching**: Each request is processed individually.  Production systems use continuous batching for throughput.
5. **Acceptance rate**: With distilgpt2→gpt2, acceptance rates reflect real model alignment, but both models are small.
