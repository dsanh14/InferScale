# Dashboard Stub

This project intentionally omits a frontend framework.  Observability is
provided through:

1. **Prometheus metrics endpoint** — each service exposes `/metrics/prometheus`
2. **JSON metrics API** — the router exposes `/metrics` with aggregated stats
3. **Structured event logs** — JSONL files in `logs/` for replay and analysis
4. **Benchmark plots** — generated as PNGs in `results/`

## Integration points for a real dashboard

| Data source | URL / path | Format |
|---|---|---|
| Live metrics | `http://localhost:8000/metrics` | JSON |
| Prometheus scrape | `http://localhost:8000/metrics/prometheus` | text/plain |
| Event traces | `logs/*_events.jsonl` | JSONL |
| Benchmark results | `results/benchmark_results.csv` | CSV |
| Latency plots | `results/latency_by_mode.png` | PNG |

A Grafana dashboard or simple Streamlit app can be wired to these sources
with minimal effort.
