"""Prometheus-style metrics collection and percentile computation.

Provides both an in-memory collector for benchmark aggregation and
optional Prometheus client integration for live scraping.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

try:
    from prometheus_client import Counter, Histogram, Gauge, REGISTRY, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class LatencyBucket:
    """Raw latency observations for percentile computation."""

    values: list[float] = field(default_factory=list)

    def record(self, value_ms: float) -> None:
        self.values.append(value_ms)

    def p50(self) -> float:
        return self._percentile(50) if self.values else 0.0

    def p95(self) -> float:
        return self._percentile(95) if self.values else 0.0

    def p99(self) -> float:
        return self._percentile(99) if self.values else 0.0

    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    def _percentile(self, pct: int) -> float:
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * pct / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]


class MetricsCollector:
    """In-memory metrics aggregator for benchmarks and observability.

    Thread-safe. Records per-mode latency, throughput, TTFT, queue wait,
    success/error counts, and speculative acceptance rates.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._latencies: dict[str, LatencyBucket] = defaultdict(LatencyBucket)
        self._ttft: dict[str, LatencyBucket] = defaultdict(LatencyBucket)
        self._queue_wait: dict[str, LatencyBucket] = defaultdict(LatencyBucket)
        self._throughput: dict[str, list[float]] = defaultdict(list)
        self._success_count: dict[str, int] = defaultdict(int)
        self._error_count: dict[str, int] = defaultdict(int)
        self._speculative_proposed: int = 0
        self._speculative_accepted: int = 0

    def record_request(self, mode: str, metrics: dict[str, Any]) -> None:
        """Record a completed request's metrics."""
        with self._lock:
            if metrics.get("success", True):
                self._success_count[mode] += 1
            else:
                self._error_count[mode] += 1

            if metrics.get("total_latency_ms") is not None:
                self._latencies[mode].record(metrics["total_latency_ms"])

            if metrics.get("ttft_ms") is not None:
                self._ttft[mode].record(metrics["ttft_ms"])

            if metrics.get("queue_wait_ms") is not None:
                self._queue_wait[mode].record(metrics["queue_wait_ms"])

            if metrics.get("tokens_per_second") is not None:
                self._throughput[mode].append(metrics["tokens_per_second"])

            if metrics.get("speculative_proposed"):
                self._speculative_proposed += metrics["speculative_proposed"]
            if metrics.get("speculative_accepted"):
                self._speculative_accepted += metrics["speculative_accepted"]

    def summary(self) -> dict[str, Any]:
        """Return a full metrics summary keyed by mode."""
        with self._lock:
            result: dict[str, Any] = {}
            all_modes = set(self._latencies) | set(self._success_count)
            for mode in sorted(all_modes):
                lat = self._latencies.get(mode, LatencyBucket())
                ttft = self._ttft.get(mode, LatencyBucket())
                tp = self._throughput.get(mode, [])
                result[mode] = {
                    "request_count": self._success_count.get(mode, 0)
                    + self._error_count.get(mode, 0),
                    "success_count": self._success_count.get(mode, 0),
                    "error_count": self._error_count.get(mode, 0),
                    "latency_p50_ms": lat.p50(),
                    "latency_p95_ms": lat.p95(),
                    "latency_p99_ms": lat.p99(),
                    "latency_mean_ms": lat.mean(),
                    "ttft_p50_ms": ttft.p50(),
                    "ttft_p95_ms": ttft.p95(),
                    "throughput_mean_tps": statistics.mean(tp) if tp else 0.0,
                }
            if self._speculative_proposed > 0:
                result["speculative_global"] = {
                    "total_proposed": self._speculative_proposed,
                    "total_accepted": self._speculative_accepted,
                    "acceptance_rate": self._speculative_accepted / self._speculative_proposed,
                }
            return result

    def reset(self) -> None:
        with self._lock:
            self._latencies.clear()
            self._ttft.clear()
            self._queue_wait.clear()
            self._throughput.clear()
            self._success_count.clear()
            self._error_count.clear()
            self._speculative_proposed = 0
            self._speculative_accepted = 0


# Singleton collector for in-process use
_global_collector = MetricsCollector()


def get_collector() -> MetricsCollector:
    return _global_collector


# Optional Prometheus metrics (created once on import if library is available)
if PROMETHEUS_AVAILABLE:
    PROM_REQUEST_LATENCY = Histogram(
        "inference_request_latency_ms",
        "End-to-end request latency in milliseconds",
        ["mode"],
        buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
    )
    PROM_REQUESTS_TOTAL = Counter(
        "inference_requests_total",
        "Total inference requests",
        ["mode", "status"],
    )
    PROM_TOKENS_PER_SECOND = Gauge(
        "inference_tokens_per_second",
        "Last observed tokens per second",
        ["mode"],
    )

    def record_prometheus(mode: str, metrics: dict[str, Any]) -> None:
        status = "success" if metrics.get("success", True) else "error"
        PROM_REQUESTS_TOTAL.labels(mode=mode, status=status).inc()
        if metrics.get("total_latency_ms") is not None:
            PROM_REQUEST_LATENCY.labels(mode=mode).observe(metrics["total_latency_ms"])
        if metrics.get("tokens_per_second") is not None:
            PROM_TOKENS_PER_SECOND.labels(mode=mode).set(metrics["tokens_per_second"])

    def prometheus_metrics_text() -> bytes:
        return generate_latest(REGISTRY)
else:

    def record_prometheus(mode: str, metrics: dict[str, Any]) -> None:
        pass

    def prometheus_metrics_text() -> bytes:
        return b"# prometheus_client not installed\n"
