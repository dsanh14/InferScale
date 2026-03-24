"""Handoff protocol between prefill and decode services.

Models the transfer of the cache artifact, including simulated transfer
latency and queue wait time that would exist in a real disaggregated
deployment (e.g. over RDMA or gRPC between prefill and decode GPUs).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class HandoffMetrics:
    """Timing data captured during the prefill-to-decode handoff."""
    queue_enter_ns: int = 0
    queue_exit_ns: int = 0
    transfer_start_ns: int = 0
    transfer_end_ns: int = 0

    @property
    def queue_wait_ms(self) -> float:
        if self.queue_exit_ns and self.queue_enter_ns:
            return (self.queue_exit_ns - self.queue_enter_ns) / 1e6
        return 0.0

    @property
    def transfer_ms(self) -> float:
        if self.transfer_end_ns and self.transfer_start_ns:
            return (self.transfer_end_ns - self.transfer_start_ns) / 1e6
        return 0.0


def simulate_handoff(artifact_size_bytes: int) -> HandoffMetrics:
    """Simulate the network transfer of a cache artifact.

    In a real system this would be an RDMA transfer or a gRPC stream.
    Here we model the latency proportionally to the artifact size.
    """
    metrics = HandoffMetrics()

    # Simulate queue wait (would depend on decode worker load)
    metrics.queue_enter_ns = time.time_ns()
    time.sleep(0.001)  # 1ms simulated queue wait
    metrics.queue_exit_ns = time.time_ns()

    # Simulate transfer time proportional to cache size
    # Assume ~10 GB/s effective bandwidth (PCIe / network)
    transfer_seconds = artifact_size_bytes / (10 * 1024**3) if artifact_size_bytes > 0 else 0.0001
    metrics.transfer_start_ns = time.time_ns()
    time.sleep(max(transfer_seconds, 0.0005))
    metrics.transfer_end_ns = time.time_ns()

    return metrics
