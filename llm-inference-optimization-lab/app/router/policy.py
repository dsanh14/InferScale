"""Routing policy — decides which backend handles a given request.

The default policy is explicit mode selection from the request.  More
sophisticated policies (cost-aware, latency-aware, load-balanced) can be
swapped in by implementing the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.common.config import ServiceConfig
from app.common.schemas import GenerateRequest


class RoutingPolicy(ABC):
    """Base class for routing strategies."""

    @abstractmethod
    def resolve(self, request: GenerateRequest) -> str:
        """Return the backend URL for the given request."""
        ...


class ExplicitModePolicy(RoutingPolicy):
    """Route based on the ``mode`` field in the request."""

    MODE_MAP: dict[str, str] = {
        "baseline": ServiceConfig.baseline_url,
        "quantized": ServiceConfig.quantized_url,
        "speculative": ServiceConfig.speculative_url,
        "disaggregated": ServiceConfig.prefill_url,
    }

    def resolve(self, request: GenerateRequest) -> str:
        url = self.MODE_MAP.get(request.mode)
        if url is None:
            raise ValueError(
                f"Unknown serving mode '{request.mode}'. "
                f"Valid modes: {list(self.MODE_MAP.keys())}"
            )
        return url


class RoundRobinPolicy(RoutingPolicy):
    """Placeholder for a load-balancing policy across replicas."""

    def __init__(self, replicas: list[str] | None = None) -> None:
        self._replicas = replicas or [ServiceConfig.baseline_url]
        self._idx = 0

    def resolve(self, request: GenerateRequest) -> str:
        url = self._replicas[self._idx % len(self._replicas)]
        self._idx += 1
        return url
