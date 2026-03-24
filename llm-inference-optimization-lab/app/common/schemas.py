"""Canonical request/response schemas shared across all services."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Unified inference request accepted by the router and every backend."""

    prompt: str
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    mode: str = "baseline"
    request_id: str | None = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    metadata: dict[str, Any] | None = None


class InferenceMetrics(BaseModel):
    """Structured metrics returned alongside every generation."""

    request_start_time: float = Field(default_factory=time.time)
    request_end_time: float | None = None
    total_latency_ms: float | None = None
    ttft_ms: float | None = None
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    tokens_per_second: float | None = None
    queue_wait_ms: float | None = None
    transfer_latency_ms: float | None = None
    gpu_memory_mb: float | None = None
    backend_name: str | None = None
    model_name: str | None = None
    success: bool = True
    error_message: str | None = None

    # Speculative-specific
    speculative_proposed: int | None = None
    speculative_accepted: int | None = None
    speculative_acceptance_rate: float | None = None

    def finalize(self) -> None:
        """Compute derived fields after generation completes."""
        self.request_end_time = time.time()
        self.total_latency_ms = (self.request_end_time - self.request_start_time) * 1000.0
        if self.output_tokens and self.total_latency_ms > 0:
            self.tokens_per_second = self.output_tokens / (self.total_latency_ms / 1000.0)
        if self.speculative_proposed and self.speculative_proposed > 0:
            self.speculative_acceptance_rate = (
                (self.speculative_accepted or 0) / self.speculative_proposed
            )


class GenerateResponse(BaseModel):
    """Unified inference response from any backend."""

    request_id: str
    mode: str
    output_text: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Standard health check response."""

    service: str
    status: str = "ok"
    model_loaded: bool = False
    device: str = "cpu"
