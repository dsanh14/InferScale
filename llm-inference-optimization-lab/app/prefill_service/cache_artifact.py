"""Cache artifact produced by the prefill phase.

In a real disaggregated system the artifact would be a serialized KV cache
transferred over the network.  Here we model it faithfully as a structured
object with timing metadata, while using a simplified representation of
the cached state.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class CacheArtifact(BaseModel):
    """Represents the intermediate state handed from prefill to decode."""

    request_id: str
    prompt: str
    prompt_tokens: int
    model_name: str
    device: str

    # Simulated KV cache fingerprint (real systems would serialize tensors)
    kv_cache_hash: str = ""
    kv_cache_size_bytes: int = 0

    prefill_latency_ms: float = 0.0
    created_at_ns: int = Field(default_factory=time.time_ns)

    generation_params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
