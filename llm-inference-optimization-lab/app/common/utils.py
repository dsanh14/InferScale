"""Shared utility helpers used across services."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Generator

import torch


def generate_request_id() -> str:
    return uuid.uuid4().hex[:16]


def get_device(preference: str = "cpu") -> str:
    """Resolve the best available device respecting the preference."""
    if preference == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preference == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gpu_memory_mb() -> float | None:
    """Return current GPU memory usage in MB, or None if not applicable."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return None


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """Context manager that records wall-clock elapsed time in milliseconds."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000.0


def count_tokens(text: str) -> int:
    """Rough whitespace-based token count for quick estimates."""
    return len(text.split())
