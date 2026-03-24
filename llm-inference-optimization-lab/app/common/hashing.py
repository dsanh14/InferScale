"""Deterministic hashing utilities for event traces and replay validation.

Mirrors the hashing logic in the C++ replay validator so Python-side
pre-computation is consistent with the C++ verification path.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_event(event_dict: dict[str, Any], stable_fields: list[str] | None = None) -> str:
    """Hash the stable (deterministic) fields of an event.

    By default, timestamps are excluded because they vary between runs.
    The replay validator uses the same approach.
    """
    if stable_fields is None:
        stable_fields = [
            "request_id",
            "service_name",
            "phase",
            "event_type",
            "sequence_no",
            "payload_hash",
        ]
    subset = {k: event_dict.get(k) for k in stable_fields}
    canonical = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def hash_trace(events: list[dict[str, Any]]) -> str:
    """Compute a cumulative hash over an ordered event trace.

    Each event hash is folded into a running state hash, so any
    divergence at position N changes the final digest.
    """
    state = hashlib.sha256()
    for evt in events:
        evt_hash = hash_event(evt)
        state.update(evt_hash.encode())
    return state.hexdigest()


def hash_text(text: str) -> str:
    """Simple SHA-256 of a string."""
    return hashlib.sha256(text.encode()).hexdigest()[:32]
