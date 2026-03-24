"""Structured event logging for deterministic replay and observability.

Every service emits events in a shared JSONL format so the C++ replay
validator can parse, hash, and compare traces across runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field

from app.common.config import LoggingConfig


class Event(BaseModel):
    """Canonical event schema consumed by all services and the replay validator."""

    timestamp_ns: int = Field(default_factory=lambda: time.time_ns())
    request_id: str = ""
    service_name: str = ""
    phase: str = ""
    event_type: str = ""
    sequence_no: int = 0
    payload_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def compute_payload_hash(self) -> str:
        """SHA-256 of the deterministic metadata content."""
        canonical = json.dumps(self.metadata, sort_keys=True, default=str)
        self.payload_hash = hashlib.sha256(canonical.encode()).hexdigest()[:32]
        return self.payload_hash


class EventLogger:
    """Thread-safe JSONL event writer.

    Maintains a per-request sequence counter so the replay validator
    can detect missing or reordered events.
    """

    def __init__(self, service_name: str, log_dir: str | None = None) -> None:
        self.service_name = service_name
        self.log_dir = Path(log_dir or LoggingConfig.event_log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._seq: dict[str, int] = {}
        self._lock = Lock()
        self._log_path = self.log_dir / f"{service_name}_events.jsonl"

    def _next_seq(self, request_id: str) -> int:
        with self._lock:
            seq = self._seq.get(request_id, 0)
            self._seq[request_id] = seq + 1
            return seq

    def log(
        self,
        request_id: str,
        phase: str,
        event_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Emit a structured event to the JSONL log file."""
        evt = Event(
            request_id=request_id,
            service_name=self.service_name,
            phase=phase,
            event_type=event_type,
            sequence_no=self._next_seq(request_id),
            metadata=metadata or {},
        )
        evt.compute_payload_hash()
        with self._lock:
            with open(self._log_path, "a") as f:
                f.write(evt.model_dump_json() + "\n")
        return evt

    def get_log_path(self) -> Path:
        return self._log_path

    def reset(self) -> None:
        """Clear sequence counters (useful between benchmark runs)."""
        with self._lock:
            self._seq.clear()
