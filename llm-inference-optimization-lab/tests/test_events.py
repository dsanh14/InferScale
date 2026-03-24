"""Tests for structured event logging and hashing."""

import json
import pytest
from pathlib import Path

from app.common.events import Event, EventLogger
from app.common.hashing import hash_event, hash_trace, hash_text


class TestEvent:
    def test_compute_payload_hash(self):
        evt = Event(
            request_id="req1",
            service_name="test",
            phase="init",
            event_type="start",
            metadata={"key": "value"},
        )
        h = evt.compute_payload_hash()
        assert len(h) == 32
        # Same metadata should produce the same hash
        evt2 = Event(metadata={"key": "value"})
        assert evt2.compute_payload_hash() == h

    def test_timestamp_populated(self):
        evt = Event()
        assert evt.timestamp_ns > 0


class TestEventLogger:
    def test_log_creates_file(self, tmp_path):
        logger = EventLogger("test_svc", log_dir=str(tmp_path))
        logger.log("req1", "init", "start", {"foo": "bar"})
        log_path = tmp_path / "test_svc_events.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["request_id"] == "req1"
        assert data["service_name"] == "test_svc"
        assert data["sequence_no"] == 0

    def test_sequence_numbers_increment(self, tmp_path):
        logger = EventLogger("test_svc", log_dir=str(tmp_path))
        logger.log("req1", "p", "e1")
        logger.log("req1", "p", "e2")
        logger.log("req1", "p", "e3")
        log_path = tmp_path / "test_svc_events.jsonl"
        lines = log_path.read_text().strip().split("\n")
        seqs = [json.loads(l)["sequence_no"] for l in lines]
        assert seqs == [0, 1, 2]

    def test_separate_request_sequences(self, tmp_path):
        logger = EventLogger("test_svc", log_dir=str(tmp_path))
        logger.log("req1", "p", "e")
        logger.log("req2", "p", "e")
        logger.log("req1", "p", "e")
        log_path = tmp_path / "test_svc_events.jsonl"
        lines = log_path.read_text().strip().split("\n")
        events = [json.loads(l) for l in lines]
        assert events[0]["sequence_no"] == 0  # req1 first
        assert events[1]["sequence_no"] == 0  # req2 first
        assert events[2]["sequence_no"] == 1  # req1 second


class TestHashing:
    def test_hash_event_deterministic(self):
        evt = {
            "request_id": "r1",
            "service_name": "svc",
            "phase": "p",
            "event_type": "e",
            "sequence_no": 0,
            "payload_hash": "h",
        }
        h1 = hash_event(evt)
        h2 = hash_event(evt)
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_trace_changes_on_mutation(self):
        events = [
            {"request_id": "r1", "service_name": "s", "phase": "p",
             "event_type": "e", "sequence_no": 0, "payload_hash": "h1"},
            {"request_id": "r1", "service_name": "s", "phase": "p",
             "event_type": "e", "sequence_no": 1, "payload_hash": "h2"},
        ]
        h1 = hash_trace(events)
        events[1]["payload_hash"] = "CHANGED"
        h2 = hash_trace(events)
        assert h1 != h2

    def test_hash_text(self):
        assert len(hash_text("hello")) == 32
        assert hash_text("hello") == hash_text("hello")
        assert hash_text("hello") != hash_text("world")
