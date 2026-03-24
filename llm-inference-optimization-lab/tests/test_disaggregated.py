"""Tests for disaggregated prefill/decode flow components."""

import pytest
import time

from app.prefill_service.cache_artifact import CacheArtifact
from app.decode_service.handoff import simulate_handoff, HandoffMetrics


class TestCacheArtifact:
    def test_construction(self):
        artifact = CacheArtifact(
            request_id="req1",
            prompt="hello world",
            prompt_tokens=2,
            model_name="gpt2",
            device="cpu",
            kv_cache_hash="abc123",
            kv_cache_size_bytes=1024,
        )
        assert artifact.request_id == "req1"
        assert artifact.prompt_tokens == 2
        assert artifact.kv_cache_size_bytes == 1024
        assert artifact.created_at_ns > 0

    def test_generation_params(self):
        artifact = CacheArtifact(
            request_id="req2",
            prompt="test",
            prompt_tokens=1,
            model_name="gpt2",
            device="cpu",
            generation_params={"max_new_tokens": 32, "temperature": 0.5},
        )
        assert artifact.generation_params["max_new_tokens"] == 32


class TestHandoff:
    def test_simulate_handoff_timing(self):
        metrics = simulate_handoff(1024)
        assert metrics.queue_wait_ms > 0
        assert metrics.transfer_ms > 0
        assert metrics.queue_enter_ns < metrics.queue_exit_ns
        assert metrics.transfer_start_ns < metrics.transfer_end_ns

    def test_handoff_metrics_properties(self):
        hm = HandoffMetrics(
            queue_enter_ns=1000000,
            queue_exit_ns=2000000,
            transfer_start_ns=3000000,
            transfer_end_ns=4000000,
        )
        assert hm.queue_wait_ms == pytest.approx(1.0)
        assert hm.transfer_ms == pytest.approx(1.0)
