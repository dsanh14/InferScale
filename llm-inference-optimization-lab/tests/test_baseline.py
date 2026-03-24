"""Tests for baseline inference service schemas and structure."""

import pytest

from app.common.schemas import GenerateRequest, GenerateResponse, InferenceMetrics


class TestGenerateRequest:
    def test_defaults(self):
        req = GenerateRequest(prompt="hello")
        assert req.max_new_tokens == 64
        assert req.temperature == 1.0
        assert req.mode == "baseline"
        assert req.request_id is not None

    def test_custom_fields(self):
        req = GenerateRequest(
            prompt="test",
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
            mode="quantized",
            request_id="custom123",
        )
        assert req.max_new_tokens == 32
        assert req.mode == "quantized"
        assert req.request_id == "custom123"


class TestInferenceMetrics:
    def test_finalize_computes_latency(self):
        m = InferenceMetrics()
        import time
        time.sleep(0.01)
        m.output_tokens = 10
        m.finalize()
        assert m.total_latency_ms is not None
        assert m.total_latency_ms > 0
        assert m.tokens_per_second is not None
        assert m.tokens_per_second > 0

    def test_speculative_acceptance_rate(self):
        m = InferenceMetrics()
        m.speculative_proposed = 10
        m.speculative_accepted = 7
        m.finalize()
        assert m.speculative_acceptance_rate == pytest.approx(0.7)


class TestGenerateResponse:
    def test_construction(self):
        resp = GenerateResponse(
            request_id="abc",
            mode="baseline",
            output_text="hello world",
            metrics={"total_latency_ms": 100},
        )
        assert resp.output_text == "hello world"
        assert resp.metrics["total_latency_ms"] == 100
