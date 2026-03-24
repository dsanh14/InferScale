"""Tests for the inference router."""

import pytest
from unittest.mock import AsyncMock, patch

from app.common.schemas import GenerateRequest, GenerateResponse
from app.router.policy import ExplicitModePolicy, RoundRobinPolicy


class TestExplicitModePolicy:
    def test_baseline_routes_correctly(self):
        policy = ExplicitModePolicy()
        req = GenerateRequest(prompt="hello", mode="baseline")
        url = policy.resolve(req)
        assert "8001" in url

    def test_quantized_routes_correctly(self):
        policy = ExplicitModePolicy()
        req = GenerateRequest(prompt="hello", mode="quantized")
        url = policy.resolve(req)
        assert "8002" in url

    def test_speculative_routes_correctly(self):
        policy = ExplicitModePolicy()
        req = GenerateRequest(prompt="hello", mode="speculative")
        url = policy.resolve(req)
        assert "8003" in url

    def test_disaggregated_routes_correctly(self):
        policy = ExplicitModePolicy()
        req = GenerateRequest(prompt="hello", mode="disaggregated")
        url = policy.resolve(req)
        assert "8004" in url

    def test_unknown_mode_raises(self):
        policy = ExplicitModePolicy()
        req = GenerateRequest(prompt="hello", mode="unknown_mode")
        with pytest.raises(ValueError, match="Unknown serving mode"):
            policy.resolve(req)


class TestRoundRobinPolicy:
    def test_round_robin_cycles(self):
        replicas = ["http://a:8001", "http://b:8001"]
        policy = RoundRobinPolicy(replicas)
        req = GenerateRequest(prompt="hello", mode="baseline")
        urls = [policy.resolve(req) for _ in range(4)]
        assert urls == ["http://a:8001", "http://b:8001", "http://a:8001", "http://b:8001"]
