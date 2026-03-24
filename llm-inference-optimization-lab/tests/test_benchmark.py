"""Tests for benchmark workload generation and analysis."""

import pytest
import json
from pathlib import Path

from app.benchmark.workloads import generate_workload_matrix, load_prompt, WorkloadItem
from app.benchmark.analysis import results_to_dataframe, compute_summary, save_results


class TestWorkloads:
    def test_default_matrix_size(self):
        items = generate_workload_matrix()
        # 3 buckets * 3 token lengths * 4 concurrency * 4 modes = 144
        assert len(items) == 144

    def test_custom_matrix(self):
        items = generate_workload_matrix(
            prompt_buckets=["short"],
            token_lengths=[32],
            concurrency_levels=[1],
            modes=["baseline"],
        )
        assert len(items) == 1
        assert items[0].mode == "baseline"
        assert items[0].max_new_tokens == 32

    def test_load_prompt_fallback(self):
        prompt = load_prompt("short")
        assert len(prompt) > 0


class TestAnalysis:
    def test_results_to_dataframe(self):
        results = [
            {"mode": "baseline", "client_latency_ms": 100, "success": True},
            {"mode": "baseline", "client_latency_ms": 150, "success": True},
        ]
        df = results_to_dataframe(results)
        assert len(df) == 2
        assert "mode" in df.columns

    def test_save_results(self, tmp_path):
        results = [
            {"mode": "baseline", "client_latency_ms": 100, "success": True},
        ]
        csv_path, json_path = save_results(results, tmp_path)
        assert csv_path.exists()
        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 1
