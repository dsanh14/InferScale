"""Centralized configuration loading from YAML files and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_yaml(name: str) -> dict[str, Any]:
    path = _PROJECT_ROOT / "configs" / name
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


_models_cfg = _load_yaml("models.yaml")
_bench_cfg = _load_yaml("benchmark.yaml")
_log_cfg = _load_yaml("logging.yaml")


class ModelConfig:
    default_model: str = _models_cfg.get("default_small_model", os.getenv("DEFAULT_MODEL", "gpt2"))
    draft_model: str = _models_cfg.get("draft_model", os.getenv("DRAFT_MODEL", "distilgpt2"))
    target_model: str = _models_cfg.get("target_model", os.getenv("TARGET_MODEL", "gpt2"))
    quantized_backend: str = _models_cfg.get("quantized_backend_mode", "portable_quantized")
    device: str = _models_cfg.get("device_preference", os.getenv("DEVICE", "cpu"))


class ServiceConfig:
    router_port: int = int(os.getenv("ROUTER_PORT", "8000"))
    baseline_port: int = int(os.getenv("BASELINE_PORT", "8001"))
    quantized_port: int = int(os.getenv("QUANTIZED_PORT", "8002"))
    speculative_port: int = int(os.getenv("SPECULATIVE_PORT", "8003"))
    prefill_port: int = int(os.getenv("PREFILL_PORT", "8004"))
    decode_port: int = int(os.getenv("DECODE_PORT", "8005"))

    baseline_url: str = os.getenv("BASELINE_URL", "http://localhost:8001")
    quantized_url: str = os.getenv("QUANTIZED_URL", "http://localhost:8002")
    speculative_url: str = os.getenv("SPECULATIVE_URL", "http://localhost:8003")
    prefill_url: str = os.getenv("PREFILL_URL", "http://localhost:8004")
    decode_url: str = os.getenv("DECODE_URL", "http://localhost:8005")


class BenchmarkConfig:
    prompt_buckets: list[str] = _bench_cfg.get("prompt_buckets", ["short", "medium", "long"])
    token_lengths: list[int] = _bench_cfg.get("token_lengths", [32, 64, 128])
    concurrency_levels: list[int] = _bench_cfg.get("concurrency_levels", [1, 2, 4, 8])
    repetitions: int = _bench_cfg.get("repetitions", 1)
    output_dir: str = _bench_cfg.get("output_dir", str(_PROJECT_ROOT / "results"))


class LoggingConfig:
    level: str = _log_cfg.get("level", os.getenv("LOG_LEVEL", "INFO"))
    event_log_dir: str = _log_cfg.get(
        "event_log_dir", os.getenv("EVENT_LOG_DIR", str(_PROJECT_ROOT / "logs"))
    )
    structured: bool = _log_cfg.get("structured", True)
