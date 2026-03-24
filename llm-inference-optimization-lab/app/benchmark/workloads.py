"""Workload definitions for benchmark sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "sample_data" / "prompts"

PROMPT_BUCKETS: dict[str, str] = {
    "short": "Explain what a GPU is in one sentence.",
    "medium": (
        "Describe the key differences between CPUs and GPUs in the context of "
        "machine learning workloads. Cover parallelism, memory bandwidth, and "
        "typical use cases for each processor type."
    ),
    "long": (
        "Write a detailed technical overview of transformer-based language models. "
        "Cover the following topics: the attention mechanism and its computational "
        "complexity, the role of positional encoding, how key-value caches work during "
        "autoregressive generation, why memory bandwidth becomes the bottleneck during "
        "the decode phase, techniques for reducing inference latency including "
        "quantization and speculative decoding, and the tradeoffs between model quality "
        "and serving efficiency."
    ),
}


def load_prompt(bucket: str) -> str:
    """Load a prompt by bucket name, falling back to built-in text."""
    path = _PROMPT_DIR / f"{bucket}.txt"
    if path.exists():
        return path.read_text().strip()
    return PROMPT_BUCKETS.get(bucket, PROMPT_BUCKETS["short"])


@dataclass
class WorkloadItem:
    """A single benchmark work item."""

    prompt_bucket: str
    prompt: str
    max_new_tokens: int
    mode: str
    concurrency: int


def generate_workload_matrix(
    prompt_buckets: list[str] | None = None,
    token_lengths: list[int] | None = None,
    concurrency_levels: list[int] | None = None,
    modes: list[str] | None = None,
) -> list[WorkloadItem]:
    """Generate the full cross-product of benchmark parameters."""
    buckets = prompt_buckets or ["short", "medium", "long"]
    tokens = token_lengths or [32, 64, 128]
    concurrencies = concurrency_levels or [1, 2, 4, 8]
    modes_list = modes or ["baseline", "quantized", "speculative", "disaggregated"]

    items: list[WorkloadItem] = []
    for bucket in buckets:
        prompt = load_prompt(bucket)
        for tok in tokens:
            for conc in concurrencies:
                for mode in modes_list:
                    items.append(
                        WorkloadItem(
                            prompt_bucket=bucket,
                            prompt=prompt,
                            max_new_tokens=tok,
                            mode=mode,
                            concurrency=conc,
                        )
                    )
    return items
