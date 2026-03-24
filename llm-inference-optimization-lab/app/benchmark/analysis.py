"""Result aggregation and statistical analysis for benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.common.logging_utils import setup_logging

logger = setup_logging("benchmark.analysis")


def results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw result dicts into a tidy DataFrame."""
    return pd.DataFrame(results)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by mode, prompt_bucket, max_new_tokens, concurrency."""
    group_cols = [c for c in ["mode", "prompt_bucket", "max_new_tokens", "concurrency"] if c in df.columns]
    if not group_cols:
        return df.describe()

    numeric_cols = ["client_latency_ms", "total_latency_ms", "ttft_ms", "tokens_per_second"]
    agg_cols = [c for c in numeric_cols if c in df.columns]

    if not agg_cols:
        return df.groupby(group_cols).size().reset_index(name="count")

    agg_funcs = {col: ["mean", "median", "min", "max", "std"] for col in agg_cols}

    summary = df.groupby(group_cols).agg(agg_funcs)
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()

    if "success" in df.columns:
        success_rate = df.groupby(group_cols)["success"].mean().reset_index(name="success_rate")
        summary = summary.merge(success_rate, on=group_cols, how="left")

    return summary


def save_results(
    results: list[dict[str, Any]],
    output_dir: str | Path,
    prefix: str = "benchmark",
) -> tuple[Path, Path]:
    """Save raw results as CSV and JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / f"{prefix}_results.csv"
    json_path = out / f"{prefix}_results.json"

    df = results_to_dataframe(results)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved JSON to {json_path}")

    return csv_path, json_path
