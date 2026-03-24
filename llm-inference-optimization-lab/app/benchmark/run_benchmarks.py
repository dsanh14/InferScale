"""Main benchmark runner — orchestrates workload sweeps and result collection."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from app.common.config import BenchmarkConfig, ServiceConfig
from app.common.logging_utils import setup_logging
from app.common.schemas import GenerateRequest
from app.common.utils import generate_request_id
from app.benchmark.analysis import results_to_dataframe, save_results, compute_summary
from app.benchmark.client import send_batch
from app.benchmark.plots import generate_all_plots
from app.benchmark.workloads import generate_workload_matrix, WorkloadItem

logger = setup_logging("benchmark.runner")


def _build_requests(item: WorkloadItem, repetitions: int = 1) -> list[GenerateRequest]:
    """Create GenerateRequest objects for a single workload item."""
    requests: list[GenerateRequest] = []
    for _ in range(repetitions):
        requests.append(
            GenerateRequest(
                prompt=item.prompt,
                max_new_tokens=item.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                mode=item.mode,
                request_id=generate_request_id(),
                metadata={
                    "prompt_bucket": item.prompt_bucket,
                    "concurrency": item.concurrency,
                },
            )
        )
    return requests


async def run_sweep(
    router_url: str | None = None,
    output_dir: str | None = None,
    modes: list[str] | None = None,
    prompt_buckets: list[str] | None = None,
    token_lengths: list[int] | None = None,
    concurrency_levels: list[int] | None = None,
    repetitions: int | None = None,
) -> list[dict[str, Any]]:
    """Run the full benchmark sweep and save results."""
    url = router_url or f"http://localhost:{ServiceConfig.router_port}"
    out = output_dir or BenchmarkConfig.output_dir
    reps = repetitions or BenchmarkConfig.repetitions

    workload = generate_workload_matrix(
        prompt_buckets=prompt_buckets or BenchmarkConfig.prompt_buckets,
        token_lengths=token_lengths or BenchmarkConfig.token_lengths,
        concurrency_levels=concurrency_levels or BenchmarkConfig.concurrency_levels,
        modes=modes,
    )

    logger.info(f"Starting benchmark sweep: {len(workload)} workload items, {reps} reps each")

    all_results: list[dict[str, Any]] = []

    for i, item in enumerate(workload):
        logger.info(
            f"[{i + 1}/{len(workload)}] mode={item.mode} bucket={item.prompt_bucket} "
            f"tokens={item.max_new_tokens} concurrency={item.concurrency}"
        )
        requests = _build_requests(item, repetitions=reps)
        batch_results = await send_batch(url, requests, concurrency=item.concurrency)

        for r in batch_results:
            r["prompt_bucket"] = item.prompt_bucket
            r["max_new_tokens"] = item.max_new_tokens
            r["concurrency"] = item.concurrency

        all_results.extend(batch_results)

    csv_path, json_path = save_results(all_results, out)
    logger.info(f"Results: {csv_path}, {json_path}")

    df = results_to_dataframe(all_results)
    summary = compute_summary(df)
    summary_path = Path(out) / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary: {summary_path}")

    generate_all_plots(df, out)

    return all_results


def main() -> None:
    """CLI entry point for the benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Runner")
    parser.add_argument("--router-url", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    args = parser.parse_args()

    results = asyncio.run(
        run_sweep(
            router_url=args.router_url,
            output_dir=args.output_dir,
            modes=args.modes,
            repetitions=args.repetitions,
        )
    )
    logger.info(f"Benchmark complete: {len(results)} total results collected")


if __name__ == "__main__":
    main()
