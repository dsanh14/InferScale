"""Benchmark visualization — generates publication-quality plots from results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.common.logging_utils import setup_logging

logger = setup_logging("benchmark.plots")

STYLE = {
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
}
plt.rcParams.update(STYLE)

MODE_COLORS = {
    "baseline": "#2196F3",
    "quantized": "#4CAF50",
    "speculative": "#FF9800",
    "disaggregated": "#9C27B0",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved plot: {path}")


def plot_latency_by_mode(df: pd.DataFrame, output_dir: Path) -> None:
    """Box plot of end-to-end latency per serving mode."""
    col = "total_latency_ms" if "total_latency_ms" in df.columns else "client_latency_ms"
    if col not in df.columns:
        logger.warning("No latency column found; skipping latency_by_mode plot")
        return

    fig, ax = plt.subplots()
    modes = sorted(df["mode"].unique())
    data = [df[df["mode"] == m][col].dropna().values for m in modes]
    colors = [MODE_COLORS.get(m, "#999999") for m in modes]

    bp = ax.boxplot(data, labels=modes, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("End-to-End Latency by Serving Mode")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_dir / "latency_by_mode.png")


def plot_throughput_by_mode(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of mean throughput (tokens/s) per serving mode."""
    if "tokens_per_second" not in df.columns:
        logger.warning("No throughput column; skipping throughput_by_mode plot")
        return

    grouped = df.groupby("mode")["tokens_per_second"].mean()
    fig, ax = plt.subplots()
    colors = [MODE_COLORS.get(m, "#999999") for m in grouped.index]
    grouped.plot(kind="bar", ax=ax, color=colors, alpha=0.8)
    ax.set_ylabel("Tokens / Second (mean)")
    ax.set_title("Throughput by Serving Mode")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    _save(fig, output_dir / "throughput_by_mode.png")


def plot_ttft_by_prompt_length(df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart of TTFT across prompt length buckets."""
    if "ttft_ms" not in df.columns or "prompt_bucket" not in df.columns:
        logger.warning("Missing ttft_ms or prompt_bucket; skipping ttft plot")
        return

    pivot = df.groupby(["prompt_bucket", "mode"])["ttft_ms"].mean().unstack(fill_value=0)
    fig, ax = plt.subplots()
    pivot.plot(kind="bar", ax=ax, color=[MODE_COLORS.get(m, "#999") for m in pivot.columns], alpha=0.8)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token by Prompt Length")
    ax.legend(title="Mode")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    _save(fig, output_dir / "ttft_by_prompt_length.png")


def plot_speculative_acceptance_rate(df: pd.DataFrame, output_dir: Path) -> None:
    """Histogram of speculative acceptance rates across requests."""
    spec = df[df["mode"] == "speculative"]
    col = None
    for c in ["speculative_acceptance_rate", "acceptance_rate"]:
        if c in spec.columns:
            col = c
            break

    if col is None or spec[col].dropna().empty:
        logger.warning("No speculative acceptance data; skipping acceptance rate plot")
        return

    fig, ax = plt.subplots()
    spec[col].dropna().plot(kind="hist", bins=20, ax=ax, color=MODE_COLORS["speculative"], alpha=0.8, edgecolor="white")
    ax.set_xlabel("Acceptance Rate")
    ax.set_ylabel("Count")
    ax.set_title("Speculative Decoding Acceptance Rate Distribution")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_dir / "speculative_acceptance_rate.png")


def generate_all_plots(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate all standard benchmark plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_latency_by_mode(df, out)
    plot_throughput_by_mode(df, out)
    plot_ttft_by_prompt_length(df, out)
    plot_speculative_acceptance_rate(df, out)
    logger.info(f"All plots saved to {out}")
