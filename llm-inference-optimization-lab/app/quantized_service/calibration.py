"""Calibration workflow abstraction for quantized model preparation.

In a production TensorRT-LLM pipeline, calibration collects activation
statistics over representative prompts to choose per-layer scale factors
for INT8 or FP8 quantization.  This module provides the workflow skeleton
that works with both the portable backend and the TensorRT adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.common.logging_utils import setup_logging

logger = setup_logging("quantized.calibration")

DEFAULT_CALIBRATION_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Explain the concept of attention in transformer models.",
    "Write a Python function that sorts a list.",
    "What are the benefits of quantized inference?",
]


@dataclass
class CalibrationConfig:
    """Parameters controlling the calibration pass."""

    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_CALIBRATION_PROMPTS))
    num_batches: int = 4
    batch_size: int = 1
    max_seq_len: int = 128
    output_dir: str = "./calibration_output"


@dataclass
class CalibrationResult:
    """Outcome of a calibration run."""

    num_prompts_used: int = 0
    scales_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = False


def run_calibration(
    backend: Any,
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """Execute the calibration workflow against the given backend.

    For the portable backend this is effectively a no-op that validates
    the interface.  For TensorRT Model Optimizer this would invoke
    modelopt.torch.quantization.calibrate().
    """
    cfg = config or CalibrationConfig()
    result = CalibrationResult()

    logger.info(f"Starting calibration with {len(cfg.prompts)} prompts")

    try:
        meta = backend.quantize_model(calibration_data=cfg.prompts)
        result.num_prompts_used = len(cfg.prompts)
        result.metadata = meta
        result.success = True
        logger.info(f"Calibration succeeded: {meta}")
    except NotImplementedError:
        logger.warning(
            "Backend does not support real calibration; returning placeholder result"
        )
        result.metadata = {"note": "calibration skipped — backend does not support it"}
    except Exception as exc:
        logger.error(f"Calibration failed: {exc}")
        result.metadata = {"error": str(exc)}

    return result
