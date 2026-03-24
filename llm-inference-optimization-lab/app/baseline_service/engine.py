"""Baseline inference engine — straightforward autoregressive generation."""

from __future__ import annotations

import time

import torch

from app.common.events import EventLogger
from app.common.logging_utils import setup_logging
from app.common.schemas import GenerateRequest, GenerateResponse, InferenceMetrics
from app.common.utils import gpu_memory_mb
from app.baseline_service.model_loader import ModelHandle

logger = setup_logging("baseline.engine")
event_logger = EventLogger("baseline")


class BaselineEngine:
    """Wraps HuggingFace generate() with metrics collection."""

    def __init__(self, handle: ModelHandle) -> None:
        self.handle = handle

    @torch.inference_mode()
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Run standard autoregressive generation and return structured output."""
        request_id = request.request_id or "unknown"
        metrics = InferenceMetrics(
            backend_name="baseline",
            model_name=self.handle.model_name,
        )
        event_logger.log(request_id, "inference", "generate_start", {
            "prompt_len": len(request.prompt),
            "max_new_tokens": request.max_new_tokens,
        })

        tokenizer = self.handle.tokenizer
        model = self.handle.model
        device = self.handle.device

        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        prompt_tokens = inputs["input_ids"].shape[1]
        metrics.prompt_tokens = prompt_tokens

        t_first = None

        def ttft_hook(*_args, **_kwargs):
            nonlocal t_first
            if t_first is None:
                t_first = time.perf_counter()

        t_start = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=max(request.temperature, 0.01),
            top_p=request.top_p,
            do_sample=request.temperature > 0.01,
        )
        t_end = time.perf_counter()

        new_tokens = outputs[0][prompt_tokens:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        metrics.output_tokens = len(new_tokens)

        if t_first is not None:
            metrics.ttft_ms = (t_first - t_start) * 1000.0
        else:
            metrics.ttft_ms = (t_end - t_start) * 1000.0 / max(len(new_tokens), 1)

        metrics.gpu_memory_mb = gpu_memory_mb()
        metrics.finalize()

        event_logger.log(request_id, "inference", "generate_done", {
            "output_tokens": metrics.output_tokens,
            "latency_ms": round(metrics.total_latency_ms or 0, 2),
        })

        return GenerateResponse(
            request_id=request_id,
            mode="baseline",
            output_text=output_text,
            metrics=metrics.model_dump(exclude_none=True),
            debug={"device": device, "prompt_tokens": prompt_tokens},
        )
