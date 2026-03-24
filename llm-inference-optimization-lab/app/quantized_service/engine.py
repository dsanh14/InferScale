"""Quantized inference engine — wraps the backend adapter with metrics."""

from __future__ import annotations

import time

import torch

from app.common.config import ModelConfig
from app.common.events import EventLogger
from app.common.logging_utils import setup_logging
from app.common.schemas import GenerateRequest, GenerateResponse, InferenceMetrics
from app.common.utils import gpu_memory_mb
from app.quantized_service.adapters import QuantizedBackend

logger = setup_logging("quantized.engine")
event_logger = EventLogger("quantized")


class QuantizedEngine:
    """Inference engine backed by a QuantizedBackend adapter."""

    def __init__(self, backend: QuantizedBackend) -> None:
        self.backend = backend
        self._tokenizer = None

    def initialize(self, model_name: str | None = None, device: str | None = None) -> None:
        model_name = model_name or ModelConfig.default_model
        device = device or ModelConfig.device
        self.backend.prepare_model(model_name, device)
        self.backend.quantize_model()
        valid = self.backend.validate_model()
        logger.info(f"Model validated: {valid}")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if hasattr(self.backend, "tokenizer") and self.backend.tokenizer is not None:
                self._tokenizer = self.backend.tokenizer
            else:
                from transformers import AutoTokenizer

                meta = self.backend.get_metadata()
                model_name = meta.get("model", ModelConfig.default_model)
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        request_id = request.request_id or "unknown"
        metrics = InferenceMetrics(
            backend_name="quantized",
            model_name=self.backend.get_metadata().get("model", "unknown"),
        )
        event_logger.log(request_id, "inference", "generate_start", {
            "prompt_len": len(request.prompt),
            "max_new_tokens": request.max_new_tokens,
        })

        device = self.backend.get_metadata().get("device", "cpu")
        inputs = self.tokenizer(request.prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        prompt_tokens = input_ids.shape[1]
        metrics.prompt_tokens = prompt_tokens

        t_start = time.perf_counter()
        output_ids = self.backend.run_inference(
            input_ids, request.max_new_tokens, request.temperature, request.top_p,
        )
        t_end = time.perf_counter()

        new_tokens = output_ids[0][prompt_tokens:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        metrics.output_tokens = len(new_tokens)
        metrics.ttft_ms = (t_end - t_start) * 1000.0 / max(len(new_tokens), 1)
        metrics.gpu_memory_mb = gpu_memory_mb()
        metrics.finalize()

        event_logger.log(request_id, "inference", "generate_done", {
            "output_tokens": metrics.output_tokens,
            "latency_ms": round(metrics.total_latency_ms or 0, 2),
        })

        return GenerateResponse(
            request_id=request_id,
            mode="quantized",
            output_text=output_text,
            metrics=metrics.model_dump(exclude_none=True),
            debug=self.backend.get_metadata(),
        )
