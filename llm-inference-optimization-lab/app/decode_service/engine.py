"""Decode engine — continues autoregressive generation from a cache artifact."""

from __future__ import annotations

import time
from typing import Any

import torch

from app.common.events import EventLogger
from app.common.logging_utils import setup_logging
from app.common.schemas import GenerateRequest, InferenceMetrics
from app.common.utils import gpu_memory_mb
from app.baseline_service.model_loader import ModelHandle
from app.decode_service.handoff import simulate_handoff, HandoffMetrics
from app.prefill_service.cache_artifact import CacheArtifact

logger = setup_logging("decode.engine")
event_logger = EventLogger("decode")


class DecodeEngine:
    """Continues generation given a prefill cache artifact."""

    def __init__(self, handle: ModelHandle) -> None:
        self.handle = handle

    @torch.inference_mode()
    def decode(
        self, request: GenerateRequest, artifact: CacheArtifact,
    ) -> dict[str, Any]:
        request_id = request.request_id or "unknown"
        event_logger.log(request_id, "decode", "start", {
            "prompt_tokens": artifact.prompt_tokens,
            "kv_cache_hash": artifact.kv_cache_hash,
        })

        # Simulate handoff transfer
        handoff = simulate_handoff(artifact.kv_cache_size_bytes)
        event_logger.log(request_id, "decode", "handoff_complete", {
            "queue_wait_ms": round(handoff.queue_wait_ms, 3),
            "transfer_ms": round(handoff.transfer_ms, 3),
        })

        tokenizer = self.handle.tokenizer
        model = self.handle.model
        device = self.handle.device

        inputs = tokenizer(artifact.prompt, return_tensors="pt").to(device)
        prompt_tokens = inputs["input_ids"].shape[1]

        max_new = artifact.generation_params.get("max_new_tokens", 64)
        temp = artifact.generation_params.get("temperature", 1.0)
        top_p = artifact.generation_params.get("top_p", 1.0)

        metrics = InferenceMetrics(
            backend_name="disaggregated_decode",
            model_name=artifact.model_name,
            prompt_tokens=prompt_tokens,
            queue_wait_ms=handoff.queue_wait_ms,
            transfer_latency_ms=handoff.transfer_ms,
        )

        t0 = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=max(temp, 0.01),
            top_p=top_p,
            do_sample=temp > 0.01,
        )
        t1 = time.perf_counter()

        new_tokens = outputs[0][prompt_tokens:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        metrics.output_tokens = len(new_tokens)
        decode_ms = (t1 - t0) * 1000.0
        metrics.ttft_ms = decode_ms / max(len(new_tokens), 1)
        metrics.gpu_memory_mb = gpu_memory_mb()
        metrics.finalize()

        event_logger.log(request_id, "decode", "done", {
            "output_tokens": metrics.output_tokens,
            "decode_ms": round(decode_ms, 2),
        })

        return {
            "output_text": output_text,
            "metrics": metrics.model_dump(exclude_none=True),
        }
