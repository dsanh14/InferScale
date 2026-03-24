"""Prefill engine — runs the prompt through the model and produces a cache artifact."""

from __future__ import annotations

import hashlib
import time

import torch

from app.common.events import EventLogger
from app.common.logging_utils import setup_logging
from app.common.schemas import GenerateRequest
from app.common.utils import gpu_memory_mb
from app.baseline_service.model_loader import ModelHandle
from app.prefill_service.cache_artifact import CacheArtifact

logger = setup_logging("prefill.engine")
event_logger = EventLogger("prefill")


class PrefillEngine:
    """Runs the prefill (prompt encoding) phase and produces a CacheArtifact."""

    def __init__(self, handle: ModelHandle) -> None:
        self.handle = handle

    @torch.inference_mode()
    def prefill(self, request: GenerateRequest) -> CacheArtifact:
        request_id = request.request_id or "unknown"
        event_logger.log(request_id, "prefill", "start", {
            "prompt_len": len(request.prompt),
        })

        tokenizer = self.handle.tokenizer
        model = self.handle.model
        device = self.handle.device

        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        prompt_tokens = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        outputs = model(**inputs, use_cache=True)
        t1 = time.perf_counter()
        prefill_ms = (t1 - t0) * 1000.0

        # Build a fingerprint of the KV cache for the replay validator
        past_kv = outputs.past_key_values
        cache_bytes = 0
        hash_parts: list[str] = []
        if past_kv is not None:
            for layer_kv in past_kv:
                for tensor in layer_kv:
                    cache_bytes += tensor.nelement() * tensor.element_size()
                    hash_parts.append(str(tensor.shape))
        kv_hash = hashlib.sha256("".join(hash_parts).encode()).hexdigest()[:32]

        artifact = CacheArtifact(
            request_id=request_id,
            prompt=request.prompt,
            prompt_tokens=prompt_tokens,
            model_name=self.handle.model_name,
            device=device,
            kv_cache_hash=kv_hash,
            kv_cache_size_bytes=cache_bytes,
            prefill_latency_ms=round(prefill_ms, 3),
            generation_params={
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
            metadata={"gpu_memory_mb": gpu_memory_mb()},
        )

        event_logger.log(request_id, "prefill", "done", {
            "prompt_tokens": prompt_tokens,
            "kv_cache_bytes": cache_bytes,
            "prefill_ms": round(prefill_ms, 2),
        })

        return artifact
