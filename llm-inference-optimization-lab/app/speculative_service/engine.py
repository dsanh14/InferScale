"""Speculative decoding engine — draft-propose, target-verify loop."""

from __future__ import annotations

import time

import torch

from app.common.events import EventLogger
from app.common.logging_utils import setup_logging
from app.common.schemas import GenerateRequest, GenerateResponse, InferenceMetrics
from app.common.utils import gpu_memory_mb
from app.speculative_service.draft_target import DraftTargetPair
from app.speculative_service.verifier import verify_draft_tokens

logger = setup_logging("speculative.engine")
event_logger = EventLogger("speculative")

DRAFT_STEPS = 5  # tokens proposed per speculation round


class SpeculativeEngine:
    """Runs the speculative decoding loop with metrics collection."""

    def __init__(self, pair: DraftTargetPair, draft_steps: int = DRAFT_STEPS) -> None:
        self.pair = pair
        self.draft_steps = draft_steps

    @torch.inference_mode()
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        request_id = request.request_id or "unknown"
        metrics = InferenceMetrics(
            backend_name="speculative",
            model_name=f"{self.pair.draft_name}+{self.pair.target_name}",
        )

        event_logger.log(request_id, "inference", "spec_start", {
            "prompt_len": len(request.prompt),
            "max_new_tokens": request.max_new_tokens,
            "draft_steps": self.draft_steps,
        })

        tokenizer = self.pair.draft_tokenizer
        device = self.pair.device

        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        prompt_tokens = input_ids.shape[1]
        metrics.prompt_tokens = prompt_tokens

        generated: list[int] = []
        total_proposed = 0
        total_accepted = 0
        t_start = time.perf_counter()
        ttft_recorded = False

        while len(generated) < request.max_new_tokens:
            current_ids = torch.cat([
                input_ids,
                torch.tensor([generated], dtype=torch.long, device=device),
            ], dim=1) if generated else input_ids

            # Draft phase: propose K tokens greedily
            draft_tokens: list[int] = []
            draft_ids = current_ids.clone()
            for _ in range(min(self.draft_steps, request.max_new_tokens - len(generated))):
                logits = self.pair.draft_model(draft_ids).logits
                next_tok = logits[0, -1].argmax(dim=-1).item()
                draft_tokens.append(next_tok)
                draft_ids = torch.cat([
                    draft_ids,
                    torch.tensor([[next_tok]], dtype=torch.long, device=device),
                ], dim=1)

            total_proposed += len(draft_tokens)

            # Verify phase
            result = verify_draft_tokens(
                self.pair.target_model, current_ids, draft_tokens, device,
            )
            total_accepted += result.accepted_count

            generated.extend(result.accepted_tokens)
            if result.bonus_token is not None and len(generated) < request.max_new_tokens:
                generated.append(result.bonus_token)

            if not ttft_recorded and generated:
                metrics.ttft_ms = (time.perf_counter() - t_start) * 1000.0
                ttft_recorded = True

            eos_id = tokenizer.eos_token_id
            if eos_id is not None and eos_id in generated:
                eos_pos = generated.index(eos_id)
                generated = generated[:eos_pos]
                break

            if result.accepted_count == 0 and result.bonus_token is None:
                break

        output_text = tokenizer.decode(generated, skip_special_tokens=True)
        metrics.output_tokens = len(generated)
        metrics.speculative_proposed = total_proposed
        metrics.speculative_accepted = total_accepted
        metrics.gpu_memory_mb = gpu_memory_mb()
        metrics.finalize()

        event_logger.log(request_id, "inference", "spec_done", {
            "output_tokens": metrics.output_tokens,
            "proposed": total_proposed,
            "accepted": total_accepted,
            "acceptance_rate": round(metrics.speculative_acceptance_rate or 0, 4),
            "latency_ms": round(metrics.total_latency_ms or 0, 2),
        })

        return GenerateResponse(
            request_id=request_id,
            mode="speculative",
            output_text=output_text,
            metrics=metrics.model_dump(exclude_none=True),
            debug={
                "draft_model": self.pair.draft_name,
                "target_model": self.pair.target_name,
                "draft_steps": self.draft_steps,
                "device": device,
            },
        )
