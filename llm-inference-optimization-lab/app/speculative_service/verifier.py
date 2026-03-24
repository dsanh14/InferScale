"""Verification logic for speculative decoding.

Implements the core accept/reject loop: the draft model proposes K tokens,
the target model scores them in a single forward pass, and we accept the
longest prefix where draft and target agree (under greedy or sampling).
"""

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Outcome of verifying a batch of draft tokens against the target."""
    accepted_tokens: list[int]
    accepted_count: int
    proposed_count: int
    bonus_token: int | None = None


def verify_draft_tokens(
    target_model: torch.nn.Module,
    input_ids: torch.Tensor,
    draft_token_ids: list[int],
    device: str = "cpu",
) -> VerificationResult:
    """Verify draft-proposed tokens using the target model.

    For each draft token at position i, we check whether the target model's
    top-1 prediction at position i matches the draft token.  The first
    mismatch ends acceptance; the target's own prediction at that position
    becomes the bonus token.
    """
    accepted: list[int] = []
    bonus: int | None = None

    if not draft_token_ids:
        return VerificationResult(
            accepted_tokens=[], accepted_count=0, proposed_count=0,
        )

    candidate_ids = torch.cat([
        input_ids,
        torch.tensor([draft_token_ids], dtype=torch.long, device=device),
    ], dim=1)

    with torch.inference_mode():
        logits = target_model(candidate_ids).logits

    prompt_len = input_ids.shape[1]

    for i, draft_tok in enumerate(draft_token_ids):
        target_tok = logits[0, prompt_len + i - 1].argmax(dim=-1).item()
        if target_tok == draft_tok:
            accepted.append(draft_tok)
        else:
            bonus = target_tok
            break
    else:
        bonus = logits[0, prompt_len + len(draft_token_ids) - 1].argmax(dim=-1).item()

    return VerificationResult(
        accepted_tokens=accepted,
        accepted_count=len(accepted),
        proposed_count=len(draft_token_ids),
        bonus_token=bonus,
    )
