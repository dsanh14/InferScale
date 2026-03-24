"""Draft and target model management for speculative decoding."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.common.config import ModelConfig
from app.common.logging_utils import setup_logging
from app.common.utils import get_device

logger = setup_logging("speculative.draft_target")


class DraftTargetPair:
    """Loads and holds the draft (small/fast) and target (large/accurate) models."""

    def __init__(
        self,
        draft_name: str | None = None,
        target_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.draft_name = draft_name or ModelConfig.draft_model
        self.target_name = target_name or ModelConfig.target_model
        self.device = get_device(device or ModelConfig.device)

        logger.info(f"Loading draft model '{self.draft_name}' on '{self.device}'")
        self.draft_tokenizer = AutoTokenizer.from_pretrained(self.draft_name)
        if self.draft_tokenizer.pad_token is None:
            self.draft_tokenizer.pad_token = self.draft_tokenizer.eos_token

        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.draft_name, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        )
        self.draft_model.to(self.device)
        self.draft_model.eval()

        logger.info(f"Loading target model '{self.target_name}' on '{self.device}'")
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_name)
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token

        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_name, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        )
        self.target_model.to(self.device)
        self.target_model.eval()

        logger.info("Draft-target pair loaded")
