"""Model loading utilities with graceful CPU/GPU fallback."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.common.config import ModelConfig
from app.common.logging_utils import setup_logging
from app.common.utils import get_device

logger = setup_logging("baseline.model_loader")


class ModelHandle:
    """Wraps a loaded model + tokenizer pair with device metadata."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str,
        model_name: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name


def load_model(
    model_name: str | None = None,
    device: str | None = None,
) -> ModelHandle:
    """Load a Hugging Face causal LM onto the best available device."""
    model_name = model_name or ModelConfig.default_model
    device = get_device(device or ModelConfig.device)
    logger.info(f"Loading model '{model_name}' on device '{device}'")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    logger.info(f"Model '{model_name}' loaded successfully on '{device}'")
    return ModelHandle(model=model, tokenizer=tokenizer, device=device, model_name=model_name)
