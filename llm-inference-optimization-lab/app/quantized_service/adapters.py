"""Backend adapter interface for quantized inference.

Defines a clean seam so the portable local backend can be swapped for
NVIDIA TensorRT-LLM / Model Optimizer without changing the engine or API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.common.config import ModelConfig
from app.common.logging_utils import setup_logging
from app.common.utils import get_device

logger = setup_logging("quantized.adapters")


class QuantizedBackend(ABC):
    """Interface every quantized backend must implement."""

    @abstractmethod
    def prepare_model(self, model_name: str, device: str) -> None:
        ...

    @abstractmethod
    def quantize_model(self, calibration_data: list[str] | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    def validate_model(self) -> bool:
        ...

    @abstractmethod
    def run_inference(
        self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float, top_p: float,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        ...


class PortableQuantizedBackend(QuantizedBackend):
    """Local CPU/GPU backend that applies torch dynamic quantization.

    Provides a real (if modest) quantization path that works on any machine.
    The performance characteristics differ from TensorRT INT8/FP8 but the
    API contract is identical.
    """

    def __init__(self) -> None:
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.device: str = "cpu"
        self.model_name: str = ""
        self._quantized = False

    def prepare_model(self, model_name: str, device: str) -> None:
        self.model_name = model_name or ModelConfig.default_model
        self.device = get_device(device or ModelConfig.device)
        logger.info(f"Loading model '{self.model_name}' for portable quantization")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        )
        self.model.eval()

    def quantize_model(self, calibration_data: list[str] | None = None) -> dict[str, Any]:
        assert self.model is not None
        if self.device == "cpu":
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8,
            )
            self._quantized = True
            logger.info("Applied torch dynamic INT8 quantization (CPU)")
        else:
            logger.info("Skipping dynamic quantization on non-CPU device (no-op)")
            self._quantized = False

        self.model.to(self.device)
        return {
            "backend": "portable_quantized",
            "quantized": self._quantized,
            "device": self.device,
        }

    def validate_model(self) -> bool:
        assert self.model is not None and self.tokenizer is not None
        test_input = self.tokenizer("Hello", return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**test_input, max_new_tokens=5)
        return out.shape[1] > test_input["input_ids"].shape[1]

    @torch.inference_mode()
    def run_inference(
        self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float, top_p: float,
    ) -> torch.Tensor:
        assert self.model is not None
        return self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            do_sample=temperature > 0.01,
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "backend": "portable_quantized",
            "model": self.model_name,
            "quantized": self._quantized,
            "device": self.device,
        }


class TensorRTModelOptAdapter(QuantizedBackend):
    """Adapter stub for NVIDIA TensorRT Model Optimizer / TensorRT-LLM.

    This class defines the integration seam.  When running on a machine
    with the NVIDIA stack installed, replace the placeholder bodies with
    real calls to:
      - modelopt.torch.quantization (for PTQ calibration)
      - tensorrt_llm.Builder / tensorrt_llm.runtime (for engine build + inference)

    See docs/nvidia_mapping.md for the full integration guide.
    """

    def prepare_model(self, model_name: str, device: str) -> None:
        # TODO: Import modelopt and load the HF checkpoint
        # TODO: Apply modelopt.torch.quantization.quantize() with INT8/FP8 config
        # TODO: Export to TensorRT-LLM checkpoint format
        raise NotImplementedError(
            "TensorRT Model Optimizer backend requires the NVIDIA modelopt "
            "and tensorrt_llm packages.  See docs/nvidia_mapping.md."
        )

    def quantize_model(self, calibration_data: list[str] | None = None) -> dict[str, Any]:
        # TODO: Run calibration with representative prompts
        # TODO: Build TensorRT engine with quantized weights
        raise NotImplementedError("TensorRT quantization not available")

    def validate_model(self) -> bool:
        # TODO: Run a short generation and compare perplexity against FP16 baseline
        raise NotImplementedError("TensorRT validation not available")

    def run_inference(
        self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float, top_p: float,
    ) -> torch.Tensor:
        # TODO: Use tensorrt_llm.runtime.GenerationSession to generate
        raise NotImplementedError("TensorRT inference not available")

    def get_metadata(self) -> dict[str, Any]:
        return {"backend": "tensorrt_model_opt", "status": "not_installed"}


def create_backend(backend_mode: str | None = None) -> QuantizedBackend:
    """Factory that returns the appropriate quantized backend."""
    mode = backend_mode or ModelConfig.quantized_backend
    if mode == "tensorrt":
        return TensorRTModelOptAdapter()
    return PortableQuantizedBackend()
