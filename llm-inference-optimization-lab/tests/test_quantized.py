"""Tests for quantized service adapter interface."""

import pytest

from app.quantized_service.adapters import (
    PortableQuantizedBackend,
    TensorRTModelOptAdapter,
    create_backend,
)
from app.quantized_service.calibration import (
    CalibrationConfig,
    CalibrationResult,
    run_calibration,
)


class TestCreateBackend:
    def test_default_is_portable(self):
        backend = create_backend("portable_quantized")
        assert isinstance(backend, PortableQuantizedBackend)

    def test_tensorrt_adapter(self):
        backend = create_backend("tensorrt")
        assert isinstance(backend, TensorRTModelOptAdapter)


class TestTensorRTStub:
    def test_prepare_raises(self):
        adapter = TensorRTModelOptAdapter()
        with pytest.raises(NotImplementedError):
            adapter.prepare_model("gpt2", "cpu")

    def test_metadata(self):
        adapter = TensorRTModelOptAdapter()
        meta = adapter.get_metadata()
        assert meta["backend"] == "tensorrt_model_opt"
        assert meta["status"] == "not_installed"


class TestCalibration:
    def test_calibration_with_not_implemented_backend(self):
        adapter = TensorRTModelOptAdapter()
        result = run_calibration(adapter)
        assert not result.success
        assert "skipped" in result.metadata.get("note", "")

    def test_calibration_config_defaults(self):
        cfg = CalibrationConfig()
        assert len(cfg.prompts) == 4
        assert cfg.num_batches == 4
