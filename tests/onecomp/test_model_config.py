"""Unit tests for ModelConfig MPS model-loading behavior.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuhki Yano
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from onecomp.model_config import ModelConfig


def _mock_loaded_model():
    model = MagicMock(name="model")
    model.parameters.side_effect = lambda: iter([torch.empty(1, dtype=torch.float16)])
    model.to.return_value = model
    return model


def test_load_model_loads_mps_model_on_cpu_then_moves_to_mps(monkeypatch):
    """MPS loads sharded checkpoints on CPU before moving the model to MPS."""
    model = _mock_loaded_model()
    config = SimpleNamespace(quantization_config=None)

    load_config = MagicMock(return_value=config)
    load_model = MagicMock(return_value=model)

    monkeypatch.setattr("onecomp.model_config.AutoConfig.from_pretrained", load_config)
    monkeypatch.setattr(
        "onecomp.model_config.AutoModelForCausalLM.from_pretrained",
        load_model,
    )

    model_config = ModelConfig(model_id="test/model", device="mps")
    loaded = model_config.load_model()

    assert loaded is model
    load_model.assert_called_once_with(
        "test/model",
        dtype=torch.float16,
        device_map="cpu",
    )
    model.to.assert_called_once_with("mps")
    model.eval.assert_called_once_with()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_load_model_preserves_explicit_no_mps_device_map(monkeypatch, device):
    """Explicit non-MPS device maps are passed through unchanged."""
    model = _mock_loaded_model()
    config = SimpleNamespace(quantization_config=None)

    monkeypatch.setattr(
        "onecomp.model_config.AutoConfig.from_pretrained",
        MagicMock(return_value=config),
    )
    load_model = MagicMock(return_value=model)
    monkeypatch.setattr(
        "onecomp.model_config.AutoModelForCausalLM.from_pretrained",
        load_model,
    )

    model_config = ModelConfig(model_id="test/model", device=device)
    model_config.load_model()

    load_model.assert_called_once_with(
        "test/model",
        dtype=torch.float16,
        device_map=device,
    )
    model.to.assert_not_called()
    model.eval.assert_called_once_with()


@pytest.mark.parametrize(
    "resolved_device, expected_device_map",
    [
        (torch.device("cpu"), "auto"),
        (torch.device("mps"), "cpu"),
    ],
)
def test_load_model_auto_preserves_device_map(monkeypatch, resolved_device, expected_device_map):
    """device='auto' keeps Transformers auto placement unless MPS is selected."""
    model = _mock_loaded_model()
    config = SimpleNamespace(quantization_config=None)
    load_model = MagicMock(return_value=model)
    monkeypatch.setattr(
        "onecomp.model_config.AutoConfig.from_pretrained",
        MagicMock(return_value=config),
    )
    monkeypatch.setattr(
        "onecomp.model_config.AutoModelForCausalLM.from_pretrained",
        load_model,
    )
    monkeypatch.setattr(
        "onecomp.model_config.get_default_device",
        MagicMock(return_value=resolved_device),
    )
    ModelConfig(model_id="test/model", device="auto").load_model()

    load_model.assert_called_once_with(
        "test/model",
        dtype=torch.float16,
        device_map=expected_device_map,
    )
    if resolved_device.type == "mps":
        model.to.assert_called_once_with(resolved_device)
    else:
        model.to.assert_not_called()
    model.eval.assert_called_once_with()
