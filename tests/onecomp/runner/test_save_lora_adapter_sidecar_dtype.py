"""Unit tests for ``Runner._save_lora_adapter_sidecar`` save dtype.

Pins the contract that the LoRA adapter sidecar is written in the base
model's runtime dtype, so the train -> save round-trip is a single
fp32 -> base-dtype rounding rather than an extra fp16 intermediate:

- ``model_config.dtype == "bfloat16"`` -> tensors saved as ``torch.bfloat16``
- ``model_config.dtype == "float16"``  -> tensors saved as ``torch.float16``

This save path expects ``model_config.dtype`` to be a concrete
``"float16"``/``"bfloat16"``, so an unexpected value (e.g. ``"auto"``) is out
of scope and intentionally raises rather than silently falling back.

The test builds a real ``LoRAGPTQLinear`` on CPU using a plain ``nn.Linear``
as a stand-in base layer (only ``in_features``/``out_features``/``parameters``
are touched), so it needs no GPU, no model download, and no full ``Runner``.

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    pytest tests/onecomp/runner/test_save_lora_adapter_sidecar_dtype.py -v
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file

from onecomp.post_process.post_process_lora_sft import LoRAGPTQLinear
from onecomp.runner import Runner
from onecomp.utils.lora import LORA_ADAPTER_SUBDIR


def _make_quantized_model() -> nn.Module:
    """A tiny container model with a single ``LoRAGPTQLinear`` submodule.

    ``LoRAGPTQLinear`` only reads ``in_features``/``out_features`` and freezes
    ``base_layer.parameters()``, so a plain ``nn.Linear`` works as the base
    layer without pulling in the real GPTQ machinery.
    """
    base_layer = nn.Linear(8, 16, bias=False)
    lora = LoRAGPTQLinear(base_layer, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    # Make lora_B non-zero so the saved tensor is not trivially all zeros
    # (zeros round-trip identically across dtypes and would weaken the check).
    nn.init.normal_(lora.lora_B.weight)

    # Nest submodules so named_modules() yields the dotted path
    # "model.layers.0.self_attn.q_proj" (add_module rejects dotted names).
    self_attn = nn.Module()
    self_attn.q_proj = lora
    layers = nn.ModuleList([nn.Module()])
    layers[0].self_attn = self_attn
    inner = nn.Module()
    inner.layers = layers
    model = nn.Module()
    model.model = inner
    return model


def _save_with_dtype(tmp_path: Path, dtype: str) -> dict:
    """Run the sidecar save on a stub Runner and return the loaded tensors."""
    stub = SimpleNamespace(
        quantized_model=_make_quantized_model(),
        model_config=SimpleNamespace(dtype=dtype),
        logger=SimpleNamespace(info=lambda *a, **k: None),
        _collect_lora_gptq_modules=Runner._collect_lora_gptq_modules,
        _remap_text_only_module_name_to_full_wrapper=(
            Runner._remap_text_only_module_name_to_full_wrapper
        ),
    )

    wrote = Runner._save_lora_adapter_sidecar(stub, str(tmp_path))
    assert wrote is True

    adapter_path = tmp_path / LORA_ADAPTER_SUBDIR / "adapter_model.safetensors"
    assert adapter_path.exists()
    return load_file(str(adapter_path))


@pytest.mark.parametrize(
    "cfg_dtype, expected",
    [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
    ],
)
def test_sidecar_saved_in_base_dtype(tmp_path, cfg_dtype, expected):
    """Saved LoRA tensors use the base model dtype."""
    tensors = _save_with_dtype(tmp_path, cfg_dtype)

    assert tensors, "No tensors written to the adapter sidecar"
    for key, tensor in tensors.items():
        assert tensor.dtype == expected, f"{key} saved as {tensor.dtype}, expected {expected}"


def test_unexpected_dtype_raises(tmp_path):
    """An out-of-scope dtype (e.g. "auto") raises rather than falling back."""
    with pytest.raises(AttributeError):
        _save_with_dtype(tmp_path, "auto")


def test_sidecar_keys_follow_peft_convention(tmp_path):
    """Keys are PEFT-style and adapter_config.json round-trips r/alpha."""
    tensors = _save_with_dtype(tmp_path, "bfloat16")

    assert set(tensors) == {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
    }

    config_path = tmp_path / LORA_ADAPTER_SUBDIR / "adapter_config.json"
    config = json.loads(config_path.read_text())
    assert config["r"] == 4
    assert config["lora_alpha"] == 8
    assert config["target_modules"] == ["q_proj"]


def test_sidecar_keys_remapped_for_full_wrapper(tmp_path):
    """save_format='full_wrapper' remaps adapter keys to the composite
    ``model.language_model.*`` namespace so vLLM's full-wrapper loader can
    match the adapter tensors to the remapped base layers."""
    stub = SimpleNamespace(
        quantized_model=_make_quantized_model(),
        model_config=SimpleNamespace(dtype="bfloat16"),
        logger=SimpleNamespace(info=lambda *a, **k: None),
        _collect_lora_gptq_modules=Runner._collect_lora_gptq_modules,
        _remap_text_only_module_name_to_full_wrapper=(
            Runner._remap_text_only_module_name_to_full_wrapper
        ),
    )

    wrote = Runner._save_lora_adapter_sidecar(stub, str(tmp_path), save_format="full_wrapper")
    assert wrote is True

    adapter_path = tmp_path / LORA_ADAPTER_SUBDIR / "adapter_model.safetensors"
    tensors = load_file(str(adapter_path))

    assert set(tensors) == {
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight",
    }
