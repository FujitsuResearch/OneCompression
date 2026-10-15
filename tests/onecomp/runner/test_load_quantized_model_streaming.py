"""Tests for the streaming path of ``load_quantized_model``: planning and placement.

Copyright 2025-2026 Fujitsu Ltd.
"""

from pathlib import Path

import pytest
import torch

from onecomp import load_quantized_model
from onecomp import quantized_model_loader_streaming as streaming
from onecomp.quantized_model_loader import QuantizedModelLoader
from tests.onecomp.fixtures.quantized_checkpoint import (
    assert_same_loaded_model,
    intercept_tensor_reads,
    write_quantized_checkpoint,
)


def _logits(model: torch.nn.Module) -> torch.Tensor:
    """Return the logits of a fixed short prompt."""
    input_ids = torch.tensor([[1, 5, 9, 3]], device=model.device)
    with torch.no_grad():
        return model(input_ids).logits


def _load_both(save_dir: Path, **kwargs) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Return the models loaded with ``load_strategy`` "streaming" and "legacy"."""
    actual, _ = load_quantized_model(str(save_dir), load_strategy="streaming", **kwargs)
    expected, _ = load_quantized_model(str(save_dir), load_strategy="legacy", **kwargs)
    return actual, expected


@pytest.mark.parametrize(
    "checkpoint",
    [
        pytest.param({}, id="qwen3-tied-4bit"),
        pytest.param(
            {"model_type": "llama", "tie_word_embeddings": False, "layer_bits": (2, 2)},
            id="llama-untied-2bit",
        ),
        pytest.param({"tie_word_embeddings": False, "layer_bits": (3, 3)}, id="qwen3-untied-3bit"),
        pytest.param(
            {"model_type": "llama", "layer_bits": (8, 8), "checkpoint_format": "gptq_v2"},
            id="llama-tied-8bit-gptq-v2",
        ),
        pytest.param({"layer_bits": (4, 8)}, id="qwen3-mixed-gptq"),
        pytest.param({"config_dtype": "bfloat16", "fp32_norm": True}, id="qwen3-bf16-fp32-norm"),
    ],
)
def test_streaming_load_matches_legacy(tmp_path, checkpoint):
    write_quantized_checkpoint(tmp_path / "model", **checkpoint)

    actual, expected = _load_both(tmp_path / "model", device_map=None)

    assert_same_loaded_model(actual, expected)
    # A float32 norm weight makes the Hugging Face RMSNorm return float32
    # activations, which the bfloat16 linear layers reject in either model.
    if not checkpoint.get("fp32_norm"):
        assert torch.equal(_logits(actual), _logits(expected))


@pytest.mark.parametrize(
    "edit_state_dict, plan_type",
    [
        pytest.param(None, streaming.StreamingPlan, id="supported"),
        pytest.param(
            lambda state_dict: state_dict.pop("model.layers.0.mlp.gate_proj.weight"),
            streaming.StreamingUnsupported,
            id="non-critical-missing-key",
        ),
    ],
)
def test_planning_reads_no_tensor_value(tmp_path, monkeypatch, edit_state_dict, plan_type):
    save_dir = tmp_path / "model"
    write_quantized_checkpoint(save_dir, edit_state_dict=edit_state_dict)
    reads = []
    intercept_tensor_reads(monkeypatch, reads.append)

    config_dict, quant_config = QuantizedModelLoader._load_config_and_quant_config(str(save_dir))
    plan = streaming.plan_streaming_load(str(save_dir), config_dict, quant_config, None, None)

    assert isinstance(plan, plan_type)
    assert reads == []


def _use_dbf_for_one_layer(quant_config: dict) -> None:
    """Switch one layer of a ``mixed_gptq`` table to the DBF method."""
    quant_config["quantization_bits"][1]["mlp.down_proj"]["method"] = "dbf"


@pytest.mark.parametrize(
    "checkpoint",
    [
        pytest.param(
            {"edit_state_dict": lambda state_dict: state_dict.pop("model.norm.weight")},
            id="missing-norm",
        ),
        pytest.param(
            {
                "edit_state_dict": lambda state_dict: state_dict.update(
                    {"model.layers.0.self_attn.k_proj.weight": torch.zeros(16, 64).half()}
                )
            },
            id="weight-shape-mismatch",
        ),
        pytest.param(
            {"layer_bits": (4, 8), "edit_quantization_config": _use_dbf_for_one_layer},
            id="mixed-layer-methods",
        ),
    ],
)
def test_streaming_raises_legacy_errors_before_reading_values(tmp_path, monkeypatch, checkpoint):
    save_dir = tmp_path / "model"
    write_quantized_checkpoint(save_dir, **checkpoint)
    with pytest.raises((RuntimeError, ValueError)) as legacy_error:
        load_quantized_model(str(save_dir), device_map=None, load_strategy="legacy")
    reads = []
    intercept_tensor_reads(monkeypatch, reads.append)

    with pytest.raises(legacy_error.type) as streaming_error:
        load_quantized_model(str(save_dir), device_map=None, load_strategy="streaming")

    assert str(streaming_error.value) == str(legacy_error.value)
    assert reads == []


@pytest.mark.parametrize(
    "num_shards, rewrite_index",
    [
        pytest.param(2, lambda index: None, id="sharded-with-index"),
        pytest.param(2, lambda index: index.unlink(), id="sharded-without-index"),
        pytest.param(1, lambda index: index.write_text("{"), id="index-not-json"),
        pytest.param(
            1, lambda index: index.write_text('{"metadata": {}}'), id="index-without-weight-map"
        ),
    ],
)
def test_streaming_reads_root_safetensors_regardless_of_index(tmp_path, num_shards, rewrite_index):
    write_quantized_checkpoint(tmp_path / "model", num_shards=num_shards)
    rewrite_index(tmp_path / "model" / "model.safetensors.index.json")

    actual, expected = _load_both(tmp_path / "model", device_map=None)

    assert_same_loaded_model(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streaming_places_the_whole_model_on_one_cuda_device(tmp_path):
    write_quantized_checkpoint(tmp_path / "model")

    actual, expected = _load_both(tmp_path / "model")

    assert actual.hf_device_map == {"": 0}
    assert not hasattr(actual, "_hf_hook")
    assert_same_loaded_model(actual, expected)
    assert torch.equal(_logits(actual), _logits(expected))
