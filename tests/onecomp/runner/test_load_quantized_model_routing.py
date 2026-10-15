"""Tests for how ``load_quantized_model`` chooses between the streaming and legacy loaders.

Copyright 2025-2026 Fujitsu Ltd.
"""

import inspect
import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, Qwen3Config, Qwen3ForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

import onecomp
from onecomp import StreamingUnsupportedError, load_quantized_model
from onecomp import quantized_model_loader_streaming as streaming
from onecomp.quantized_model_loader import QuantizedModelLoader
from onecomp.utils.lora import LORA_ADAPTER_SUBDIR
from tests.onecomp.fixtures.quantized_checkpoint import (
    assert_same_loaded_model,
    intercept_tensor_reads,
    write_quantized_checkpoint,
)


def test_load_strategy_is_added_as_a_keyword_only_argument_defaulting_to_auto():
    keyword_defaults = {
        "torch_dtype": None,
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": True,
        "load_strategy": "auto",
    }
    for function in (QuantizedModelLoader.load_quantized_model, onecomp.load_quantized_model):
        parameters = inspect.signature(function).parameters
        assert list(parameters) == ["save_directory", *keyword_defaults]
        assert parameters["save_directory"].default is inspect.Parameter.empty
        for name, default in keyword_defaults.items():
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default == default


@pytest.mark.parametrize("load_strategy", ["invalid", None, 0, ["auto"]])
def test_invalid_load_strategy_raises_before_the_path_is_checked(tmp_path, load_strategy):
    with pytest.raises(ValueError, match="load_strategy"):
        load_quantized_model(str(tmp_path / "missing"), load_strategy=load_strategy)


@pytest.mark.parametrize("load_strategy", ["auto", "streaming", "legacy"])
def test_missing_directory_raises_file_not_found_for_every_strategy(tmp_path, load_strategy):
    with pytest.raises(FileNotFoundError):
        load_quantized_model(str(tmp_path / "missing"), load_strategy=load_strategy)


def _loader_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "onecomp.quantized_model_loader" and record.levelno >= logging.WARNING
    ]


def _write_with_lora_sidecar(save_dir: Path) -> None:
    write_quantized_checkpoint(save_dir)
    adapter_dir = save_dir / LORA_ADAPTER_SUBDIR
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    prefix = "base_model.model.model.layers.0.mlp.down_proj"
    save_file(
        {
            f"{prefix}.lora_A.weight": torch.full((4, 128), 0.01),
            f"{prefix}.lora_B.weight": torch.full((64, 4), 0.01),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )


def _write_with_key_in_two_files(save_dir: Path) -> None:
    write_quantized_checkpoint(save_dir)
    save_file({"model.norm.weight": torch.ones(64).half()}, str(save_dir / "extra.safetensors"))


@pytest.mark.parametrize(
    "write_checkpoint, device_map",
    [
        pytest.param(
            lambda save_dir: write_quantized_checkpoint(
                save_dir,
                edit_state_dict=lambda state_dict: state_dict.pop(
                    "model.layers.0.mlp.gate_proj.weight"
                ),
            ),
            None,
            id="non-critical-missing-key",
        ),
        pytest.param(_write_with_lora_sidecar, None, id="lora-sidecar"),
        pytest.param(
            lambda save_dir: write_quantized_checkpoint(
                save_dir, edit_quantization_config=lambda config: config.update(rotated=True)
            ),
            None,
            id="rotated",
        ),
        pytest.param(
            lambda save_dir: write_quantized_checkpoint(save_dir, quant_method="dbf"),
            None,
            id="dbf",
        ),
        pytest.param(write_quantized_checkpoint, {}, id="device-map-dict"),
        pytest.param(_write_with_key_in_two_files, None, id="key-in-two-files"),
    ],
)
def test_unsupported_checkpoint_falls_back_under_auto_and_stops_under_streaming(
    tmp_path, caplog, write_checkpoint, device_map
):
    save_dir = tmp_path / "model"
    write_checkpoint(save_dir)
    caplog.set_level(logging.WARNING, logger="onecomp.quantized_model_loader")
    torch.manual_seed(0)
    expected, _ = load_quantized_model(
        str(save_dir), device_map=device_map, load_strategy="legacy"
    )
    expected_rng_state = torch.get_rng_state()
    legacy_warnings = _loader_warnings(caplog)
    caplog.clear()

    torch.manual_seed(0)
    actual, _ = load_quantized_model(str(save_dir), device_map=device_map)

    assert torch.equal(torch.get_rng_state(), expected_rng_state)
    assert_same_loaded_model(actual, expected)
    fallback_warning, *warnings_after_fallback = _loader_warnings(caplog)
    assert warnings_after_fallback == legacy_warnings

    with patch.object(QuantizedModelLoader, "_load_quantized_model_legacy") as legacy:
        with pytest.raises(StreamingUnsupportedError) as error:
            load_quantized_model(str(save_dir), device_map=device_map, load_strategy="streaming")
    legacy.assert_not_called()
    assert error.value.reason in fallback_warning


def test_custom_registration_is_left_to_the_legacy_loader(tmp_path, monkeypatch):
    constructed = []

    class CustomQwen3Config(Qwen3Config):
        pass

    class CustomQwen3ForCausalLM(Qwen3ForCausalLM):
        config_class = CustomQwen3Config

        def __init__(self, config):
            super().__init__(config)
            constructed.append(type(self))
            # Uses the CPU RNG, and .item() fails on the meta device.
            self.register_buffer("noise", torch.rand(4), persistent=False)
            self.lm_head.weight.sum().item()

    save_dir = tmp_path / "model"
    write_quantized_checkpoint(save_dir)
    monkeypatch.setitem(CONFIG_MAPPING._extra_content, "qwen3", CustomQwen3Config)
    monkeypatch.setitem(
        MODEL_FOR_CAUSAL_LM_MAPPING._extra_content, CustomQwen3Config, CustomQwen3ForCausalLM
    )

    with pytest.raises(StreamingUnsupportedError):
        load_quantized_model(str(save_dir), device_map=None, load_strategy="streaming")
    assert constructed == []

    torch.manual_seed(0)
    expected, _ = load_quantized_model(str(save_dir), device_map=None, load_strategy="legacy")
    expected_rng_state = torch.get_rng_state()
    torch.manual_seed(0)
    actual, _ = load_quantized_model(str(save_dir), device_map=None)

    assert type(actual) is CustomQwen3ForCausalLM
    assert torch.equal(torch.get_rng_state(), expected_rng_state)
    assert_same_loaded_model(actual, expected)


def test_legacy_strategy_does_not_plan_a_streaming_load(tmp_path):
    write_quantized_checkpoint(tmp_path / "model")

    with patch.object(streaming, "plan_streaming_load") as plan:
        load_quantized_model(str(tmp_path / "model"), device_map=None, load_strategy="legacy")

    plan.assert_not_called()


def test_read_failure_after_planning_propagates_without_legacy_retry(tmp_path, monkeypatch):
    write_quantized_checkpoint(tmp_path / "model")

    def fail(key):
        raise OSError(f"cannot read {key}")

    intercept_tensor_reads(monkeypatch, fail)

    with patch.object(QuantizedModelLoader, "_load_quantized_model_legacy") as legacy:
        with pytest.raises(OSError, match="cannot read"):
            load_quantized_model(str(tmp_path / "model"), device_map=None)
    legacy.assert_not_called()


def test_streaming_unsupported_error_is_a_public_value_error_with_reason():
    from onecomp.quantized_model_loader import StreamingUnsupportedError as defined_error

    error = StreamingUnsupportedError("no streaming for this")

    assert StreamingUnsupportedError is defined_error
    assert isinstance(error, ValueError)
    assert error.reason == "no streaming for this"
    assert "no streaming for this" in str(error)
