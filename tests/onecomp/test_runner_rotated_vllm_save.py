"""Unit tests for Runner rotated-vLLM save metadata handling.

Copyright 2025-2026 Fujitsu Ltd.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from onecomp.runner import Runner
from onecomp.utils import unfuse_moe as unfuse_moe_module
from onecomp.pre_process import rotation_utils


class _FakeLayer:
    pass


class _FakeModel:
    def __init__(self, *, num_hidden_layers=4, num_experts=0):
        self.config = SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            num_experts=num_experts,
            quantization_config=None,
        )
        self.down_proj = _FakeLayer()

    def named_modules(self):
        return [("model.layers.0.mlp.down_proj", self.down_proj)]


class _FakeModelConfig:
    def __init__(self, *, rotated: bool, fp32_had: bool = False, model=None):
        self._rotated = rotated
        self.fp32_had = fp32_had
        self._model = model or _FakeModel()

    def load_model(self, device_map=None):
        assert device_map == "cpu"
        return self._model

    def load_tokenizer(self):
        return "tokenizer"

    def has_additional_data(self):
        return self._rotated


class _FakeQuantizer:
    def __init__(self, quant_method="gptq"):
        self.results = {
            "model.layers.1.mlp.down_proj": object(),
            "model.layers.0.self_attn.q_proj": object(),
        }
        self._quant_method = quant_method

    def get_quant_config(self):
        return {"quant_method": self._quant_method, "group_size": 128}

    def apply_results_to_model(self, model, **kwargs):
        self.apply_kwargs = kwargs

    def finalize_quant_config_for_save(
        self,
        *,
        quant_config,
        quantized_layer_names,
        num_hidden_layers,
    ):
        quant_config = dict(quant_config)
        quant_config["finalized_names"] = list(quantized_layer_names)
        quant_config["finalized_num_hidden_layers"] = num_hidden_layers
        return quant_config


def test_create_quantized_model_switches_rotated_gptq_to_mixed(monkeypatch):
    register_calls = []
    patch_calls = []
    fake_model = _FakeModel(num_hidden_layers=6)
    runner = Runner(
        model_config=_FakeModelConfig(rotated=True, fp32_had=True, model=fake_model),
        quantizer=_FakeQuantizer("gptq"),
    )

    monkeypatch.setattr(unfuse_moe_module, "unfuse_moe_experts", lambda model, logger: False)
    monkeypatch.setattr(
        rotation_utils,
        "register_online_hadamard_hooks",
        lambda model, fp32_had=False, layers_cls=None: register_calls.append(
            (model, fp32_had, layers_cls)
        ) or [object()],
    )
    monkeypatch.setattr(
        Runner,
        "_patch_k_eq_v_for_vllm",
        lambda self, model, quant_config: patch_calls.append((model, dict(quant_config))),
    )

    model, tokenizer = runner.create_quantized_model()
    qcfg = model.config.quantization_config

    assert tokenizer == "tokenizer"
    assert qcfg["quant_method"] == "mixed_gptq"
    assert qcfg["rotated"] is True
    assert qcfg["fp32_had"] is True
    assert qcfg["modules_in_block_to_quantize"] == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.mlp.down_proj",
    ]
    assert qcfg["quantized_layer_names"] == qcfg["modules_in_block_to_quantize"]
    assert qcfg["finalized_num_hidden_layers"] == 6
    assert register_calls[0][0] is fake_model
    assert register_calls[0][1] is True
    assert register_calls[0][2] == [_FakeLayer]
    assert patch_calls[0][0] is fake_model
    assert patch_calls[0][1]["quant_method"] == "mixed_gptq"


def test_create_quantized_model_keeps_plain_gptq_unless_rotation_or_moe(monkeypatch):
    runner = Runner(
        model_config=_FakeModelConfig(rotated=False, fp32_had=False),
        quantizer=_FakeQuantizer("gptq"),
    )
    register_hook = MagicMock()

    monkeypatch.setattr(unfuse_moe_module, "unfuse_moe_experts", lambda model, logger: False)
    monkeypatch.setattr(rotation_utils, "register_online_hadamard_hooks", register_hook)
    monkeypatch.setattr(Runner, "_patch_k_eq_v_for_vllm", lambda self, model, quant_config: None)

    model, _tokenizer = runner.create_quantized_model()

    assert model.config.quantization_config["quant_method"] == "gptq"
    assert model.config.quantization_config["rotated"] is False
    assert model.config.quantization_config["fp32_had"] is False
    register_hook.assert_not_called()


def test_create_quantized_model_switches_moe_gptq_to_mixed(monkeypatch):
    fake_model = _FakeModel(num_hidden_layers=3, num_experts=8)
    runner = Runner(
        model_config=_FakeModelConfig(rotated=False, model=fake_model),
        quantizer=_FakeQuantizer("gptq"),
    )

    monkeypatch.setattr(unfuse_moe_module, "unfuse_moe_experts", lambda model, logger: False)
    monkeypatch.setattr(Runner, "_patch_k_eq_v_for_vllm", lambda self, model, quant_config: None)

    model, _tokenizer = runner.create_quantized_model()

    assert model.config.quantization_config["quant_method"] == "mixed_gptq"
    assert model.config.quantization_config["rotated"] is False


def test_create_quantized_model_keeps_existing_mixed_gptq_without_extra_switch(monkeypatch):
    fake_model = _FakeModel(num_hidden_layers=5)
    runner = Runner(
        model_config=_FakeModelConfig(rotated=True, fp32_had=True, model=fake_model),
        quantizer=_FakeQuantizer("mixed_gptq"),
    )
    register_calls = []

    monkeypatch.setattr(unfuse_moe_module, "unfuse_moe_experts", lambda model, logger: False)
    monkeypatch.setattr(
        rotation_utils,
        "register_online_hadamard_hooks",
        lambda model, fp32_had=False, layers_cls=None: register_calls.append(
            (model, fp32_had, layers_cls)
        ) or [object()],
    )
    monkeypatch.setattr(Runner, "_patch_k_eq_v_for_vllm", lambda self, model, quant_config: None)

    model, _tokenizer = runner.create_quantized_model()

    assert model.config.quantization_config["quant_method"] == "mixed_gptq"
    assert model.config.quantization_config["rotated"] is True
    assert model.config.quantization_config["fp32_had"] is True
    assert len(register_calls) == 1
