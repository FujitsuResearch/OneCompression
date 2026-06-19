"""Tests for VLM state_dict key remapping in QuantizedModelLoader.

Gemma3 VLMs saved via save_pretrained store language-model weights under
model.language_model.model.layers. while from_config builds
model.language_model.layers.*.  Without remapping, load_state_dict
silently skips those tensors.

Copyright 2025-2026 Fujitsu Ltd.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file


class _FakeGemma3LikeModel(nn.Module):
    """Minimal stand-in for a Gemma3 VLM text stack (from_config layout)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        self.model.language_model.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)


class _FakeGemma3VLMWithAttn(nn.Module):
    """Gemma3-like VLM with a quantizable q_proj under language_model."""

    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.model.language_model.layers = nn.ModuleList([layer])


class _FakeLlamaLikeModel(nn.Module):
    """Plain CausalLM layout (model.layers.*) without VLM wrappers."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        self.lm_head = nn.Linear(4, 8, bias=False)


def _layer_sd_by_prefix(state_dict: dict, module_name: str) -> dict:
    """Mirror the direct-prefix branch of _replace_quantized_layers._get_layer_sd."""
    prefix = module_name + "."
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _make_tiny_gptq_state_dict(
    in_features: int,
    out_features: int,
    *,
    saved_module_name: str,
) -> dict:
    """Build a minimal GPTQ checkpoint entry under *saved_module_name*."""
    num_groups = max(in_features // 128, 1)
    qweight = torch.ones(out_features, in_features // 8, dtype=torch.int32)
    scales = torch.ones(num_groups, out_features, dtype=torch.float16)
    qzeros = torch.zeros(num_groups, out_features // 8, dtype=torch.int32)
    return {
        f"{saved_module_name}.qweight": qweight,
        f"{saved_module_name}.scales": scales,
        f"{saved_module_name}.qzeros": qzeros,
    }


def _gptq_quant_config(module_name: str) -> dict:
    return {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "groupsize": 128,
        "desc_act": False,
        "actorder": False,
        "checkpoint_format": "gptq",
        "modules_in_block_to_quantize": [module_name],
    }


def _write_gemma3_vlm_save_dir(save_dir: Path, state_dict: dict) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": "gemma3",
        "text_config": {"model_type": "gemma3_text", "vocab_size": 8, "hidden_size": 4},
        "vision_config": {"model_type": "siglip_vision_model", "hidden_size": 4},
        "quantization_config": {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "modules_in_block_to_quantize": [
                "model.language_model.layers.0.weight",
            ],
        },
    }
    (save_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file(state_dict, str(save_dir / "model.safetensors"))


def test_resolve_state_dict_key_gemma3_language_model_wrapper():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    model_keys = {
        "model.language_model.layers.0.self_attn.q_proj.qweight",
        "model.language_model.embed_tokens.weight",
    }
    ckpt = "model.language_model.model.layers.0.self_attn.q_proj.qweight"
    assert (
        QuantizedModelLoader._resolve_state_dict_key(ckpt, model_keys)
        == "model.language_model.layers.0.self_attn.q_proj.qweight"
    )
    ckpt_embed = "model.language_model.model.embed_tokens.weight"
    assert (
        QuantizedModelLoader._resolve_state_dict_key(ckpt_embed, model_keys)
        == "model.language_model.embed_tokens.weight"
    )


def test_resolve_state_dict_key_quantized_tensor_without_model_key():
    """GPTQ buffers are remapped before nn.Linear is replaced."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    # Empty model still has Linear.weight, not qweight/scales.
    model_keys = {"model.language_model.layers.0.self_attn.q_proj.weight"}
    ckpt = "model.language_model.model.layers.0.self_attn.q_proj.qweight"
    assert (
        QuantizedModelLoader._resolve_state_dict_key(ckpt, model_keys)
        == "model.language_model.layers.0.self_attn.q_proj.qweight"
    )


def test_remap_state_dict_keys_rewrites_quantized_tensors_before_layer_swap():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    model = _FakeGemma3LikeModel()
    qweight = torch.arange(4, dtype=torch.int32)
    ckpt = {
        "model.language_model.model.layers.0.self_attn.q_proj.qweight": qweight,
        "model.language_model.model.layers.0.self_attn.q_proj.scales": qweight.float(),
    }
    remapped = QuantizedModelLoader._remap_state_dict_keys(ckpt, model)
    assert "model.language_model.layers.0.self_attn.q_proj.qweight" in remapped
    assert "model.language_model.model.layers.0.self_attn.q_proj.qweight" not in remapped
    assert torch.equal(remapped["model.language_model.layers.0.self_attn.q_proj.qweight"], qweight)


def test_resolve_state_dict_key_without_model_prefix():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    model_keys = {"model.language_model.layers.0.input_layernorm.weight"}
    ckpt = "language_model.model.layers.0.input_layernorm.weight"
    assert (
        QuantizedModelLoader._resolve_state_dict_key(ckpt, model_keys)
        == "model.language_model.layers.0.input_layernorm.weight"
    )


def test_remap_state_dict_keys_noop_when_already_aligned():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    model = _FakeGemma3LikeModel()
    state_dict = dict(model.named_parameters())
    remapped = QuantizedModelLoader._remap_state_dict_keys(state_dict, model)
    assert remapped is state_dict or remapped == state_dict


def test_remap_state_dict_keys_rewrites_gemma3_checkpoint(tmp_path):
    from onecomp.quantized_model_loader import QuantizedModelLoader

    model = _FakeGemma3LikeModel()
    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    ckpt = {
        "model.language_model.model.layers.0.weight": weight,
        "model.language_model.model.embed_tokens.weight": torch.ones(8, 4),
    }
    remapped = QuantizedModelLoader._remap_state_dict_keys(ckpt, model)
    assert "model.language_model.layers.0.weight" in remapped
    assert "model.language_model.model.layers.0.weight" not in remapped
    assert torch.equal(remapped["model.language_model.layers.0.weight"], weight)


def test_load_quantized_model_applies_remap_before_state_dict(tmp_path):
    """End-to-end: remapped keys reach load_state_dict."""
    from types import SimpleNamespace

    from onecomp.quantized_model_loader import QuantizedModelLoader

    ckpt_key = "model.language_model.model.embed_tokens.weight"
    model_key = "model.language_model.embed_tokens.weight"
    save_dir = tmp_path / "saved"
    tensor = torch.ones(8, 4)
    _write_gemma3_vlm_save_dir(save_dir, {ckpt_key: tensor})

    captured: dict = {}

    class _RecordingModel(_FakeGemma3LikeModel):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False)

        def load_state_dict(self, state_dict, *args, **kwargs):
            captured.update(state_dict)
            return super().load_state_dict(state_dict, *args, **kwargs)

    with (
        patch.object(
            QuantizedModelLoader,
            "_build_empty_model_from_config",
            return_value=_RecordingModel(),
        ),
        patch(
            "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained",
            return_value=object(),
        ),
        patch.object(
            QuantizedModelLoader,
            "_replace_quantized_layers",
            lambda *a, **k: None,
        ),
        patch.object(
            QuantizedModelLoader,
            "_cast_fp16_to_target_dtype",
            return_value=[],
        ),
    ):
        QuantizedModelLoader.load_quantized_model(str(save_dir), device_map="")

    assert model_key in captured
    assert torch.equal(captured[model_key], tensor)


def test_remap_enables_direct_layer_sd_prefix_lookup():
    """After remap, _get_layer_sd hits q_proj without suffix fallback."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    module_name = "model.language_model.layers.0.self_attn.q_proj"
    saved_module_name = "model.language_model.model.layers.0.self_attn.q_proj"
    model = _FakeGemma3VLMWithAttn()
    ckpt = _make_tiny_gptq_state_dict(128, 128, saved_module_name=saved_module_name)
    remapped = QuantizedModelLoader._remap_state_dict_keys(ckpt, model)

    layer_sd = _layer_sd_by_prefix(remapped, module_name)
    assert layer_sd.keys() >= {"qweight", "scales", "qzeros"}
    assert not any(key.startswith("model.language_model.model.") for key in remapped)


def test_remap_replace_and_load_quantized_layer_pipeline():
    """remap → _replace_quantized_layers → load_state_dict fills GPTQ buffers."""
    from onecomp.quantized_model_loader import QuantizedModelLoader
    from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

    module_name = "model.language_model.layers.0.self_attn.q_proj"
    saved_module_name = "model.language_model.model.layers.0.self_attn.q_proj"
    model = _FakeGemma3VLMWithAttn()
    ckpt = _make_tiny_gptq_state_dict(128, 128, saved_module_name=saved_module_name)
    remapped = QuantizedModelLoader._remap_state_dict_keys(ckpt, model)
    quant_config = _gptq_quant_config(module_name)

    QuantizedModelLoader._replace_quantized_layers(model, remapped, quant_config)
    model.load_state_dict(remapped, strict=False, assign=True)

    q_proj = model.model.language_model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, GPTQLinear)
    assert q_proj.qweight.sum().item() != 0


def test_remap_state_dict_keys_noop_for_plain_llm_checkpoint():
    """Standard model.layers.* checkpoints must not be rewritten."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    model = _FakeLlamaLikeModel()
    ckpt = {
        "model.layers.0.self_attn.q_proj.qweight": torch.ones(4, 1, dtype=torch.int32),
        "model.layers.0.self_attn.q_proj.scales": torch.ones(1, 4, dtype=torch.float16),
        "model.embed_tokens.weight": torch.randn(8, 4),
    }
    remapped = QuantizedModelLoader._remap_state_dict_keys(ckpt, model)
    assert set(remapped.keys()) == set(ckpt.keys())
    for key in ckpt:
        assert torch.equal(remapped[key], ckpt[key])
