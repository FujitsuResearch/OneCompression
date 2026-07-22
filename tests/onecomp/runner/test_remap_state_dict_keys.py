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
    """Return the tensor fields stored directly under ``module_name.`` in
    *state_dict*, keyed by field name (e.g. ``"qweight"``)."""
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


def test_resolve_state_dict_key_returns_none_for_quantized_buffer_not_yet_in_model():
    """GPTQ buffers (qweight/scales/...) are not resolved here.

    Before _replace_quantized_layers() runs, the model still has a plain
    nn.Linear (``.weight``), so no candidate for a ``.qweight`` checkpoint
    key exists in model_keys yet. Quantized-buffer resolution is handled
    later, in _replace_quantized_layers(), not by this generic key
    resolver.
    """
    from onecomp.quantized_model_loader import QuantizedModelLoader

    # Empty model still has Linear.weight, not qweight/scales.
    model_keys = {"model.language_model.layers.0.self_attn.q_proj.weight"}
    ckpt = "model.language_model.model.layers.0.self_attn.q_proj.qweight"
    assert QuantizedModelLoader._resolve_state_dict_key(ckpt, model_keys) is None


def test_remap_state_dict_keys_leaves_quantized_tensors_for_layer_swap():
    """_remap_state_dict_keys() alone must not touch quantized-buffer keys.

    Their source-prefix -> target-prefix move is deferred to
    _replace_quantized_layers(), which has the quantized module names from
    quant_config needed to resolve them correctly (see
    test_remap_replace_and_load_quantized_layer_pipeline for the full
    pipeline).
    """
    from onecomp.quantized_model_loader import QuantizedModelLoader

    model = _FakeGemma3LikeModel()
    qweight = torch.arange(4, dtype=torch.int32)
    ckpt = {
        "model.language_model.model.layers.0.self_attn.q_proj.qweight": qweight,
        "model.language_model.model.layers.0.self_attn.q_proj.scales": qweight.float(),
    }
    remapped = QuantizedModelLoader._remap_state_dict_keys(ckpt, model)
    assert "model.language_model.model.layers.0.self_attn.q_proj.qweight" in remapped
    assert torch.equal(remapped["model.language_model.model.layers.0.self_attn.q_proj.qweight"], qweight)


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
    _write_gemma3_vlm_save_dir(
        save_dir,
        {
            ckpt_key: tensor,
            # Required so the post-load critical-key check
            # (_check_load_state_dict_result) does not flag lm_head as missing.
            "lm_head.weight": torch.ones(8, 4),
        },
    )

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
            # _replace_quantized_layers returns the (possibly materialized)
            # state_dict; the caller reassigns it.
            lambda model, state_dict, quant_config: state_dict,
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


def test_find_layer_state_resolves_quantized_tensors_by_suffix():
    """_replace_quantized_layers locates quantized tensors via _find_layer_state.

    _remap_state_dict_keys() leaves quantized-buffer keys under their
    original (checkpoint) prefix (see
    test_remap_state_dict_keys_leaves_quantized_tensors_for_layer_swap).
    _find_layer_state() is the mechanism that then locates them for a
    given *target* module name by layer-suffix match, even though no key
    with that exact prefix exists yet.
    """
    from onecomp.quantized_model_loader import QuantizedModelLoader

    module_name = "model.language_model.layers.0.self_attn.q_proj"
    saved_module_name = "model.language_model.model.layers.0.self_attn.q_proj"
    model = _FakeGemma3VLMWithAttn()
    ckpt = _make_tiny_gptq_state_dict(128, 128, saved_module_name=saved_module_name)
    remapped = QuantizedModelLoader._remap_state_dict_keys(ckpt, model)

    # No exact-prefix match yet: the checkpoint still uses the
    # unremapped saved_module_name prefix at this point.
    assert not _layer_sd_by_prefix(remapped, module_name)

    sd_prefix_map = QuantizedModelLoader._build_state_dict_prefix_map(remapped)
    layer_sd, source_prefix = QuantizedModelLoader._find_layer_state(
        module_name, remapped, sd_prefix_map
    )
    assert source_prefix == saved_module_name
    assert layer_sd.keys() >= {"qweight", "scales", "qzeros"}


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


def test_resolve_state_dict_key_suffix_fallback_not_capped_at_eight_components():
    """The generic suffix fallback must consider the full key depth.

    For a 9+ component key whose last 8 components are ambiguous (shared
    by two model keys), only the full, un-truncated key disambiguates the
    match. A depth cap on the fallback would make it return None instead
    of resolving correctly.
    """
    from onecomp.quantized_model_loader import QuantizedModelLoader

    ckpt = "zzz.aaa.bbb.ccc.blocks.0.attn.proj.weight"  # 9 components
    model_keys = {
        # Shares the same last-8-component suffix as `ckpt` ("aaa.bbb.ccc.blocks.0.attn.proj.weight"),
        # so any fallback limited to <=8 trailing components is ambiguous between these two.
        "zzz.aaa.bbb.ccc.blocks.0.attn.proj.weight",
        "yyy.aaa.bbb.ccc.blocks.0.attn.proj.weight",
    }
    assert (
        QuantizedModelLoader._resolve_state_dict_key(ckpt, model_keys)
        == "zzz.aaa.bbb.ccc.blocks.0.attn.proj.weight"
    )


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


# ---------------------------------------------------------------------------
# _flatten_module_names
# ---------------------------------------------------------------------------


def test_flatten_module_names_passthrough_for_flat_list():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    names = ["model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj"]
    assert QuantizedModelLoader._flatten_module_names(names) == names


def test_flatten_module_names_flattens_nested_lists():
    """Some quant configs group module names per-block, e.g. mixed_gptq."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    nested = [
        ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"],
        ["model.layers.0.mlp.up_proj"],
    ]
    assert QuantizedModelLoader._flatten_module_names(nested) == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.mlp.up_proj",
    ]


# ---------------------------------------------------------------------------
# _resolve_module_name
# ---------------------------------------------------------------------------


def test_resolve_module_name_exact_match():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    name_to_module = {"model.layers.0.mlp.up_proj": nn.Linear(4, 4)}
    assert (
        QuantizedModelLoader._resolve_module_name("model.layers.0.mlp.up_proj", name_to_module)
        == "model.layers.0.mlp.up_proj"
    )


def test_resolve_module_name_no_match_returns_none():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    name_to_module = {"model.layers.0.mlp.up_proj": nn.Linear(4, 4)}
    assert (
        QuantizedModelLoader._resolve_module_name("model.layers.5.mlp.up_proj", name_to_module)
        is None
    )


def test_resolve_module_name_unique_suffix_match_across_prefix_drift():
    """config uses model.layers.*, actual model uses model.language_model.layers.*."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    name_to_module = {"model.language_model.layers.0.mlp.up_proj": nn.Linear(4, 4)}
    assert (
        QuantizedModelLoader._resolve_module_name("model.layers.0.mlp.up_proj", name_to_module)
        == "model.language_model.layers.0.mlp.up_proj"
    )


def test_resolve_module_name_ambiguous_prefers_language_model_hit():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    name_to_module = {
        "model.language_model.layers.0.mlp.up_proj": nn.Linear(4, 4),
        "model.vision.layers.0.mlp.up_proj": nn.Linear(4, 4),
    }
    assert (
        QuantizedModelLoader._resolve_module_name("model.layers.0.mlp.up_proj", name_to_module)
        == "model.language_model.layers.0.mlp.up_proj"
    )


def test_resolve_module_name_ambiguous_without_unique_language_model_hit_raises():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    name_to_module = {
        "model.encoder_a.layers.0.mlp.up_proj": nn.Linear(4, 4),
        "model.encoder_b.layers.0.mlp.up_proj": nn.Linear(4, 4),
    }
    with pytest.raises(RuntimeError, match="Ambiguous module suffix match"):
        QuantizedModelLoader._resolve_module_name("model.layers.0.mlp.up_proj", name_to_module)


# ---------------------------------------------------------------------------
# _materialize_layer_state_dict
# ---------------------------------------------------------------------------


def test_materialize_layer_state_dict_moves_tensors_to_target_prefix():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    qweight = torch.arange(4, dtype=torch.int32)
    state_dict = {
        "src.qweight": qweight,
        "src.scales": qweight.float(),
        "unrelated.weight": torch.zeros(2),
    }
    result = QuantizedModelLoader._materialize_layer_state_dict(
        state_dict,
        source_prefix="src",
        target_prefix="dst",
        layer_sd={"qweight": qweight, "scales": qweight.float()},
    )
    assert result is state_dict  # mutated in place
    assert "dst.qweight" in state_dict and "dst.scales" in state_dict
    assert "src.qweight" not in state_dict and "src.scales" not in state_dict
    assert torch.equal(state_dict["dst.qweight"], qweight)
    # Untouched keys outside the layer must survive.
    assert "unrelated.weight" in state_dict


def test_materialize_layer_state_dict_noop_when_prefixes_match():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    qweight = torch.arange(4, dtype=torch.int32)
    state_dict = {"same.qweight": qweight}
    result = QuantizedModelLoader._materialize_layer_state_dict(
        state_dict,
        source_prefix="same",
        target_prefix="same",
        layer_sd={"qweight": qweight},
    )
    assert result is state_dict
    assert state_dict == {"same.qweight": qweight}


def test_materialize_layer_state_dict_raises_on_empty_layer_sd():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    with pytest.raises(RuntimeError, match="No layer state found"):
        QuantizedModelLoader._materialize_layer_state_dict(
            {}, source_prefix="src", target_prefix="dst", layer_sd={}
        )


def test_materialize_layer_state_dict_raises_on_key_collision():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    qweight = torch.arange(4, dtype=torch.int32)
    state_dict = {
        "src.qweight": qweight,
        # Some other layer already occupies the target key.
        "dst.qweight": torch.zeros(4, dtype=torch.int32),
    }
    with pytest.raises(RuntimeError, match="collision"):
        QuantizedModelLoader._materialize_layer_state_dict(
            state_dict,
            source_prefix="src",
            target_prefix="dst",
            layer_sd={"qweight": qweight},
        )


# ---------------------------------------------------------------------------
# _replace_quantized_layers: fails fast when a quantized layer cannot be
# resolved, instead of leaving it randomly initialized.
# ---------------------------------------------------------------------------


def test_replace_quantized_layers_raises_when_module_name_unresolvable():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    model = _FakeLlamaLikeModel()  # only has model.layers.0
    quant_config = _gptq_quant_config("model.layers.5")  # layer 5 does not exist
    with pytest.raises(RuntimeError, match="Failed to replace/load all quantized layers"):
        QuantizedModelLoader._replace_quantized_layers(model, {}, quant_config)


def test_replace_quantized_layers_raises_when_state_unresolvable():
    from onecomp.quantized_model_loader import QuantizedModelLoader

    model = _FakeLlamaLikeModel()  # model.layers.0 exists as an nn.Linear
    quant_config = _gptq_quant_config("model.layers.0")
    # No qweight/scales/qzeros anywhere in the state_dict for this layer.
    state_dict = {"unrelated.weight": torch.zeros(2)}
    with pytest.raises(RuntimeError, match="Failed to replace/load all quantized layers"):
        QuantizedModelLoader._replace_quantized_layers(model, state_dict, quant_config)


# ---------------------------------------------------------------------------
# _check_load_state_dict_result
# ---------------------------------------------------------------------------


def test_check_load_state_dict_result_passes_when_nothing_critical():
    from types import SimpleNamespace

    from onecomp.quantized_model_loader import QuantizedModelLoader

    incompat = SimpleNamespace(missing_keys=["model.some_vlm_only_adapter.weight"], unexpected_keys=[])
    # Should not raise.
    QuantizedModelLoader._check_load_state_dict_result(incompat)


def test_check_load_state_dict_result_raises_on_critical_missing():
    from types import SimpleNamespace

    from onecomp.quantized_model_loader import QuantizedModelLoader

    incompat = SimpleNamespace(
        missing_keys=["model.layers.0.mlp.up_proj.qweight"], unexpected_keys=[]
    )
    with pytest.raises(RuntimeError, match="Critical state_dict mismatch"):
        QuantizedModelLoader._check_load_state_dict_result(incompat)


def test_check_load_state_dict_result_raises_on_critical_unexpected():
    from types import SimpleNamespace

    from onecomp.quantized_model_loader import QuantizedModelLoader

    incompat = SimpleNamespace(missing_keys=[], unexpected_keys=["lm_head.weight"])
    with pytest.raises(RuntimeError, match="Critical state_dict mismatch"):
        QuantizedModelLoader._check_load_state_dict_result(incompat)


def test_check_load_state_dict_result_expected_missing_suppresses_critical_flag():
    """lm_head.weight is legitimately absent from a tied-embedding checkpoint
    (see test_load_tied_embeddings.py); the caller passes it via
    expected_missing when it knows tie_weights() will restore it.
    """
    from types import SimpleNamespace

    from onecomp.quantized_model_loader import QuantizedModelLoader

    incompat = SimpleNamespace(missing_keys=["lm_head.weight"], unexpected_keys=[])
    # Should not raise when lm_head.weight is expected to be missing.
    QuantizedModelLoader._check_load_state_dict_result(
        incompat, expected_missing={"lm_head.weight"}
    )


# ---------------------------------------------------------------------------
# _assert_quantized_modules_loaded
# ---------------------------------------------------------------------------


def test_assert_quantized_modules_loaded_passes_for_valid_gptq_layer():
    from onecomp.quantized_model_loader import QuantizedModelLoader
    from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

    layer_sd = _make_tiny_gptq_state_dict(128, 128, saved_module_name="x")
    layer_sd = {k[len("x.") :]: v for k, v in layer_sd.items()}  # strip prefix -> field names
    layer = GPTQLinear.from_saved_state(
        layer_sd, in_features=128, out_features=128, wbits=4, groupsize=128, empty=False
    )
    model = nn.Module()
    model.q_proj = layer
    QuantizedModelLoader._assert_quantized_modules_loaded(model)  # should not raise


def test_assert_quantized_modules_loaded_raises_for_all_zero_gptq_buffer():
    """An all-zero buffer means the layer was replaced but never filled
    with real weights (e.g. a checkpoint/config key mismatch); this must
    be caught rather than silently producing a model with garbage
    output."""
    from onecomp.quantized_model_loader import QuantizedModelLoader
    from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

    layer_sd = _make_tiny_gptq_state_dict(128, 128, saved_module_name="x")
    layer_sd = {k[len("x.") :]: v for k, v in layer_sd.items()}
    # empty=True zeroes all buffers, simulating a layer whose weights never loaded.
    layer = GPTQLinear.from_saved_state(
        layer_sd, in_features=128, out_features=128, wbits=4, groupsize=128, empty=True
    )
    model = nn.Module()
    model.q_proj = layer
    with pytest.raises(RuntimeError, match="Invalid quantized module buffers"):
        QuantizedModelLoader._assert_quantized_modules_loaded(model)


def test_assert_quantized_modules_loaded_passes_for_valid_dbf_layer():
    """DoubleBinaryLinear's real attributes are scaling0/scaling2/scaling4
    and bp1/bp3 (see dbf_layer.py); there is no plain ``bp`` attribute. A
    normally-constructed DBF layer must pass this check.
    """
    from onecomp.quantized_model_loader import QuantizedModelLoader
    from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear

    layer = DoubleBinaryLinear(
        dbf_Da=torch.ones(4),
        dbf_A=torch.ones(4, 4),
        dbf_mid=torch.ones(4),
        dbf_B=torch.ones(4, 4),
        dbf_Db=torch.ones(4),
        use_gemlite=False,
    )
    model = nn.Module()
    model.layer = layer
    QuantizedModelLoader._assert_quantized_modules_loaded(model)  # should not raise


def test_assert_quantized_modules_loaded_raises_for_all_zero_dbf_buffer():
    from onecomp.quantized_model_loader import QuantizedModelLoader
    from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear

    layer = DoubleBinaryLinear(
        dbf_Da=torch.ones(4),
        dbf_A=torch.ones(4, 4),
        dbf_mid=torch.ones(4),
        dbf_B=torch.ones(4, 4),
        dbf_Db=torch.ones(4),
        use_gemlite=False,
    )
    with torch.no_grad():
        layer.bp1.zero_()
        layer.bp3.zero_()
    model = nn.Module()
    model.layer = layer
    with pytest.raises(RuntimeError, match="Invalid quantized module buffers"):
        QuantizedModelLoader._assert_quantized_modules_loaded(model)
