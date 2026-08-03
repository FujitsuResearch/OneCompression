"""Regression tests for loading MDBF-quantized models from safetensors.

MDBF is the only quantizer whose per-layer tensors are *nested*: a
``MultipathMDBFLinear`` keeps one ``MDBFLinear`` submodule per pass, so its
checkpoint keys look like ``<layer>.paths.{p}.A_amp`` rather than the flat
``<layer>.qweight`` (GPTQ) / ``<layer>.bp1`` (DBF) layout.  Both the tensor
collection that precedes ``from_saved_state`` and the post-load validation
have to cope with that nesting.  Tests use tiny CPU-only
``LlamaForCausalLM`` instances so they do not depend on CUDA, network
access, or downloaded weights.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from onecomp.quantized_model_loader import QuantizedModelLoader
from onecomp.quantizer.mdbf.config import resolve_mdbf_paths
from onecomp.quantizer.mdbf.initialize import MDBFParams
from onecomp.quantizer.mdbf.mdbf_layer import MDBFLinear, MultipathMDBFLinear

# Layers replaced by MDBF in the tiny test model (one attention, one MLP
# projection) - enough to cover both square and rectangular weight shapes.
TARGET_SUFFIXES = ("self_attn.q_proj", "mlp.down_proj")
MDBF_RANK = 8
MDBF_PATHS = 2


def _make_params(n: int, m: int, r: int, l: int, seed: int) -> MDBFParams:
    """Build deterministic, non-degenerate MDBF parameters for one path.

    Args:
        n: Output features.
        m: Input features.
        r: Decomposition rank.
        l: Multi-scale amplitude rank.
        seed: RNG seed making the tensors reproducible across runs.

    Returns:
        MDBFParams with +-1 sign matrices and strictly positive amplitudes.
    """
    g = torch.Generator().manual_seed(seed)

    def _sign(*shape: int) -> torch.Tensor:
        return torch.where(torch.randn(*shape, generator=g) > 0, 1.0, -1.0)

    def _amp(*shape: int) -> torch.Tensor:
        # Offset away from 0 so an "all-zero buffer" check cannot pass by luck.
        return torch.rand(*shape, generator=g) + 0.5

    return MDBFParams(
        A_sign=_sign(n, r),
        B_sign=_sign(r, m),
        A_amp=_amp(n, l),
        B_amp=_amp(m, l),
        Q_U_amp=_amp(r, l),
        Q_V_amp=_amp(r, l),
    )


def _build_mdbf_model(*, with_bias: bool) -> tuple[torch.nn.Module, Any, list[str]]:
    """Build a tiny Llama whose target linears are MultipathMDBFLinear.

    Args:
        with_bias: Whether the replaced linears carry a bias buffer.

    Returns:
        (model, config, quantized_layer_names)
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        vocab_size=32,
        tie_word_embeddings=False,
        attention_bias=with_bias,
        mlp_bias=with_bias,
    )
    config.torch_dtype = torch.float16
    model = LlamaForCausalLM(config).to(torch.float16).eval()

    name_to_module = dict(model.named_modules())
    quantized_names: list[str] = []
    for layer_idx in range(config.num_hidden_layers):
        for suffix in TARGET_SUFFIXES:
            name = f"model.layers.{layer_idx}.{suffix}"
            quantized_names.append(name)
            parent_name, _, child_name = name.rpartition(".")
            parent = name_to_module[parent_name]
            linear = getattr(parent, child_name)
            bias = linear.bias.detach().clone() if linear.bias is not None else None
            params_list = [
                _make_params(
                    linear.out_features,
                    linear.in_features,
                    MDBF_RANK,
                    1,
                    seed=1000 * layer_idx + 7 * p + len(suffix),
                )
                for p in range(MDBF_PATHS)
            ]
            setattr(
                parent,
                child_name,
                MultipathMDBFLinear(params_list, bias=bias, use_gemlite=False),
            )

    return model, config, quantized_names


def _write_save_dir(
    save_dir: Path,
    config: Any,
    state_dict: dict,
    quantized_names: list[str],
    *,
    record_paths: bool = True,
) -> None:
    """Persist an MDBF checkpoint the loader can consume.

    Args:
        save_dir: Directory to write config.json and model.safetensors into.
        config: The model's ``PretrainedConfig``.
        state_dict: Tensors to save.
        quantized_names: Layers recorded as MDBF-quantized.
        record_paths: Whether to record ``P`` the way the quantizer does.
            Set False to emulate a hand-written or partial config that omits it.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = config.to_dict()
    cfg_dict["torch_dtype"] = "float16"
    cfg_dict["quantization_config"] = {
        "quant_method": "mdbf",
        "bits": 2.0,
        "l": 1,
        "modules_in_block_to_quantize": quantized_names,
    }
    if record_paths:
        cfg_dict["quantization_config"]["P"] = MDBF_PATHS
    (save_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
    save_file(
        {k: v.contiguous() for k, v in state_dict.items()},
        str(save_dir / "model.safetensors"),
    )


def _load(save_dir: Path) -> tuple[torch.nn.Module, Any]:
    """Call ``load_quantized_model`` with the tokenizer load patched out."""
    with patch(
        "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained",
        return_value=object(),
    ):
        return QuantizedModelLoader.load_quantized_model(
            str(save_dir),
            device_map="",
            local_files_only=True,
        )


@pytest.mark.parametrize("with_bias", [False, True])
def test_mdbf_checkpoint_round_trips_through_loader(tmp_path: Path, with_bias: bool) -> None:
    """A saved MDBF model reloads with bit-identical buffers and outputs.

    Without bias the layer prefix owns only nested ``paths.{p}.*`` keys;
    with bias it owns a direct key as well.  Both must resolve to the full
    nested tensor set, otherwise ``from_saved_state`` rebuilds an empty layer
    (or the loader fails outright).
    """
    reference, config, quantized_names = _build_mdbf_model(with_bias=with_bias)
    save_dir = tmp_path / "mdbf_model"
    _write_save_dir(save_dir, config, reference.state_dict(), quantized_names)

    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        expected_logits = reference(input_ids).logits.float()

    model, _ = _load(save_dir)

    loaded_modules = dict(model.named_modules())
    for name in quantized_names:
        module = loaded_modules[name]
        assert isinstance(module, MultipathMDBFLinear)
        assert len(module.paths) == MDBF_PATHS
        assert (module.bias is not None) == with_bias

    # assign=True must have installed the checkpoint tensors themselves, so
    # every amplitude/sign buffer matches the saved model bit for bit.  This is
    # the strict half of the round-trip: a layer left at its empty-model zeros
    # fails here even if the forward pass still produces plausible numbers.
    saved = reference.state_dict()
    loaded = model.state_dict()
    assert set(loaded) == set(saved)
    for key, tensor in loaded.items():
        assert torch.equal(tensor, saved[key]), f"buffer mismatch for {key}"

    # Logits are compared with a tolerance rather than bit-exactly: identical
    # fp16 buffers can still reduce in a different order once the tensors come
    # back with a different memory layout, which costs 1-2 ulp.
    with torch.no_grad():
        actual_logits = model(input_ids).logits.float()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0.0, atol=1e-3)


def test_load_rejects_checkpoint_missing_a_whole_path(tmp_path: Path) -> None:
    """Dropping the last path must fail rather than load a truncated layer.

    ``from_saved_state`` derives P from the path indices it is given, so a
    checkpoint that lost ``paths.{P-1}.*`` rebuilds as a valid-looking layer
    with fewer passes - every remaining buffer is correctly populated, so no
    post-load buffer check can notice.  Only the config's recorded P can.
    """
    reference, config, quantized_names = _build_mdbf_model(with_bias=False)
    victim = quantized_names[0]
    state_dict = {
        key: tensor
        for key, tensor in reference.state_dict().items()
        if not key.startswith(f"{victim}.paths.{MDBF_PATHS - 1}.")
    }
    save_dir = tmp_path / "mdbf_model"
    _write_save_dir(save_dir, config, state_dict, quantized_names)

    with pytest.raises(ValueError, match="Incomplete MDBF checkpoint"):
        _load(save_dir)


def test_validate_saved_state_rejects_out_of_range_path_index() -> None:
    """A right-sized but wrongly-indexed path set is still incomplete.

    ``paths.0`` + ``paths.5`` with P=2 has the expected count, so the count
    alone cannot stand in for the full comparison.
    """
    layer_sd = {"paths.0.A_amp": torch.zeros(1), "paths.5.A_amp": torch.zeros(1)}

    with pytest.raises(ValueError, match="Incomplete MDBF checkpoint"):
        MultipathMDBFLinear.validate_saved_state(
            layer_sd,
            layer_name="model.layers.0.mlp.down_proj",
            expected_paths=2,
            expects_bias=False,
        )


def test_load_accepts_complete_checkpoint_when_config_omits_p(tmp_path: Path) -> None:
    """A config that records no P still loads a complete checkpoint.

    ``P`` has been written by ``MDBF.get_quant_config`` since MDBF landed, so
    this only covers hand-written or partial configs: they keep working, they
    just lose the path-count check.  What happens to an *incomplete*
    checkpoint in that case is deliberately not pinned here - the skip is a
    back-compat concession, not a promise to accept damaged tensors.
    """
    reference, config, quantized_names = _build_mdbf_model(with_bias=False)
    save_dir = tmp_path / "mdbf_model"
    _write_save_dir(save_dir, config, reference.state_dict(), quantized_names, record_paths=False)

    model, _ = _load(save_dir)

    loaded_modules = dict(model.named_modules())
    for name in quantized_names:
        assert len(loaded_modules[name].paths) == MDBF_PATHS


def test_load_rejects_checkpoint_missing_bias(tmp_path: Path) -> None:
    """Dropping the bias must fail rather than load a bias-less layer.

    A missing ``bias`` key is indistinguishable from "this layer has no
    bias" to ``from_saved_state``; the model's own ``nn.Linear`` is the only
    source of truth for which one it is.
    """
    reference, config, quantized_names = _build_mdbf_model(with_bias=True)
    victim = quantized_names[0]
    state_dict = {
        key: tensor for key, tensor in reference.state_dict().items() if key != f"{victim}.bias"
    }
    save_dir = tmp_path / "mdbf_model"
    _write_save_dir(save_dir, config, state_dict, quantized_names)

    with pytest.raises(ValueError, match="bias mismatch"):
        _load(save_dir)


def test_load_rejects_unexpected_bias_in_checkpoint(tmp_path: Path) -> None:
    """A bias the model has no place for is a mismatch too, not a silent drop."""
    reference, config, quantized_names = _build_mdbf_model(with_bias=False)
    victim = quantized_names[0]
    state_dict = dict(reference.state_dict())
    state_dict[f"{victim}.bias"] = torch.zeros(
        dict(reference.named_modules())[victim].n, dtype=torch.float16
    )
    save_dir = tmp_path / "mdbf_model"
    _write_save_dir(save_dir, config, state_dict, quantized_names)

    with pytest.raises(ValueError, match="bias mismatch"):
        _load(save_dir)


def test_resolve_mdbf_paths_returns_none_when_unrecorded() -> None:
    """A config with no P yields None so the caller skips validation.

    An explicit null is treated the same as an absent key: both mean the
    config records nothing to validate against.
    """
    assert resolve_mdbf_paths({"bits": 2.0}) is None
    assert resolve_mdbf_paths({"bits": 2.0, "P": None}) is None


# 2.0 covers non-int numerics, True the bool-is-an-int trap, 0 the range check.
@pytest.mark.parametrize("bad", [0, 2.0, True])
def test_resolve_mdbf_paths_rejects_invalid_values(bad: Any) -> None:
    """P must be a positive integer; anything else is a corrupt config."""
    with pytest.raises(ValueError, match="must be an integer > 0"):
        resolve_mdbf_paths({"P": bad})


def test_build_state_dict_prefix_map_registers_nested_prefixes() -> None:
    """Nested MDBF keys are reachable from the quantized layer prefix."""
    layer = "model.layers.0.mlp.down_proj"
    state_dict = {
        f"{layer}.paths.0.A_amp": torch.zeros(1),
        f"{layer}.paths.1.A_amp": torch.zeros(1),
        f"{layer}.bias": torch.zeros(1),
    }

    prefix_map = QuantizedModelLoader._build_state_dict_prefix_map(state_dict)

    assert sorted(prefix_map[layer]) == sorted(state_dict)
    assert prefix_map[f"{layer}.paths.0"] == [f"{layer}.paths.0.A_amp"]

    layer_sd, source_prefix = QuantizedModelLoader._find_layer_state(layer, state_dict, prefix_map)
    assert source_prefix == layer
    assert sorted(layer_sd) == ["bias", "paths.0.A_amp", "paths.1.A_amp"]


@pytest.mark.parametrize(
    "field",
    ["A_sign_packed", "B_sign_packed", "A_amp", "B_amp", "Q_U_amp", "Q_V_amp"],
)
def test_check_load_state_dict_result_treats_mdbf_buffers_as_critical(field: str) -> None:
    """A missing MDBF tensor must fail the load instead of warning."""
    incompat = SimpleNamespace(
        missing_keys=[f"model.layers.0.mlp.down_proj.paths.0.{field}"],
        unexpected_keys=[],
    )

    with pytest.raises(RuntimeError, match="Critical state_dict mismatch"):
        QuantizedModelLoader._check_load_state_dict_result(incompat)


def _wrap_in_module(layer: torch.nn.Module) -> torch.nn.Module:
    """Return a container module holding *layer*, for named_modules() walks."""
    container = torch.nn.Module()
    container.down_proj = layer
    return container


def _saved_layer_state() -> dict:
    """Build the per-layer state_dict of a small MultipathMDBFLinear."""
    params_list = [_make_params(6, 4, 3, 1, seed=10 + p) for p in range(MDBF_PATHS)]
    return MultipathMDBFLinear(params_list, use_gemlite=False).state_dict()


def test_assert_quantized_modules_loaded_rejects_zeroed_mdbf() -> None:
    """An MDBF layer left at its empty-model zeros is reported as invalid.

    ``empty=True`` is exactly the state a layer stays in when the checkpoint
    tensors never reach it, which is the failure this check exists to catch.
    """
    layer = MultipathMDBFLinear.from_saved_state(
        _saved_layer_state(),
        in_features=4,
        out_features=6,
        empty=True,
    )

    with pytest.raises(RuntimeError, match="all_zero"):
        QuantizedModelLoader._assert_quantized_modules_loaded(_wrap_in_module(layer))


def test_assert_quantized_modules_loaded_rejects_mdbf_missing_buffer() -> None:
    """A path missing one of its buffers is reported rather than ignored."""
    layer = MultipathMDBFLinear.from_saved_state(
        _saved_layer_state(),
        in_features=4,
        out_features=6,
        empty=False,
    )
    path: MDBFLinear = layer.paths[0]
    del path._buffers["A_sign_packed"]

    with pytest.raises(RuntimeError, match="missing"):
        QuantizedModelLoader._assert_quantized_modules_loaded(_wrap_in_module(layer))


def test_resolve_mdbf_layer_bits_uses_saved_layer_name(tmp_path: Path) -> None:
    """Per-layer config lookups use the checkpoint's name, not the model's.

    The loader iterates checkpoint-side names and resolves each to a module
    path, so the two can differ.  Here the checkpoint stores the quantized
    layers under ``model.decoder.layers.*`` while the from_config model
    exposes ``model.layers.*``; ``module_target_bits`` is keyed by the
    checkpoint name only, so the override lands solely if that name is what
    reaches ``resolve_mdbf_layer_bits`` - the sibling GPTQ/DBF branches pass
    the same one.
    """
    reference, config, model_names = _build_mdbf_model(with_bias=False)
    saved_names = {
        name: name.replace("model.layers.", "model.decoder.layers.") for name in model_names
    }

    state_dict = {}
    for key, tensor in reference.state_dict().items():
        for model_name, saved_name in saved_names.items():
            if key.startswith(f"{model_name}."):
                key = saved_name + key[len(model_name) :]
                break
        state_dict[key] = tensor

    save_dir = tmp_path / "mdbf_model"
    _write_save_dir(save_dir, config, state_dict, sorted(saved_names.values()))

    cfg_path = save_dir / "config.json"
    cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_dict["quantization_config"]["module_target_bits"] = {
        saved_names["model.layers.0.self_attn.q_proj"]: 3.0,
    }
    cfg_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    model, _ = _load(save_dir)

    loaded_modules = dict(model.named_modules())
    assert isinstance(loaded_modules["model.layers.0.self_attn.q_proj"], MultipathMDBFLinear)
    assert loaded_modules["model.layers.0.self_attn.q_proj"].target_bits == 3.0
    assert loaded_modules["model.layers.1.self_attn.q_proj"].target_bits == 2.0
    assert loaded_modules["model.layers.0.mlp.down_proj"].target_bits == 2.0
