"""Regression tests for issue #64 Issue 3 (non-quantized dtype handling).

Verifies that ``QuantizedModelLoader.load_quantized_model`` keeps every
non-quantized parameter of the loaded model in the dtype recorded in
``config.json`` (i.e. ``model.config.torch_dtype``), regardless of
whether ``load_state_dict(..., assign=True)`` could find the parameter's
key in the safetensors checkpoint.

Two cooperating mechanisms guarantee this:

* ``_build_empty_model_from_config`` honours the ``torch_dtype`` /
  ``dtype`` field of the saved config when no explicit ``torch_dtype`` is
  passed by the caller, so the empty model starts in the same dtype as
  the checkpoint.
* ``_cast_fp16_to_target_dtype`` is invoked at the end of the load and
  normalises any leftover ``float16`` params/buffers of non-quantized
  modules.  Quantized layers (``GPTQLinear``, ``DoubleBinaryLinear``)
  and ``float32`` params (e.g. fp32 LayerNorm in mixed-precision
  models) are deliberately preserved.

Tests use tiny CPU-only ``LlamaForCausalLM`` instances so they do not
depend on CUDA, network access, or downloaded weights.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file


def _build_tiny_llama(dtype: torch.dtype, *, tie_word_embeddings: bool = False):
    """Construct a tiny ``LlamaForCausalLM`` on CPU with the given dtype."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        vocab_size=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    config.torch_dtype = dtype
    model = LlamaForCausalLM(config).to(dtype)
    model.eval()
    return model, config


def _write_save_dir(
    config,
    state_dict: dict,
    save_dir: Path,
    *,
    config_torch_dtype: str = "bfloat16",
) -> None:
    """Persist ``state_dict`` and a minimal ``config.json`` for the loader."""
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = config.to_dict()
    cfg_dict["torch_dtype"] = config_torch_dtype
    cfg_dict["quantization_config"] = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "modules_in_block_to_quantize": [],
    }
    (save_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
    save_file(state_dict, str(save_dir / "model.safetensors"))


def _load(save_dir: Path, **kwargs):
    """Call ``load_quantized_model`` with the tokenizer load patched out."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    with patch(
        "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained",
        return_value=object(),
    ):
        return QuantizedModelLoader.load_quantized_model(
            str(save_dir),
            device_map="",
            local_files_only=True,
            **kwargs,
        )


def test_build_empty_model_uses_config_torch_dtype():
    """``_build_empty_model_from_config`` honours the saved config dtype.

    Isolated unit test for the change-A root fix: when no explicit
    ``torch_dtype`` is passed, the empty model must be built in the
    dtype declared by ``config.json`` (here ``bfloat16``) rather than
    the legacy ``torch.float16`` fallback.  This ensures any parameter
    whose key is missing from the checkpoint inherits the correct dtype
    from the empty initialisation, independently of the safety-net cast.
    """
    from transformers import LlamaConfig

    from onecomp.quantized_model_loader import QuantizedModelLoader

    config = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        vocab_size=64,
    )
    config_dict = config.to_dict()
    config_dict["torch_dtype"] = "bfloat16"

    model = QuantizedModelLoader._build_empty_model_from_config(config_dict, torch_dtype=None)

    sample = next(model.parameters())
    assert sample.dtype == torch.bfloat16


def test_build_empty_model_falls_back_to_fp16_when_config_is_silent():
    """No config dtype and no caller dtype => preserve legacy fp16 default."""
    from transformers import LlamaConfig

    from onecomp.quantized_model_loader import QuantizedModelLoader

    config = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        vocab_size=64,
    )
    config_dict = config.to_dict()
    config_dict.pop("torch_dtype", None)
    config_dict.pop("dtype", None)

    model = QuantizedModelLoader._build_empty_model_from_config(config_dict, torch_dtype=None)

    sample = next(model.parameters())
    assert sample.dtype == torch.float16


def test_load_quantized_model_uses_config_dtype_for_empty_model(tmp_path):
    """Missing-key parameters keep ``config.torch_dtype`` (bf16), not fp16.

    Reproduces the issue #64 Issue 3 failure mode: when
    ``load_state_dict(..., assign=True)`` cannot find a key in the
    checkpoint (as happens for some VLM submodules where the path
    prefix differs between checkpoint and ``from_config``), the
    parameter previously stayed at the empty-model fp16 default.
    With the fix, the empty model is built in the dtype declared by
    ``config.json`` so the missing parameter ends up in bf16 too.
    """
    model, config = _build_tiny_llama(dtype=torch.bfloat16)
    state_dict = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    dropped_key = "model.layers.0.mlp.down_proj.weight"
    assert dropped_key in state_dict
    del state_dict[dropped_key]

    save_dir = tmp_path / "missing_key"
    _write_save_dir(config, state_dict, save_dir)

    loaded_model, _ = _load(save_dir)

    dropped = loaded_model.model.layers[0].mlp.down_proj.weight
    assert dropped.dtype == torch.bfloat16


def test_load_quantized_model_safety_net_casts_fp16_to_target(tmp_path):
    """A weight saved as fp16 must be cast back to ``config.torch_dtype``.

    Even when the bug above is unreachable (state_dict key matches),
    the safety net still corrects fp16 leftovers introduced by
    ``assign=True`` swapping the empty Parameter for a fp16 tensor.
    """
    model, config = _build_tiny_llama(dtype=torch.bfloat16)
    state_dict = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    fp16_key = "model.layers.0.self_attn.q_proj.weight"
    assert fp16_key in state_dict
    state_dict[fp16_key] = state_dict[fp16_key].to(torch.float16).contiguous()

    save_dir = tmp_path / "fp16_leftover"
    _write_save_dir(config, state_dict, save_dir)

    loaded_model, _ = _load(save_dir)

    cast = loaded_model.model.layers[0].self_attn.q_proj.weight
    assert cast.dtype == torch.bfloat16


def test_load_quantized_model_safety_net_preserves_fp32(tmp_path):
    """fp32 parameters (e.g. mixed-precision LayerNorm) must NOT be cast."""
    model, config = _build_tiny_llama(dtype=torch.bfloat16)
    state_dict = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    norm_key = "model.norm.weight"
    assert norm_key in state_dict
    state_dict[norm_key] = state_dict[norm_key].to(torch.float32).contiguous()

    save_dir = tmp_path / "fp32_norm"
    _write_save_dir(config, state_dict, save_dir)

    loaded_model, _ = _load(save_dir)

    assert loaded_model.model.norm.weight.dtype == torch.float32
    assert loaded_model.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16


def test_cast_fp16_to_target_dtype_skips_quantized_layers():
    """Quantized layers (GPTQ / DBF / MDBF / OneBit) must not be cast.

    GPTQ stores its ``scales`` (and similar metadata) in fp16 by design;
    dragging the safety-net cast through quantized modules would
    silently corrupt their format.  This test pins the skip behaviour
    using stub instances that bypass the heavy real ``__init__``.
    MDBF uses a real ``MultipathMDBFLinear`` because its fp16 amplitude
    buffers live on nested ``MDBFLinear`` children, and the skip applies
    per visited module (skipping a parent does not skip its children).
    The OneBit stub keeps its ``a``/``b`` scaling vectors as fp16
    buffers (not parameters), covering the buffer branch of the helper.
    """
    from onecomp.quantized_model_loader import QuantizedModelLoader
    from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear
    from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
    from onecomp.quantizer.mdbf.initialize import MDBFParams
    from onecomp.quantizer.mdbf.mdbf_layer import MultipathMDBFLinear
    from onecomp.quantizer.onebit.onebit_layer import OneBitLinear

    class _StubGPTQ(GPTQLinear):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.scales = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float16))

    class _StubDBF(DoubleBinaryLinear):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.scaling0 = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float16))

    class _StubOneBit(OneBitLinear):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.register_buffer("a", torch.ones(4, dtype=torch.float16))
            self.register_buffer("b", torch.ones(4, dtype=torch.float16))

    def _make_mdbf():
        params = MDBFParams(
            A_sign=torch.ones(4, 2),
            B_sign=torch.ones(2, 4),
            A_amp=torch.ones(4, 1),
            B_amp=torch.ones(4, 1),
            Q_U_amp=torch.ones(2, 1),
            Q_V_amp=torch.ones(2, 1),
        )
        return MultipathMDBFLinear([params], use_gemlite=False)

    class _Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.regular = torch.nn.Linear(4, 4)
            self.regular.weight.data = self.regular.weight.data.to(torch.float16)
            self.regular.bias.data = self.regular.bias.data.to(torch.float16)
            self.gptq = _StubGPTQ()
            self.dbf = _StubDBF()
            self.mdbf = _make_mdbf()
            self.onebit = _StubOneBit()

    model = _Wrapper()
    converted = QuantizedModelLoader._cast_fp16_to_target_dtype(model, torch.bfloat16)

    assert model.regular.weight.dtype == torch.bfloat16
    assert model.regular.bias.dtype == torch.bfloat16
    assert model.gptq.scales.dtype == torch.float16
    assert model.dbf.scaling0.dtype == torch.float16
    for amp_name in ("A_amp", "B_amp", "Q_U_amp", "Q_V_amp"):
        assert getattr(model.mdbf.paths[0], amp_name).dtype == torch.float16
    assert model.onebit.a.dtype == torch.float16
    assert model.onebit.b.dtype == torch.float16
    # The helper returns a list of fully-qualified names so callers
    # (and tests) can see exactly which submodules were touched.
    assert isinstance(converted, list)
    assert sorted(converted) == ["regular.bias", "regular.weight"]


def test_cast_fp16_to_target_dtype_is_noop_for_fp16_target():
    """Calling the helper with target=fp16 must do nothing."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    layer = torch.nn.Linear(4, 4)
    layer.weight.data = layer.weight.data.to(torch.float16)
    layer.bias.data = layer.bias.data.to(torch.float16)

    converted = QuantizedModelLoader._cast_fp16_to_target_dtype(layer, torch.float16)
    assert converted == []
    assert layer.weight.dtype == torch.float16


def test_cast_fp16_to_target_dtype_returns_buffer_names():
    """fp16 buffers must also appear in the returned list of names."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    class _WithBuffer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_mean", torch.zeros(4, dtype=torch.float16))

    class _Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = _WithBuffer()

    model = _Root()
    converted = QuantizedModelLoader._cast_fp16_to_target_dtype(model, torch.bfloat16)

    assert converted == ["child.running_mean"]
    assert model.child.running_mean.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "config_value, expected",
    [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
        ("auto", None),
        ("", None),
        (None, None),
        ("not_a_dtype", None),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_resolve_dtype_from_config(config_value, expected):
    """``_resolve_dtype_from_config`` accepts strings and torch.dtype values."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    config = {"torch_dtype": config_value}
    assert QuantizedModelLoader._resolve_dtype_from_config(config) == expected


def test_resolve_dtype_from_config_falls_back_to_dtype_key():
    """When ``torch_dtype`` is missing, ``dtype`` is used as a fallback."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    config = {"dtype": "bfloat16"}
    assert QuantizedModelLoader._resolve_dtype_from_config(config) == torch.bfloat16
