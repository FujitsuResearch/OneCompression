"""Regression test for tied-embedding handling in ``load_quantized_model``.

Pins the fix for issue #64 Issue 2: when a model has
``tie_word_embeddings=True`` and ``lm_head`` is excluded from
quantization, the loader must re-establish the ``lm_head <-> embed_tokens``
weight tie that ``load_state_dict(..., assign=True)`` breaks.  Otherwise
``lm_head.weight`` keeps the freshly initialised dtype (typically
``float16``) while every other parameter is whatever dtype was stored
in the safetensors checkpoint (typically ``bfloat16``), and the final
``lm_head`` matmul fails with a dtype mismatch.

The test uses a tiny CPU-only ``LlamaForCausalLM`` so it does not need
CUDA, network access or downloaded weights.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file


def _build_tiny_tied_llama(dtype: torch.dtype):
    """Build a tiny Llama-style model with ``tie_word_embeddings=True``."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        vocab_size=64,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config).to(dtype)
    model.eval()
    assert model.config.tie_word_embeddings is True
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    return model, config


def _write_quantized_save_dir(model, config, save_dir: Path) -> None:
    """Persist a tiny model in the layout consumed by ``load_quantized_model``.

    Writes ``config.json`` (with a minimal ``quantization_config`` so the
    loader does not bail out) and a safetensors checkpoint that mirrors
    HF's behaviour for tied embeddings (``lm_head.weight`` is *not*
    serialized when it shares memory with ``embed_tokens.weight``).
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = config.to_dict()
    cfg_dict["quantization_config"] = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "modules_in_block_to_quantize": [],
    }
    (save_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    state_dict: dict[str, torch.Tensor] = {}
    embed_ptr = model.model.embed_tokens.weight.data_ptr()
    for k, v in model.state_dict().items():
        if (
            getattr(model.config, "tie_word_embeddings", False)
            and k == "lm_head.weight"
            and v.data_ptr() == embed_ptr
        ):
            continue
        state_dict[k] = v.detach().clone().contiguous()
    save_file(state_dict, str(save_dir / "model.safetensors"))


def test_load_quantized_model_reties_lm_head_for_tied_embeddings(tmp_path):
    """``lm_head.weight`` and dtype must match ``embed_tokens`` after load."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    model, config = _build_tiny_tied_llama(dtype=torch.bfloat16)
    save_dir = tmp_path / "tied_save"
    _write_quantized_save_dir(model, config, save_dir)

    with patch(
        "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained",
        return_value=object(),
    ):
        loaded_model, _ = QuantizedModelLoader.load_quantized_model(
            str(save_dir),
            torch_dtype=torch.bfloat16,
            device_map="",
            local_files_only=True,
        )

    assert loaded_model.config.tie_word_embeddings is True
    assert loaded_model.lm_head.weight.dtype == torch.bfloat16
    assert loaded_model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert (
        loaded_model.lm_head.weight.data_ptr() == loaded_model.model.embed_tokens.weight.data_ptr()
    ), "lm_head should be re-tied to embed_tokens after assign-load"


def test_load_quantized_model_forward_is_finite_for_tied_model(tmp_path):
    """Forward pass on the loaded model must succeed without dtype mismatch.

    Reproduces the exact failure mode reported in issue #64: when the
    saved checkpoint stores ``embed_tokens`` as ``bfloat16`` but the
    loader-built empty model defaults to ``float16``, the un-fixed code
    leaves ``lm_head.weight`` as ``float16`` and ``F.linear`` raises
    ``RuntimeError: expected mat1 and mat2 to have the same dtype,
    but got: c10::BFloat16 != c10::Half``.

    By passing no explicit ``torch_dtype`` we exercise that exact path.
    """
    from onecomp.quantized_model_loader import QuantizedModelLoader

    model, config = _build_tiny_tied_llama(dtype=torch.bfloat16)
    save_dir = tmp_path / "tied_save_forward"
    _write_quantized_save_dir(model, config, save_dir)

    with patch(
        "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained",
        return_value=object(),
    ):
        loaded_model, _ = QuantizedModelLoader.load_quantized_model(
            str(save_dir),
            device_map="",
            local_files_only=True,
        )
    loaded_model.eval()

    assert loaded_model.lm_head.weight.dtype == loaded_model.model.embed_tokens.weight.dtype

    inputs = torch.randint(0, config.vocab_size, (1, 4))
    with torch.no_grad():
        logits = loaded_model(inputs).logits

    assert torch.isfinite(logits).all()


def test_load_quantized_model_does_not_tie_when_disabled(tmp_path):
    """When ``tie_word_embeddings=False`` the loader must not call ``tie_weights``."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from onecomp.quantized_model_loader import QuantizedModelLoader

    config = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        vocab_size=64,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()

    save_dir = tmp_path / "untied_save"
    _write_quantized_save_dir(model, config, save_dir)

    with patch(
        "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained",
        return_value=object(),
    ):
        loaded_model, _ = QuantizedModelLoader.load_quantized_model(
            str(save_dir),
            torch_dtype=torch.bfloat16,
            device_map="",
            local_files_only=True,
        )

    assert loaded_model.config.tie_word_embeddings is False
    assert (
        loaded_model.lm_head.weight.data_ptr() != loaded_model.model.embed_tokens.weight.data_ptr()
    )


# ---------------------------------------------------------------------------
# Unit tests for ``_retie_lm_head_if_needed``
#
# Decides whether lm_head should be re-tied to embed_tokens after an
# assign-load, tells _check_load_state_dict_result to tolerate a missing
# lm_head.weight when that's the case, and performs the actual
# model.tie_weights() call.
# ---------------------------------------------------------------------------


def test_retie_lm_head_if_needed_ties_when_tied_and_untied_in_memory(tmp_path):
    from onecomp.quantized_model_loader import QuantizedModelLoader

    model, _ = _build_tiny_tied_llama(dtype=torch.bfloat16)
    # Simulate the post assign-load state: tie_weights() has not run yet in
    # this test, so lm_head and embed_tokens do not (necessarily) share
    # storage; assert the call re-establishes it.
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()

    incompat = SimpleNamespace(missing_keys=["lm_head.weight"], unexpected_keys=[])
    QuantizedModelLoader._retie_lm_head_if_needed(model, incompat)

    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_retie_lm_head_if_needed_noop_when_not_tied():
    from transformers import LlamaConfig, LlamaForCausalLM

    from onecomp.quantized_model_loader import QuantizedModelLoader

    config = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        vocab_size=64,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    lm_head_ptr_before = model.lm_head.weight.data_ptr()

    incompat = SimpleNamespace(missing_keys=[], unexpected_keys=[])
    QuantizedModelLoader._retie_lm_head_if_needed(model, incompat)

    # Untouched: no retie happened.
    assert model.lm_head.weight.data_ptr() == lm_head_ptr_before
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()


def test_retie_lm_head_if_needed_still_raises_for_unrelated_critical_missing():
    """A critical missing key other than lm_head.weight must still raise,
    even when the model is tied and lm_head.weight itself is expected to
    be missing."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from onecomp.quantized_model_loader import QuantizedModelLoader

    config = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        vocab_size=64,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()

    incompat = SimpleNamespace(
        missing_keys=["lm_head.weight", "model.embed_tokens.weight"], unexpected_keys=[]
    )
    with pytest.raises(RuntimeError, match="Critical state_dict mismatch"):
        QuantizedModelLoader._retie_lm_head_if_needed(model, incompat)


# ---------------------------------------------------------------------------
# Unit tests for ``_should_retie_word_embeddings``
#
# Multi-config VLMs (e.g. Llama 3.2-Vision) place ``tie_word_embeddings``
# inside ``text_config`` rather than at the top level.  The helper must
# detect both shapes so the post-load re-tie is invoked even for those
# nested configs.  Pure unit tests below exercise the helper directly so
# they do not need a real VLM checkpoint.
# ---------------------------------------------------------------------------


def test_should_retie_word_embeddings_top_level_true():
    """A top-level ``tie_word_embeddings=True`` triggers the re-tie."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    cfg = SimpleNamespace(tie_word_embeddings=True)
    assert QuantizedModelLoader._should_retie_word_embeddings(cfg) is True


def test_should_retie_word_embeddings_nested_text_config_true():
    """``text_config.tie_word_embeddings=True`` (e.g. Llama 3.2-Vision)."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    text_cfg = SimpleNamespace(tie_word_embeddings=True)
    outer = SimpleNamespace(tie_word_embeddings=False, text_config=text_cfg)
    assert QuantizedModelLoader._should_retie_word_embeddings(outer) is True


def test_should_retie_word_embeddings_all_false():
    """All sub-configs say False -> no re-tie."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    text_cfg = SimpleNamespace(tie_word_embeddings=False)
    vision_cfg = SimpleNamespace(tie_word_embeddings=False)
    outer = SimpleNamespace(
        tie_word_embeddings=False,
        text_config=text_cfg,
        vision_config=vision_cfg,
    )
    assert QuantizedModelLoader._should_retie_word_embeddings(outer) is False


def test_should_retie_word_embeddings_ignores_unrelated_attrs():
    """Sub-attributes without ``tie_word_embeddings`` must not crash the walk."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    outer = SimpleNamespace(
        tie_word_embeddings=False,
        unrelated="just a string",
        another_unrelated=12345,
        nested_no_flag=SimpleNamespace(some_other_field=True),
    )
    assert QuantizedModelLoader._should_retie_word_embeddings(outer) is False
