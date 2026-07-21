"""Tests for ``Runner``'s ``save_format`` handling in ``save_quantized_model``.

``save_format="full_wrapper"`` remaps a Qwen3.6 text-only quantized checkpoint
(``model.layers.*``) to the composite ``model.language_model.*`` namespace
that vLLM's composite ``Qwen3_5ForConditionalGeneration`` loader expects. It
is deliberately scoped to Qwen3.6 (``model_type`` ``qwen3_5_text`` /
``qwen3_5_moe_text``): the config/state_dict namespace helpers below gate the
remap, and any other model - including other VLM architectures whose original
config also has ``text_config`` - falls through to a ``RuntimeError`` instead
of being converted.

These tests use plain stand-ins (``SimpleNamespace`` / small stub classes)
rather than real Qwen3.6 models, matching the pattern used by
``test_remap_state_dict_keys.py`` for VLM key-remap tests.

Copyright 2025-2026 Fujitsu Ltd.
"""

from logging import getLogger
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from onecomp.runner import Runner


def _make_runner_stub(*, load_config=None):
    """Build a ``Runner`` instance with only the fields these helpers use."""
    runner = Runner.__new__(Runner)
    runner.logger = getLogger("test_save_format_full_wrapper")
    if load_config is not None:
        runner.model_config = SimpleNamespace(load_config=load_config)
    return runner


def _fake_model(*, model_type, state_dict_keys, text_config=None, quantization_config=None):
    config = SimpleNamespace(
        model_type=model_type,
        text_config=text_config,
        quantization_config=quantization_config,
    )
    state_dict = {k: torch.zeros(1) for k in state_dict_keys}
    return SimpleNamespace(
        config=config,
        state_dict=lambda: dict(state_dict),
        named_modules=lambda: [(k.rsplit(".", 1)[0], None) for k in state_dict],
    )


# ---------------------------------------------------------------------------
# _detect_weight_namespace
# ---------------------------------------------------------------------------


def test_detect_weight_namespace_text_only():
    runner = _make_runner_stub()
    model = _fake_model(model_type="qwen3_5_text", state_dict_keys=["model.layers.0.weight"])
    assert runner._detect_weight_namespace(model) == "text_only"


def test_detect_weight_namespace_full_wrapper_prefix():
    runner = _make_runner_stub()
    model = _fake_model(
        model_type="qwen3_5", state_dict_keys=["model.language_model.model.layers.0.weight"]
    )
    assert runner._detect_weight_namespace(model) == "full_language_model"


def test_detect_weight_namespace_direct_language_model_prefix():
    runner = _make_runner_stub()
    model = _fake_model(model_type="qwen3_5", state_dict_keys=["model.language_model.layers.0.weight"])
    assert runner._detect_weight_namespace(model) == "full_language_model"


def test_detect_weight_namespace_bare_language_model_prefix():
    runner = _make_runner_stub()
    model = _fake_model(model_type="qwen3_5", state_dict_keys=["language_model.layers.0.weight"])
    assert runner._detect_weight_namespace(model) == "full_language_model"


def test_detect_weight_namespace_substring_language_model_layers():
    """e.g. vision_tower.language_model.layers.* for other composite archs."""
    runner = _make_runner_stub()
    model = _fake_model(
        model_type="gemma3", state_dict_keys=["vision_tower.language_model.layers.0.weight"]
    )
    assert runner._detect_weight_namespace(model) == "full_language_model"


def test_detect_weight_namespace_unknown():
    runner = _make_runner_stub()
    model = _fake_model(model_type="llama", state_dict_keys=["transformer.h.0.weight"])
    assert runner._detect_weight_namespace(model) == "unknown"


# ---------------------------------------------------------------------------
# _detect_config_namespace
# ---------------------------------------------------------------------------


def test_detect_config_namespace_qwen36_text_only():
    runner = _make_runner_stub()
    for model_type in ("qwen3_5_text", "qwen3_5_moe_text"):
        model = _fake_model(model_type=model_type, state_dict_keys=[])
        assert runner._detect_config_namespace(model) == "text_only"


def test_detect_config_namespace_qwen36_full_language_model():
    runner = _make_runner_stub()
    for model_type in ("qwen3_5", "qwen3_5_moe"):
        model = _fake_model(model_type=model_type, state_dict_keys=[])
        assert runner._detect_config_namespace(model) == "full_language_model"


def test_detect_config_namespace_other_composite_via_text_config():
    """Non-Qwen composite configs are detected as full_language_model, never
    text_only - only the two hardcoded Qwen3.6 model_types resolve to
    text_only (see module docstring)."""
    runner = _make_runner_stub()
    model = _fake_model(
        model_type="gemma3", state_dict_keys=[], text_config=SimpleNamespace(model_type="gemma3_text")
    )
    assert runner._detect_config_namespace(model) == "full_language_model"


def test_detect_config_namespace_unknown_for_plain_model():
    runner = _make_runner_stub()
    model = _fake_model(model_type="llama", state_dict_keys=[])
    assert runner._detect_config_namespace(model) == "unknown"


# ---------------------------------------------------------------------------
# _assert_config_state_dict_namespace_consistent
# ---------------------------------------------------------------------------


def test_namespace_consistent_passes_when_matching():
    runner = _make_runner_stub()
    model = _fake_model(model_type="qwen3_5_text", state_dict_keys=["model.layers.0.weight"])
    runner._assert_config_state_dict_namespace_consistent(model)  # should not raise


def test_namespace_consistent_noop_when_state_dict_namespace_unknown():
    runner = _make_runner_stub()
    model = _fake_model(model_type="qwen3_5_text", state_dict_keys=["something.else.weight"])
    runner._assert_config_state_dict_namespace_consistent(model)  # should not raise


def test_namespace_consistent_raises_on_mismatch():
    runner = _make_runner_stub()
    model = _fake_model(
        model_type="qwen3_5_text", state_dict_keys=["model.language_model.layers.0.weight"]
    )
    with pytest.raises(RuntimeError, match="namespace mismatch"):
        runner._assert_config_state_dict_namespace_consistent(model)


# ---------------------------------------------------------------------------
# _assert_quant_config_matches_model_namespace
# ---------------------------------------------------------------------------


def test_quant_config_check_passes_when_no_quantization_config():
    runner = _make_runner_stub()
    model = _fake_model(model_type="llama", state_dict_keys=["model.layers.0.weight"])
    runner._assert_quant_config_matches_model_namespace(model)  # should not raise


def test_quant_config_check_passes_when_modules_present():
    runner = _make_runner_stub()
    model = _fake_model(
        model_type="qwen3_5_text",
        state_dict_keys=["model.layers.0.mlp.down_proj.qweight"],
        quantization_config={"modules_in_block_to_quantize": ["model.layers.0.mlp.down_proj"]},
    )
    runner._assert_quant_config_matches_model_namespace(model)  # should not raise


def test_quant_config_check_raises_when_module_missing():
    runner = _make_runner_stub()
    model = _fake_model(
        model_type="qwen3_5_text",
        state_dict_keys=["model.layers.0.mlp.down_proj.qweight"],
        quantization_config={"modules_in_block_to_quantize": ["model.layers.5.mlp.down_proj"]},
    )
    with pytest.raises(RuntimeError, match="missing="):
        runner._assert_quant_config_matches_model_namespace(model)


# ---------------------------------------------------------------------------
# _prepare_model_for_quantized_save
# ---------------------------------------------------------------------------


def test_prepare_model_for_quantized_save_raises_on_unknown_format():
    runner = _make_runner_stub()
    model = _fake_model(model_type="llama", state_dict_keys=["model.layers.0.weight"])
    with pytest.raises(ValueError, match="Unknown save_format"):
        runner._prepare_model_for_quantized_save(model, save_format="bogus")


def test_prepare_model_for_quantized_save_auto_returns_state_dict_unchanged():
    runner = _make_runner_stub()
    model = _fake_model(model_type="llama", state_dict_keys=["model.layers.0.weight"])
    sentinel = {"model.layers.0.weight": torch.zeros(1)}
    result = runner._prepare_model_for_quantized_save(
        model, save_format="auto", state_dict=sentinel
    )
    assert result is sentinel


def test_prepare_model_for_quantized_save_native_raises_on_namespace_mismatch():
    # cfg_ns="full_language_model" / sd_ns="text_only" does not match the
    # text_only+full_language_model combination that
    # _restore_original_composite_config_if_needed auto-fixes, so this
    # reaches _assert_config_state_dict_namespace_consistent's raise.
    runner = _make_runner_stub()
    model = _fake_model(model_type="qwen3_5", state_dict_keys=["model.layers.0.weight"])
    with pytest.raises(RuntimeError, match="namespace mismatch"):
        runner._prepare_model_for_quantized_save(model, save_format="native")


# ---------------------------------------------------------------------------
# _prepare_full_wrapper_quantized_save: the Qwen3.6-only scoping behavior
# ---------------------------------------------------------------------------


def test_full_wrapper_noop_when_already_composite():
    runner = _make_runner_stub()
    model = _fake_model(
        model_type="qwen3_5", state_dict_keys=["model.language_model.layers.0.weight"]
    )
    sentinel = {"model.language_model.layers.0.weight": torch.zeros(1)}
    result = runner._prepare_full_wrapper_quantized_save(model, state_dict=sentinel)
    assert result is sentinel


def test_full_wrapper_raises_when_original_config_not_composite():
    """A plain (non-VLM) model has no text_config on its original config, so
    full_wrapper must refuse rather than silently no-op or corrupt it."""
    runner = _make_runner_stub(load_config=lambda: SimpleNamespace(model_type="llama"))
    model = _fake_model(model_type="llama", state_dict_keys=["model.layers.0.weight"])
    with pytest.raises(RuntimeError, match="not composite and has no text_config"):
        runner._prepare_full_wrapper_quantized_save(model)


def test_full_wrapper_raises_for_non_qwen36_composite_model():
    """Gemma3 (and other non-Qwen VLMs) has a composite original config
    (text_config is set), but _detect_config_namespace only recognizes
    text_only for the two hardcoded Qwen3.6 model_types - so this must raise,
    not silently convert. This is the scoping behavior documented for users
    (docs/user-guide/vllm-inference.md, save_quantized_model docstring)."""
    runner = _make_runner_stub(
        load_config=lambda: SimpleNamespace(
            model_type="gemma3", text_config=SimpleNamespace(model_type="gemma3_text")
        )
    )
    model = _fake_model(model_type="gemma3_text", state_dict_keys=["model.layers.0.weight"])
    with pytest.raises(RuntimeError, match="consistent text-only checkpoints only"):
        runner._prepare_full_wrapper_quantized_save(model)


def test_full_wrapper_raises_when_no_quantization_config():
    runner = _make_runner_stub(
        load_config=lambda: SimpleNamespace(
            model_type="qwen3_5", text_config=SimpleNamespace(model_type="qwen3_5_text")
        )
    )
    model = _fake_model(model_type="qwen3_5_text", state_dict_keys=["model.layers.0.weight"])
    with pytest.raises(RuntimeError, match="no quantization_config"):
        runner._prepare_full_wrapper_quantized_save(model)


def test_full_wrapper_remaps_qwen36_text_only_checkpoint():
    composite_config = SimpleNamespace(
        model_type="qwen3_5", text_config=SimpleNamespace(model_type="qwen3_5_text")
    )
    runner = _make_runner_stub(load_config=lambda: composite_config)
    model = _fake_model(
        model_type="qwen3_5_text",
        state_dict_keys=[
            "model.layers.0.mlp.down_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ],
        quantization_config={
            "modules_in_block_to_quantize": ["model.layers.0.mlp.down_proj"],
        },
    )

    result = runner._prepare_full_wrapper_quantized_save(model)

    assert set(result) == {
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    }
    # model.config is replaced with the (deep-copied) original composite config...
    assert model.config is not composite_config
    assert model.config.model_type == "qwen3_5"
    # ...with the quantization_config module names remapped to match it.
    assert model.config.quantization_config["modules_in_block_to_quantize"] == [
        "model.language_model.layers.0.mlp.down_proj"
    ]


# ---------------------------------------------------------------------------
# _remap_text_only_quant_config_to_full_wrapper / _remap_text_only_state_dict_to_full_wrapper
# ---------------------------------------------------------------------------


def test_remap_quant_config_renames_layers_embed_tokens_and_norm():
    # remap_name() matches on "model.embed_tokens." / "model.norm." (with a
    # trailing dot before the leaf field), so bare "model.embed_tokens" /
    # "model.norm" (no further suffix) are intentionally left unchanged.
    quant_config = {
        "modules_in_block_to_quantize": ["model.layers.0.mlp.down_proj"],
        "quantized_layer_names": ["model.embed_tokens.weight", "model.norm.weight"],
        "quant_method": "gptq",
    }
    remapped = Runner._remap_text_only_quant_config_to_full_wrapper(quant_config)
    assert remapped["modules_in_block_to_quantize"] == ["model.language_model.layers.0.mlp.down_proj"]
    assert remapped["quantized_layer_names"] == [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    ]
    assert remapped["quant_method"] == "gptq"
    # Original dict must not be mutated in place.
    assert quant_config["modules_in_block_to_quantize"] == ["model.layers.0.mlp.down_proj"]


def test_remap_state_dict_prefixes_model_keeps_lm_head():
    weight = torch.arange(4, dtype=torch.float32)
    state_dict = {
        "model.layers.0.weight": weight,
        "model.embed_tokens.weight": weight,
        "lm_head.weight": weight,
    }
    remapped = Runner._remap_text_only_state_dict_to_full_wrapper(state_dict)
    assert set(remapped) == {
        "model.language_model.layers.0.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    }
    assert torch.equal(remapped["lm_head.weight"], weight)


def test_remap_state_dict_raises_on_key_collision():
    state_dict = {
        "model.layers.0.weight": torch.zeros(1),
        # Already-composite key that collides with the remapped target above.
        "model.language_model.layers.0.weight": torch.ones(1),
    }
    with pytest.raises(RuntimeError, match="collision"):
        Runner._remap_text_only_state_dict_to_full_wrapper(state_dict)


# ---------------------------------------------------------------------------
# save_quantized_model(save_format="full_wrapper"): end-to-end config restore
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def save_pretrained(self, save_directory):
        del save_directory


class _FakeFullWrapperModel:
    """Minimal stand-in exercising the save_pretrained(state_dict=...) path."""

    def __init__(self, config, state_dict):
        self.config = config
        self._state_dict = state_dict
        self.save_calls = []

    def state_dict(self):
        return dict(self._state_dict)

    def named_modules(self):
        return [(k.rsplit(".", 1)[0], None) for k in self._state_dict]

    def save_pretrained(self, save_directory, state_dict=None):
        self.save_calls.append(
            {"state_dict": state_dict, "config_model_type": self.config.model_type}
        )


def test_save_quantized_model_full_wrapper_remaps_and_restores_config(tmp_path):
    """save_quantized_model(save_format="full_wrapper") must:
    - pass the remapped, composite-namespace state_dict to save_pretrained(),
    - but restore self.quantized_model.config to the original in-memory
      object afterwards, so the Runner never leaves the model mutated as a
      side effect of saving (see runner.py's try/finally around
      _prepare_model_for_quantized_save)."""
    text_only_config = SimpleNamespace(
        model_type="qwen3_5_text",
        quantization_config={
            "modules_in_block_to_quantize": ["model.layers.0.mlp.down_proj"],
        },
    )
    composite_config = SimpleNamespace(
        model_type="qwen3_5", text_config=SimpleNamespace(model_type="qwen3_5_text")
    )
    model = _FakeFullWrapperModel(
        text_only_config,
        {
            "model.layers.0.mlp.down_proj.weight": torch.zeros(2, 2),
            "model.embed_tokens.weight": torch.zeros(4, 2),
            "lm_head.weight": torch.zeros(4, 2),
        },
    )

    runner = Runner.__new__(Runner)
    runner.logger = getLogger("test_save_format_full_wrapper")
    runner.quantized_model = model
    runner.model_config = SimpleNamespace(
        dtype="float16",
        load_tokenizer=lambda: _FakeTokenizer(),
        get_model_id_or_path=lambda: None,
        load_config=lambda: composite_config,
    )

    save_dir = tmp_path / "qwen36_full_wrapper"
    with patch.object(
        Runner, "_save_processor_files_if_available", lambda self, save_directory: None
    ):
        runner.save_quantized_model(str(save_dir), save_format="full_wrapper")

    assert len(model.save_calls) == 1
    call = model.save_calls[0]
    assert set(call["state_dict"]) == {
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    }
    # save_pretrained() saw the remapped composite config...
    assert call["config_model_type"] == "qwen3_5"
    # ...but it is restored to the original text-only object afterwards.
    assert model.config is text_only_config
    assert model.config.model_type == "qwen3_5_text"


def test_save_quantized_model_full_wrapper_raises_for_non_qwen36_model(tmp_path):
    """A plain (non-Qwen3.6) model must fail save_format="full_wrapper"
    instead of silently writing a checkpoint vLLM won't be able to load."""
    plain_config = SimpleNamespace(model_type="llama", quantization_config=None)
    model = _FakeFullWrapperModel(plain_config, {"model.layers.0.weight": torch.zeros(1)})

    runner = Runner.__new__(Runner)
    runner.logger = getLogger("test_save_format_full_wrapper")
    runner.quantized_model = model
    runner.model_config = SimpleNamespace(
        dtype="float16",
        load_tokenizer=lambda: _FakeTokenizer(),
        get_model_id_or_path=lambda: None,
        load_config=lambda: SimpleNamespace(model_type="llama"),
    )

    save_dir = tmp_path / "not_qwen36"
    with pytest.raises(RuntimeError, match="not composite and has no text_config"):
        runner.save_quantized_model(str(save_dir), save_format="full_wrapper")

    # Must not have called save_pretrained, and config must be untouched.
    assert model.save_calls == []
    assert model.config is plain_config


def test_save_quantized_model_default_format_is_noop_for_non_qwen_model(tmp_path):
    """The default save_format ("auto") must be a true no-op for an ordinary
    (non-Qwen3.6, non-composite) model: no state_dict remap and no config
    swap, not even transiently during the call. This is the regression
    guarantee that adding save_format="full_wrapper" did not change the
    existing default behavior for every other model (Llama, plain Qwen3,
    Gemma-text, ...)."""
    plain_config = SimpleNamespace(model_type="llama", quantization_config=None)
    model = _FakeFullWrapperModel(plain_config, {"model.layers.0.weight": torch.zeros(1)})

    runner = Runner.__new__(Runner)
    runner.logger = getLogger("test_save_format_full_wrapper")
    runner.quantized_model = model
    runner.model_config = SimpleNamespace(
        dtype="float16",
        load_tokenizer=lambda: _FakeTokenizer(),
        get_model_id_or_path=lambda: None,
        load_config=lambda: SimpleNamespace(model_type="llama"),
    )

    save_dir = tmp_path / "plain_llama_default"
    # No save_format kwarg: exercises the actual default, not just "auto"
    # passed explicitly.
    runner.save_quantized_model(str(save_dir))

    assert len(model.save_calls) == 1
    call = model.save_calls[0]
    # state_dict=None means "use model.state_dict() as-is" (see
    # _prepare_model_for_quantized_save's docstring) - i.e. no remap applied.
    assert call["state_dict"] is None
    assert call["config_model_type"] == "llama"
    # Config object identity is preserved - it was never swapped, not even
    # temporarily inside the try/finally.
    assert model.config is plain_config
