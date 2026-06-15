"""Unit tests for quantization_config schema validation.

Covers the shared schema-validation helpers in
:mod:`onecomp.utils.quant_config`:

- :func:`validate_quant_config` — dict-level schema check.
- :func:`validate_quantized_model_config` — model-level wrapper that pulls
  ``quantization_config`` off ``model.config`` and applies the same check.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest

from onecomp.utils.quant_config import (
    validate_quant_config,
    validate_quantized_model_config,
)
from tests.onecomp.fixtures.quant_config import (
    FakeConfig as _FakeModelConfig,
    valid_quant_config,
)


class _FakeConfigWithoutQuant:
    """``model.config`` that lacks a ``quantization_config`` attribute."""


class _FakeModelWithoutConfig:
    """Model stand-in that lacks a ``.config`` attribute."""


class _FakeModel:
    """Minimal model exposing a ``.config`` attribute."""

    def __init__(self, config):
        self.config = config


# ---------------------------------------------------------------------------
# validate_quant_config
# ---------------------------------------------------------------------------


def test_validate_quant_config_returns_same_object_for_valid_config():
    """A valid config is returned unchanged (same object)."""
    config = valid_quant_config()
    result = validate_quant_config(config, context="save")
    assert result is config


@pytest.mark.parametrize(
    "invalid_config",
    [
        None,
        [],
        {"modules_in_block_to_quantize": []},  # quant_method missing
        {"quant_method": "", "modules_in_block_to_quantize": []},  # empty
        {"quant_method": "gptq"},  # modules_in_block_to_quantize missing
    ],
    ids=[
        "none",
        "list",
        "missing-quant_method",
        "empty-quant_method",
        "missing-modules",
    ],
)
def test_validate_quant_config_rejects_invalid_schema(invalid_config):
    """Dict-level invalid schemas are rejected with ValueError."""
    with pytest.raises(ValueError):
        validate_quant_config(invalid_config, context="save")


# ---------------------------------------------------------------------------
# validate_quantized_model_config
# ---------------------------------------------------------------------------


def test_validate_config_missing_config_attr():
    """A model lacking a ``.config`` attribute is rejected."""
    model = _FakeModelWithoutConfig()
    with pytest.raises(ValueError):
        validate_quantized_model_config(model, context="post_process")


def test_validate_config_none_config_attr():
    """A model whose ``config`` is None is rejected."""
    model = _FakeModel(config=None)
    with pytest.raises(ValueError):
        validate_quantized_model_config(model, context="post_process")


def test_validate_config_missing_quantization_config_attr():
    """A ``config`` without ``quantization_config`` is rejected."""
    model = _FakeModel(config=_FakeConfigWithoutQuant())
    with pytest.raises(ValueError):
        validate_quantized_model_config(model, context="post_process")


def test_validate_config_not_a_dict():
    """A non-dict ``quantization_config`` is rejected."""
    model = _FakeModel(config=_FakeModelConfig(quantization_config="not-a-dict"))
    with pytest.raises(ValueError):
        validate_quantized_model_config(model, context="post_process")


def test_validate_config_missing_quant_method():
    """A missing ``quant_method`` is rejected via the model wrapper."""
    model = _FakeModel(
        config=_FakeModelConfig(
            quantization_config={"modules_in_block_to_quantize": []}
        )
    )
    with pytest.raises(ValueError):
        validate_quantized_model_config(model, context="post_process")


def test_validate_config_empty_quant_method():
    """An empty ``quant_method`` is rejected via the model wrapper."""
    model = _FakeModel(
        config=_FakeModelConfig(
            quantization_config={
                "quant_method": "",
                "modules_in_block_to_quantize": [],
            }
        )
    )
    with pytest.raises(ValueError):
        validate_quantized_model_config(model, context="post_process")


def test_validate_config_missing_modules():
    """A missing ``modules_in_block_to_quantize`` is rejected."""
    model = _FakeModel(
        config=_FakeModelConfig(quantization_config={"quant_method": "gptq"})
    )
    with pytest.raises(ValueError):
        validate_quantized_model_config(model, context="post_process")


def test_validate_config_returns_same_dict():
    """Model-level validation returns the same dict object."""
    config = valid_quant_config()
    model = _FakeModel(config=_FakeModelConfig(quantization_config=config))
    result = validate_quantized_model_config(model, context="post_process")
    assert result is config


def test_validate_config_context_in_message():
    """The context label appears in the raised error message."""
    model = _FakeModel(config=None)
    with pytest.raises(ValueError, match="post_process"):
        validate_quantized_model_config(model, context="post_process")
