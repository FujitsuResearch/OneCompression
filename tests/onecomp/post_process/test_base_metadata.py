"""Unit tests for ``PostQuantizationProcess.build_metadata``.

Verifies the JSON-serializable audit-metadata contract of
:meth:`onecomp.post_process._base.PostQuantizationProcess.build_metadata`:
``name`` defaulting, concrete ``class`` recording, exclusion of ``name`` and
private fields from ``config``, tuple normalization, non-serializable coercion,
nested-dataclass expansion, and end-to-end JSON serializability.

Copyright 2025-2026 Fujitsu Ltd.
"""

import json
from dataclasses import dataclass, field

from onecomp.post_process._base import PostQuantizationProcess
from onecomp.post_process.blockwise_ptq import BlockWisePTQ

# ---------------------------------------------------------------------------
# Concrete test doubles
# ---------------------------------------------------------------------------


@dataclass
class _SimpleProcess(PostQuantizationProcess):
    learning_rate: float = 0.1
    epochs: int = 3

    def _run(self, quantized_model, model_config) -> None:  # pragma: no cover
        pass


@dataclass
class _PrivateFieldProcess(PostQuantizationProcess):
    public_value: int = 5
    _private_value: int = 9

    def _run(self, quantized_model, model_config) -> None:  # pragma: no cover
        pass


@dataclass
class _TupleFieldProcess(PostQuantizationProcess):
    shape: tuple = (1, 2, 3)

    def _run(self, quantized_model, model_config) -> None:  # pragma: no cover
        pass


@dataclass
class _NonSerializableFieldProcess(PostQuantizationProcess):
    payload: object = field(default_factory=object)

    def _run(self, quantized_model, model_config) -> None:  # pragma: no cover
        pass


@dataclass
class _InnerConfig:
    alpha: int = 1
    beta: str = "x"


@dataclass
class _NestedFieldProcess(PostQuantizationProcess):
    inner: _InnerConfig = field(default_factory=_InnerConfig)

    def _run(self, quantized_model, model_config) -> None:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# name / class
# ---------------------------------------------------------------------------


def test_name_defaults_to_class_name():
    """``name`` defaults to the class name when not set explicitly."""
    metadata = _SimpleProcess().build_metadata()
    assert metadata["name"] == "_SimpleProcess"


def test_explicit_name_is_preserved():
    """An explicitly supplied ``name`` is preserved."""
    metadata = _SimpleProcess(name="custom-step").build_metadata()
    assert metadata["name"] == "custom-step"


def test_class_is_concrete_type_name():
    """``class`` records the concrete type name regardless of ``name``."""
    metadata = _SimpleProcess(name="custom-step").build_metadata()
    assert metadata["class"] == "_SimpleProcess"


# ---------------------------------------------------------------------------
# config contents
# ---------------------------------------------------------------------------


def test_config_excludes_name():
    """``config`` does not duplicate the top-level ``name``."""
    metadata = _SimpleProcess(name="custom-step").build_metadata()
    assert "name" not in metadata["config"]


def test_config_excludes_private_fields():
    """Fields whose name starts with ``_`` are excluded from ``config``."""
    metadata = _PrivateFieldProcess().build_metadata()
    assert "_private_value" not in metadata["config"]


def test_config_records_public_fields():
    """Public fields are recorded in ``config``."""
    metadata = _SimpleProcess(learning_rate=0.25, epochs=7).build_metadata()
    assert metadata["config"]["learning_rate"] == 0.25
    assert metadata["config"]["epochs"] == 7


def test_tuple_field_is_normalized_to_list():
    """A tuple field is normalized to a JSON list."""
    metadata = _TupleFieldProcess(shape=(4, 5)).build_metadata()
    assert metadata["config"]["shape"] == [4, 5]


def test_non_serializable_field_is_coerced_to_string():
    """A JSON-incompatible value is coerced to a string."""
    metadata = _NonSerializableFieldProcess().build_metadata()
    assert isinstance(metadata["config"]["payload"], str)


def test_metadata_is_json_serializable():
    """The whole metadata dict is ``json.dumps``-able."""
    metadata = _NonSerializableFieldProcess().build_metadata()
    # Must not raise.
    json.dumps(metadata)


def test_nested_dataclass_field_is_expanded():
    """A nested dataclass field is recorded as a dict."""
    metadata = _NestedFieldProcess(inner=_InnerConfig(alpha=9, beta="y")).build_metadata()
    assert metadata["config"]["inner"] == {"alpha": 9, "beta": "y"}


# ---------------------------------------------------------------------------
# regression: real post-process
# ---------------------------------------------------------------------------


def test_blockwise_ptq_metadata_contains_key_hyperparameters():
    """``BlockWisePTQ`` records its key hyperparameters (regression)."""
    metadata = BlockWisePTQ(lr=2e-4, epochs=5, cbq_enable=True).build_metadata()
    config = metadata["config"]
    assert metadata["class"] == "BlockWisePTQ"
    assert config["lr"] == 2e-4
    assert config["epochs"] == 5
    assert config["cbq_enable"] is True
