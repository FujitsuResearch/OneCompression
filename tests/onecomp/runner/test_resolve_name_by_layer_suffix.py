"""Unit tests for ``QuantizedModelLoader._resolve_name_by_layer_suffix``.

Pins the ambiguity-handling contract of the shared name-resolution helper:

- ``on_ambiguous="first"`` (default, used by the quantized-layer load path
  for VLM tied/shared submodules) returns the first matching candidate.
- ``on_ambiguous="error"`` (used by the LoRA re-wrap path, where colliding
  candidates can be *distinct* layers) raises ``ValueError`` rather than
  silently guessing which layer to target.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest

from onecomp.quantized_model_loader import QuantizedModelLoader


def test_exact_match_returns_name():
    """An exact key match short-circuits before suffix resolution."""
    candidates = {"model.layers.0.self_attn.q_proj": object()}
    resolved = QuantizedModelLoader._resolve_name_by_layer_suffix(
        "model.layers.0.self_attn.q_proj", candidates
    )
    assert resolved == "model.layers.0.self_attn.q_proj"


def test_unique_suffix_match():
    """A single suffix hit is resolved regardless of the differing prefix."""
    candidates = {"language_model.model.layers.0.self_attn.q_proj": object()}
    resolved = QuantizedModelLoader._resolve_name_by_layer_suffix(
        "model.layers.0.self_attn.q_proj", candidates
    )
    assert resolved == "language_model.model.layers.0.self_attn.q_proj"


def test_no_match_returns_none():
    """No suffix hit (and no exact match) resolves to ``None``."""
    candidates = {"model.layers.0.mlp.gate_proj": object()}
    resolved = QuantizedModelLoader._resolve_name_by_layer_suffix(
        "model.layers.0.self_attn.q_proj", candidates
    )
    assert resolved is None


def test_no_layer_suffix_returns_none():
    """A name without a ``layers.N.`` segment never falls back to suffix match."""
    candidates = {"vision_tower.embeddings.patch_embedding": object()}
    resolved = QuantizedModelLoader._resolve_name_by_layer_suffix(
        "embeddings.patch_embedding", candidates
    )
    assert resolved is None


def test_ambiguous_first_returns_leading_hit():
    """With ``on_ambiguous="first"`` the first matching candidate is returned."""
    # dict preserves insertion order, so hits[0] is the language_model entry.
    candidates = {
        "language_model.model.layers.0.self_attn.q_proj": object(),
        "vision.model.layers.0.self_attn.q_proj": object(),
    }
    resolved = QuantizedModelLoader._resolve_name_by_layer_suffix(
        "model.layers.0.self_attn.q_proj", candidates, on_ambiguous="first"
    )
    assert resolved == "language_model.model.layers.0.self_attn.q_proj"


def test_ambiguous_first_is_default():
    """``on_ambiguous`` defaults to ``"first"``."""
    candidates = {
        "language_model.model.layers.0.self_attn.q_proj": object(),
        "vision.model.layers.0.self_attn.q_proj": object(),
    }
    resolved = QuantizedModelLoader._resolve_name_by_layer_suffix(
        "model.layers.0.self_attn.q_proj", candidates
    )
    assert resolved == "language_model.model.layers.0.self_attn.q_proj"


def test_ambiguous_error_raises():
    """With ``on_ambiguous="error"`` an ambiguous suffix raises ``ValueError``."""
    candidates = {
        "language_model.model.layers.0.self_attn.q_proj": object(),
        "vision.model.layers.0.self_attn.q_proj": object(),
    }
    with pytest.raises(ValueError) as excinfo:
        QuantizedModelLoader._resolve_name_by_layer_suffix(
            "model.layers.0.self_attn.q_proj", candidates, on_ambiguous="error"
        )
    # The message must name both colliding candidates for fast diagnosis.
    message = str(excinfo.value)
    assert "language_model.model.layers.0.self_attn.q_proj" in message
    assert "vision.model.layers.0.self_attn.q_proj" in message


def test_error_mode_unique_match_does_not_raise():
    """``on_ambiguous="error"`` only affects the ambiguous case."""
    candidates = {"language_model.model.layers.0.self_attn.q_proj": object()}
    resolved = QuantizedModelLoader._resolve_name_by_layer_suffix(
        "model.layers.0.self_attn.q_proj", candidates, on_ambiguous="error"
    )
    assert resolved == "language_model.model.layers.0.self_attn.q_proj"
