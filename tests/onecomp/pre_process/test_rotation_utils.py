"""Tests for rotation utility predicates.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest

from onecomp.pre_process.rotation_utils import is_online_hadamard_target


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mlp.down_proj", True),
        ("model.layers.0.mlp.down_proj", True),
        ("model.layers.0.mlp.gate_proj", False),
        ("down_proj", False),
        ("model.layers.0.block_sparse_moe.experts.0.down_proj", False),
    ],
)
def test_is_online_hadamard_target(name, expected):
    """Online Hadamard targets are limited to dense MLP down_proj paths."""
    assert is_online_hadamard_target(name) is expected