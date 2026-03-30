"""Tests for fused-group validation in manual assignment mode.

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

import pytest

from onecomp.quantizer.autobit._autobit import AutoBitQuantizer
from onecomp.quantizer.gptq import GPTQ


def _make_quantizer(**overrides):
    defaults = dict(
        assignment_strategy="manual",
        enable_fused_groups=True,
        quantizers=[GPTQ(wbits=4)],
    )
    defaults.update(overrides)
    return AutoBitQuantizer(**defaults)


def test_consistent_keywords_passes():
    """QKV all match 'self_attn' -> same quantizer -> OK."""
    ab = _make_quantizer(
        quantizers=[
            GPTQ(wbits=4, include_layer_keywords=["self_attn"]),
            GPTQ(wbits=3, include_layer_keywords=["mlp"]),
        ]
    )
    ab.validate_params()


def test_inconsistent_qkv_raises():
    """q/k match one quantizer, v matches another with different bits."""
    ab = _make_quantizer(
        quantizers=[
            GPTQ(wbits=4, include_layer_keywords=["q_proj", "k_proj"]),
            GPTQ(wbits=2, include_layer_keywords=["v_proj"]),
        ]
    )
    with pytest.raises(ValueError, match="mixed bit-widths"):
        ab.validate_params()


def test_same_bits_different_quantizers_passes():
    """Different quantizer objects but same wbits -> OK."""
    ab = _make_quantizer(
        quantizers=[
            GPTQ(wbits=4, include_layer_keywords=["q_proj"]),
            GPTQ(wbits=4, include_layer_keywords=["k_proj", "v_proj"]),
        ]
    )
    ab.validate_params()


def test_catchall_quantizer_consistent():
    """First quantizer has no keywords (matches all) -> all fused members resolve to it."""
    ab = _make_quantizer(quantizers=[GPTQ(wbits=4)])
    ab.validate_params()
