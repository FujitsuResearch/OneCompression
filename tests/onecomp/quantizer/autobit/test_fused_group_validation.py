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


def test_bitpack_flag_propagates_to_supported_gptq():
    """AutoBit bitpack flag is copied to supported GPTQ candidates."""
    q2 = GPTQ(wbits=2, bitpack_on_quantize=False)
    q4 = GPTQ(wbits=4, bitpack_on_quantize=False)
    ab = _make_quantizer(bitpack_on_quantize=True, quantizers=[q2, q4])

    ab.validate_params()

    assert q2.bitpack_on_quantize is True
    assert q4.bitpack_on_quantize is True


def test_bitpack_flag_disables_unsupported_gptq_before_child_validation():
    """Unsupported GPTQ wbits are left unpacked when AutoBit bitpack is enabled."""
    q4 = GPTQ(wbits=4, bitpack_on_quantize=False)
    q5 = GPTQ(wbits=5, bitpack_on_quantize=True)
    ab = _make_quantizer(
        bitpack_on_quantize=True,
        enable_fused_groups=False,
        quantizers=[q4, q5],
    )

    ab.validate_params()

    assert q4.bitpack_on_quantize is True
    assert q5.bitpack_on_quantize is False


def test_invalid_autobit_bitpack_flag_raises():
    """AutoBit validates its own bitpack flag before child validation."""
    ab = _make_quantizer(bitpack_on_quantize="yes")

    with pytest.raises(ValueError, match="Invalid parameter 'bitpack_on_quantize'"):
        ab.validate_params()
