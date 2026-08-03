"""Tests for fused-group validation in manual assignment mode.

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

import logging

import pytest
import torch

from onecomp.quantizer.autobit._autobit import AutoBitQuantizer
from onecomp.quantizer.autobit.dbf_fallback import inject_dbf
from onecomp.quantizer.dbf import DBF
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


def test_bitpack_flag_raises_for_unsupported_gptq_wbits():
    """Unsupported GPTQ wbits raise when AutoBit bitpack is enabled."""
    q4 = GPTQ(wbits=4, bitpack_on_quantize=False)
    q5 = GPTQ(wbits=5, bitpack_on_quantize=False)
    ab = _make_quantizer(
        bitpack_on_quantize=True,
        enable_fused_groups=False,
        quantizers=[q4, q5],
    )

    with pytest.raises(ValueError, match="bitpack_on_quantize=True"):
        ab.validate_params()

    assert q4.bitpack_on_quantize is True
    assert q5.bitpack_on_quantize is True


def test_bitpack_disabled_allows_unsupported_gptq_wbits():
    """Unsupported GPTQ wbits are allowed when AutoBit bitpack is disabled."""
    q5 = GPTQ(wbits=5, bitpack_on_quantize=True)
    ab = _make_quantizer(
        bitpack_on_quantize=False,
        enable_fused_groups=False,
        quantizers=[q5],
    )

    ab.validate_params()

    assert q5.bitpack_on_quantize is False


def test_invalid_autobit_bitpack_flag_raises():
    """AutoBit rejects a non-bool bitpack_on_quantize value."""
    ab = _make_quantizer(enable_fused_groups=False, bitpack_on_quantize="yes")
    with pytest.raises(ValueError, match="Invalid parameter 'bitpack_on_quantize'"):
        ab.validate_params()


@pytest.mark.parametrize("flag", [True, False])
def test_bitpack_flag_propagates_to_dbf_candidate(flag):
    """AutoBit bitpack flag is synced onto DBF candidates during validation."""
    # Start the DBF candidate with the opposite flag to prove it gets synced.
    dbf = DBF(target_bits=1.5, bitpack_on_quantize=not flag)
    ab = _make_quantizer(
        bitpack_on_quantize=flag,
        enable_fused_groups=False,
        quantizers=[dbf],
    )

    ab.validate_params()

    assert dbf.bitpack_on_quantize is flag


@pytest.mark.parametrize("flag", [True, False])
def test_inject_dbf_propagates_bitpack_on_quantize(flag):
    """DBF fallback quantizers created by inject_dbf inherit the bitpack flag."""
    # in_features=64 keeps 1-bit GPTQ effective bpw (1 + 17/64 ≈ 1.27) below the
    # 2.0 threshold so inject_dbf actually creates a DBF fallback.
    lin = torch.nn.Linear(64, 8, bias=False)
    gptq = GPTQ(wbits=1, bitpack_on_quantize=False)
    quantizers = [gptq]
    assignments = [("model.layers.0.self_attn.q_proj", lin, gptq)]

    inject_dbf(
        assignments,
        quantizers,
        threshold=2.0,
        logger=logging.getLogger(__name__),
        bitpack_on_quantize=flag,
    )

    dbf_fallbacks = [q for q in quantizers if isinstance(q, DBF)]
    assert dbf_fallbacks, "expected inject_dbf to create a DBF fallback quantizer"
    assert all(q.bitpack_on_quantize is flag for q in dbf_fallbacks)
