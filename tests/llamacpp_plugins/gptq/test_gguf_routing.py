"""Unit tests for the llama.cpp mixed-bit GPTQ plugin (no model / GPU needed).

Covers:
  - select_gguf_route: per-module (bits, sym) -> GGUF type + routing class
  - pack_gptq_linear: 4/8-bit GPTQ codes pack *bit-exactly* into GGUF legacy
    blocks (verified against gguf's own dequantizer)
  - _route_layer: act-order layers fall back to a K-quant

Copyright 2025-2026 Fujitsu Ltd.
"""

import numpy as np
import pytest

try:
    import gguf.quants as GQ
    from gguf.constants import GGMLQuantizationType as T

    from llamacpp_plugins.gptq.constants import (
        ROUTE_DENSE,
        ROUTE_DIRECT,
        ROUTE_KQUANT,
        select_gguf_route,
    )
    from onecomp.cpu.export.blocks import pack_gptq_linear

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

_needs = pytest.mark.skipif(not _HAS_DEPS, reason="gguf / onecomp not installed")


@_needs
@pytest.mark.parametrize(
    "bits,sym,route,tname",
    [
        (4, True, ROUTE_DIRECT, "Q4_0"),
        (4, False, ROUTE_DIRECT, "Q4_1"),
        (8, True, ROUTE_DIRECT, "Q8_0"),
        (2, True, ROUTE_KQUANT, "Q2_K"),
        (3, True, ROUTE_KQUANT, "Q3_K"),
        (8, False, ROUTE_DENSE, "F16"),
    ],
)
def test_select_gguf_route(bits, sym, route, tname):
    r, gt = select_gguf_route(bits, sym)
    assert r == route
    assert gt.name == tname


def _dequant_to_logical(packed, qtype, out_f, in_f):
    w = GQ.dequantize(np.ascontiguousarray(packed), qtype).astype(np.float32)
    return w.reshape(out_f, in_f)


@_needs
def test_pack_q4_0_bit_exact():
    rng = np.random.default_rng(0)
    out_f, in_f, gs = 8, 64, 32
    ng = in_f // gs
    q = rng.integers(0, 16, size=(out_f, in_f)).astype(np.int32)  # 4-bit codes
    scales = (rng.random((ng, out_f)).astype(np.float32) + 0.1) * 0.05
    zeros = np.full((ng, out_f), 8.0, dtype=np.float32)  # symmetric

    packed, qtype = pack_gptq_linear(q, scales, zeros, wbits=4, sym=True, groupsize=gs)
    assert qtype == T.Q4_0
    w = _dequant_to_logical(packed, qtype, out_f, in_f)

    g_idx = np.arange(in_f) // gs
    # GGUF stores the block scale as fp16, so the lossless reference must too.
    d16 = scales.astype(np.float16).astype(np.float32)
    ref = d16[g_idx, :].T * (q.astype(np.float32) - 8.0)
    assert np.max(np.abs(w - ref)) == 0.0


@_needs
def test_pack_q4_1_bit_exact():
    rng = np.random.default_rng(1)
    out_f, in_f, gs = 8, 64, 32
    ng = in_f // gs
    q = rng.integers(0, 16, size=(out_f, in_f)).astype(np.int32)
    scales = (rng.random((ng, out_f)).astype(np.float32) + 0.1) * 0.05
    zeros = rng.integers(0, 16, size=(ng, out_f)).astype(np.float32)  # asymmetric

    packed, qtype = pack_gptq_linear(q, scales, zeros, wbits=4, sym=False, groupsize=gs)
    assert qtype == T.Q4_1
    w = _dequant_to_logical(packed, qtype, out_f, in_f)

    g_idx = np.arange(in_f) // gs
    # Q4_1 dequant = fp16(d) * q + fp16(-d*z); mirror the fp16 storage exactly.
    d = scales[g_idx, :].T
    d16 = d.astype(np.float16).astype(np.float32)
    m16 = (-(d * zeros[g_idx, :].T)).astype(np.float16).astype(np.float32)
    ref = d16 * q.astype(np.float32) + m16
    assert np.max(np.abs(w - ref)) == 0.0


@_needs
def test_pack_q8_0_bit_exact():
    rng = np.random.default_rng(2)
    out_f, in_f, gs = 8, 64, 32
    ng = in_f // gs
    q = rng.integers(0, 256, size=(out_f, in_f)).astype(np.int32)  # 8-bit codes
    scales = (rng.random((ng, out_f)).astype(np.float32) + 0.1) * 0.01
    zeros = np.full((ng, out_f), 128.0, dtype=np.float32)

    packed, qtype = pack_gptq_linear(q, scales, zeros, wbits=8, sym=True, groupsize=gs)
    assert qtype == T.Q8_0
    w = _dequant_to_logical(packed, qtype, out_f, in_f)

    g_idx = np.arange(in_f) // gs
    d16 = scales.astype(np.float16).astype(np.float32)
    ref = d16[g_idx, :].T * (q.astype(np.float32) - 128.0)
    assert np.max(np.abs(w - ref)) == 0.0


@_needs
def test_q8_0_requires_zero_128():
    rng = np.random.default_rng(3)
    from onecomp.cpu.export.blocks import UnsupportedGPTQLayout

    q = rng.integers(0, 256, size=(8, 64)).astype(np.int32)
    scales = np.ones((2, 8), dtype=np.float32) * 0.01
    zeros = np.full((2, 8), 100.0, dtype=np.float32)  # not 128 -> must reject
    with pytest.raises(UnsupportedGPTQLayout):
        pack_gptq_linear(q, scales, zeros, wbits=8, sym=True, groupsize=32)


@_needs
def test_actorder_falls_back_to_kquant():
    from llamacpp_plugins.gptq.llamacpp_plugin import ModulePlan, _route_layer

    class _L:  # minimal stand-in for GPTQLayer
        wbits = 4
        sym = True
        actorder = True

    route, gtype_name, reason = _route_layer(_L())
    assert route == ROUTE_KQUANT
    assert gtype_name == "Q4_K"
    assert "actorder" in reason


class _Layer:
    """Minimal GPTQLayer stand-in for routing-feasibility tests."""

    def __init__(self, wbits, sym, zeros_val, in_features=64, groupsize=32):
        self.wbits = wbits
        self.sym = sym
        self.actorder = False
        self.in_features = in_features
        self.groupsize = groupsize
        self.zeros = np.full((in_features // groupsize, 8), float(zeros_val), dtype=np.float32)


@_needs
def test_route_direct_when_feasible():
    from llamacpp_plugins.gptq.llamacpp_plugin import _route_layer

    route, gt, _ = _route_layer(_Layer(4, True, 8))  # Q4_0 needs zero==8
    assert (route, gt) == (ROUTE_DIRECT, "Q4_0")
    route, gt, _ = _route_layer(_Layer(8, True, 128))  # Q8_0 needs zero==128
    assert (route, gt) == (ROUTE_DIRECT, "Q8_0")
    route, gt, _ = _route_layer(_Layer(4, False, 5))  # Q4_1: any zero ok
    assert (route, gt) == (ROUTE_DIRECT, "Q4_1")


@_needs
def test_route_demotes_infeasible_direct_to_kquant():
    from llamacpp_plugins.gptq.llamacpp_plugin import _route_layer

    # 4-bit sym but zero point != 8 -> cannot pack Q4_0 losslessly -> K-quant.
    route, gt, reason = _route_layer(_Layer(4, True, 7))
    assert route == ROUTE_KQUANT and gt == "Q4_K" and "infeasible" in reason
    # 8-bit sym with zero != 128 -> Q6_K fallback.
    route, gt, _ = _route_layer(_Layer(8, True, 100))
    assert route == ROUTE_KQUANT and gt == "Q6_K"
    # input dim not a multiple of 32 -> demote even with correct zero point.
    route, gt, _ = _route_layer(_Layer(4, True, 8, in_features=48, groupsize=16))
    assert route == ROUTE_KQUANT
