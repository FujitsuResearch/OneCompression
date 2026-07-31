"""Regression tests for the qzero=0 corruption fix in GPTQLinear.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

These tests exist to pin exactly two changes in
``onecomp/quantizer/gptq/gptq_layer.py``:

1. ``_pack_rows`` masks each value with ``(1 << wbits) - 1`` before shift/OR.
   Without the mask, a signed ``-1`` (which occurs in the AutoGPTQ v1 encoding
   ``qzero - 1`` when ``qzero == 0``) OR's its sign-extended high bits into
   neighboring slots of the packed INT32 word, corrupting every slot it shares
   a word with.

2. ``GPTQLinear.forward`` applies the same mask after the v1 ``+1``
   restoration. Without the mask, ``stored = 2^wbits - 1`` (the encoding of
   ``qzero == 0``) overflows to ``2^wbits`` instead of wrapping to ``0``.
   The restoration is gated on ``checkpoint_format != "gptq_v2"``; the v2 path
   must remain untouched.

Test Contents
-------------
test_pack_rows_does_not_corrupt_neighbors_on_signed_minus_one:
    Pins fix 1. Parametrized over wbits in {2, 3, 4, 8}. For every slot
    position within a packed INT32 word, placing ``-1`` at that slot must
    round-trip via ``pack_zeros``/``unpack_zeros`` as ``2^wbits - 1`` without
    disturbing its neighbors. Exhaustive per-position iteration covers both
    the same-word branch and the 3-bit cross-word branch.

test_forward_restores_qzero_zero:
    Pins fix 2 (and fix 1 in combination when ``pack_weights=True``).
    Parametrized over wbits in {2, 3, 4, 8} and ``pack_weights`` in
    {True, False}. Builds a ``GPTQLinear`` with ``qzero=0`` in the first 4
    output columns and verifies forward output matches a reference dequant
    within FP16 tolerance. The ``pack_weights=False`` cases isolate fix 2
    from fix 1.

test_forward_v2_checkpoint_leaves_zeros_unshifted:
    Pins the ``_v1`` branch gate in fix 2. Parametrized over wbits in
    {2, 3, 4, 8}. Loads a ``gptq_v2`` checkpoint (qzeros stored as raw
    integers, no ``-1`` offset) via ``from_saved_state`` and verifies forward
    does not apply ``+1`` to the zeros. Guards against a future refactor
    accidentally shifting every v2 output by one quantization bin.

test_forward_v1_saved_state_restores_qzero_zero:
    Covers the actual production path: ``save_quantized_model`` stores qzeros
    with the v1 ``-1`` offset, then ``from_saved_state(checkpoint_format="gptq")``
    reconstructs the layer. Verifies that forward correctly recovers qzero=0
    (stored as ``2^wbits - 1``) and that the output matches a reference
    dequantization. Parametrized over wbits in {2, 3, 4, 8}. Protects against
    constructor-side and loader-side logic diverging after a refactor.

test_forward_restores_zero_columns_exactly:
    Direct implementation-level pin for fix 2. Uses ``qweight=0``, ``scale=1``,
    and a one-hot input so that each output column becomes exactly
    ``-restored_zero`` inside ``GPTQLinear.forward``. This lets the test assert
    that bug columns restore to exactly 0, rather than recomputing the wrap
    formula in the test body. Parametrized over wbits in {2, 3, 4, 8} and
    ``pack_weights`` in {True, False}, so both packed and unpacked forward
    branches are covered.
"""

import pytest
import torch

from onecomp.quantizer.gptq.gptq_layer import (
    GPTQLinear,
    _normalize_wbits,
    is_packable_wbits,
    normalize_scale_zero,
    pack_int_weights,
    pack_zeros,
    unpack_zeros,
)


def _build_layer(wbits, pack_weights, device, *, with_qzero_zero=True):
    """Construct a small GPTQLinear with optional qzero=0 columns.

    Shapes (in_features=64, out_features=32, groupsize=32) satisfy the
    divisibility constraints of every packable bit width, including the
    3-bit cross-word packing path.
    """
    torch.manual_seed(0)
    groupsize = 32
    in_features = groupsize * 2
    out_features = 32
    num_groups = in_features // groupsize
    vmax = (1 << wbits) - 1

    qweight = torch.randint(
        0, vmax + 1, (out_features, in_features), dtype=torch.int32, device=device
    )
    scales = torch.rand(num_groups, out_features, device=device, dtype=torch.float32) * 0.05 + 0.02
    zeros = torch.full(
        (num_groups, out_features), float(min(vmax, 5)), device=device, dtype=torch.float32
    )
    if with_qzero_zero:
        # Exercise the v1 -1 offset round trip through pack/unpack_in_place.
        zeros[:, :4] = 0.0

    layer = GPTQLinear(
        in_features=in_features,
        out_features=out_features,
        wbits=wbits,
        groupsize=groupsize,
        actorder=False,
        quantized_weight=qweight,
        scale=scales,
        zero=zeros,
        bias=None,
        device=device,
        pack_weights=pack_weights,
        use_gemlite=False,
    )
    x = torch.randn(2, in_features, device=device, dtype=torch.float16)
    return layer, x


@pytest.mark.parametrize("wbits", [2, 3, 4, 8])
def test_pack_rows_does_not_corrupt_neighbors_on_signed_minus_one(wbits):
    """Fix 1: at every slot position within a packed INT32 word, placing ``-1``
    must round-trip as ``2^wbits - 1`` without disturbing the other slots.

    Iterating over every slot position exhaustively covers both the same-word
    branch and the 3-bit cross-word branch (which straddles two INT32 words at
    slot positions 10 and 21). This catches a hypothetical half-applied fix
    that masks only some positions.
    """
    pack_factor = 32 if wbits == 3 else 32 // wbits
    cols = 3
    mask = (1 << wbits) - 1
    other = min(mask - 1, 5)  # any valid non-zero representative

    for pos in range(pack_factor):
        values = torch.full((pack_factor, cols), other, dtype=torch.int32)
        values[pos, :] = -1  # triggers the sign-extension bug at this slot

        packed = pack_zeros(values.t().contiguous(), wbits).t().contiguous()
        unpacked = unpack_zeros(packed.t().contiguous(), wbits, pack_factor).t().contiguous()

        expected = values.clone()
        expected[pos, :] = mask  # -1 mod 2^wbits

        assert torch.equal(unpacked, expected), (
            f"wbits={wbits}, pos={pos}: neighbor corruption, " f"got={unpacked[:, 0].tolist()}"
        )


@pytest.mark.parametrize("wbits", [2, 3, 4, 8])
@pytest.mark.parametrize("pack_weights", [True, False])
def test_forward_restores_qzero_zero(wbits, pack_weights):
    """Fix 2 (and, for ``pack_weights=True``, fix 1 in combination):
    ``GPTQLinear.forward`` must dequantize ``qzero=0`` columns as
    ``scale * qweight``.

    Before the fix the output diverged by ``O(scale * 2^wbits)`` either
    because neighbor slots were corrupted (pack path) or because
    ``zeros + 1`` overflowed instead of wrapping modularly.
    """
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    groupsize = 32
    in_features = groupsize * 2
    out_features = 32
    num_groups = in_features // groupsize
    vmax = (1 << wbits) - 1

    qweight = torch.randint(
        0, vmax + 1, (out_features, in_features), dtype=torch.int32, device=device
    )
    scales = torch.rand(num_groups, out_features, device=device, dtype=torch.float32) * 0.05 + 0.02
    # First 4 output columns trigger the bug (qzero=0); the rest use any valid non-zero.
    zeros = torch.full(
        (num_groups, out_features), float(min(vmax, 5)), device=device, dtype=torch.float32
    )
    zeros[:, :4] = 0.0

    layer = GPTQLinear(
        in_features=in_features,
        out_features=out_features,
        wbits=wbits,
        groupsize=groupsize,
        actorder=False,
        quantized_weight=qweight,
        scale=scales,
        zero=zeros,
        bias=None,
        device=device,
        pack_weights=pack_weights,
        use_gemlite=False,
    )

    ref_w = torch.zeros(out_features, in_features, dtype=torch.float32, device=device)
    for g in range(num_groups):
        s, e = g * groupsize, (g + 1) * groupsize
        ref_w[:, s:e] = scales[g].unsqueeze(1) * (qweight[:, s:e].float() - zeros[g].unsqueeze(1))

    x = torch.randn(2, in_features, device=device, dtype=torch.float16)
    ref_out = torch.nn.functional.linear(x, ref_w.to(torch.float16))
    layer_out = layer(x)
    assert torch.allclose(ref_out, layer_out, rtol=0.02, atol=0.3), (
        f"wbits={wbits}, pack_weights={pack_weights}: "
        f"max |diff|={(ref_out - layer_out).abs().max().item()}"
    )


@pytest.mark.parametrize("wbits", [2, 3, 4, 8])
def test_forward_v2_checkpoint_leaves_zeros_unshifted(wbits):
    """Fix 2 gates the modular ``+1`` restoration on ``checkpoint_format !=
    "gptq_v2"``. A v2 checkpoint stores qzeros as raw integers (no ``-1``
    offset at save time), so forward must consume them as-is; any accidental
    ``+1`` would shift every output by one quantization bin.
    """
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    groupsize = 32
    in_features = groupsize * 2
    out_features = 32
    num_groups = in_features // groupsize
    vmax = (1 << wbits) - 1

    qweight = torch.randint(
        0, vmax + 1, (out_features, in_features), dtype=torch.int32, device=device
    )
    scales = torch.rand(num_groups, out_features, device=device, dtype=torch.float32) * 0.05 + 0.02
    # v2 convention: stored = raw qzero, with no -1 offset. All values must lie
    # in [0, vmax]; we pick values in [1, vmax] to keep it independent of fix 2.
    zeros_raw = torch.randint(
        1, vmax + 1, (num_groups, out_features), dtype=torch.int32, device=device
    )

    state_dict = {
        "qweight": pack_int_weights(qweight, wbits),
        "scales": scales.to(torch.float16),
        "qzeros": pack_zeros(zeros_raw, wbits),
        "g_idx": (torch.arange(in_features, device=device, dtype=torch.int32) // groupsize),
    }
    layer = GPTQLinear.from_saved_state(
        state_dict,
        in_features=in_features,
        out_features=out_features,
        wbits=wbits,
        groupsize=groupsize,
        actorder=False,
        empty=False,
        checkpoint_format="gptq_v2",
    )

    ref_w = torch.zeros(out_features, in_features, dtype=torch.float32, device=device)
    for g in range(num_groups):
        s, e = g * groupsize, (g + 1) * groupsize
        ref_w[:, s:e] = scales[g].unsqueeze(1) * (
            qweight[:, s:e].float() - zeros_raw[g].unsqueeze(1).float()
        )

    x = torch.randn(2, in_features, device=device, dtype=torch.float16)
    ref_out = torch.nn.functional.linear(x, ref_w.to(torch.float16))
    layer_out = layer(x)
    assert torch.allclose(ref_out, layer_out, rtol=0.02, atol=0.3), (
        f"wbits={wbits}: v2 path drifted "
        f"(max |diff|={(ref_out - layer_out).abs().max().item()})"
    )


@pytest.mark.parametrize("wbits", [2, 3, 4, 8])
def test_forward_v1_saved_state_restores_qzero_zero(wbits):
    """Fix 2 via the ``from_saved_state(checkpoint_format="gptq")`` path.

    Manually builds v1-style saved tensors and feeds them to
    ``from_saved_state``. This covers the loader-side contract for a gptq v1
    checkpoint: qzeros are already packed and already stored with the v1 ``-1``
    offset. Verifies that ``forward`` correctly restores qzero=0 (packed as
    ``2^wbits - 1``) to 0, not to ``2^wbits``. Protects against constructor-
    side and loader-side logic diverging after a future refactor.
    """
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    groupsize = 32
    in_features = groupsize * 2
    out_features = 32
    num_groups = in_features // groupsize
    vmax = (1 << wbits) - 1

    qweight = torch.randint(
        0, vmax + 1, (out_features, in_features), dtype=torch.int32, device=device
    )
    scales = torch.rand(num_groups, out_features, device=device, dtype=torch.float32) * 0.05 + 0.02
    # First 4 output columns trigger the bug (qzero=0); the rest use a valid non-zero.
    zeros_raw = torch.full(
        (num_groups, out_features), float(min(vmax, 5)), device=device, dtype=torch.float32
    )
    zeros_raw[:, :4] = 0.0

    # Simulate the v1 offset applied by the GPTQLinear constructor / save_quantized_model:
    # stored = round(raw_zero) - 1; for raw_zero=0, stored = -1 which packs as 2^wbits-1.
    zeros_v1 = zeros_raw.round().to(torch.int32) - 1

    state_dict = {
        "qweight": pack_int_weights(qweight, wbits),
        "scales": scales.to(torch.float16),
        "qzeros": pack_zeros(zeros_v1, wbits),
        "g_idx": torch.arange(in_features, device=device, dtype=torch.int32) // groupsize,
    }
    layer = GPTQLinear.from_saved_state(
        state_dict,
        in_features=in_features,
        out_features=out_features,
        wbits=wbits,
        groupsize=groupsize,
        actorder=False,
        empty=False,
        checkpoint_format="gptq",
    )

    ref_w = torch.zeros(out_features, in_features, dtype=torch.float32, device=device)
    for g in range(num_groups):
        s, e = g * groupsize, (g + 1) * groupsize
        ref_w[:, s:e] = scales[g].unsqueeze(1) * (
            qweight[:, s:e].float() - zeros_raw[g].unsqueeze(1)
        )

    x = torch.randn(2, in_features, device=device, dtype=torch.float16)
    ref_out = torch.nn.functional.linear(x, ref_w.to(torch.float16))
    layer_out = layer(x)
    assert torch.allclose(ref_out, layer_out, rtol=0.02, atol=0.3), (
        f"wbits={wbits}: v1 from_saved_state path drifted "
        f"(max |diff|={(ref_out - layer_out).abs().max().item()})"
    )


@pytest.mark.parametrize("wbits", [2, 3, 4, 8])
@pytest.mark.parametrize("pack_weights", [True, False])
def test_forward_restores_zero_columns_exactly(wbits, pack_weights):
    """Direct implementation-level pin for fix 2.

    Choose ``qweight=0``, ``scale=1``, and a one-hot input so that each output
    column equals exactly ``-restored_zero`` inside ``GPTQLinear.forward``.
    This makes the forward result an exact probe of the restoration logic:
    bug columns must produce 0 and control columns must produce -1.

    Unlike the previous helper-style check, this test does not recompute the
    wrap formula in the test body. It asserts observable outputs from the
    production implementation for both packed and unpacked qzeros paths.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    in_features = 32  # 3-bit weight packing requires in_features % 32 == 0
    out_features = 32  # 3-bit zero packing requires out_features % 32 == 0
    groupsize = 32

    qweight = torch.zeros((out_features, in_features), dtype=torch.int32, device=device)
    scales = torch.ones((1, out_features), dtype=torch.float32, device=device)

    # First 4 columns are the bug columns: raw qzero=0 must restore exactly to 0.
    # Remaining columns use raw qzero=1 as a control, which should dequantize to -1
    # under the chosen qweight/scale/input setup.
    zeros = torch.ones((1, out_features), dtype=torch.float32, device=device)
    zeros[:, :4] = 0.0

    layer = GPTQLinear(
        in_features=in_features,
        out_features=out_features,
        wbits=wbits,
        groupsize=groupsize,
        actorder=False,
        quantized_weight=qweight,
        scale=scales,
        zero=zeros,
        bias=None,
        device=device,
        pack_weights=pack_weights,
        use_gemlite=False,
    )

    # Probe a single input column so each output equals one dequantized weight value.
    x = torch.zeros((1, in_features), device=device, dtype=torch.float16)
    x[0, 0] = 1.0

    out = layer(x).float().squeeze(0)
    assert torch.all(out[:4] == 0), (
        f"wbits={wbits}, pack_weights={pack_weights}: expected bug columns to restore "
        f"to exact zero, got {out[:4].tolist()}"
    )
    assert torch.all(out[4:] == -1), (
        f"wbits={wbits}, pack_weights={pack_weights}: expected control columns to "
        f"dequantize to -1, got sample {out[4:8].tolist()}"
    )


@pytest.mark.parametrize("wbits", [2, 3, 4, 8])
def test_pack_unpack_in_place_roundtrip_preserves_forward(wbits):
    """A pack → unpack → pack round trip must leave forward output unchanged.

    The dequantization math is identical in the packed and unpacked branches,
    so the output should be (near) bit-identical. A broken pack/unpack would
    corrupt qweight/qzeros and shift the output by ``O(scale * 2^wbits)``.
    Columns with qzero=0 are included to exercise the v1 ``-1`` offset.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer, x = _build_layer(wbits, pack_weights=True, device=device)

    assert layer._weight_is_packed is True
    ref = layer(x).clone()

    layer.unpack_in_place()
    assert layer._weight_is_packed is False
    out_unpacked = layer(x)
    assert (ref - out_unpacked).abs().max().item() < 1e-3, (
        f"wbits={wbits}: unpack_in_place changed forward output "
        f"(max |diff|={(ref - out_unpacked).abs().max().item()})"
    )

    layer.pack_in_place()
    assert layer._weight_is_packed is True
    out_repacked = layer(x)
    assert (ref - out_repacked).abs().max().item() < 1e-3, (
        f"wbits={wbits}: pack_in_place after unpack changed forward output "
        f"(max |diff|={(ref - out_repacked).abs().max().item()})"
    )


@pytest.mark.parametrize("wbits", [2, 3, 4, 8])
def test_pack_in_place_noop_when_already_packed(wbits):
    """``pack_in_place`` on an already-packed layer must be a no-op: the buffers
    are left untouched (same tensor objects) and the flag stays True."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer, _ = _build_layer(wbits, pack_weights=True, device=device)

    qweight_before, qzeros_before = layer.qweight, layer.qzeros
    layer.pack_in_place()

    assert layer._weight_is_packed is True
    assert layer.qweight is qweight_before
    assert layer.qzeros is qzeros_before


@pytest.mark.parametrize("wbits", [2, 3, 4, 8])
def test_unpack_in_place_noop_when_already_unpacked(wbits):
    """``unpack_in_place`` on an already-unpacked layer must be a no-op."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer, _ = _build_layer(wbits, pack_weights=False, device=device)

    assert layer._weight_is_packed is False
    qweight_before, qzeros_before = layer.qweight, layer.qzeros
    layer.unpack_in_place()

    assert layer._weight_is_packed is False
    assert layer.qweight is qweight_before
    assert layer.qzeros is qzeros_before


def test_pack_in_place_noop_for_non_packable_wbits():
    """A non-packable bit width (JointQ 1-bit) has no AutoGPTQ packing layout,
    so ``pack_in_place`` must safely leave the layer unpacked."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1-bit layers are constructed unpacked (pack_weights=False); no qzero=0
    # columns since the v1 offset is orthogonal to this guard.
    layer, _ = _build_layer(1, pack_weights=False, device=device, with_qzero_zero=False)

    assert layer._weight_is_packed is False
    qweight_before, qzeros_before = layer.qweight, layer.qzeros
    layer.pack_in_place()

    assert layer._weight_is_packed is False, "pack_in_place must not pack a non-packable wbits"
    assert layer.qweight is qweight_before
    assert layer.qzeros is qzeros_before


def test_is_packable_wbits():
    """The shared predicate accepts supported integral values without rounding."""
    assert all(is_packable_wbits(w) for w in (2, 3, 4, 8, 2.0, 3.0, 4.0, 8.0))
    assert not is_packable_wbits(1)
    assert not is_packable_wbits(2.5)
    assert not is_packable_wbits(16)


def test_normalize_wbits_accepts_integral_rejects_others():
    """Integral values normalize to int; all other inputs are rejected."""

    class IntSubclass(int):
        pass

    assert _normalize_wbits(2) == 2 and type(_normalize_wbits(2)) is int
    normalized = _normalize_wbits(2.0)
    assert normalized == 2 and type(normalized) is int
    assert type(_normalize_wbits(IntSubclass(2))) is int
    for bad in (2.5, True, False, "2", float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _normalize_wbits(bad)


def test_normalize_scale_zero_layouts():
    """Every scale/zero layout OneComp produces maps to (num_groups, out_features)."""
    out_features = 4
    values = torch.arange(out_features, dtype=torch.float32)

    # (out_features,) → (1, out_features)
    normalized = normalize_scale_zero(values, out_features)
    assert normalized.shape == (1, out_features)
    assert torch.equal(normalized[0], values)

    # (out_features, 1) → (1, out_features)
    normalized = normalize_scale_zero(values.reshape(out_features, 1), out_features)
    assert normalized.shape == (1, out_features)
    assert torch.equal(normalized[0], values)

    # (num_groups, out_features) is already normalized
    grouped = torch.arange(2 * out_features, dtype=torch.float32).reshape(2, out_features)
    assert normalize_scale_zero(grouped, out_features) is grouped


def test_normalize_scale_zero_transposes_only_the_out_features_column():
    """``out_features`` alone decides which 2-D layout is a per-channel column.

    The transpose is deliberately asymmetric: ``(out_features, 1)`` is a
    per-channel column and must be transposed, while the already-normalized
    ``(1, out_features)`` must not be. Shapes matching neither documented
    layout pass through unvalidated.
    """
    out_features = 4

    already_normalized = torch.zeros(1, out_features)
    assert normalize_scale_zero(already_normalized, out_features) is already_normalized

    # Same (n, 1) shape as a per-channel column, but n != out_features → left alone.
    not_per_channel = torch.zeros(2, 1)
    assert normalize_scale_zero(not_per_channel, out_features) is not_per_channel

    unknown = torch.zeros(3, 7)
    assert normalize_scale_zero(unknown, out_features) is unknown


def _minimal_gptq_kwargs(shape_wbits, device="cpu"):
    """Constructor kwargs (minus ``wbits``) sized for an integral bit width."""
    groupsize = 32
    in_features = groupsize * 2
    out_features = 32
    num_groups = in_features // groupsize
    vmax = (1 << shape_wbits) - 1
    return {
        "in_features": in_features,
        "out_features": out_features,
        "groupsize": groupsize,
        "actorder": False,
        "quantized_weight": torch.zeros(
            out_features, in_features, dtype=torch.int32, device=device
        ),
        "scale": torch.ones(num_groups, out_features, dtype=torch.float32, device=device),
        "zero": torch.full(
            (num_groups, out_features), float(min(vmax, 5)), dtype=torch.float32, device=device
        ),
        "bias": None,
        "device": device,
        "pack_weights": True,
        "use_gemlite": False,
    }


def test_gptqlinear_init_accepts_integral_float_wbits():
    """``GPTQLinear(wbits=2.0)`` constructs and stores a builtin int width."""
    layer = GPTQLinear(wbits=2.0, **_minimal_gptq_kwargs(2))
    assert layer.wbits == 2
    assert type(layer.wbits) is int


def test_gptqlinear_init_rejects_non_integral_float_wbits():
    """``GPTQLinear(wbits=2.5)`` raises before the ``1 << wbits`` bit-shift."""
    with pytest.raises(ValueError):
        GPTQLinear(wbits=2.5, **_minimal_gptq_kwargs(2))


def _packed_state_dict(shape_wbits, device="cpu"):
    """Build a v1-style packed state_dict for an integral bit width."""
    groupsize = 32
    in_features = groupsize * 2
    out_features = 32
    num_groups = in_features // groupsize
    vmax = (1 << shape_wbits) - 1
    qweight = torch.zeros(out_features, in_features, dtype=torch.int32, device=device)
    zeros_raw = torch.full(
        (num_groups, out_features), min(vmax, 5), dtype=torch.int32, device=device
    )
    state_dict = {
        "qweight": pack_int_weights(qweight, shape_wbits),
        "scales": torch.ones(num_groups, out_features, dtype=torch.float16, device=device),
        "qzeros": pack_zeros(zeros_raw, shape_wbits),
        "g_idx": torch.arange(in_features, device=device, dtype=torch.int32) // groupsize,
    }
    return state_dict, in_features, out_features, groupsize


def test_from_saved_state_accepts_integral_float_wbits():
    """``from_saved_state(wbits=2.0)`` normalizes to a builtin int width."""
    state_dict, in_features, out_features, groupsize = _packed_state_dict(2)
    layer = GPTQLinear.from_saved_state(
        state_dict,
        in_features=in_features,
        out_features=out_features,
        wbits=2.0,
        groupsize=groupsize,
        actorder=False,
        empty=False,
        checkpoint_format="gptq",
    )
    assert layer.wbits == 2
    assert type(layer.wbits) is int


def test_from_saved_state_rejects_non_integral_float_wbits():
    """``from_saved_state(wbits=2.5)`` raises like the constructor."""
    state_dict, in_features, out_features, groupsize = _packed_state_dict(2)
    with pytest.raises(ValueError):
        GPTQLinear.from_saved_state(
            state_dict,
            in_features=in_features,
            out_features=out_features,
            wbits=2.5,
            groupsize=groupsize,
            actorder=False,
            empty=False,
            checkpoint_format="gptq",
        )
