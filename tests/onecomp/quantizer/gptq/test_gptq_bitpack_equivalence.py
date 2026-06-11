"""Equivalence tests for GPTQ per-module bitpacking (``bitpack_on_quantize``).

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Background
----------
GPTQ supports two storage modes for an already-quantized module:

  - Normal mode        (``bitpack_on_quantize=False``): keep the integer
    ``qweight``/``qzeros`` unpacked; packing happens later, at save time.
  - Per-module bitpack (``bitpack_on_quantize=True``):  pack ``qweight``/``qzeros``
    immediately after the module is quantized (see the packing block in
    ``GPTQ.quantize_layer``, onecomp/quantizer/gptq/_gptq.py).

Bit-packing only changes the *storage layout* of the already-quantized integers:
pack -> unpack is a lossless, bit-exact round-trip. Therefore the *dequantized*
weights reconstructed from each mode (via ``GPTQResult.compute_dequantized_weight``)
must be **bit-identical** -- an exact ``torch.equal``, not merely close. Any
non-zero difference points to a bug in the packing path (e.g. the qzeros v1
``-1/+1`` offset or shape normalization), not quantization noise.

Why ``run_gptq`` is mocked
--------------------------
``run_gptq`` is NOT bit-reproducible run-to-run (its sequential error feedback
accumulates floating-point reduction differences -- confirmed on both CPU and
GPU for ``groupsize=-1``). So quantizing twice -- once per mode -- and comparing
would conflate that nondeterminism with bit-packing differences and make the
check meaningless.

To isolate *only* the packing path while still exercising the **real** inline
packing code in ``quantize_layer``, these tests mock ``run_gptq`` to return one
fixed integer quantization result, then drive the real ``quantize_layer`` in both
modes against that same result. Any difference is then attributable solely to
pack/unpack + the qzeros v1 offset restoration.
"""

from unittest import mock

import pytest
import torch

from onecomp.quantizer.gptq._gptq import GPTQ

# Bit-widths supported by the GPTQ bit-packer. Others (5, 6, 7, ...) cannot be
# packed, so per-module bitpacking is not applicable to them.
WBITS = [2, 3, 4, 8]

# (groupsize, actorder, sym) combinations to sweep.
CONFIGS = [
    pytest.param(-1, False, True, id="perchannel-sym"),
    pytest.param(-1, False, False, id="perchannel-asym"),
    pytest.param(-1, True, False, id="perchannel-asym-actorder"),
    pytest.param(128, False, False, id="grouped-asym"),
    pytest.param(128, True, False, id="grouped-asym-actorder"),
]

OUT_FEATURES = 256
IN_FEATURES = 512

# run_gptq target to patch (module-level function called by quantize_layer).
RUN_GPTQ_TARGET = "onecomp.quantizer.gptq._gptq.run_gptq"


def make_fake_quant_result(wbits, groupsize, actorder, seed=0):
    """Build a deterministic, self-consistent integer quantization result.

    Mirrors the shapes/dtypes that ``run_gptq`` returns:
      - qweight: (out_features, in_features) int32, values in [0, 2^wbits - 1]
      - scales : (out_features, 1) f16 for per-channel, else (num_groups, out) f16
      - qzeros : same shape as scales, int32, values in [0, 2^wbits - 1]
      - perm   : (in_features,) long if actorder else None
    """
    gen = torch.Generator().manual_seed(seed)
    maxq = (1 << wbits) - 1

    qweight = torch.randint(
        0, maxq + 1, (OUT_FEATURES, IN_FEATURES), generator=gen, dtype=torch.int32
    )

    if groupsize == -1:
        scales = (torch.rand(OUT_FEATURES, 1, generator=gen) * 0.1 + 0.01).to(torch.float16)
        qzeros = torch.randint(0, maxq + 1, (OUT_FEATURES, 1), generator=gen, dtype=torch.int32)
    else:
        num_groups = (IN_FEATURES + groupsize - 1) // groupsize
        scales = (torch.rand(num_groups, OUT_FEATURES, generator=gen) * 0.1 + 0.01).to(
            torch.float16
        )
        qzeros = torch.randint(
            0, maxq + 1, (num_groups, OUT_FEATURES), generator=gen, dtype=torch.int32
        )

    perm = torch.randperm(IN_FEATURES, generator=gen) if actorder else None

    return {"qweight": qweight, "scales": scales, "qzeros": qzeros, "perm": perm}


def _fresh_copy(result_dict):
    """Return a deep-ish copy so the two quantize_layer calls cannot alias/mutate
    each other's tensors via the shared mock return value."""
    return {
        k: (v.clone() if torch.is_tensor(v) else v) for k, v in result_dict.items()
    }


def _quantize_both_modes(wbits, groupsize, actorder, sym):
    """Run the real ``quantize_layer`` in both modes against one fixed result."""
    fake = make_fake_quant_result(wbits, groupsize, actorder)
    module = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)

    common = dict(wbits=wbits, groupsize=groupsize, actorder=actorder, sym=sym)
    q_unpacked = GPTQ(bitpack_on_quantize=False, **common)
    q_packed = GPTQ(bitpack_on_quantize=True, **common)
    q_unpacked.validate_params()
    q_packed.validate_params()

    # side_effect hands out a fresh copy on every call so neither result aliases
    # the other's tensors.
    with mock.patch(RUN_GPTQ_TARGET, side_effect=lambda *a, **k: _fresh_copy(fake)):
        result_unpacked = q_unpacked.quantize_layer(module, None, hessian=None)
        result_packed = q_packed.quantize_layer(module, None, hessian=None)

    return result_unpacked, result_packed


@pytest.mark.parametrize("groupsize, actorder, sym", CONFIGS)
@pytest.mark.parametrize("wbits", WBITS)
def test_bitpack_on_quantize_matches_unpacked_dequant(wbits, groupsize, actorder, sym):
    """Per-module bitpacking must reconstruct bit-identical dequantized weights.

    Drives the real ``quantize_layer`` packing path (bitpack_on_quantize=True)
    and the unpacked path against one fixed quantization result, then compares
    the dequantized weights. The match must be exact, not approximate.
    """
    result_unpacked, result_packed = _quantize_both_modes(wbits, groupsize, actorder, sym)

    # Confirm the two results genuinely took different storage paths.
    assert result_unpacked.qweight_is_packed is False
    assert result_unpacked.qzeros_is_packed is False
    assert result_packed.qweight_is_packed is True
    assert result_packed.qzeros_is_packed is True

    w_unpacked = result_unpacked.compute_dequantized_weight()
    w_packed = result_packed.compute_dequantized_weight()

    assert w_unpacked.shape == w_packed.shape == (OUT_FEATURES, IN_FEATURES)
    assert torch.equal(w_unpacked, w_packed), (
        f"wbits={wbits}, groupsize={groupsize}, actorder={actorder}, sym={sym}: "
        f"dequantized weights differ; max abs diff="
        f"{(w_unpacked.float() - w_packed.float()).abs().max().item():.3e}. "
        "Bit-packing is a lossless storage transform, so any difference indicates "
        "a bug in the pack/unpack path (qzeros v1 -1/+1 offset or shape handling)."
    )


@pytest.mark.parametrize("groupsize, actorder, sym", CONFIGS)
@pytest.mark.parametrize("wbits", WBITS)
def test_bitpack_preserves_integer_weights(wbits, groupsize, actorder, sym):
    """The packed qweight must unpack back to the original integer weights.

    Guards the lossless round-trip at the result level (independent of scales /
    zero-points), so a corruption in the qweight pack/unpack path is caught even
    if it happened to cancel out in the dequantized comparison above.
    """
    from onecomp.quantizer.gptq.gptq_layer import unpack_int_weights

    result_unpacked, result_packed = _quantize_both_modes(wbits, groupsize, actorder, sym)

    restored = unpack_int_weights(
        result_packed.qweight.to(torch.int32),
        wbits,
        result_packed.qweight_original_shape,
    )
    assert torch.equal(restored, result_unpacked.qweight)
