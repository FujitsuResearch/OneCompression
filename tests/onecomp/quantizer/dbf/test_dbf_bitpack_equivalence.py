"""Equivalence tests for DBF per-module bitpacking (``bitpack_on_quantize``).

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Background
----------
DBF supports two storage modes for an already-quantized module:

  - Normal mode        (``bitpack_on_quantize=False``): keep ``dbf_A``/``dbf_B``
    as unpacked +/-1 float16 matrices in ``DBFResult``.
  - Per-module bitpack (``bitpack_on_quantize=True``): pack ``dbf_A``/``dbf_B``
    immediately after the module is quantized.

Bit-packing only changes the storage layout of the already-quantized binary
factors: pack -> unpack is a lossless, bit-exact round-trip. Therefore the
weights reconstructed from each mode (via ``DBFResult.compute_dequantized_weight``)
must be bit-identical when both modes receive the same DBF factors.
"""

import os
import sys
from unittest import mock

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from test_module import QuantizeTestHelper

from onecomp.quantizer.dbf._dbf import DBF, DBFResult
from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear, pack_binary


def _make_pm1(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    """Build a deterministic +/-1 float16 matrix for pack/unpack tests."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (rows, cols), generator=g) * 2 - 1).to(torch.float16)


def _make_dbf_result(out_dim, mid_dim, in_dim, *, packed, seed=0) -> DBFResult:
    """Build a DBFResult from random +/-1 factors, packed or unpacked."""
    A = _make_pm1(out_dim, mid_dim, seed=seed)
    B = _make_pm1(mid_dim, in_dim, seed=seed + 1)
    g = torch.Generator().manual_seed(seed + 2)
    Da = torch.randn(out_dim, generator=g).to(torch.float16)
    mid = torch.randn(mid_dim, generator=g).to(torch.float16)
    Db = torch.randn(in_dim, generator=g).to(torch.float16)
    kwargs = dict(is_dbf_quantized=True, dbf_Da=Da, dbf_mid=mid, dbf_Db=Db)
    if packed:
        kwargs.update(
            dbf_A=pack_binary(A),
            dbf_B=pack_binary(B),
            dbf_A_is_packed=True,
            dbf_B_is_packed=True,
            dbf_A_original_shape=(out_dim, mid_dim),
            dbf_B_original_shape=(mid_dim, in_dim),
        )
    else:
        kwargs.update(dbf_A=A, dbf_B=B)
    return DBFResult(**kwargs)


def _quantize_both_modes():
    """Quantize two identical layers in packed and unpacked modes."""
    helper = QuantizeTestHelper()
    helper.set_deterministic()

    # Two layers with identical weights so each quantizer runs on a pristine layer.
    layer_p = helper.make_linear(8, 8, device="cpu", dtype=torch.float32)
    layer_u = helper.make_linear(8, 8, device="cpu", dtype=torch.float32)
    layer_u.weight.data.copy_(layer_p.weight.data)
    inp = helper.make_input(device="cpu", dtype=torch.float32)

    params = {"target_bits": 1.0, "iters": 1, "balance_iters": 1}
    h_p = DBF().calculate_hessian(layer_p, inp)
    h_u = DBF().calculate_hessian(layer_u, inp)

    helper.seed_everything(123)
    result_packed = DBF(bitpack_on_quantize=True, **params).quantize_layer(
        layer_p, inp, hessian=h_p
    )
    helper.seed_everything(123)
    result_unpacked = DBF(bitpack_on_quantize=False, **params).quantize_layer(
        layer_u, inp, hessian=h_u
    )

    return result_unpacked, result_packed


# ---------------------------------------------------------------------------
# run_dbf-mocked equivalence: isolate ONLY the pack path.
#
# The seed-based ``_quantize_both_modes`` above relies on DBF being fully
# deterministic run-to-run, so the two quantization passes it compares are only
# equal if that assumption holds. To pin the pack/unpack path independently of
# any DBF nondeterminism, these helpers mock ``run_dbf`` to return one fixed
# factor set and drive the real ``quantize_layer`` pack block in both modes
# against that same result. Any difference is then attributable solely to
# pack/unpack, not to quantization nondeterminism.
# ---------------------------------------------------------------------------

# run_dbf is imported into _dbf as a module-level name, so patch it there.
RUN_DBF_TARGET = "onecomp.quantizer.dbf._dbf.run_dbf"

OUT_DIM = 16
MID_DIM = 12
IN_DIM = 20


def make_fake_dbf_result(seed=0):
    """Build a deterministic, self-consistent unpacked ``run_dbf`` result.

    Mirrors the keys/dtypes ``run_dbf`` returns: +/-1 float16 ``dbf_A``/``dbf_B``
    and float16 scaling vectors, all on CPU, with ``is_dbf_quantized=True``.
    """
    g = torch.Generator().manual_seed(seed)
    A = (torch.randint(0, 2, (OUT_DIM, MID_DIM), generator=g) * 2 - 1).to(torch.float16)
    B = (torch.randint(0, 2, (MID_DIM, IN_DIM), generator=g) * 2 - 1).to(torch.float16)
    Da = torch.randn(OUT_DIM, generator=g).to(torch.float16)
    mid = torch.randn(MID_DIM, generator=g).to(torch.float16)
    Db = torch.randn(IN_DIM, generator=g).to(torch.float16)
    return {
        "dbf_Da": Da,
        "dbf_A": A,
        "dbf_mid": mid,
        "dbf_B": B,
        "dbf_Db": Db,
        "is_dbf_quantized": True,
    }


def _fresh_copy(result_dict):
    """Return a shallow-cloned copy so the two ``quantize_layer`` calls cannot
    alias/mutate each other's tensors via the shared mock return value."""
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in result_dict.items()}


def _quantize_both_modes_mocked():
    """Drive the real ``quantize_layer`` in both modes against one fixed result."""
    fake = make_fake_dbf_result()
    module = torch.nn.Linear(IN_DIM, OUT_DIM, bias=False)

    params = {"target_bits": 1.0, "iters": 1, "balance_iters": 1}
    q_unpacked = DBF(bitpack_on_quantize=False, **params)
    q_packed = DBF(bitpack_on_quantize=True, **params)
    q_unpacked.validate_params()
    q_packed.validate_params()

    # side_effect hands out a fresh copy per call so neither result aliases the
    # other's tensors.
    with mock.patch(RUN_DBF_TARGET, side_effect=lambda *a, **k: _fresh_copy(fake)):
        result_unpacked = q_unpacked.quantize_layer(module, None, hessian=None)
        result_packed = q_packed.quantize_layer(module, None, hessian=None)

    return result_unpacked, result_packed


def test_bitpack_on_quantize_matches_unpacked_dequant_mocked():
    """Pack path only: real quantize_layer, one fixed run_dbf result, both modes.

    Unlike the seed-based test, this does not depend on DBF determinism -- both
    modes consume the identical mocked factor set, so any dequant difference is
    attributable solely to the pack/unpack path.
    """
    result_unpacked, result_packed = _quantize_both_modes_mocked()

    assert result_unpacked.dbf_A_is_packed is False
    assert result_unpacked.dbf_B_is_packed is False
    assert result_packed.dbf_A_is_packed is True
    assert result_packed.dbf_B_is_packed is True

    w_unpacked = result_unpacked.compute_dequantized_weight()
    w_packed = result_packed.compute_dequantized_weight()

    assert torch.equal(w_unpacked, w_packed), (
        "DBF dequantized weights differ between unpacked and bitpacked storage "
        "for one fixed run_dbf result; "
        f"max abs diff={(w_unpacked.float() - w_packed.float()).abs().max().item():.3e}. "
        "Bit-packing is a lossless storage transform, so any difference indicates "
        "a bug in the pack/unpack path, not quantization noise."
    )


def test_bitpack_preserves_binary_factors_mocked():
    """The packed factors must unpack back to the original +/-1 factors.

    Guards the lossless round-trip at the result level against one fixed
    run_dbf result, so a corruption in the pack/unpack path is caught even if it
    happened to cancel out in the dequantized comparison above.
    """
    result_unpacked, result_packed = _quantize_both_modes_mocked()

    restored_A, restored_B = result_packed.get_unpacked_binary_factors()
    assert torch.equal(restored_A, result_unpacked.dbf_A)
    assert torch.equal(restored_B, result_unpacked.dbf_B)


def test_bitpack_on_quantize_matches_unpacked_dequant():
    """Per-module bitpacking must reconstruct bit-identical dequantized weights."""
    result_unpacked, result_packed = _quantize_both_modes()

    assert result_unpacked.dbf_A_is_packed is False
    assert result_unpacked.dbf_B_is_packed is False
    assert result_packed.dbf_A_is_packed is True
    assert result_packed.dbf_B_is_packed is True

    w_unpacked = result_unpacked.compute_dequantized_weight()
    w_packed = result_packed.compute_dequantized_weight()

    assert torch.equal(w_unpacked, w_packed), (
        "DBF dequantized weights differ between unpacked and bitpacked storage; "
        f"max abs diff={(w_unpacked.float() - w_packed.float()).abs().max().item():.3e}."
    )


def test_bitpack_preserves_binary_factors():
    """Packed binary factors must unpack back to the normal-mode factors."""
    result_unpacked, result_packed = _quantize_both_modes()

    unpacked_A, unpacked_B = result_unpacked.get_unpacked_binary_factors()
    packed_A, packed_B = result_packed.get_unpacked_binary_factors()

    assert torch.equal(packed_A, unpacked_A)
    assert torch.equal(packed_B, unpacked_B)


def test_compute_dequantized_weight_equivalence_synthetic():
    """Hand-built packed/unpacked DBFResults dequantize identically."""
    unpacked = _make_dbf_result(6, 5, 7, packed=False, seed=3)
    packed = _make_dbf_result(6, 5, 7, packed=True, seed=3)

    assert torch.equal(
        unpacked.compute_dequantized_weight(),
        packed.compute_dequantized_weight(),
    )


def test_packed_unpacked_inference_layers_match():
    """Packed and unpacked results build equivalent inference layers."""
    out_dim, mid_dim, in_dim = 6, 5, 8
    packed = _make_dbf_result(out_dim, mid_dim, in_dim, packed=True, seed=4)
    unpacked = _make_dbf_result(out_dim, mid_dim, in_dim, packed=False, seed=4)

    dbl_p = DoubleBinaryLinear.from_quantization_result(packed, use_gemlite=False)
    dbl_u = DoubleBinaryLinear.from_quantization_result(unpacked, use_gemlite=False)

    assert torch.equal(dbl_p.bp1, dbl_u.bp1)
    assert torch.equal(dbl_p.bp3, dbl_u.bp3)

    x = torch.randn(2, in_dim, dtype=torch.float16)
    with torch.no_grad():
        assert torch.equal(dbl_p(x), dbl_u(x))