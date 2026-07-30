"""Helpers for DBF quantize_layer bitpack tests.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import os
import sys

import pytest
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from test_module import QuantizeTestHelper

from onecomp.analyzer.cumulative_error import _update_weights
from onecomp.quantizer.dbf._dbf import DBF, DBFResult
from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear, pack_binary_factor


def _make_layer_and_input():
    """Create a small deterministic Linear layer and calibration input."""
    helper = QuantizeTestHelper()
    helper.set_deterministic()
    helper.seed_everything(123)
    layer = helper.make_linear(8, 8, device="cpu", dtype=torch.float32)
    inp = helper.make_input(device="cpu", dtype=torch.float32)
    return layer, inp


def _quantize_layer(*, bitpack_on_quantize=True):
    """Quantize the shared 8x8 fixture and return layer, input, and result."""
    layer, inp = _make_layer_and_input()
    quantizer = DBF(
        target_bits=1.0,
        iters=1,
        balance_iters=1,
        bitpack_on_quantize=bitpack_on_quantize,
    )
    hessian = quantizer.calculate_hessian(layer, inp)
    result = quantizer.quantize_layer(layer, inp, hessian=hessian)
    return layer, inp, result


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
            dbf_A=pack_binary_factor(A),
            dbf_B=pack_binary_factor(B),
            dbf_A_is_packed=True,
            dbf_B_is_packed=True,
            dbf_A_original_shape=(out_dim, mid_dim),
            dbf_B_original_shape=(mid_dim, in_dim),
        )
    else:
        kwargs.update(dbf_A=A, dbf_B=B)
    return DBFResult(**kwargs)


class _Tiny(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)


def test_bitpack_on_quantize_default_is_true():
    """DBF defaults to packing the binary factors at quantize time."""
    assert DBF().bitpack_on_quantize is True


def test_quantize_layer_bitpack_stores_packed_result():
    """Verify quantize_layer stores packed binary factors and shape metadata."""
    layer, _inp, result = _quantize_layer(bitpack_on_quantize=True)

    assert result.dbf_A_is_packed is True
    assert result.dbf_B_is_packed is True
    assert result.dbf_A_original_shape is not None
    assert result.dbf_B_original_shape is not None
    assert result.dbf_A.dtype == torch.uint8
    assert result.dbf_B.dtype == torch.uint8
    assert result.dbf_A.device == torch.device("cpu")
    assert result.dbf_B.device == torch.device("cpu")
    assert result.dbf_A.ndim == 1
    assert result.dbf_B.ndim == 1

    dbf_A, dbf_B = result.get_unpacked_binary_factors()
    assert dbf_A.shape == result.dbf_A_original_shape
    assert dbf_B.shape == result.dbf_B_original_shape
    assert dbf_A.dtype == torch.float16
    assert dbf_B.dtype == torch.float16
    assert torch.all((dbf_A == 1) | (dbf_A == -1))
    assert torch.all((dbf_B == 1) | (dbf_B == -1))

    dequant = result.compute_dequantized_weight()
    assert dequant.shape == layer.weight.shape
    assert dequant.dtype == torch.float16
    assert dequant.device == torch.device("cpu")


def test_quantize_layer_unpacked_mode_keeps_float16_result():
    """With packing disabled, the legacy unpacked float16 factors are kept."""
    _layer, _inp, result = _quantize_layer(bitpack_on_quantize=False)

    assert result.dbf_A_is_packed is False
    assert result.dbf_B_is_packed is False
    assert result.dbf_A.dtype == torch.float16
    assert result.dbf_B.dtype == torch.float16
    assert result.dbf_A.ndim == 2
    assert result.dbf_B.ndim == 2


def test_from_quantization_result_accepts_bitpacked_result():
    """Verify DoubleBinaryLinear accepts packed quantize_layer results."""
    _layer, inp, result = _quantize_layer(bitpack_on_quantize=True)

    dbl = DoubleBinaryLinear.from_quantization_result(result, use_gemlite=False)

    assert torch.equal(dbl.bp1, result.dbf_B)
    assert torch.equal(dbl.bp3, result.dbf_A)
    assert dbl._bp1_shape == tuple(result.dbf_B_original_shape)
    assert dbl._bp3_shape == tuple(result.dbf_A_original_shape)

    x = inp.to(torch.float16)
    ref_weight = result.compute_dequantized_weight().to(torch.float16)
    ref_out = torch.nn.functional.linear(x, ref_weight)
    layer_out = dbl(x)

    assert torch.allclose(layer_out, ref_out, atol=5e-2, rtol=5e-2)


def test_update_weights_accepts_bitpacked_result():
    """Downstream weight updates consume a packed DBFResult without error."""
    out_dim, mid_dim, in_dim = 6, 5, 8
    result = _make_dbf_result(out_dim, mid_dim, in_dim, packed=True, seed=7)
    model = _Tiny(in_dim, out_dim)

    _update_weights(model, {"proj": result}, ["proj"])

    expected = result.compute_dequantized_weight().to(model.proj.weight.dtype)
    assert model.proj.weight.shape == (out_dim, in_dim)
    assert torch.equal(model.proj.weight.data, expected)


def test_update_weights_packed_unpacked_equivalent():
    """Packed and unpacked DBF results update model weights identically."""
    out_dim, mid_dim, in_dim = 6, 5, 8
    packed = _make_dbf_result(out_dim, mid_dim, in_dim, packed=True, seed=7)
    unpacked = _make_dbf_result(out_dim, mid_dim, in_dim, packed=False, seed=7)

    model_p = _Tiny(in_dim, out_dim)
    model_u = _Tiny(in_dim, out_dim)
    _update_weights(model_p, {"proj": packed}, ["proj"])
    _update_weights(model_u, {"proj": unpacked}, ["proj"])

    assert torch.equal(model_p.proj.weight.data, model_u.proj.weight.data)


def test_bitpack_missing_shape_raises_clear_error():
    """Packed flag without original shape is an explicit error."""
    result = _make_dbf_result(4, 3, 5, packed=True, seed=1)
    result.dbf_A_original_shape = None

    with pytest.raises(ValueError, match="dbf_A_original_shape"):
        result.get_unpacked_binary_factors()


def test_bitpack_on_quantize_invalid_type_raises():
    """Non-bool bitpack_on_quantize raises a clear ValueError."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))

    with pytest.raises(ValueError, match="bitpack_on_quantize"):
        DBF(bitpack_on_quantize="yes").setup(model)
