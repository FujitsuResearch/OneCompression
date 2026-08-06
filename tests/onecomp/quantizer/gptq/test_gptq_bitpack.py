"""Helpers for GPTQ quantize_layer bitpack tests.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import pytest
import torch

from onecomp.quantizer.gptq._gptq import GPTQ
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

_DEVICES = ["cpu", "cuda"]


def _skip_if_cuda_unavailable(device):
    """Skip CUDA parametrized cases when CUDA is unavailable."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _make_layer_and_input(device):
    """Create a small deterministic Linear layer and calibration input."""
    torch.manual_seed(123)
    layer = torch.nn.Linear(8, 8, bias=False, device=device, dtype=torch.float32)
    inp = torch.randn(2, 3, 8, device=device, dtype=torch.float32)
    return layer, inp


def _make_quantizer(bitpack_on_quantize):
    """Create the baseline 4-bit GPTQ quantizer used by bitpack tests."""
    return GPTQ(wbits=4, groupsize=-1, bitpack_on_quantize=bitpack_on_quantize)


def _quantize_layer(device, *, bitpack_on_quantize=True):
    """Quantize the shared 8x8 fixture and return layer, input, and result."""
    _skip_if_cuda_unavailable(device)
    layer, inp = _make_layer_and_input(device)
    quantizer = _make_quantizer(bitpack_on_quantize)
    # GPTQ has flag_nsamples=False, so the sample count is not needed here.
    hessian, _ = quantizer.calculate_hessian(layer, inp)
    result = quantizer.quantize_layer(layer, inp, hessian=hessian)
    return layer, inp, result


def _quantize_layer_pair(device):
    """Create packed and unpacked results from the same layer, input, and Hessian."""
    _skip_if_cuda_unavailable(device)
    layer, inp = _make_layer_and_input(device)
    unpacked_quantizer = _make_quantizer(bitpack_on_quantize=False)
    packed_quantizer = _make_quantizer(bitpack_on_quantize=True)
    hessian, _ = unpacked_quantizer.calculate_hessian(layer, inp)
    unpacked_result = unpacked_quantizer.quantize_layer(layer, inp, hessian=hessian.clone())
    packed_result = packed_quantizer.quantize_layer(layer, inp, hessian=hessian.clone())
    return layer, inp, unpacked_result, packed_result


@pytest.mark.parametrize("device", _DEVICES)
def test_quantize_layer_bitpack_stores_packed_result(device):
    """Verify quantize_layer stores packed tensors and unpack metadata."""
    layer, _inp, result = _quantize_layer(device, bitpack_on_quantize=True)

    assert result.qweight_is_packed is True
    assert result.qzeros_is_packed is True
    assert result.qweight_original_shape == tuple(layer.weight.shape)
    assert result.qweight.dtype == torch.int32
    assert result.qweight.device == torch.device("cpu")
    assert result.qzeros.dtype == torch.int32
    assert result.qzeros.device == torch.device("cpu")
    assert result.qweight.shape != layer.weight.shape
    assert result.qweight.shape == torch.Size([1, 8])
    assert result.qzeros.shape == torch.Size([1, 1])

    dequant = result.compute_dequantized_weight()
    assert dequant.shape == layer.weight.shape
    assert dequant.dtype == torch.float16
    assert dequant.device == torch.device("cpu")


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("pack_weights", [True, False])
def test_from_quantization_result_accepts_bitpacked_result(device, pack_weights):
    """Verify GPTQLinear accepts packed quantize_layer results."""
    layer, inp, result = _quantize_layer(device, bitpack_on_quantize=True)

    gptq_linear = GPTQLinear.from_quantization_result(
        result,
        device=device,
        pack_weights=pack_weights,
        use_gemlite=False,
    )

    if pack_weights:
        assert gptq_linear.qweight.shape == result.qweight.shape
        assert torch.equal(gptq_linear.qweight.cpu(), result.qweight)
        assert gptq_linear.qzeros.shape == result.qzeros.shape
        assert torch.equal(gptq_linear.qzeros.cpu(), result.qzeros)
    else:
        assert gptq_linear.qweight.shape == layer.weight.shape

    x = inp.to(torch.float16)
    ref_weight = result.compute_dequantized_weight().to(device).to(torch.float16)
    ref_out = torch.nn.functional.linear(x, ref_weight)
    layer_out = gptq_linear(x)

    assert torch.allclose(layer_out, ref_out, rtol=0.02, atol=0.3)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("wbits", [1, 15])
def test_bitpack_unsupported_wbits_raise_clear_error(device, wbits):
    """Verify bitpack rejects unsupported wbits with a clear error."""
    _skip_if_cuda_unavailable(device)
    layer, inp = _make_layer_and_input(device)
    quantizer = GPTQ(wbits=wbits, bitpack_on_quantize=True)
    hessian, _ = quantizer.calculate_hessian(layer, inp)

    with pytest.raises(
        ValueError,
        match="bitpack_on_quantize=True supports only wbits",
    ):
        quantizer.quantize_layer(layer, inp, hessian=hessian)


@pytest.mark.parametrize("device", _DEVICES)
def test_bitpack_unaligned_shape_raises_assertion_error(device):
    """Verify bitpack rejects unaligned shapes via the packer assertion."""
    _skip_if_cuda_unavailable(device)
    layer = torch.nn.Linear(4, 4, bias=False, device=device, dtype=torch.float32)
    inp = torch.randn(1, 1, 4, device=device, dtype=torch.float32)
    quantizer = GPTQ(wbits=4, bitpack_on_quantize=True)
    hessian, _ = quantizer.calculate_hessian(layer, inp)

    with pytest.raises(AssertionError, match="rows \\(4\\) must be divisible"):
        quantizer.quantize_layer(layer, inp, hessian=hessian)
