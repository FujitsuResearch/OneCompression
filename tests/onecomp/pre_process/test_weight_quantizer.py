"""Tests for WeightQuantizer and cross-quantizer consistency.

Verifies that WeightQuantizer (training-time RTN proxy), RTN's
pseudo_quantize_tensor (final quantizer), and GPTQExcecutor (GPTQ quantizer)
produce identical scale, zero, and quantized outputs for the same input weight.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yusei Kawakami

Usage::

    # All tests (CPU only, no GPU required)
    pytest tests/onecomp/pre_process/test_weight_quantizer.py -v

    # WeightQuantizer unit tests only
    pytest tests/onecomp/pre_process/test_weight_quantizer.py -v -k "Unit"

    # Cross-quantizer consistency tests (WQ↔RTN, WQ↔GPTQ, RTN↔GPTQ)
    pytest tests/onecomp/pre_process/test_weight_quantizer.py -v -k "Consistency"

    # MSE grid search tests only
    pytest tests/onecomp/pre_process/test_weight_quantizer.py -v -k "MSE"
"""

import os
import pytest
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from onecomp.pre_process.quant_models import STEQuantize, WeightQuantizer
from onecomp.quantizer.gptq._gptq import GPTQExcecutor
from onecomp.quantizer.rtn.quantizer import pseudo_quantize_tensor

# ============================================================
# Helper
# ============================================================


def _make_weight(out_features=8, in_features=16, seed=42):
    """Create a reproducible random weight tensor."""
    torch.manual_seed(seed)
    return torch.randn(out_features, in_features)


# ============================================================
# WeightQuantizer unit tests
# ============================================================


class TestWeightQuantizerUnit:
    """Standalone unit tests for WeightQuantizer."""

    def test_configure_sets_attributes(self):
        wq = WeightQuantizer()
        wq.configure(
            bits=4,
            perchannel=True,
            sym=False,
            weight_groupsize=-1,
            mse=True,
            norm=3.0,
            grid=200,
            maxshrink=0.9,
        )
        assert wq.bits == 4
        assert wq.perchannel is True
        assert wq.sym is False
        assert wq.weight_groupsize == -1
        assert wq.mse is True
        assert wq.norm == 3.0
        assert wq.grid == 200
        assert wq.maxshrink == 0.9
        assert wq.maxq.item() == 15

    def test_configure_defaults(self):
        wq = WeightQuantizer()
        wq.configure(bits=4)
        assert wq.bits == 4
        assert wq.perchannel is True
        assert wq.sym is True
        assert wq.weight_groupsize == -1
        assert wq.mse is False
        assert wq.norm == 2.4
        assert wq.grid == 100
        assert wq.maxshrink == 0.8
        assert wq.maxq.item() == 15

    def test_enabled_after_configure(self):
        wq = WeightQuantizer()
        assert not wq.enabled()
        wq.configure(bits=4)
        assert wq.enabled()

    def test_not_ready_before_find_params(self):
        wq = WeightQuantizer()
        wq.configure(bits=4)
        assert not wq.ready()

    def test_ready_after_find_params(self):
        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=False)
        w = _make_weight()
        wq.find_params(w)
        assert wq.ready()

    def test_quantize_returns_unchanged_when_not_ready(self):
        wq = WeightQuantizer()
        wq.configure(bits=4)
        w = _make_weight()
        out = wq.quantize(w)
        assert torch.equal(out, w)

    def test_quantize_returns_unchanged_when_bits_16(self):
        wq = WeightQuantizer()
        wq.configure(bits=16)
        w = _make_weight()
        wq.find_params(w)
        out = wq.quantize(w)
        assert torch.equal(out, w)

    @pytest.mark.parametrize("sym", [True, False])
    def test_find_params_produces_finite_scale(self, sym):
        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=sym)
        w = _make_weight()
        wq.find_params(w)
        assert torch.all(torch.isfinite(wq.scale))
        assert torch.all(wq.scale > 0)

    @pytest.mark.parametrize("sym", [True, False])
    def test_quantize_output_shape(self, sym):
        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=sym)
        w = _make_weight()
        wq.find_params(w)
        out = wq.quantize(w)
        assert out.shape == w.shape

    def test_perchannel_true_scale_shape(self):
        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=False)
        w = _make_weight(out_features=8, in_features=16)
        wq.find_params(w)
        assert wq.scale.shape == (8, 1)
        assert wq.zero.shape == (8, 1)

    def test_perchannel_false_scale_shape(self):
        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=False, sym=False)
        w = _make_weight(out_features=8, in_features=16)
        wq.find_params(w)
        assert wq.scale.numel() == 1
        assert wq.zero.numel() == 1

    def test_groupwise_scale_shape(self):
        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=False, weight_groupsize=4)
        w = _make_weight(out_features=8, in_features=16)
        wq.find_params(w)
        num_groups = 16 // 4  # 4
        # weight_groupsize > 0: find_params reshapes x to (out, num_groups, group_size)
        # then computes scale with keepdim=True on last dim → (out, num_groups, 1)
        # the `weight_groupsize > 0` branch is a pass (no further reshape)
        assert wq.scale.shape == (8, num_groups, 1)
        assert wq.zero.shape == (8, num_groups, 1)

    def test_symmetric_zero_is_midpoint(self):
        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=True)
        w = _make_weight()
        wq.find_params(w)
        expected = (2**4 - 1 + 1) / 2  # 8.0
        assert torch.allclose(wq.zero, torch.full_like(wq.zero, expected))

    def test_dead_zone_all_zeros(self):
        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=False)
        w = torch.zeros(4, 8)
        wq.find_params(w)
        assert torch.all(torch.isfinite(wq.scale))
        assert torch.all(wq.scale > 0)


# ============================================================
# STEQuantize tests
# ============================================================


class TestSTEQuantize:
    """Tests for STEQuantize autograd function."""

    def test_forward_matches_manual(self):
        x = torch.tensor([[0.0, 0.5, -0.5, 1.5]])
        scale = torch.tensor([[0.2]])
        zero = torch.tensor([[8.0]])
        maxq = torch.tensor(15)
        out = STEQuantize.apply(x, scale, zero, maxq)
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        expected = scale * (q - zero)
        assert torch.equal(out, expected)

    def test_backward_passes_gradient_through(self):
        x = torch.randn(4, 8, requires_grad=True)
        scale = torch.tensor([[0.1]])
        zero = torch.tensor([[8.0]])
        maxq = torch.tensor(15)
        out = STEQuantize.apply(x, scale, zero, maxq)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.equal(x.grad, torch.ones_like(x))


# ============================================================
# WeightQuantizer ↔ RTN pseudo_quantize_tensor consistency
# ============================================================


class TestWeightQuantizerRTNConsistency:
    """Verify WeightQuantizer and RTN pseudo_quantize_tensor produce identical results.

    This is the core verification for the quantization logic unification:
    the training-time proxy (WeightQuantizer) must match the final quantizer
    (RTN) exactly.
    """

    @pytest.mark.parametrize("sym", [True, False])
    def test_scale_zero_match_perchannel(self, sym):
        """perchannel mode: scale and zero must be identical."""
        w = _make_weight()

        _, rtn_scale, rtn_zero, _ = pseudo_quantize_tensor(
            w, n_bit=4, zero_point=not sym, perchannel=True, mse=False
        )

        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=sym, mse=False)
        wq.find_params(w)

        assert torch.equal(wq.scale.squeeze(), rtn_scale.squeeze())
        assert torch.equal(wq.zero.squeeze(), rtn_zero.squeeze())

    @pytest.mark.parametrize("sym", [True, False])
    def test_dequantized_output_match_perchannel(self, sym):
        """Dequantized weights must be identical between the two paths."""
        w = _make_weight()

        rtn_dequant, _, _, _ = pseudo_quantize_tensor(
            w, n_bit=4, zero_point=not sym, perchannel=True, mse=False
        )

        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=sym, mse=False)
        wq.find_params(w)
        wq_dequant = wq.quantize(w)

        assert torch.equal(wq_dequant, rtn_dequant)

    @pytest.mark.parametrize("sym", [True, False])
    def test_scale_zero_match_with_mse(self, sym):
        """With MSE grid search enabled, results must still match."""
        w = _make_weight()

        _, rtn_scale, rtn_zero, _ = pseudo_quantize_tensor(
            w, n_bit=4, zero_point=not sym, perchannel=True, mse=True
        )

        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=sym, mse=True)
        wq.find_params(w)

        assert torch.equal(wq.scale.squeeze(), rtn_scale.squeeze())
        assert torch.equal(wq.zero.squeeze(), rtn_zero.squeeze())

    @pytest.mark.parametrize("sym", [True, False])
    def test_dequantized_output_match_with_mse(self, sym):
        w = _make_weight()

        rtn_dequant, _, _, _ = pseudo_quantize_tensor(
            w, n_bit=4, zero_point=not sym, perchannel=True, mse=True
        )

        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=sym, mse=True)
        wq.find_params(w)
        wq_dequant = wq.quantize(w)

        assert torch.equal(wq_dequant, rtn_dequant)

    def test_groupwise_match(self):
        """Group-wise quantization must also match between the two paths."""
        w = _make_weight(out_features=8, in_features=16)
        group_size = 4

        rtn_dequant, _, _, _ = pseudo_quantize_tensor(
            w, n_bit=4, q_group_size=group_size, zero_point=True, mse=False
        )

        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=False, weight_groupsize=group_size, mse=False)
        wq.find_params(w)
        wq_dequant = wq.quantize(w)

        assert torch.equal(wq_dequant, rtn_dequant)

    @pytest.mark.parametrize("sym", [True, False])
    def test_groupwise_match_with_mse(self, sym):
        """Group-wise + MSE quantization must also match."""
        w = _make_weight(out_features=8, in_features=16)
        group_size = 4

        rtn_dequant, rtn_scale, rtn_zero, _ = pseudo_quantize_tensor(
            w,
            n_bit=4,
            q_group_size=group_size,
            zero_point=not sym,
            mse=True,
        )

        wq = WeightQuantizer()
        wq.configure(
            bits=4,
            perchannel=True,
            sym=sym,
            weight_groupsize=group_size,
            mse=True,
        )
        wq.find_params(w)
        wq_dequant = wq.quantize(w)

        assert torch.equal(wq.scale.squeeze(), rtn_scale.squeeze())
        assert torch.equal(wq.zero.squeeze(), rtn_zero.squeeze())
        assert torch.equal(wq_dequant, rtn_dequant)


# ============================================================
# WeightQuantizer ↔ GPTQExcecutor consistency
# ============================================================


class TestWeightQuantizerGPTQConsistency:
    """Verify WeightQuantizer and GPTQExcecutor produce identical scale/zero.

    GPTQExcecutor is the reference implementation. WeightQuantizer and
    RTN are aligned to match its quantization logic.
    """

    @pytest.mark.parametrize("sym", [True, False])
    def test_scale_zero_match(self, sym):
        w = _make_weight()

        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=sym, mse=False)
        wq.find_params(w)

        gptq = GPTQExcecutor()
        gptq.configure(bits=4, perchannel=True, sym=sym, mse=False)
        gptq.find_params(w.clone(), weight=True)

        assert torch.equal(wq.scale.squeeze(), gptq.scale.squeeze())
        assert torch.equal(wq.zero.squeeze(), gptq.zero.squeeze())

    @pytest.mark.parametrize("sym", [True, False])
    def test_dequantized_output_match(self, sym):
        w = _make_weight()

        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=sym, mse=False)
        wq.find_params(w)
        wq_dequant = wq.quantize(w)

        gptq = GPTQExcecutor()
        gptq.configure(bits=4, perchannel=True, sym=sym, mse=False)
        gptq.find_params(w.clone(), weight=True)
        gptq_dequant = gptq.quantize(w.clone())

        assert torch.equal(wq_dequant, gptq_dequant)

    @pytest.mark.parametrize("sym", [True, False])
    def test_scale_zero_match_with_mse(self, sym):
        w = _make_weight()

        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=sym, mse=True)
        wq.find_params(w)

        gptq = GPTQExcecutor()
        gptq.configure(bits=4, perchannel=True, sym=sym, mse=True)
        gptq.find_params(w.clone(), weight=True)

        assert torch.equal(wq.scale.squeeze(), gptq.scale.squeeze())
        assert torch.equal(wq.zero.squeeze(), gptq.zero.squeeze())

    @pytest.mark.parametrize("sym", [True, False])
    def test_dequantized_output_match_with_mse(self, sym):
        w = _make_weight()

        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=sym, mse=True)
        wq.find_params(w)
        wq_dequant = wq.quantize(w)

        gptq = GPTQExcecutor()
        gptq.configure(bits=4, perchannel=True, sym=sym, mse=True)
        gptq.find_params(w.clone(), weight=True)
        gptq_dequant = gptq.quantize(w.clone())

        assert torch.equal(wq_dequant, gptq_dequant)


# ============================================================
# RTN pseudo_quantize_tensor ↔ GPTQExcecutor consistency
# ============================================================


class TestRTNGPTQConsistency:
    """Verify RTN pseudo_quantize_tensor and GPTQExcecutor produce identical scale/zero.

    This completes the triangle: if A==B and A==C then B==C, but testing
    directly catches bugs that pair-wise tests might miss.
    """

    @pytest.mark.parametrize("sym", [True, False])
    def test_scale_zero_match(self, sym):
        w = _make_weight()

        _, rtn_scale, rtn_zero, _ = pseudo_quantize_tensor(
            w, n_bit=4, zero_point=not sym, perchannel=True, mse=False
        )

        gptq = GPTQExcecutor()
        gptq.configure(bits=4, perchannel=True, sym=sym, mse=False)
        gptq.find_params(w.clone(), weight=True)

        assert torch.equal(rtn_scale.squeeze(), gptq.scale.squeeze())
        assert torch.equal(rtn_zero.squeeze(), gptq.zero.squeeze())

    @pytest.mark.parametrize("sym", [True, False])
    def test_dequantized_output_match(self, sym):
        w = _make_weight()

        rtn_dequant, _, _, _ = pseudo_quantize_tensor(
            w, n_bit=4, zero_point=not sym, perchannel=True, mse=False
        )

        gptq = GPTQExcecutor()
        gptq.configure(bits=4, perchannel=True, sym=sym, mse=False)
        gptq.find_params(w.clone(), weight=True)
        gptq_dequant = gptq.quantize(w.clone())

        assert torch.equal(rtn_dequant, gptq_dequant)

    @pytest.mark.parametrize("sym", [True, False])
    def test_scale_zero_match_with_mse(self, sym):
        w = _make_weight()

        _, rtn_scale, rtn_zero, _ = pseudo_quantize_tensor(
            w, n_bit=4, zero_point=not sym, perchannel=True, mse=True
        )

        gptq = GPTQExcecutor()
        gptq.configure(bits=4, perchannel=True, sym=sym, mse=True)
        gptq.find_params(w.clone(), weight=True)

        assert torch.equal(rtn_scale.squeeze(), gptq.scale.squeeze())
        assert torch.equal(rtn_zero.squeeze(), gptq.zero.squeeze())

    @pytest.mark.parametrize("sym", [True, False])
    def test_dequantized_output_match_with_mse(self, sym):
        w = _make_weight()

        rtn_dequant, _, _, _ = pseudo_quantize_tensor(
            w, n_bit=4, zero_point=not sym, perchannel=True, mse=True
        )

        gptq = GPTQExcecutor()
        gptq.configure(bits=4, perchannel=True, sym=sym, mse=True)
        gptq.find_params(w.clone(), weight=True)
        gptq_dequant = gptq.quantize(w.clone())

        assert torch.equal(rtn_dequant, gptq_dequant)


# ============================================================
# MSE grid search behaviour
# ============================================================


class TestMSEGridSearch:
    """Verify MSE grid search behaviour for RTN and WeightQuantizer."""

    @pytest.mark.parametrize("sym", [True, False])
    def test_mse_produces_lower_error_rtn(self, sym):
        w = _make_weight()
        dq_no, _, _, _ = pseudo_quantize_tensor(w, n_bit=4, zero_point=not sym, mse=False)
        dq_mse, _, _, _ = pseudo_quantize_tensor(w, n_bit=4, zero_point=not sym, mse=True)
        err_no = (dq_no - w).pow(2).sum()
        err_mse = (dq_mse - w).pow(2).sum()
        assert err_mse <= err_no + 1e-6

    @pytest.mark.parametrize("sym", [True, False])
    def test_mse_produces_lower_error_weight_quantizer(self, sym):
        w = _make_weight()

        wq_no = WeightQuantizer()
        wq_no.configure(bits=4, perchannel=True, sym=sym, mse=False)
        wq_no.find_params(w)
        dq_no = wq_no.quantize(w)

        wq_mse = WeightQuantizer()
        wq_mse.configure(bits=4, perchannel=True, sym=sym, mse=True)
        wq_mse.find_params(w)
        dq_mse = wq_mse.quantize(w)

        err_no = (dq_no - w).pow(2).sum()
        err_mse = (dq_mse - w).pow(2).sum()
        assert err_mse <= err_no + 1e-6

    @pytest.mark.parametrize("sym", [True, False])
    def test_mse_does_not_increase_error(self, sym):
        """MSE search should never produce higher quantisation error than the default range."""
        w = _make_weight()
        dq_no, _, _, _ = pseudo_quantize_tensor(w, n_bit=4, zero_point=not sym, mse=False)
        dq_mse, _, _, _ = pseudo_quantize_tensor(w, n_bit=4, zero_point=not sym, mse=True)
        err_no = (dq_no - w).pow(2).sum()
        err_mse = (dq_mse - w).pow(2).sum()
        assert err_mse <= err_no + 1e-6
