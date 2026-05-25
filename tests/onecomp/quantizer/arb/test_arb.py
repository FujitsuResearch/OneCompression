"""Tests for the ARB quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.
"""

import os
import sys

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from test_module import BaseQuantizeSpec

from onecomp.quantizer.arb._arb import ARB, ARBResult


class TestARB(BaseQuantizeSpec):
    """Test cases for ARB quantization."""

    __test__ = True
    quantizer_cls = ARB
    result_cls = ARBResult
    default_parameter_for_test = {
        "arb_iters": 1,
        "split_points": 2,
    }
    boundary_parameters = [
        # arb_iters: int >= 0 (validated by validate_params), no explicit upper
        {"arb_iters": 0},  # arb_iters lower boundary
        {"arb_iters": 100},  # arb_iters large value (no explicit upper bound)
        # split_points: int >= 1 (validated by validate_params), no explicit upper
        {"split_points": 1},  # split_points lower boundary
        {"split_points": 100},  # split_points large value (no explicit upper bound)
        # verbose: bool
        {"verbose": True},
        {"verbose": False},
        # all class defaults
        {"arb_iters": 15, "split_points": 2, "verbose": False},
        # all minimum
        {"arb_iters": 0, "split_points": 1, "verbose": False},
        # all maximum
        {"arb_iters": 100, "split_points": 100, "verbose": True},
    ]
    abnormal_parameters = [
        {"arb_iters": -1},  # below lower boundary (arb_iters >= 0)
        {"split_points": 0},  # below lower boundary (split_points >= 1)
    ]

    def check_quantize_layer(
        self,
        result,
        layer: torch.nn.Module,
    ):
        """Validate types, shapes, and devices of quantize_layer outputs."""
        assert isinstance(result, self.result_cls)
        for attr in [
            "quantized_weight",
            "alpha",
            "mu",
        ]:
            assert hasattr(result, attr)

        for attr in [
            "quantized_weight",
            "alpha",
            "mu",
        ]:
            tensor = getattr(result, attr)
            assert isinstance(tensor, torch.Tensor)

        # quantized_weight: INT8, {±1}
        assert result.quantized_weight.dtype == torch.int8
        assert result.quantized_weight.device == torch.device("cpu")
        assert torch.all((result.quantized_weight == 1) | (result.quantized_weight == -1))

        assert result.alpha.dtype == torch.float16
        assert result.alpha.device == torch.device("cpu")
        assert result.mu.dtype == torch.float16
        assert result.mu.device == torch.device("cpu")

        assert result.quantized_weight.shape == layer.weight.shape
        assert result.alpha.shape[0] == layer.weight.shape[0]
        assert result.mu.shape[0] == layer.weight.shape[0]

    def check_equal_results(self, r1, r2):
        """Validate equality of quantization result objects."""
        assert torch.equal(r1.dequantized_weight, r2.dequantized_weight)
        assert torch.equal(r1.quantized_weight, r2.quantized_weight)
        assert torch.equal(r1.alpha, r2.alpha)
        assert torch.equal(r1.mu, r2.mu)

    def check_quantize_error(self, error, max_error):
        """Validate that quantization error is within tolerance."""
        assert error < 0.4
        assert max_error < 1.71

    def check_forward_error(
        self,
        error_original_vs_dequantized,
        error_dequantized_vs_applied,
        max_error_dequantized_vs_applied,
    ):
        """Validate forward errors."""
        print(
            "[ARB forward error] "
            f"original_vs_arb(rel={error_original_vs_dequantized:.8f}), "
            f"arb_vs_arbl(max={max_error_dequantized_vs_applied:.8f}), "
            f"arb_vs_arbl(rel={error_dequantized_vs_applied:.8f})"
        )

        assert max_error_dequantized_vs_applied < 1e-2, (
            f"ARB dequantized vs applied max error too large: "
            f"{max_error_dequantized_vs_applied}"
        )

    def apply_quantized_weights(self, module, result, device):
        """Apply quantized weights to a module."""
        module.weight.data = result.dequantized_weight.to(device)

    def test_forward_error(self, helper):
        """Skip forward error test (no inference layer support)."""
        import pytest

        pytest.skip("ARB does not support create_inference_layer")
