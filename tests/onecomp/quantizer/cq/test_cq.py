"""Tests for the CQ quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.
"""

import os
import sys

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from test_module import BaseQuantizeSpec

from onecomp.quantizer.cq._cq import CQ, CQResult


class TestCQ(BaseQuantizeSpec):
    """Test cases for CQ quantization."""

    __test__ = True
    quantizer_cls = CQ
    result_cls = CQResult
    default_parameter_for_test = {
        "each_row": True,
    }
    boundary_parameters = [
        # each_row: bool, only parameter
        {"each_row": False},  # each_row boundary (global clustering)
        {"each_row": True},  # each_row boundary (row-wise clustering)
    ]
    abnormal_parameters = [
        # CQ does not have parameters with invalid values to test
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
            "threshold",
            "left_mean",
            "right_mean",
        ]:
            assert hasattr(result, attr)

        for attr in [
            "quantized_weight",
            "threshold",
            "left_mean",
            "right_mean",
        ]:
            tensor = getattr(result, attr)
            assert isinstance(tensor, torch.Tensor)

        # quantized_weight: INT8, {0, 1}
        assert result.quantized_weight.dtype == torch.int8
        assert result.quantized_weight.device == torch.device("cpu")
        assert torch.all((result.quantized_weight == 0) | (result.quantized_weight == 1))

        assert result.threshold.dtype == torch.float32
        assert result.threshold.device == torch.device("cpu")
        assert result.left_mean.dtype == torch.float32
        assert result.left_mean.device == torch.device("cpu")
        assert result.right_mean.dtype == torch.float32
        assert result.right_mean.device == torch.device("cpu")

        assert result.quantized_weight.shape == layer.weight.shape

    def check_equal_results(self, r1, r2):
        """Validate equality of quantization result objects."""
        assert torch.equal(r1.dequantized_weight, r2.dequantized_weight)
        assert torch.equal(r1.quantized_weight, r2.quantized_weight)
        assert torch.equal(r1.threshold, r2.threshold)
        assert torch.equal(r1.left_mean, r2.left_mean)
        assert torch.equal(r1.right_mean, r2.right_mean)

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
            "[CQ forward error] "
            f"original_vs_cq(rel={error_original_vs_dequantized:.8f}), "
            f"cq_vs_cql(max={max_error_dequantized_vs_applied:.8f}), "
            f"cq_vs_cql(rel={error_dequantized_vs_applied:.8f})"
        )

        assert max_error_dequantized_vs_applied < 1e-2, (
            f"CQ dequantized vs applied max error too large: "
            f"{max_error_dequantized_vs_applied}"
        )

    def apply_quantized_weights(self, module, result, device):
        """Apply quantized weights to a module."""
        module.weight.data = result.dequantized_weight.to(device)

    def test_forward_error(self, helper):
        """Skip forward error test (no inference layer support)."""
        import pytest

        pytest.skip("CQ does not support create_inference_layer")

    def test_parameters_abnormal_values_raise(self, params):
        """Skip abnormal values test."""
        import pytest

        pytest.skip("CQ does not have parameters with invalid values to test")
