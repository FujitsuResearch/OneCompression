"""Tests for the QBB quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.
"""

import os
import sys

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from test_module import BaseQuantizeSpec

from onecomp.quantizer.qbb._qbb import QBB, QBBResult


class TestQBB(BaseQuantizeSpec):
    """Test cases for QBB quantization."""

    __test__ = True
    quantizer_cls = QBB
    result_cls = QBBResult
    default_parameter_for_test = {
        "wbits": 1,
        "iters_per_basis": 1,
        "lr": 1e-4,
    }
    boundary_parameters = [
        # wbits: int >= 1 (validated by validate_params), no explicit upper
        {"wbits": 1},  # wbits lower boundary
        {"wbits": 32},  # wbits large value (no explicit upper bound)
        # iters_per_basis: int >= 0 (validated by validate_params), no explicit upper
        {"iters_per_basis": 0},  # iters_per_basis lower boundary
        {"iters_per_basis": 100},  # iters_per_basis large value (no explicit upper bound)
        # lr: float >= 0 (validated by validate_params), no explicit upper
        {"lr": 0.0},  # lr lower boundary
        {"lr": 1.0},  # lr large value (no explicit upper bound)
        # ste_type: str in {clipped, identity, tanh} (validated by validate_params)
        {"ste_type": "clipped"},
        {"ste_type": "identity"},
        {"ste_type": "tanh"},
        # use_progressive_quantization: bool
        {"use_progressive_quantization": False},
        # progressive_bits: int >= 2 when enabled (validated by validate_params), no explicit upper
        {
            "wbits": 4,
            "use_progressive_quantization": True,
            "progressive_bits": 2,  # progressive_bits lower boundary
        },
        {
            "wbits": 8,
            "use_progressive_quantization": True,
            "progressive_bits": 16,  # progressive_bits large value (no explicit upper bound)
        },
        # all class defaults
        {
            "wbits": 4,
            "iters_per_basis": 1000,
            "lr": 1e-4,
            "ste_type": "clipped",
            "use_progressive_quantization": False,
            "progressive_bits": 2,
        },
        # all minimum
        {
            "wbits": 1,
            "iters_per_basis": 0,
            "lr": 0.0,
            "ste_type": "clipped",
            "use_progressive_quantization": False,
            "progressive_bits": 2,
        },
        # all maximum
        {
            "wbits": 32,
            "iters_per_basis": 100,
            "lr": 1.0,
            "ste_type": "tanh",
            "use_progressive_quantization": True,
            "progressive_bits": 16,
        },
    ]
    abnormal_parameters = [
        {"wbits": 0},  # below lower boundary (wbits >= 1)
        {"iters_per_basis": -1},  # below lower boundary (iters_per_basis >= 0)
        {"lr": -0.01},  # below lower boundary (lr >= 0)
        {"ste_type": "invalid"},  # not in allowed set
        {
            "use_progressive_quantization": True,
            "progressive_bits": 1,  # below lower boundary (progressive_bits >= 2)
        },
    ]

    def check_quantize_layer(
        self,
        result,
        layer: torch.nn.Module,
    ):
        """Validate types, shapes, and devices of quantize_layer outputs."""
        assert isinstance(result, self.result_cls)
        for attr in [
            "quantized_weight_list",
            "alpha_list",
        ]:
            assert hasattr(result, attr)

        # quantized_weight_list: list of binary tensors {±1}
        assert isinstance(result.quantized_weight_list, list)
        assert len(result.quantized_weight_list) == result.wbits

        for bw in result.quantized_weight_list:
            assert isinstance(bw, torch.Tensor)
            assert bw.dtype == torch.int8
            assert bw.device == torch.device("cpu")
            assert torch.all((bw == 1) | (bw == -1))
            assert bw.shape == layer.weight.shape

        # alpha_list: list of scale coefficients
        assert isinstance(result.alpha_list, list)
        assert len(result.alpha_list) == result.wbits

        for alpha in result.alpha_list:
            assert isinstance(alpha, torch.Tensor)
            assert alpha.dtype == torch.float16
            assert alpha.device == torch.device("cpu")

    def check_equal_results(self, r1, r2):
        """Validate equality of quantization result objects."""
        assert torch.equal(r1.dequantized_weight, r2.dequantized_weight)
        assert len(r1.quantized_weight_list) == len(r2.quantized_weight_list)
        for bw1, bw2 in zip(r1.quantized_weight_list, r2.quantized_weight_list):
            assert torch.equal(bw1, bw2)
        assert len(r1.alpha_list) == len(r2.alpha_list)
        for a1, a2 in zip(r1.alpha_list, r2.alpha_list):
            assert torch.equal(a1, a2)

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
            "[QBB forward error] "
            f"original_vs_qbb(rel={error_original_vs_dequantized:.8f}), "
            f"qbb_vs_qbbl(max={max_error_dequantized_vs_applied:.8f}), "
            f"qbb_vs_qbbl(rel={error_dequantized_vs_applied:.8f})"
        )

        assert max_error_dequantized_vs_applied < 1e-2, (
            f"QBB dequantized vs applied max error too large: "
            f"{max_error_dequantized_vs_applied}"
        )

    def apply_quantized_weights(self, module, result, device):
        """Apply quantized weights to a module."""
        module.weight.data = result.dequantized_weight.to(device)

    def test_forward_error(self, helper):
        """Skip forward error test (no inference layer support)."""
        import pytest

        pytest.skip("QBB does not support create_inference_layer")
