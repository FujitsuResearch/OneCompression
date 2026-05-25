"""Tests for the QUIP quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.
"""

import os
import sys

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from test_module import BaseQuantizeSpec

from onecomp.quantizer.quip._quip import QUIP, QUIPResult


class TestQUIP(BaseQuantizeSpec):
    """Test cases for QUIP quantization."""

    __test__ = True
    quantizer_cls = QUIP
    result_cls = QUIPResult
    default_parameter_for_test = {
        "wbits": 1,
        "percdamp": 0.01,
        "incoh_mode": "had",
    }
    boundary_parameters = [
        # wbits: int in 1..63 (validated by validate_params)
        {"wbits": 1},  # wbits lower boundary
        {"wbits": 63},  # wbits upper boundary
        # percdamp: float >= 0 (validated by validate_params), no explicit upper
        {"percdamp": 0.0},  # percdamp lower boundary
        {"percdamp": 1.0},  # percdamp large value (no explicit upper bound)
        # incoh_mode: "kron" or "had" (validated by validate_params)
        {"incoh_mode": "kron"},
        {"incoh_mode": "had"},
        # all class defaults
        {"wbits": 4, "percdamp": 0.01, "incoh_mode": "kron"},
        # all minimum
        {"wbits": 1, "percdamp": 0.0, "incoh_mode": "had"},
        # all maximum
        {"wbits": 63, "percdamp": 1.0, "incoh_mode": "kron"},
    ]
    abnormal_parameters = [
        {"wbits": 0},  # below lower boundary (wbits >= 1)
        {"wbits": 64},  # above upper boundary (wbits <= 63)
        {"percdamp": -0.01},  # below lower boundary (percdamp >= 0)
        {"incoh_mode": "invalid"},  # not in {kron, had}
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
            "scale",
            "zero",
            "maxq",
        ]:
            assert hasattr(result, attr)

        for attr in [
            "quantized_weight",
            "scale",
            "maxq",
        ]:
            tensor = getattr(result, attr)
            assert isinstance(tensor, torch.Tensor)

        assert result.quantized_weight.dtype == torch.int32
        assert result.quantized_weight.device == torch.device("cpu")
        assert result.scale.dtype == torch.float16
        assert result.scale.device == torch.device("cpu")
        if result.zero is not None:
            assert isinstance(result.zero, torch.Tensor)
            assert result.zero.dtype == torch.float16
            assert result.zero.device == torch.device("cpu")
        assert result.maxq.dtype == torch.float16
        assert result.maxq.device == torch.device("cpu")

    def check_equal_results(self, r1, r2):
        """Validate equality of quantization result objects."""
        assert torch.equal(r1.dequantized_weight, r2.dequantized_weight)
        assert torch.equal(r1.scale, r2.scale)
        if r1.zero is not None and r2.zero is not None:
            assert torch.equal(r1.zero, r2.zero)
        else:
            assert r1.zero is None and r2.zero is None

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
            "[QUIP forward error] "
            f"original_vs_quip(rel={error_original_vs_dequantized:.8f}), "
            f"quip_vs_quipl(max={max_error_dequantized_vs_applied:.8f}), "
            f"quip_vs_quipl(rel={error_dequantized_vs_applied:.8f})"
        )

        assert max_error_dequantized_vs_applied < 1e-2, (
            f"QUIP dequantized vs applied max error too large: "
            f"{max_error_dequantized_vs_applied}"
        )

    def apply_quantized_weights(self, module, result, device):
        """Apply quantized weights to a module."""
        module.weight.data = result.dequantized_weight.to(device)

    def test_forward_error(self, helper):
        """Skip forward error test (no inference layer support)."""
        import pytest

        pytest.skip("QUIP does not support create_inference_layer")
