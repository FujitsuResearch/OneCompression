"""Tests for the RTN quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.
"""

import logging
import os
import sys

import pytest
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from onecomp.quantizer.rtn._rtn import RTN, RTNResult

from test_module import BaseQuantizeSpec


class TestRTN(BaseQuantizeSpec):
    """Test cases for RTN quantization."""

    __test__ = True
    quantizer_cls = RTN
    result_cls = RTNResult
    default_parameter_for_test = {}
    boundary_parameters = [
        # wbits: int in 1..64 (validated by validate_params)
        {"wbits": 1},  # wbits lower boundary
        {"wbits": 64},  # wbits upper boundary
        # groupsize: -1 or >=1 (validated by validate_params), no explicit upper
        {"groupsize": -1},  # groupsize (no grouping)
        {"groupsize": 1},  # groupsize positive lower boundary
        {"groupsize": 4},  # groupsize large value (must divide in_features=4)
        # sym: bool
        {"sym": True},
        {"sym": False},
        # mse: bool
        {"mse": True},
        {"mse": False},
        # grid: int >= 1 (validated when mse=True), no explicit upper
        {"mse": True, "grid": 1},  # grid lower boundary
        {"mse": True, "grid": 10000},  # grid large value
        # norm: float > 0 (validated when mse=True), no explicit upper
        {"mse": True, "norm": 1e-5},  # norm near-zero positive
        {"mse": True, "norm": 100.0},  # norm large value
        # grid/norm: not validated when mse=False
        {"grid": 0, "mse": False},  # allowed when mse=False
        {"norm": 0.0, "mse": False},  # allowed when mse=False
        # combo: all bools True
        {"mse": True, "sym": True},
        # all class defaults
        {
            "wbits": 4,
            "groupsize": -1,
            "sym": False,
            "mse": False,
            "norm": 2.4,
            "grid": 100,
        },
        # all minimum
        {
            "wbits": 1,
            "groupsize": -1,
            "sym": False,
            "mse": False,
            "norm": 1e-5,
            "grid": 1,
        },
        # all maximum
        {
            "wbits": 64,
            "groupsize": 4,
            "sym": True,
            "mse": True,
            "norm": 100.0,
            "grid": 10000,
        },
    ]
    abnormal_parameters = [
        {"wbits": 0},  # below lower boundary (wbits >= 1)
        {"wbits": 65},  # above upper boundary (wbits <= 64)
        {"groupsize": 0},  # between -1 and 1 (invalid)
        {"groupsize": -2},  # just below -1
        {"grid": 0, "mse": True},  # below lower boundary (grid >= 1 when mse=True)
        {"norm": 0.0, "mse": True},  # boundary value (norm > 0, strict)
        {"norm": -1.0, "mse": True},  # negative norm
    ]
    logger = logging.getLogger(__name__)

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
        ]:
            assert hasattr(result, attr)

        for attr in [
            "quantized_weight",
            "scale",
            "zero",
        ]:
            tensor = getattr(result, attr)
            assert isinstance(tensor, torch.Tensor)

        assert result.quantized_weight.dtype == torch.int32
        assert result.quantized_weight.device == torch.device("cpu")
        assert result.scale.dtype == torch.float16
        assert result.scale.device == torch.device("cpu")
        assert result.zero.dtype == torch.float16
        assert result.zero.device == torch.device("cpu")

        assert result.quantized_weight.shape == layer.weight.shape

    def check_equal_results(self, r1, r2):
        """Validate equality of quantization result objects."""
        assert torch.equal(r1.dequantized_weight, r2.dequantized_weight)
        assert torch.equal(r1.quantized_weight, r2.quantized_weight)
        assert torch.equal(r1.scale, r2.scale)
        assert torch.equal(r1.zero, r2.zero)

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
        self.logger.info(
            "[RTN forward error] "
            f"original_vs_rtn(rel={error_original_vs_dequantized:.8f}), "
            f"rtn_vs_rtnl(max={max_error_dequantized_vs_applied:.8f}), "
            f"rtn_vs_rtnl(rel={error_dequantized_vs_applied:.8f})"
        )

        assert max_error_dequantized_vs_applied < 1e-2, (
            f"RTN dequantized vs applied max error too large: "
            f"{max_error_dequantized_vs_applied}"
        )

    def apply_quantized_weights(self, module, result, device):
        """Apply quantized weights to a module."""
        module.weight.data = result.dequantized_weight.to(device)

    def test_forward_error(self, helper):
        """Skip forward error test (no inference layer support)."""
        pytest.skip("RTN does not support create_inference_layer")
