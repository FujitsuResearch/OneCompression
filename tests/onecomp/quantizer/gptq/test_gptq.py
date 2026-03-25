"""Tests for the GPTQ quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import sys
import os
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from onecomp.quantizer.gptq._gptq import GPTQ, GPTQResult

from test_module import BaseQuantizeSpec


class TestGPTQ(BaseQuantizeSpec):
    """Test cases for GPTQ quantization."""

    __test__ = True
    quantizer_cls = GPTQ
    result_cls = GPTQResult
    default_parameter_for_test = {}
    boundary_parameters = [
        {"actorder": True},
        {"mse": True},
        {"sym": False},
        {"actorder": True, "mse": True, "sym": False},
        {
            "blocksize": 1,
            "percdamp": 3.95e-4,
            "wbits": 1,
            "groupsize": -1,
            "q_grid": 1,
            "q_norm": 1e-5,
            "mse": True,
            "sym": False,
        },
    ]
    abnormal_parameters = [
        {"blocksize": -1},
        {"percdamp": -0.1},
        {"wbits": 0},
        {"groupsize": 0},
        {"q_grid": -1, "mse": True},
        {"q_norm": 0.0, "mse": True},
    ]

    def check_quantize_layer(
        self,
        result,
        layer: torch.nn.Module,
    ):
        """Validate types, shapes, and devices of quantize_layer outputs."""
        assert isinstance(result, self.result_cls)
        for attr in [
            "qweight",
            "scales",
            "qzeros",
            "perm",
        ]:
            assert hasattr(result, attr)

        for attr in [
            "qweight",
            "scales",
            "qzeros",
        ]:
            tensor = getattr(result, attr)
            assert isinstance(tensor, torch.Tensor)

        assert result.qweight.dtype == torch.int32
        assert result.qweight.device == torch.device("cpu")
        assert result.scales.dtype == torch.float16
        assert result.scales.device == torch.device("cpu")
        assert result.qzeros.dtype == torch.int32
        assert result.qzeros.device == torch.device("cpu")

        assert result.qweight.shape == layer.weight.shape

        if result.actorder:
            assert isinstance(result.perm, torch.Tensor)
            assert result.perm.ndim == 1
        else:
            assert result.perm is None

    def check_equal_results(self, r1, r2):
        """Validate equality of quantization result objects."""
        assert torch.equal(r1.compute_dequantized_weight(), r2.compute_dequantized_weight())
        assert torch.equal(r1.qweight, r2.qweight)
        assert torch.equal(r1.scales, r2.scales)
        assert torch.equal(r1.qzeros, r2.qzeros)
        if r1.perm is None or r2.perm is None:
            assert r1.perm is None and r2.perm is None
        else:
            assert torch.equal(r1.perm, r2.perm)

    def check_quantize_error(self, error, max_error):
        """Validate that quantization error is within tolerance.

        Thresholds are set for FP16 dequantized weights returned by
        compute_dequantized_weight().
        """
        assert error < 0.6
        assert max_error < 2.5

    def check_forward_error(
        self,
        error_original_vs_dequantized,
        error_dequantized_vs_applied,
        max_error_dequantized_vs_applied,
    ):
        """Validate forward errors."""
        print(
            "[GPTQ forward error] "
            f"original_vs_gptq(rel={error_original_vs_dequantized:.8f}), "
            f"gptq_vs_gptql(max={max_error_dequantized_vs_applied:.8f}), "
            f"gptq_vs_gptql(rel={error_dequantized_vs_applied:.8f})"
        )

        assert max_error_dequantized_vs_applied < 1e-2, (
            f"GPTQ dequantized vs applied max error too large: "
            f"{max_error_dequantized_vs_applied}"
        )

    def apply_quantized_weights(self, module, result, device):
        """Apply quantized weights to a module."""
        dtype = module.weight.data.dtype
        module.weight.data = result.compute_dequantized_weight().to(device).to(dtype)
