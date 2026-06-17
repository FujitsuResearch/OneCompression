"""Tests for the OneBit quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.
"""

import os
import sys

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from test_module import BaseQuantizeSpec

from onecomp.quantizer.onebit._onebit import Onebit, OnebitResult


class TestOnebit(BaseQuantizeSpec):
    """Test cases for OneBit quantization."""

    __test__ = True
    quantizer_cls = Onebit
    result_cls = OnebitResult
    default_parameter_for_test = {
        "iters": 1,
        "balance_iters": 1,
    }
    boundary_parameters = [
        # iters: int >= 0 (validated by validate_params), no explicit upper
        {"iters": 0},  # iters lower boundary
        {"iters": 100},  # iters large value (no explicit upper bound)
        # balance_iters: int >= 1 when use_balancing=True (validated by validate_params)
        {"balance_iters": 1},  # balance_iters lower boundary (use_balancing=True default)
        {"balance_iters": 100},  # balance_iters large value (no explicit upper bound)
        # balance_iters: not validated when use_balancing=False
        {"use_balancing": False, "balance_iters": 0},  # balance_iters=0 allowed when balancing off
        # balance_alpha: float > 0 when use_balancing=True (validated by validate_params)
        {"balance_alpha": 1e-10},  # balance_alpha lower boundary (near zero, positive)
        {"balance_alpha": 100.0},  # balance_alpha large value (no explicit upper bound)
        # balance_alpha: not validated when use_balancing=False
        {
            "use_balancing": False,
            "balance_alpha": 0.0,
        },  # balance_alpha=0 allowed when balancing off
        # use_importance_scaling: bool
        {"use_importance_scaling": True},
        {"use_importance_scaling": False},
        # use_balancing: bool
        {"use_balancing": True},
        {"use_balancing": False},
        # all class defaults
        {
            "iters": 10,
            "use_importance_scaling": True,
            "use_balancing": True,
            "balance_iters": 40,
            "balance_alpha": 1.0,
        },
        # all minimum (use_balancing=False skips balance_* validation)
        {
            "iters": 0,
            "use_importance_scaling": False,
            "use_balancing": False,
            "balance_iters": 0,
            "balance_alpha": 1e-10,
        },
        # all maximum
        {
            "iters": 100,
            "use_importance_scaling": True,
            "use_balancing": True,
            "balance_iters": 100,
            "balance_alpha": 100.0,
        },
    ]
    abnormal_parameters = [
        {"iters": -1},  # below lower boundary (iters >= 0)
        {"balance_iters": 0},  # below lower boundary (balance_iters >= 1 when use_balancing=True)
        {"balance_iters": -1},  # below lower boundary (balance_iters >= 1 when use_balancing=True)
        {"balance_alpha": 0},  # boundary value (balance_alpha > 0 when use_balancing=True)
    ]

    def check_quantize_layer(
        self,
        result,
        layer: torch.nn.Module,
    ):
        """Validate types, shapes, and devices of quantize_layer outputs."""
        assert isinstance(result, self.result_cls)
        for attr in [
            "a",
            "b",
            "sign",
        ]:
            assert hasattr(result, attr)

        for attr in [
            "a",
            "b",
            "sign",
        ]:
            tensor = getattr(result, attr)
            assert isinstance(tensor, torch.Tensor)

        assert result.sign.dtype == torch.float16
        assert result.sign.device == torch.device("cpu")
        assert torch.all((result.sign == 1) | (result.sign == -1))

        assert result.a.dtype == torch.float16
        assert result.a.device == torch.device("cpu")
        assert result.a.ndim == 1
        assert result.a.shape[0] == layer.weight.shape[0]

        assert result.b.dtype == torch.float16
        assert result.b.device == torch.device("cpu")
        assert result.b.ndim == 1
        assert result.b.shape[0] == layer.weight.shape[1]

        assert result.sign.shape == layer.weight.shape

    def check_equal_results(self, r1, r2):
        """Validate equality of quantization result objects."""
        assert torch.equal(r1.compute_dequantized_weight(), r2.compute_dequantized_weight())
        assert torch.equal(r1.a, r2.a)
        assert torch.equal(r1.b, r2.b)
        assert torch.equal(r1.sign, r2.sign)

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
            "[OneBit forward error] "
            f"original_vs_onebit(rel={error_original_vs_dequantized:.8f}), "
            f"onebit_vs_onebitl(max={max_error_dequantized_vs_applied:.8f}), "
            f"onebit_vs_onebitl(rel={error_dequantized_vs_applied:.8f})"
        )

        assert max_error_dequantized_vs_applied < 1e-2, (
            f"OneBit dequantized vs applied max error too large: "
            f"{max_error_dequantized_vs_applied}"
        )

    def apply_quantized_weights(self, module, result, device):
        """Apply quantized weights to a module."""
        dtype = module.weight.data.dtype
        module.weight.data = result.compute_dequantized_weight().to(device).to(dtype)
