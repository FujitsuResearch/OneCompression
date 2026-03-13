"""Tests for the DBF quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import sys
import os
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from onecomp.quantizer.dbf._dbf import DBF, DBFResult

from test_module import BaseQuantizeSpec


class TestDBF(BaseQuantizeSpec):
    """Test cases for DBF quantization."""

    __test__ = True
    quantizer_cls = DBF
    result_cls = DBFResult
    default_parameter_for_test = {
        "target_bits": 1.0,
        "iters": 1,
        "balance_iters": 1,
    }
    boundary_parameters = [
        {"target_bits": 4.0},
        {"balance_mode": "l2"},
        {"use_balancing": False},
        {"use_balancing": False, "use_adaptive_rho": False},
        {
            "target_bits": 1.0,
            "iters": 1,
            "reg": 0.0,
            "balance_iters": 1,
            "balance_alpha": 1.0,
        },
    ]
    abnormal_parameters = [
        {"target_bits": -1.0},
        {"iters": -1},
        {"reg": -1.0},
        {"balance_iters": -1},
        {"balance_alpha": -1.0},
    ]

    def check_quantize_layer(
        self,
        result: dict,
        layer: torch.nn.Module,
    ):
        """Validate types, shapes, and device of quantize_layer outputs."""
        assert isinstance(result, self.result_cls)
        for attr in [
            "is_dbf_quantized",
            "dbf_Da",
            "dbf_A",
            "dbf_mid",
            "dbf_B",
            "dbf_Db",
        ]:
            assert hasattr(result, attr)

        assert isinstance(result.is_dbf_quantized, bool)
        for attr in ["dbf_Da", "dbf_A", "dbf_mid", "dbf_B", "dbf_Db"]:
            tensor = getattr(result, attr)
            assert isinstance(tensor, torch.Tensor)

        assert result.dbf_Da.dtype == torch.float16
        assert result.dbf_Da.device == torch.device("cpu")
        assert result.dbf_A.dtype == torch.float16
        assert result.dbf_A.device == torch.device("cpu")
        assert result.dbf_mid.dtype == torch.float16
        assert result.dbf_mid.device == torch.device("cpu")
        assert result.dbf_B.dtype == torch.float16
        assert result.dbf_B.device == torch.device("cpu")
        assert result.dbf_Db.dtype == torch.float16
        assert result.dbf_Db.device == torch.device("cpu")

        W_reconstructed = result.dbf_A @ torch.diag(result.dbf_mid) @ result.dbf_B
        assert W_reconstructed.shape == layer.weight.shape

        assert torch.all((-1 <= result.dbf_A) & (result.dbf_A <= 1))
        assert torch.all((-1 <= result.dbf_B) & (result.dbf_B <= 1))

    def check_equal_results(self, r1, r2):
        """Validate equality of quantization result objects."""
        assert torch.equal(r1.dequantized_weight, r2.dequantized_weight)
        assert r1.is_dbf_quantized == r2.is_dbf_quantized
        assert torch.equal(r1.dbf_A, r2.dbf_A)
        assert torch.equal(r1.dbf_B, r2.dbf_B)
        assert torch.equal(r1.dbf_mid, r2.dbf_mid)

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
            "[DBF forward error] "
            f"original_vs_dbf(rel={error_original_vs_dequantized:.8f}), "
            f"dbf_vs_dbl(max={max_error_dequantized_vs_applied:.8f}), "
            f"dbf_vs_dbl(rel={error_dequantized_vs_applied:.8f})"
        )

        assert max_error_dequantized_vs_applied < 1e-2, (
            f"DBF dequantized vs applied max error too large: "
            f"{max_error_dequantized_vs_applied}"
        )

    def apply_quantized_weights(self, module, result, device):
        """Apply quantized weights to a module."""
        module.weight.data = result.dequantized_weight.to(device)
        module.dbf_A = result.dbf_A.to(device)
        module.dbf_B = result.dbf_B.to(device)
        module.dbf_mid = result.dbf_mid.to(device)
        module.is_dbf_quantized = True
