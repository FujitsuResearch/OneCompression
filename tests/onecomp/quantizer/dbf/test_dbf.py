"""Tests for the DBF quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import logging
import os
import sys

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from test_module import BaseQuantizeSpec

from onecomp.quantizer.dbf._dbf import DBF, DBFResult


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
        # target_bits: float > 0 (validated by validate_params), no explicit upper
        {"target_bits": 1e-10},  # target_bits lower boundary (near zero, positive)
        {"target_bits": 100.0},  # target_bits large value (no explicit upper bound)
        # iters: int >= 1 (validated by validate_params), no explicit upper
        {"iters": 1},  # iters lower boundary
        {"iters": 100},  # iters large value (no explicit upper bound)
        # reg: float >= 0 (validated by validate_params), no explicit upper
        {"reg": 0.0},  # reg lower boundary
        {"reg": 100.0},  # reg large value (no explicit upper bound)
        # use_balancing: bool
        {"use_balancing": True},
        {"use_balancing": False},
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
        # balance_mode: str in {l1, l2} when use_balancing=True (validated by validate_params)
        {"balance_mode": "l1"},
        {"balance_mode": "l2"},
        # balance_mode: not validated when use_balancing=False
        {
            "use_balancing": False,
            "balance_mode": "invalid",
        },  # invalid mode allowed when balancing off
        # use_adaptive_rho: bool
        {"use_adaptive_rho": True},
        {"use_adaptive_rho": False},
        # combo: bools False
        {"use_balancing": False, "use_adaptive_rho": False},
        # combo: numerics at lower bounds
        {
            "target_bits": 1.0,
            "iters": 1,
            "reg": 0.0,
            "balance_iters": 1,
            "balance_alpha": 1.0,
        },
        # all class defaults
        {
            "target_bits": 1.5,
            "iters": 600,
            "reg": 3e-2,
            "use_balancing": True,
            "balance_iters": 40,
            "balance_alpha": 1.0,
            "balance_mode": "l1",
            "use_adaptive_rho": True,
        },
        # all minimum (use_balancing=False skips balance_* validation)
        {
            "target_bits": 1e-10,
            "iters": 1,
            "reg": 0.0,
            "use_balancing": False,
            "balance_iters": 0,
            "balance_alpha": 1e-10,
            "balance_mode": "l1",
            "use_adaptive_rho": False,
        },
        # all maximum
        {
            "target_bits": 100.0,
            "iters": 100,
            "reg": 100.0,
            "use_balancing": True,
            "balance_iters": 100,
            "balance_alpha": 100.0,
            "balance_mode": "l2",
            "use_adaptive_rho": True,
        },
    ]
    abnormal_parameters = [
        {"target_bits": 0.0},  # boundary value (target_bits > 0, strict)
        {"iters": 0},  # below lower boundary (iters >= 1)
        {"reg": -0.01},  # below lower boundary (reg >= 0)
        {"balance_iters": 0},  # below lower boundary (balance_iters >= 1 when use_balancing=True)
        {"balance_iters": -1},  # below lower boundary (balance_iters >= 1 when use_balancing=True)
        {"balance_alpha": 0.0},  # boundary value (balance_alpha > 0 when use_balancing=True)
        {"balance_mode": "invalid"},  # not in {l1, l2} when use_balancing=True
    ]
    logger = logging.getLogger(__name__)

    def check_quantize_layer(
        self,
        result: dict,
        layer: torch.nn.Module,
    ):
        """Validate types, shapes, and device of quantize_layer outputs.

        Handles both representations of the binary factors:
        - ``bitpack_on_quantize=True``  : dbf_A / dbf_B are packed uint8 (1D).
        - ``bitpack_on_quantize=False`` : dbf_A / dbf_B are unpacked ±1 float16.
        """
        assert isinstance(result, self.result_cls)
        for attr in [
            "is_dbf_quantized",
            "dbf_Da",
            "dbf_A",
            "dbf_mid",
            "dbf_B",
            "dbf_Db",
            "dbf_A_is_packed",
            "dbf_B_is_packed",
        ]:
            assert hasattr(result, attr)

        assert isinstance(result.is_dbf_quantized, bool)
        for attr in ["dbf_Da", "dbf_A", "dbf_mid", "dbf_B", "dbf_Db"]:
            tensor = getattr(result, attr)
            assert isinstance(tensor, torch.Tensor)

        # Scaling vectors are always unpacked float16 CPU tensors.
        assert result.dbf_Da.dtype == torch.float16
        assert result.dbf_Da.device == torch.device("cpu")
        assert result.dbf_mid.dtype == torch.float16
        assert result.dbf_mid.device == torch.device("cpu")
        assert result.dbf_Db.dtype == torch.float16
        assert result.dbf_Db.device == torch.device("cpu")

        # Binary factor storage depends on the bitpack flag.
        if result.dbf_A_is_packed:
            assert result.dbf_A.dtype == torch.uint8
            assert result.dbf_A.device == torch.device("cpu")
            assert result.dbf_A.ndim == 1
            assert result.dbf_A_original_shape is not None
        else:
            assert result.dbf_A.dtype == torch.float16
            assert result.dbf_A.device == torch.device("cpu")

        if result.dbf_B_is_packed:
            assert result.dbf_B.dtype == torch.uint8
            assert result.dbf_B.device == torch.device("cpu")
            assert result.dbf_B.ndim == 1
            assert result.dbf_B_original_shape is not None
        else:
            assert result.dbf_B.dtype == torch.float16
            assert result.dbf_B.device == torch.device("cpu")

        # Recover the unpacked ±1 factors via the representation-agnostic helper
        # and validate shapes/values uniformly.
        dbf_A, dbf_B = result.get_unpacked_binary_factors()
        W_reconstructed = dbf_A.float() @ torch.diag(result.dbf_mid.float()) @ dbf_B.float()
        assert W_reconstructed.shape == layer.weight.shape

        assert torch.all((-1 <= dbf_A) & (dbf_A <= 1))
        assert torch.all((-1 <= dbf_B) & (dbf_B <= 1))

    def check_equal_results(self, r1, r2):
        """Validate equality of quantization result objects."""
        assert torch.equal(r1.compute_dequantized_weight(), r2.compute_dequantized_weight())
        assert r1.is_dbf_quantized == r2.is_dbf_quantized
        # Packed-state metadata must match for reproducibility.
        assert r1.dbf_A_is_packed == r2.dbf_A_is_packed
        assert r1.dbf_B_is_packed == r2.dbf_B_is_packed
        # Compare via the representation-agnostic helper so the check holds for
        # both packed (uint8) and unpacked (float16) storage.
        a1, b1 = r1.get_unpacked_binary_factors()
        a2, b2 = r2.get_unpacked_binary_factors()
        assert torch.equal(a1, a2)
        assert torch.equal(b1, b2)
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
        self.logger.info(
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
        dtype = module.weight.data.dtype
        module.weight.data = result.compute_dequantized_weight().to(device).to(dtype)
        # Store the unpacked ±1 factors so the attributes are usable regardless
        # of whether the result kept them packed.
        dbf_A, dbf_B = result.get_unpacked_binary_factors(device=device)
        module.dbf_A = dbf_A
        module.dbf_B = dbf_B
        module.dbf_mid = result.dbf_mid.to(device)
        module.is_dbf_quantized = True
