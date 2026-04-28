"""Tests for the JointQ quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.
"""

import sys
import os
import torch
import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from onecomp.quantizer.jointq._jointq import JointQ

    HAS_JOINTQ = True
except ImportError:
    HAS_JOINTQ = False

from onecomp.quantizer._quantizer import QuantizationResult
from test_module import BaseQuantizeSpec


@pytest.mark.skipif(not HAS_JOINTQ, reason="jointq package not installed")
class TestJointQ(BaseQuantizeSpec):
    """Test cases for JointQ quantization.

    Note: JointQ requires the external `jointq` package.
    JointQ returns a plain tensor (auto-wrapped as QuantizationResult).
    """

    __test__ = True
    quantizer_cls = JointQ if HAS_JOINTQ else None
    result_cls = QuantizationResult
    default_parameter_for_test = {
        "bits": 1,
        "symmetric": False,
        "group_size": 1,
    }
    boundary_parameters = [
        # bits: int >= 1 (validated by validate_params), no explicit upper
        {"bits": 1},  # bits lower boundary
        {"bits": 4},  # bits upper boundary (jointq core supports 1-4)
        # symmetric: bool
        {"symmetric": True},
        {"symmetric": False},
        # group_size: int >= 1 (validated by validate_params), no explicit upper
        {"group_size": 1},  # group_size lower boundary
        {"group_size": 128},  # group_size large value (no explicit upper bound)
        # log_level: int in 0..2 (validated by validate_params)
        {"log_level": 0},  # log_level lower boundary
        {"log_level": 2},  # log_level upper boundary
        # ils_enabled: bool
        {"ils_enabled": True},
        {"ils_enabled": False},
        # ILS sub-params: lower boundaries (no explicit upper)
        {"ils_num_iterations": 1},  # ils_num_iterations lower boundary
        {"ils_num_clones": 1},  # ils_num_clones lower boundary
        {"ils_num_channels": 1},  # ils_num_channels lower boundary
        # ILS sub-params: large values (no explicit upper)
        {"ils_num_iterations": 100},  # ils_num_iterations large value (no explicit upper bound)
        {"ils_num_clones": 100},  # ils_num_clones large value (no explicit upper bound)
        {"ils_num_channels": 10000},  # ils_num_channels large value (no explicit upper bound)
        # ILS combo: lower boundaries
        {"ils_enabled": True, "ils_num_iterations": 1, "ils_num_clones": 1, "ils_num_channels": 1},
        # ils_num_channels: None is also valid (auto-detect)
        {
            "ils_enabled": True,
            "ils_num_iterations": 1,
            "ils_num_clones": 1,
            "ils_num_channels": None,
        },
        # all class defaults
        {
            "bits": 4,
            "symmetric": False,
            "group_size": 128,
            "log_level": 1,
            "ils_enabled": True,
            "ils_num_iterations": 10,
            "ils_num_clones": 8,
            "ils_num_channels": 512,
        },
        # all minimum
        {
            "bits": 1,
            "symmetric": False,
            "group_size": 1,
            "log_level": 0,
            "ils_enabled": False,
            "ils_num_iterations": 1,
            "ils_num_clones": 1,
            "ils_num_channels": 1,
        },
        # all maximum
        {
            "bits": 4,
            "symmetric": True,
            "group_size": 128,
            "log_level": 2,
            "ils_enabled": True,
            "ils_num_iterations": 100,
            "ils_num_clones": 100,
            "ils_num_channels": 10000,
        },
        # lambda_mode: fixed_lambda (explicit)
        {"lambda_mode": "fixed_lambda"},
        # lambda_mode: incremental_lambda with defaults
        {"lambda_mode": "incremental_lambda"},
        # lambda_mode: incremental_lambda with custom lambda_list
        {"lambda_mode": "incremental_lambda", "lambda_list": [0.0, 0.1, 0.5]},
        # lambda_mode: incremental_lambda with single-element lambda_list
        {"lambda_mode": "incremental_lambda", "lambda_list": [0.5]},
        # lambda_mode: incremental_lambda with custom eps values
        {
            "lambda_mode": "incremental_lambda",
            "incremental_eps_y": 0.0,
            "incremental_eps_w": 0.0,
        },
        # lambda_mode: incremental_lambda with large eps values
        {
            "lambda_mode": "incremental_lambda",
            "incremental_eps_y": 1.0,
            "incremental_eps_w": 1.0,
        },
        {
            "lambda_mode": "incremental_lambda",
            "incremental_initial_skip_ew_threshold": 1.0,
        },
        {
            "lambda_mode": "incremental_lambda",
            "incremental_initial_skip_ew_threshold": None,
        },
    ]
    abnormal_parameters = [
        {"bits": 0},  # below lower boundary (bits >= 1)
        {"group_size": 0},  # below lower boundary (group_size >= 1)
        {"log_level": -1},  # below lower boundary (log_level in 0..2)
        {"log_level": 3},  # above upper boundary (log_level in 0..2)
        {
            "ils_enabled": True,
            "ils_num_iterations": 0,
        },  # below lower boundary (ils_num_iterations >= 1)
        {"ils_enabled": True, "ils_num_clones": 0},  # below lower boundary (ils_num_clones >= 1)
        {
            "ils_enabled": True,
            "ils_num_channels": 0,
        },  # below lower boundary (ils_num_channels >= 1 or None)
        {"lambda_mode": "invalid_mode"},  # invalid lambda_mode
        {
            "lambda_mode": "incremental_lambda",
            "lambda_list": [],
        },  # empty lambda_list
        {
            "lambda_mode": "incremental_lambda",
            "lambda_list": [-1.0, 0.5],
        },  # negative value in lambda_list
        {
            "lambda_mode": "incremental_lambda",
            "incremental_eps_y": -0.1,
        },  # negative eps_y
        {
            "lambda_mode": "incremental_lambda",
            "incremental_eps_w": -0.1,
        },  # negative eps_w
        {
            "lambda_mode": "incremental_lambda",
            "incremental_initial_skip_ew_threshold": -0.1,
        },  # negative initial Ew threshold
    ]

    def make_quantizer(self, **params):
        """Pin pre-change regularization defaults for backward-compat.

        Tests in this class were authored when JointQ defaults were
        regularization_lambda=0.2 and regularization_mode="identity".
        The library defaults have since changed; explicitly inject the
        old values when callers do not specify them so existing tests
        keep their original behavior. Callers can still override.
        """
        params.setdefault("regularization_lambda", 0.2)
        params.setdefault("regularization_mode", "identity")
        return self.quantizer_cls(**params)

    def check_quantize_layer(
        self,
        result,
        layer: torch.nn.Module,
    ):
        """Validate types, shapes, and devices of quantize_layer outputs."""
        assert isinstance(result, self.result_cls)
        dw = result.compute_dequantized_weight()
        assert isinstance(dw, torch.Tensor)
        assert dw.shape == layer.weight.shape
        assert dw.device == torch.device("cpu")

    def check_equal_results(self, r1, r2):
        """Validate equality of quantization result objects."""
        assert torch.equal(r1.compute_dequantized_weight(), r2.compute_dequantized_weight())

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
            "[JointQ forward error] "
            f"original_vs_jointq(rel={error_original_vs_dequantized:.8f}), "
            f"jointq_vs_jointql(max={max_error_dequantized_vs_applied:.8f}), "
            f"jointq_vs_jointql(rel={error_dequantized_vs_applied:.8f})"
        )

        assert max_error_dequantized_vs_applied < 1e-2, (
            f"JointQ dequantized vs applied max error too large: "
            f"{max_error_dequantized_vs_applied}"
        )

    def apply_quantized_weights(self, module, result, device):
        """Apply quantized weights to a module."""
        dw = result.compute_dequantized_weight().to(device=device, dtype=module.weight.dtype)
        module.weight.data = dw

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_quantize_layer_returns(self, device, helper):
        """Skip CPU; JointQ is GPU-based."""
        if device == "cpu":
            pytest.skip("JointQ is GPU-based, skipping CPU test")
        super().test_quantize_layer_returns(device, helper)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_quantize_layer_reproducibility(self, device, helper):
        """Skip CPU; JointQ is GPU-based."""
        if device == "cpu":
            pytest.skip("JointQ is GPU-based, skipping CPU test")
        super().test_quantize_layer_reproducibility(device, helper)

    def test_cpu_gpu_output_match(self, helper):
        """Skip; JointQ is GPU-based."""
        pytest.skip("JointQ is GPU-based, CPU/GPU comparison not applicable")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_parameters_boundary(self, params, helper):
        """Override to use a larger layer on CUDA, compatible with group_size up to 128."""
        device = "cuda"
        layer = helper.make_linear(128, 128, device=device, dtype=torch.float32)
        inp = helper.make_input(batch=1, seq=1, hidden=128, device=device, dtype=torch.float32)

        q = self.make_quantizer(**params)
        hessian = q.calculate_hessian(layer, inp)
        result = q.quantize_layer(layer, inp, hessian=hessian)
        self.check_quantize_layer(result, layer)

    def test_forward_error(self, helper):
        """Skip forward error test (no inference layer support)."""
        pytest.skip("JointQ does not support create_inference_layer")

    # ------------------------------------------------------------------
    # incremental_lambda mode tests
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_incremental_lambda_basic(self, helper):
        """Verify incremental_lambda mode completes and returns a valid result."""
        device = "cuda"
        layer = helper.make_linear(128, 128, device=device, dtype=torch.float32)
        inp = helper.make_input(batch=1, seq=1, hidden=128, device=device, dtype=torch.float32)

        q = self.make_quantizer(
            bits=4,
            symmetric=False,
            group_size=128,
            lambda_mode="incremental_lambda",
            lambda_list=[0.01, 0.5, 1.0],
            log_level=1,
        )
        hessian = q.calculate_hessian(layer, inp)
        result = q.quantize_layer(layer, inp, hessian=hessian)
        self.check_quantize_layer(result, layer)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_incremental_lambda_single_step(self, helper):
        """Verify incremental_lambda with a single lambda value works."""
        device = "cuda"
        layer = helper.make_linear(128, 128, device=device, dtype=torch.float32)
        inp = helper.make_input(batch=1, seq=1, hidden=128, device=device, dtype=torch.float32)

        q = self.make_quantizer(
            bits=4,
            symmetric=False,
            group_size=128,
            lambda_mode="incremental_lambda",
            lambda_list=[0.2],
        )
        hessian = q.calculate_hessian(layer, inp)
        result = q.quantize_layer(layer, inp, hessian=hessian)
        self.check_quantize_layer(result, layer)

    def test_accept_candidate_both_decrease(self):
        """Rule 1: both Ew and Ey decrease -> accept."""
        accepted, reason = JointQ._accept_candidate(
            ew_prev=0.10, ey_prev=0.05, ew_new=0.08, ey_new=0.04,
            eps_y=0.03, eps_w=0.10,
        )
        assert accepted is True
        assert "both decreased" in reason

    def test_accept_candidate_ew_increase(self):
        """Rule 2: Ew increased -> reject."""
        accepted, reason = JointQ._accept_candidate(
            ew_prev=0.10, ey_prev=0.05, ew_new=0.11, ey_new=0.04,
            eps_y=0.03, eps_w=0.10,
        )
        assert accepted is False
        assert "Ew increased" in reason

    def test_accept_candidate_tradeoff_accept(self):
        """Rule 3: Ey worsened within eps_y, Ew improved by at least eps_w -> accept."""
        accepted, reason = JointQ._accept_candidate(
            ew_prev=0.20, ey_prev=0.05, ew_new=0.17, ey_new=0.051,
            eps_y=0.03, eps_w=0.10,
        )
        assert accepted is True
        assert "tolerance" in reason

    def test_accept_candidate_tradeoff_reject_ey_too_large(self):
        """Rule 3 fail: Ey worsened beyond eps_y -> reject."""
        accepted, reason = JointQ._accept_candidate(
            ew_prev=0.20, ey_prev=0.05, ew_new=0.17, ey_new=0.06,
            eps_y=0.03, eps_w=0.10,
        )
        assert accepted is False
        assert "eps_y" in reason

    def test_accept_candidate_tradeoff_reject_ew_insufficient(self):
        """Rule 3 fail: Ew did not improve enough -> reject."""
        accepted, reason = JointQ._accept_candidate(
            ew_prev=0.20, ey_prev=0.05, ew_new=0.19, ey_new=0.051,
            eps_y=0.03, eps_w=0.10,
        )
        assert accepted is False
        assert "eps_w" in reason

    def test_accept_candidate_ew_equal_ey_equal(self):
        """Edge case: Ew and Ey unchanged -> reject (no strict decrease)."""
        accepted, _ = JointQ._accept_candidate(
            ew_prev=0.10, ey_prev=0.05, ew_new=0.10, ey_new=0.05,
            eps_y=0.03, eps_w=0.10,
        )
        assert accepted is False

    def test_accept_candidate_no_eps(self):
        """Without eps parameters, rule 3 cannot fire; only rules 1-2 apply."""
        # Ey worsened, Ew improved, but no eps -> reject
        accepted, _ = JointQ._accept_candidate(
            ew_prev=0.20, ey_prev=0.05, ew_new=0.17, ey_new=0.051,
        )
        assert accepted is False
        # Both decrease -> still accept
        accepted, _ = JointQ._accept_candidate(
            ew_prev=0.20, ey_prev=0.05, ew_new=0.17, ey_new=0.04,
        )
        assert accepted is True

    def test_initial_zero_lambda_skip_guard_default(self):
        """Initial zero-lambda skip guard defaults to 30% Ew."""
        q = self.make_quantizer(lambda_mode="incremental_lambda")
        assert q.incremental_initial_skip_ew_threshold == 0.3
