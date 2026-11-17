"""Tests for the JointQ quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.
"""

import os
import sys

import pytest
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from onecomp.quantizer.jointq._jointq import JointQ

    HAS_JOINTQ = True
except ImportError:
    HAS_JOINTQ = False

from test_module import BaseQuantizeSpec

from onecomp.quantizer._quantizer import QuantizationResult


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
        "bits": 2,
        "symmetric": False,
        "group_size": 1,
    }
    # JointQ requires in_features divisible by pack_factor (32 // wbits = 16 for wbits=2).
    _forward_error_features = 32

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
        # regularization retry parameters
        {"regularization_retry_lambda": 0.01},
        {"regularization_retry_lambda": None},
        {"regularization_retry_threshold": 0.0},
        {"regularization_retry_threshold": 0.2},
        {"regularization_retry_threshold": None},
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
        {"regularization_retry_lambda": 0.0},  # must be > 0 or None
        {"regularization_retry_lambda": -0.01},  # must be > 0 or None
        {"regularization_retry_threshold": -0.01},  # must be >= 0 or None
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
        result = self._quantize_with_calculated_hessian(q, layer, inp)
        self.check_quantize_layer(result, layer)


@pytest.mark.skipif(not HAS_JOINTQ, reason="jointq package not installed")
def test_regularization_lambda_by_layer_resolves_exact_and_suffix():
    """Per-layer lambda overrides support full names and longest suffix matches."""
    q = JointQ(
        regularization_lambda=0.2,
        regularization_lambda_by_layer={
            "model.layers.0.self_attn.q_proj": 0.05,
            "self_attn.q_proj": 0.0,
            "q_proj": 0.3,
        },
    )

    assert q._resolve_regularization_lambda("model.layers.0.self_attn.q_proj") == (
        0.05,
        True,
    )
    assert q._resolve_regularization_lambda("model.layers.1.self_attn.q_proj") == (
        0.0,
        True,
    )
    assert q._resolve_regularization_lambda("model.layers.1.mlp.down_proj") == (
        0.2,
        False,
    )


@pytest.mark.skipif(not HAS_JOINTQ, reason="jointq package not installed")
def test_regularization_lambda_by_layer_zero_keeps_retry_enabled():
    """An explicit per-layer 0.0 should still allow fallback retry."""
    q = JointQ(
        regularization_retry_lambda=0.01,
        regularization_retry_threshold=0.2,
    )

    assert q._should_retry_with_regularization(0.0)


@pytest.mark.skipif(not HAS_JOINTQ, reason="jointq package not installed")
@pytest.mark.parametrize(
    "overrides",
    [
        ["self_attn.q_proj"],
        {1: 0.0},
        {"self_attn.q_proj": -0.1},
        {"self_attn.q_proj": None},
    ],
)
def test_regularization_lambda_by_layer_validation(overrides):
    q = JointQ(regularization_lambda_by_layer=overrides)

    with pytest.raises(ValueError, match="regularization_lambda_by_layer"):
        q.validate_params()
