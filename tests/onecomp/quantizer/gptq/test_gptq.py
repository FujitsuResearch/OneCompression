"""Tests for the GPTQ quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import logging
import os
import sys

import pytest
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from test_module import BaseQuantizeSpec

from onecomp.quantizer.gptq._gptq import GPTQ, GPTQResult
from onecomp.quantizer.gptq.config import resolve_gptq_layer_wbits


class TestGPTQ(BaseQuantizeSpec):
    """Test cases for GPTQ quantization."""

    __test__ = True
    quantizer_cls = GPTQ
    result_cls = GPTQResult
    default_parameter_for_test = {"bitpack_on_quantize": False}
    boundary_parameters = [
        # blocksize: int >= 1 (validated by validate_params), no explicit upper
        {"blocksize": 1},  # blocksize lower boundary
        {"blocksize": 1024},  # blocksize large value (no explicit upper bound)
        # percdamp: float >= 3.95e-4 (validated by validate_params), no explicit upper
        {"percdamp": 3.95e-4},  # percdamp lower boundary
        {"percdamp": 1.0},  # percdamp large value (no explicit upper bound)
        # wbits: int in 1..15 (validated by validate_params)
        {"wbits": 1},  # wbits lower boundary
        {"wbits": 15},  # wbits upper boundary
        # groupsize: -1 or >=1 (validated by validate_params), no explicit upper
        {"groupsize": -1},  # groupsize (no grouping)
        {"groupsize": 1},  # groupsize positive lower boundary
        {"groupsize": 1024},  # groupsize large value (no explicit upper bound)
        # actorder: bool
        {"actorder": True},
        {"actorder": False},
        # mse: bool
        {"mse": True},
        {"mse": False},
        # sym: bool
        {"sym": True},
        {"sym": False},
        # q_grid: int >= 1 (validated by validate_params when mse=True), no explicit upper
        {"q_grid": 1, "mse": True},  # q_grid lower boundary
        {"q_grid": 10000, "mse": True},  # q_grid large value (no explicit upper bound)
        # q_norm: float > 0 (validated by validate_params when mse=True), no explicit upper
        {"q_norm": 1e-5, "mse": True},  # q_norm near-zero positive
        {"q_norm": 100.0, "mse": True},  # q_norm large value (no explicit upper bound)
        # q_grid/q_norm: not validated when mse=False
        {"q_grid": 0, "mse": False},  # allowed when mse=False
        {"q_norm": 0.0, "mse": False},  # allowed when mse=False
        # combo: all bools True
        {"actorder": True, "mse": True, "sym": True},
        # all class defaults
        {
            "blocksize": 128,
            "percdamp": 0.01,
            "wbits": 4,
            "groupsize": -1,
            "actorder": False,
            "mse": False,
            "sym": False,
            "q_grid": 600,
            "q_norm": 2.4,
        },
        # all minimum
        {
            "blocksize": 1,
            "percdamp": 3.95e-4,
            "wbits": 1,
            "groupsize": -1,
            "actorder": False,
            "mse": False,
            "sym": False,
            "q_grid": 1,
            "q_norm": 1e-5,
        },
        # all maximum
        {
            "blocksize": 1024,
            "percdamp": 1.0,
            "wbits": 15,
            "groupsize": 1024,
            "actorder": True,
            "mse": True,
            "sym": True,
            "q_grid": 10000,
            "q_norm": 100.0,
        },
    ]
    abnormal_parameters = [
        {"blocksize": 0},  # below lower boundary (blocksize >= 1)
        {"percdamp": 0.0},  # below lower boundary (percdamp >= 3.95e-4)
        {"wbits": 0, "sym": True},  # below lower boundary (wbits >= 1)
        {"wbits": 16},  # above upper boundary (wbits <= 15)
        {"groupsize": 0},  # between -1 and 1 (invalid)
        {"groupsize": -2},  # just below -1
        {"q_grid": 0, "mse": True},  # below lower boundary (q_grid >= 1)
        {"q_norm": 0.0, "mse": True},  # boundary value (q_norm > 0, strict)
    ]
    logger = logging.getLogger(__name__)

    def make_quantizer(self, **params):
        """Instantiate GPTQ with unpacked quantize_layer results by default."""
        merged = {**self.default_parameter_for_test, **params}
        return self.quantizer_cls(**merged)

    @pytest.mark.parametrize("wbits", [2, 3, 4, 8])
    def test_bitpack_supported_wbits_pass(self, wbits):
        """GPTQ bitpacking accepts the packer-supported bit widths."""
        self.make_quantizer(wbits=wbits, bitpack_on_quantize=True).validate_params()

    @pytest.mark.parametrize("wbits", [1, 5, 7, 9, 15])
    def test_bitpack_unsupported_wbits_raise(self, wbits):
        """GPTQ bitpacking rejects valid GPTQ widths that the packer cannot store."""
        q = self.make_quantizer(wbits=wbits, bitpack_on_quantize=True)

        with pytest.raises(ValueError, match="bitpack_on_quantize=True"):
            q.validate_params()

    def test_bitpack_disabled_allows_pack_unsupported_wbits(self):
        """Unpacked GPTQ allows valid GPTQ widths unsupported by the packer."""
        self.make_quantizer(wbits=5, bitpack_on_quantize=False).validate_params()

    @pytest.mark.parametrize(
        "params",
        [
            {"wbits": 15},
            {"mlp_wbits": 15},
            {"module_wbits": {"model.layers.0.mlp.down_proj": 15}},
        ],
    )
    def test_wbits_upper_boundary_passes(self, params):
        """wbits, mlp_wbits, and module_wbits accept the shared 15-bit upper bound."""
        self.make_quantizer(bitpack_on_quantize=False, **params).validate_params()

    @pytest.mark.parametrize(
        "params",
        [
            {"wbits": 16},
            {"mlp_wbits": 16},
            {"module_wbits": {"model.layers.0.mlp.down_proj": 16}},
        ],
    )
    def test_wbits_above_upper_boundary_raise(self, params):
        """wbits, mlp_wbits, and module_wbits reject 16 bits and above."""
        q = self.make_quantizer(bitpack_on_quantize=False, **params)

        with pytest.raises(ValueError, match="1\\.\\.15"):
            q.validate_params()

    @pytest.mark.parametrize(
        "quant_config",
        [
            {"bits": 16},
            {"bits": 4, "mlp_wbits": 16},
            {
                "bits": 4,
                "module_wbits": {"model.layers.0.mlp.down_proj": 16},
            },
            {
                "bits": 4,
                "quantization_bits": [
                    {"mlp.down_proj": {"bits": 16, "method": "gptq"}},
                ],
            },
        ],
    )
    def test_resolve_gptq_layer_wbits_rejects_above_15(self, quant_config):
        """Saved GPTQ quantization_config uses the same 15-bit upper bound."""
        with pytest.raises(ValueError, match="1\\.\\.15"):
            resolve_gptq_layer_wbits("model.layers.0.mlp.down_proj", quant_config)

    @pytest.mark.parametrize("float_bits", [2.0, 2.5])
    def test_resolve_gptq_layer_wbits_rejects_float_bits(self, float_bits):
        """Saved per-layer bit widths reject integral and non-integral floats."""
        quant_config = {
            "bits": 4,
            "quantization_bits": [
                {"mlp.down_proj": {"bits": float_bits, "method": "gptq"}},
            ],
        }
        with pytest.raises(ValueError, match="1\\.\\.15"):
            resolve_gptq_layer_wbits("model.layers.0.mlp.down_proj", quant_config)

    @pytest.mark.parametrize(
        "override_params",
        [
            {"mlp_wbits": 5},
            {"module_wbits": {"model.layers.0.mlp.down_proj": 5}},
        ],
    )
    def test_bitpack_unsupported_override_wbits_raise(self, override_params):
        """Mixed-bit overrides are also pack widths when bitpacking is enabled."""
        q = self.make_quantizer(wbits=4, bitpack_on_quantize=True, **override_params)

        with pytest.raises(ValueError, match="bitpack_on_quantize=True"):
            q.validate_params()

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

        assert result.qweight_is_packed is False
        assert result.qzeros_is_packed is False
        assert result.qweight_original_shape == tuple(layer.weight.shape)
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
