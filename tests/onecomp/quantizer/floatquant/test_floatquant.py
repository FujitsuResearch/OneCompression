"""Tests for the FloatQuant quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import logging
import os
import sys

import pytest
import torch
from torch import nn
from transformers import Conv1D

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from onecomp.quantizer.floatquant._floatquant import FloatQuant, FloatQuantResult

logger = logging.getLogger(__name__)

IN_FEATURES = 64
OUT_FEATURES = 32


def make_layer(seed=123, in_features=IN_FEATURES, out_features=OUT_FEATURES):
    """Create a deterministic linear layer for test use."""
    torch.manual_seed(seed)
    return nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)


def make_input(seed=123, batch=2, seq=4, hidden=IN_FEATURES):
    """Create a deterministic random input for test use."""
    torch.manual_seed(seed + 1)
    return torch.randn(batch, seq, hidden, dtype=torch.float32)


class TestFP4Defaults:
    """Tests for format-dependent defaults and flags."""

    def test_default_block_sizes(self):
        """block_size defaults depend on the format."""
        assert FloatQuant(fmt="nvfp4").block_size == 16
        assert FloatQuant(fmt="mxfp4").block_size == 32
        assert FloatQuant(fmt="fp8").block_size == -1

    def test_explicit_block_size_kept(self):
        """An explicitly provided block_size is not overridden."""
        assert FloatQuant(fmt="nvfp4", block_size=8).block_size == 8

    def test_flags_follow_use_hessian(self):
        """flag_calibration and flag_hessian follow use_hessian."""
        q = FloatQuant(use_hessian=False)
        assert q.flag_calibration is False
        assert q.flag_hessian is False

        q = FloatQuant(use_hessian=True)
        assert q.flag_calibration is True
        assert q.flag_hessian is True


class TestFP4ValidateParams:
    """Tests for parameter validation."""

    @pytest.mark.parametrize(
        "params",
        [
            {"fmt": "nvfp4"},
            {"fmt": "mxfp4"},
            {"fmt": "fp8"},
            {"fmt": "nvfp4", "block_size": 8},
            {"fmt": "mxfp4", "block_size": 16},
            {"fmt": "nvfp4", "use_hessian": True},
            {"fmt": "nvfp4", "use_hessian": True, "scale_timing": "static"},
            {
                "fmt": "nvfp4",
                "use_hessian": True,
                "scale_timing": "static",
                "scale_objective": "diag_wmse",
            },
            {
                "fmt": "nvfp4",
                "use_hessian": True,
                "scale_timing": "in_loop",
                "scale_objective": "conditional",
            },
            {
                "fmt": "nvfp4",
                "use_hessian": True,
                "scale_timing": "in_loop",
                "scale_candidate_strategy": "adaptive",
            },
            {
                "fmt": "mxfp4",
                "use_hessian": True,
                "scale_timing": "static",
                "scale_candidate_strategy": "full",
            },
            {"fmt": "mxfp4", "scale_timing": "static"},
            {"fmt": "nvfp4", "use_hessian": True, "blocksize": 1},
            {"fmt": "nvfp4", "use_hessian": True, "percdamp": 0.05},
        ],
    )
    def test_valid_parameters(self, params):
        """Valid parameter combinations pass setup()."""
        q = FloatQuant(**params)
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        q.setup(model)

    @pytest.mark.parametrize(
        "params",
        [
            {"fmt": "int4"},  # unsupported format
            {"fmt": ""},  # empty format
            {"fmt": "nvfp4", "block_size": 0},  # below lower boundary
            {"fmt": "nvfp4", "block_size": -1},  # per-channel not valid for fp4
            {"fmt": "mxfp4", "block_size": -3},  # negative block size
            {"fmt": "fp8", "block_size": 16},  # fp8 uses per-channel scales
            {"fmt": "nvfp4", "use_hessian": "yes"},  # non-bool use_hessian
            {"fmt": "nvfp4", "scale_timing": "later"},  # unsupported timing
            {"fmt": "nvfp4", "scale_objective": "loss"},  # unsupported objective
            {"fmt": "nvfp4", "scale_candidate_strategy": "random"},  # unsupported strategy
            {
                "fmt": "nvfp4",
                "scale_objective": "diag_wmse",
            },  # weighted objective needs a Hessian
            {
                "fmt": "nvfp4",
                "use_hessian": True,
                "scale_timing": "static",
                "scale_objective": "conditional",
            },  # conditional uses in-loop block metrics
            {"fmt": "nvfp4", "use_hessian": True, "blocksize": 0},  # invalid blocksize
            {"fmt": "nvfp4", "use_hessian": True, "percdamp": 0.0},  # invalid percdamp
        ],
    )
    def test_abnormal_parameters_raise(self, params):
        """Abnormal parameter values raise on setup()."""
        q = FloatQuant(**params)
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        with pytest.raises(Exception):
            q.setup(model)


class TestFP4QuantizeLayer:
    """Integration tests for quantize_layer."""

    @pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4", "fp8"])
    @pytest.mark.parametrize("use_hessian", [False, True])
    def test_quantize_layer_returns(self, fmt, use_hessian):
        """quantize_layer returns a well-formed FloatQuantResult for all formats."""
        layer = make_layer()
        inp = make_input()

        q = FloatQuant(fmt=fmt, use_hessian=use_hessian)
        hessian = q.calculate_hessian(layer, inp)
        result = q.quantize_layer(layer, inp, hessian=hessian)

        assert isinstance(result, FloatQuantResult)
        assert result.fmt == fmt
        assert result.dequantized_weight.shape == layer.weight.shape
        assert result.dequantized_weight.device == torch.device("cpu")
        assert torch.isfinite(result.dequantized_weight).all()

        if fmt == "fp8":
            assert result.codes is None
            assert result.block_scales.shape == (OUT_FEATURES, 1)
            assert result.tensor_scale is None
        else:
            assert result.codes is not None
            assert result.codes.dtype == torch.int8
            assert result.codes.shape == layer.weight.shape
            num_blocks = IN_FEATURES // result.block_size
            assert result.block_scales.shape == (OUT_FEATURES, num_blocks)
            if fmt == "nvfp4":
                assert result.tensor_scale is not None
            else:
                assert result.tensor_scale is None

    @pytest.mark.parametrize(
        "fmt,max_rel_error", [("nvfp4", 0.15), ("mxfp4", 0.25), ("fp8", 0.05)]
    )
    def test_quantization_error_bound(self, fmt, max_rel_error):
        """Relative Frobenius weight error stays within the expected bound."""
        layer = make_layer(seed=0, in_features=128, out_features=128)
        q = FloatQuant(fmt=fmt)
        result = q.quantize_layer(layer)
        w = layer.weight.data
        w_hat = result.dequantized_weight
        rel = (torch.norm(w - w_hat) / torch.norm(w)).item()
        logger.info("[FloatQuant %s] relative weight error: %.6f", fmt, rel)
        assert rel < max_rel_error

    @pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4", "fp8"])
    def test_hessian_reduces_output_error(self, fmt):
        """Error compensation does not increase the layer output error."""
        layer = make_layer(seed=7, in_features=128, out_features=64)
        inp = make_input(seed=7, batch=4, seq=16, hidden=128)
        x = inp.reshape(-1, 128)

        q_direct = FloatQuant(fmt=fmt, use_hessian=False)
        direct = q_direct.quantize_layer(layer)

        q_hess = FloatQuant(fmt=fmt, use_hessian=True)
        hessian = q_hess.calculate_hessian(layer, inp)
        compensated = q_hess.quantize_layer(layer, inp, hessian=hessian)

        w = layer.weight.data
        err_direct = torch.norm(x @ (w - direct.dequantized_weight).T).item()
        err_hess = torch.norm(x @ (w - compensated.dequantized_weight).T).item()
        logger.info(
            "[FloatQuant %s] output error direct=%.6f, hessian=%.6f", fmt, err_direct, err_hess
        )
        assert err_hess <= err_direct * 1.05

    @pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
    @pytest.mark.parametrize("use_hessian", [False, True])
    def test_compute_dequantized_weight_matches(self, fmt, use_hessian):
        """compute_dequantized_weight reproduces the stored dequantized weight."""
        layer = make_layer()
        inp = make_input()

        q = FloatQuant(fmt=fmt, use_hessian=use_hessian)
        hessian = q.calculate_hessian(layer, inp)
        result = q.quantize_layer(layer, inp, hessian=hessian)

        recomputed = result.compute_dequantized_weight()
        stored = result.dequantized_weight.to(torch.float16)
        assert torch.allclose(recomputed, stored, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("use_hessian", [False, True])
    def test_reproducibility(self, use_hessian):
        """Identical layers and inputs produce identical results."""
        layer1 = make_layer()
        layer2 = make_layer()
        layer2.weight.data.copy_(layer1.weight.data)
        inp = make_input()

        q = FloatQuant(fmt="nvfp4", use_hessian=use_hessian)
        h1 = q.calculate_hessian(layer1, inp)
        h2 = q.calculate_hessian(layer2, inp)
        r1 = q.quantize_layer(layer1, inp, hessian=h1)
        r2 = q.quantize_layer(layer2, inp, hessian=h2)

        assert torch.equal(r1.dequantized_weight, r2.dequantized_weight)
        assert torch.equal(r1.codes, r2.codes)
        assert torch.equal(r1.block_scales, r2.block_scales)

    def test_use_hessian_requires_hessian(self):
        """use_hessian=True without a Hessian raises ValueError."""
        layer = make_layer()
        q = FloatQuant(fmt="nvfp4", use_hessian=True)
        with pytest.raises(ValueError):
            q.quantize_layer(layer, hessian=None)

    def test_indivisible_block_size_raises(self):
        """A block size that does not divide in_features raises ValueError."""
        layer = make_layer(in_features=24)
        q = FloatQuant(fmt="nvfp4", block_size=16)
        with pytest.raises(ValueError):
            q.quantize_layer(layer)

    @pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
    @pytest.mark.parametrize(
        "scale_timing,scale_objective",
        [
            ("static", "mse"),
            ("static", "diag_wmse"),
            ("in_loop", "diag_wmse"),
            ("in_loop", "conditional"),
        ],
    )
    def test_hessian_scale_modes_return_finite_results(self, fmt, scale_timing, scale_objective):
        """Static, in-loop, and conditional sweeps are valid GPTQ modes."""
        layer = make_layer(seed=19, in_features=64, out_features=32)
        inp = make_input(seed=19, batch=3, seq=8, hidden=64)
        q = FloatQuant(
            fmt=fmt,
            use_hessian=True,
            scale_timing=scale_timing,
            scale_objective=scale_objective,
            blocksize=32,
        )
        hessian = q.calculate_hessian(layer, inp)
        result = q.quantize_layer(layer, inp, hessian=hessian)
        assert torch.isfinite(result.dequantized_weight).all()
        assert result.codes.shape == layer.weight.shape

    def test_scale_timing_static_enables_direct_sweep(self):
        """Direct quantization can request the static sweep without scale_search."""
        layer = make_layer(seed=23, in_features=64, out_features=32)
        q = FloatQuant(fmt="nvfp4", scale_timing="static")
        result = q.quantize_layer(layer)
        assert torch.isfinite(result.dequantized_weight).all()

    @pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
    @pytest.mark.parametrize("strategy", ["full", "adaptive"])
    def test_nonlocal_candidate_strategies_return_finite_results(self, fmt, strategy):
        """Full-grid and adaptive candidate strategies are valid sweep ablations."""
        layer = make_layer(seed=31, in_features=64, out_features=32)
        inp = make_input(seed=31, batch=3, seq=8, hidden=64)
        q = FloatQuant(
            fmt=fmt,
            use_hessian=True,
            scale_timing="static",
            scale_objective="mse",
            scale_candidate_strategy=strategy,
            blocksize=32,
        )
        hessian = q.calculate_hessian(layer, inp)
        result = q.quantize_layer(layer, inp, hessian=hessian)
        assert torch.isfinite(result.dequantized_weight).all()

    @pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
    def test_full_candidate_strategy_is_not_worse_than_local_mse(self, fmt):
        """Full-grid MSE search is a superset of the local candidate window."""
        layer = make_layer(seed=37, in_features=64, out_features=32)
        local = FloatQuant(fmt=fmt, scale_timing="static", scale_candidate_strategy="local")
        full = FloatQuant(fmt=fmt, scale_timing="static", scale_candidate_strategy="full")
        local_result = local.quantize_layer(layer)
        full_result = full.quantize_layer(layer)
        local_err = torch.norm(layer.weight.data - local_result.dequantized_weight).item()
        full_err = torch.norm(layer.weight.data - full_result.dequantized_weight).item()
        assert full_err <= local_err + 1e-6

    @pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
    def test_static_mse_sweep_changes_hessian_block_scales(self, fmt):
        """Static MSE mode must sweep candidates, not fall back to defaults."""
        layer = make_layer(seed=0, in_features=64, out_features=32)
        inp = make_input(seed=0, batch=3, seq=8, hidden=64)
        q_default = FloatQuant(fmt=fmt, use_hessian=True, scale_timing="none", blocksize=32)
        hessian = q_default.calculate_hessian(layer, inp)
        default_result = q_default.quantize_layer(layer, inp, hessian=hessian.clone())
        q_static = FloatQuant(
            fmt=fmt,
            use_hessian=True,
            scale_timing="static",
            scale_objective="mse",
            blocksize=32,
        )
        static_result = q_static.quantize_layer(layer, inp, hessian=hessian.clone())
        assert not torch.equal(default_result.block_scales, static_result.block_scales)

    def test_scale_timing_none_overrides_scale_search(self):
        """Explicitly disabling scale sweeps should beat the legacy flag."""
        layer = make_layer(seed=29, in_features=64, out_features=32)
        disabled = FloatQuant(fmt="nvfp4", scale_search=True, scale_timing="none")
        default = FloatQuant(fmt="nvfp4", scale_search=False)
        disabled_result = disabled.quantize_layer(layer)
        default_result = default.quantize_layer(layer)
        torch.testing.assert_close(
            disabled_result.dequantized_weight,
            default_result.dequantized_weight,
        )


class TestFP4QuantConfig:
    """Tests for the save-path quantization config."""

    @pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4", "fp8"])
    def test_quant_method_records_format(self, fmt):
        """quant_method is the dedicated name; fmt records the format."""
        config = FloatQuant(fmt=fmt).get_quant_config()
        assert config["quant_method"] == "onecomp_fake_quant"
        assert config["fmt"] == fmt

    def test_block_size_recorded_for_fp4_formats(self):
        """Block size is recorded for block-scaled formats."""
        config = FloatQuant(fmt="nvfp4").get_quant_config()
        assert config["block_size"] == 16
        config = FloatQuant(fmt="mxfp4").get_quant_config()
        assert config["block_size"] == 32
        config = FloatQuant(fmt="fp8").get_quant_config()
        assert "block_size" not in config

    def test_scale_mode_recorded(self):
        """Scale timing/objective metadata is saved for experiment traceability."""
        config = FloatQuant(
            fmt="nvfp4",
            use_hessian=True,
            scale_timing="in_loop",
            scale_objective="conditional",
            scale_candidate_strategy="adaptive",
        ).get_quant_config()
        assert config["scale_timing"] == "in_loop"
        assert config["scale_objective"] == "conditional"
        assert config["scale_candidate_strategy"] == "adaptive"

    def test_create_inference_layer(self):
        """create_inference_layer returns a Linear with dequantized weights."""
        layer = make_layer()
        q = FloatQuant(fmt="nvfp4")
        result = q.quantize_layer(layer)
        inference_layer = q.create_inference_layer(result, layer)
        assert isinstance(inference_layer, nn.Linear)
        assert torch.allclose(
            inference_layer.weight.data,
            result.dequantized_weight.to(torch.float16),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_loader_skips_fake_quant_checkpoints(self):
        """The model loader leaves fake-quant checkpoints untouched."""
        from onecomp.quantized_model_loader import QuantizedModelLoader

        model = nn.Sequential(nn.Linear(8, 8, bias=False))
        original = model[0]
        quant_config = FloatQuant(fmt="nvfp4").get_quant_config()
        quant_config["modules_in_block_to_quantize"] = ["0"]
        QuantizedModelLoader._replace_quantized_layers(model, {}, quant_config)
        assert model[0] is original


class TestConv1DOrientation:
    """Conv1D modules store W^T; every format must yield (out, in) weights."""

    @pytest.mark.parametrize("fmt", ["fp8", "nvfp4"])
    def test_create_inference_layer_from_conv1d(self, fmt):
        torch.manual_seed(0)
        nx, nf = 32, 48  # in_features, out_features (non-square)
        module = Conv1D(nf, nx)
        quantizer = FloatQuant(fmt=fmt, use_hessian=False)
        quantizer.validate_params()
        result = quantizer.quantize_layer(module)

        assert result.weight_transposed
        assert result.dequantized_weight.shape == (nx, nf)
        weight = result.compute_dequantized_weight()
        assert weight.shape == (nf, nx)
        assert torch.equal(weight, result.dequantized_weight.t().to(torch.float16))

        layer = quantizer.create_inference_layer(result, module)
        assert layer.weight.shape == (nf, nx)
        x = torch.randn(2, nx, dtype=torch.float16)
        expected = x @ result.dequantized_weight.to(torch.float16) + module.bias.to(torch.float16)
        assert torch.allclose(layer(x), expected, atol=1e-2, rtol=1e-2)

    def test_linear_result_not_transposed(self):
        torch.manual_seed(1)
        module = torch.nn.Linear(32, 16)
        quantizer = FloatQuant(fmt="fp8", use_hessian=False)
        quantizer.validate_params()
        result = quantizer.quantize_layer(module)
        assert not result.weight_transposed
        assert torch.equal(
            result.compute_dequantized_weight(),
            result.dequantized_weight.to(torch.float16),
        )
