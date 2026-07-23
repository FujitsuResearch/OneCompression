"""Tests for the vLLM-independent config resolution of the FloatQuant plugin.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest

from vllm_plugins.floatquant.config import (
    CANONICAL_FAKE_QUANT_METHOD,
    LEGACY_FAKE_QUANT_METHODS,
    resolve_fake_quant_config,
)


def _fake_quant_config(method, fmt=None, **extra):
    config = {
        "quant_method": method,
        "fmt": fmt or (method if method != CANONICAL_FAKE_QUANT_METHOD else "nvfp4"),
        "use_hessian": False,
        "checkpoint_format": "fake_quant",
    }
    config.update(extra)
    return config


class TestResolveFakeQuantConfig:
    """Resolution of onecomp FloatQuant fake-quant quantization configs."""

    @pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4", "fp8"])
    def test_canonical_config_resolves(self, fmt):
        block_size = {"nvfp4": 16, "mxfp4": 32, "fp8": None}[fmt]
        config = _fake_quant_config(CANONICAL_FAKE_QUANT_METHOD, fmt=fmt)
        if block_size is not None:
            config["block_size"] = block_size
            config["group_size"] = block_size
        resolved = resolve_fake_quant_config(config)
        assert resolved["quant_method"] == CANONICAL_FAKE_QUANT_METHOD
        assert resolved["fmt"] == fmt
        assert resolved["block_size"] == block_size
        assert resolved["use_hessian"] is False

    @pytest.mark.parametrize("method", LEGACY_FAKE_QUANT_METHODS)
    def test_legacy_method_normalizes_to_canonical(self, method):
        """Pre-migration checkpoints used the format name as quant_method."""
        resolved = resolve_fake_quant_config(_fake_quant_config(method))
        assert resolved["quant_method"] == CANONICAL_FAKE_QUANT_METHOD
        assert resolved["fmt"] == method

    def test_runner_saved_nvfp4_config(self):
        # Exactly what Runner.save_quantized_model writes for FloatQuant(fmt="nvfp4").
        config = {
            "quant_method": "onecomp_fake_quant",
            "fmt": "nvfp4",
            "use_hessian": False,
            "checkpoint_format": "fake_quant",
            "block_size": 16,
            "group_size": 16,
            "modules_in_block_to_quantize": ["model.layers.0.self_attn.q_proj"],
            "quantized_layer_names": ["model.layers.0.self_attn.q_proj"],
            "rotated": False,
            "fp32_had": False,
        }
        resolved = resolve_fake_quant_config(config)
        assert resolved == {
            "quant_method": "onecomp_fake_quant",
            "fmt": "nvfp4",
            "block_size": 16,
            "use_hessian": False,
        }

    def test_block_size_falls_back_to_group_size(self):
        config = _fake_quant_config("nvfp4", group_size=16)
        assert resolve_fake_quant_config(config)["block_size"] == 16

    def test_hessian_flag_preserved(self):
        config = _fake_quant_config("mxfp4", block_size=32, use_hessian=True)
        assert resolve_fake_quant_config(config)["use_hessian"] is True

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            resolve_fake_quant_config(_fake_quant_config("int4"))

    def test_native_checkpoint_rejected(self):
        # A native FP8 checkpoint (e.g. from save_vllm_fp8_model) must not be
        # routed through the fake-quant plugin.
        with pytest.raises(ValueError, match="fake-quant"):
            resolve_fake_quant_config({"quant_method": "fp8", "activation_scheme": "dynamic"})

    def test_missing_checkpoint_format_rejected(self):
        config = _fake_quant_config("nvfp4")
        del config["checkpoint_format"]
        with pytest.raises(ValueError):
            resolve_fake_quant_config(config)
