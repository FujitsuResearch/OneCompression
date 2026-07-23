"""Tests for the FloatQuant fake-quant vLLM plugin registration.

Requires vLLM; skipped automatically when it is not installed.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest

vllm = pytest.importorskip("vllm")

from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: E402

from vllm_plugins.floatquant.vllm_plugin import OneCompFakeQuantConfig  # noqa: E402


def _fake_quant_config(fmt="nvfp4", **extra):
    config = {
        "quant_method": "onecomp_fake_quant",
        "fmt": fmt,
        "use_hessian": False,
        "checkpoint_format": "fake_quant",
    }
    config.update(extra)
    return config


class TestOneCompFakeQuantConfig:
    """The registered config class must be self-consistent with its key."""

    def test_get_name_matches_registered_method(self):
        """get_name() must return the key used in register_quantization_config.

        Regression test: a stray ``METHOD = "fp8"`` once shadowed the
        dedicated name, which broke vLLM's name consistency and collided
        with the built-in fp8 handler.
        """
        assert OneCompFakeQuantConfig.get_name() == "onecomp_fake_quant"

    def test_registered_in_vllm(self):
        from vllm.model_executor.layers.quantization import get_quantization_config

        assert get_quantization_config("onecomp_fake_quant") is OneCompFakeQuantConfig

    def test_builtin_fp8_not_overridden(self):
        """Built-in fp8 must keep resolving to vLLM's own handler."""
        from vllm.model_executor.layers.quantization import get_quantization_config

        assert get_quantization_config("fp8") is not OneCompFakeQuantConfig

    @pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4", "fp8"])
    def test_from_config_resolves_all_formats(self, fmt):
        cfg = OneCompFakeQuantConfig.from_config(_fake_quant_config(fmt=fmt))
        assert cfg.fmt == fmt
        assert cfg.get_name() == "onecomp_fake_quant"

    def test_linear_layers_map_to_unquantized(self):
        import torch
        from vllm.model_executor.layers.linear import LinearBase

        cfg = OneCompFakeQuantConfig.from_config(_fake_quant_config())
        # Constructing a real vLLM Linear requires an initialised TP group;
        # get_quant_method only dispatches on isinstance, so an
        # uninitialised instance is sufficient here.
        layer = LinearBase.__new__(LinearBase)
        method = cfg.get_quant_method(layer, prefix="")
        assert isinstance(method, UnquantizedLinearMethod)
        assert cfg.get_quant_method(torch.nn.LayerNorm(8), prefix="") is None
