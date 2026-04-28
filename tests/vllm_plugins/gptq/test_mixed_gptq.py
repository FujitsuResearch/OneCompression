"""Unit tests for the MixedGPTQConfig vLLM plugin.

Covers:
  - _resolve_group_size priority: per-module direct > params dict > global
  - from_config round-trip parsing
  - get_quant_method dispatches correct group_size per module
  - _build_quantization_bits preserves per-quantizer group_size (no vLLM needed)

Copyright 2025-2026 Fujitsu Ltd.

"""

from unittest.mock import MagicMock

import pytest

try:
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
    from vllm.model_executor.layers.quantization.gptq_marlin import (
        GPTQMarlinLinearMethod,
    )

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

try:
    from onecomp.quantizer.autobit._autobit import AutoBitQuantizer
    from onecomp.quantizer.gptq import GPTQ as _GPTQ

    _HAS_ONECOMP = True
except ImportError:
    _HAS_ONECOMP = False

_needs_vllm = pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
_needs_onecomp = pytest.mark.skipif(not _HAS_ONECOMP, reason="onecomp deps not installed")

if _HAS_VLLM:
    from vllm_plugins.gptq.vllm_plugin import MixedGPTQConfig


# ---------------------------------------------------------------------------
# Fixtures (shared by vLLM-dependent tests)
# ---------------------------------------------------------------------------

QUANTIZATION_BITS_MIXED_GS = [
    {
        "self_attn.q_proj": {"bits": 4, "method": "gptq", "group_size": 128},
        "self_attn.k_proj": {"bits": 4, "method": "gptq", "group_size": 128},
        "self_attn.v_proj": {"bits": 4, "method": "gptq", "group_size": 128},
        "self_attn.o_proj": {"bits": 4, "method": "gptq", "group_size": 128},
        "mlp.gate_proj": {"bits": 2, "method": "gptq", "group_size": 32},
        "mlp.up_proj": {"bits": 2, "method": "gptq", "group_size": 32},
        "mlp.down_proj": {"bits": 2, "method": "gptq", "group_size": 32},
    },
]


def _make_config(quantization_bits=None, group_size=-1):
    return MixedGPTQConfig(
        quantization_bits=quantization_bits or QUANTIZATION_BITS_MIXED_GS,
        group_size=group_size,
        desc_act=False,
        sym=True,
    )


def _make_linear_layer(prefix):
    """Create a mock that isinstance-checks as LinearBase."""
    layer = MagicMock(spec=LinearBase)
    return layer


# ===========================================================================
# Tests that do NOT require vLLM
# ===========================================================================


@_needs_onecomp
class TestBuildQuantizationBitsGroupSize:
    """Verify _build_quantization_bits from AutoBitQuantizer preserves per-quantizer group_size.

    These tests only depend on onecomp, not vLLM.
    """

    @pytest.fixture
    def autobit_quantizer(self):
        """Build an AutoBitQuantizer with mixed group-size candidates after assignment."""
        candidates = [
            _GPTQ(wbits=4, groupsize=128),
            _GPTQ(wbits=4, groupsize=32),
        ]
        quantizer = AutoBitQuantizer(
            assignment_strategy="activation_aware",
            target_bit=4.0,
            quantizers=candidates,
        )
        quantizer._name_to_quantizer = {
            "model.layers.0.self_attn.q_proj": candidates[0],
            "model.layers.0.self_attn.k_proj": candidates[0],
            "model.layers.0.self_attn.v_proj": candidates[0],
            "model.layers.0.mlp.gate_proj": candidates[1],
            "model.layers.0.mlp.up_proj": candidates[1],
            "model.layers.0.mlp.down_proj": candidates[1],
            "model.layers.1.self_attn.q_proj": candidates[0],
            "model.layers.1.self_attn.k_proj": candidates[0],
            "model.layers.1.self_attn.v_proj": candidates[0],
            "model.layers.1.mlp.gate_proj": candidates[1],
            "model.layers.1.mlp.up_proj": candidates[1],
            "model.layers.1.mlp.down_proj": candidates[1],
        }
        return quantizer

    def test_quantization_bits_has_per_module_group_size(self, autobit_quantizer):
        bits = autobit_quantizer._build_quantization_bits(num_layers=2)

        assert len(bits) == 2
        for layer_cfg in bits:
            assert layer_cfg["self_attn.q_proj"]["params"]["group_size"] == 128
            assert layer_cfg["mlp.gate_proj"]["params"]["group_size"] == 32
            assert layer_cfg["mlp.down_proj"]["params"]["group_size"] == 32

    def test_empty_assignment(self, autobit_quantizer):
        autobit_quantizer._name_to_quantizer = {}
        bits = autobit_quantizer._build_quantization_bits(num_layers=2)
        assert bits == []

    def test_layer_count_pads_empty_layers(self, autobit_quantizer):
        """num_layers > assigned layers → missing indices become empty dicts."""
        bits = autobit_quantizer._build_quantization_bits(num_layers=5)
        assert len(bits) == 5
        assert bits[0] != {}
        assert bits[2] == {}
        assert bits[4] == {}

    @_needs_vllm
    def test_vllm_plugin_resolves_saved_group_size(self, autobit_quantizer):
        """Full round-trip: AutoBit saves → MixedGPTQConfig reads → correct gs."""
        bits = autobit_quantizer._build_quantization_bits(num_layers=2)
        raw_config = {
            "quant_method": "mixed_gptq",
            "quantization_bits": bits,
            "group_size": -1,
            "desc_act": False,
            "sym": True,
        }
        cfg = MixedGPTQConfig.from_config(raw_config)

        layer = _make_linear_layer("model.layers.0.self_attn.q_proj")
        attn_method = cfg.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
        assert not isinstance(attn_method, UnquantizedLinearMethod)

        layer = _make_linear_layer("model.layers.0.mlp.gate_proj")
        mlp_method = cfg.get_quant_method(layer, "model.layers.0.mlp.gate_proj")
        assert not isinstance(mlp_method, UnquantizedLinearMethod)

        def _extract_gs(method):
            return method.quant_config.group_size

        assert _extract_gs(attn_method) == 128
        assert _extract_gs(mlp_method) == 32


# ===========================================================================
# Tests that require vLLM
# ===========================================================================


@_needs_vllm
class TestResolveGroupSize:
    """Verify per-module > params > global priority."""

    def test_direct_per_module(self):
        cfg = _make_config(group_size=256)
        mod_cfg = {"bits": 4, "method": "gptq", "group_size": 128}
        assert cfg._resolve_group_size(mod_cfg) == 128

    def test_params_dict(self):
        cfg = _make_config(group_size=256)
        mod_cfg = {"bits": 4, "method": "gptq", "params": {"group_size": 64}}
        assert cfg._resolve_group_size(mod_cfg) == 64

    def test_direct_takes_priority_over_params(self):
        cfg = _make_config(group_size=256)
        mod_cfg = {
            "bits": 4,
            "method": "gptq",
            "group_size": 128,
            "params": {"group_size": 64},
        }
        assert cfg._resolve_group_size(mod_cfg) == 128

    def test_global_fallback(self):
        cfg = _make_config(group_size=256)
        mod_cfg = {"bits": 4, "method": "gptq"}
        assert cfg._resolve_group_size(mod_cfg) == 256

    def test_none_mod_cfg(self):
        cfg = _make_config(group_size=256)
        assert cfg._resolve_group_size(None) == 256


@_needs_vllm
class TestFromConfig:
    """Verify config dict is correctly parsed."""

    def test_basic_round_trip(self):
        raw = {
            "quant_method": "mixed_gptq",
            "quantization_bits": QUANTIZATION_BITS_MIXED_GS,
            "group_size": -1,
            "desc_act": False,
            "sym": True,
        }
        cfg = MixedGPTQConfig.from_config(raw)
        assert cfg.group_size == -1
        assert cfg.sym is True
        assert len(cfg.quantization_bits) == 1

    def test_from_config_with_per_module_group_size(self):
        quantization_bits = [
            {
                "self_attn.q_proj": {
                    "bits": 4,
                    "method": "gptq",
                    "params": {"group_size": 128},
                },
                "mlp.gate_proj": {
                    "bits": 4,
                    "method": "gptq",
                    "params": {"group_size": 32},
                },
                "mlp.up_proj": {
                    "bits": 4,
                    "method": "gptq",
                    "params": {"group_size": 32},
                },
            },
        ]
        raw = {
            "quant_method": "mixed_gptq",
            "quantization_bits": quantization_bits,
            "group_size": -1,
            "desc_act": False,
            "sym": True,
        }
        cfg = MixedGPTQConfig.from_config(raw)
        assert cfg._resolve_group_size(quantization_bits[0]["self_attn.q_proj"]) == 128
        assert cfg._resolve_group_size(quantization_bits[0]["mlp.gate_proj"]) == 32

    def test_legacy_layer_bits_fallback(self):
        raw = {
            "quant_method": "mixed_gptq",
            "layer_bits": [4, 4, 2],
            "group_size": 128,
        }
        cfg = MixedGPTQConfig.from_config(raw)
        assert len(cfg.quantization_bits) == 3
        assert cfg.group_size == 128


@_needs_vllm
class TestGetQuantMethodGroupSize:
    """Verify get_quant_method passes the correct group_size per module."""

    def _cfg_two_layers(self):
        """Two-layer config: attn uses gs=128, mlp uses gs=32."""
        layer = {
            "self_attn.q_proj": {"bits": 4, "method": "gptq", "group_size": 128},
            "self_attn.k_proj": {"bits": 4, "method": "gptq", "group_size": 128},
            "self_attn.v_proj": {"bits": 4, "method": "gptq", "group_size": 128},
            "self_attn.o_proj": {"bits": 4, "method": "gptq", "group_size": 128},
            "mlp.gate_proj": {"bits": 4, "method": "gptq", "group_size": 32},
            "mlp.up_proj": {"bits": 4, "method": "gptq", "group_size": 32},
            "mlp.down_proj": {"bits": 4, "method": "gptq", "group_size": 32},
        }
        return _make_config(quantization_bits=[layer, layer], group_size=-1)

    def test_attn_module_gets_gs128(self):
        cfg = self._cfg_two_layers()
        layer = _make_linear_layer("model.layers.0.self_attn.o_proj")
        method = cfg.get_quant_method(layer, "model.layers.0.self_attn.o_proj")

        assert not isinstance(method, UnquantizedLinearMethod)
        if isinstance(method, GPTQMarlinLinearMethod):
            assert method.quant_config.group_size == 128
        elif isinstance(method, GPTQLinearMethod):
            assert method.quant_config.group_size == 128

    def test_mlp_module_gets_gs32(self):
        cfg = self._cfg_two_layers()
        layer = _make_linear_layer("model.layers.0.mlp.down_proj")
        method = cfg.get_quant_method(layer, "model.layers.0.mlp.down_proj")

        assert not isinstance(method, UnquantizedLinearMethod)
        if isinstance(method, GPTQMarlinLinearMethod):
            assert method.quant_config.group_size == 32
        elif isinstance(method, GPTQLinearMethod):
            assert method.quant_config.group_size == 32

    def test_different_layers_same_dispatch(self):
        """Layer 0 and layer 1 should both dispatch correctly."""
        cfg = self._cfg_two_layers()
        for layer_idx in (0, 1):
            layer = _make_linear_layer(f"model.layers.{layer_idx}.mlp.gate_proj")
            method = cfg.get_quant_method(layer, f"model.layers.{layer_idx}.mlp.gate_proj")
            assert not isinstance(method, UnquantizedLinearMethod)

    def test_non_linear_returns_none(self):
        cfg = self._cfg_two_layers()
        non_linear = MagicMock()  # not LinearBase
        result = cfg.get_quant_method(non_linear, "model.layers.0.self_attn.o_proj")
        assert result is None

    def test_non_layer_prefix_returns_unquantized(self):
        cfg = self._cfg_two_layers()
        layer = _make_linear_layer("model.embed_tokens")
        method = cfg.get_quant_method(layer, "model.embed_tokens")
        assert isinstance(method, UnquantizedLinearMethod)

    def test_bits2_uses_exllama(self):
        """bits=2 should use GPTQLinearMethod (Exllama), not Marlin."""
        quantization_bits = [
            {
                "self_attn.q_proj": {"bits": 2, "method": "gptq", "group_size": 64},
                "self_attn.k_proj": {"bits": 2, "method": "gptq", "group_size": 64},
                "self_attn.v_proj": {"bits": 2, "method": "gptq", "group_size": 64},
            },
        ]
        cfg = _make_config(quantization_bits=quantization_bits, group_size=-1)
        layer = _make_linear_layer("model.layers.0.self_attn.q_proj")
        method = cfg.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
        assert isinstance(method, GPTQLinearMethod)
        assert method.quant_config.group_size == 64
