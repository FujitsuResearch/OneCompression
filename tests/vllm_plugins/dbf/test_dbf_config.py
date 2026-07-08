"""Smoke tests for the DBF vLLM quantization config (DbfConfig).

Covers config parsing (from_config) and per-module dispatch
(get_quant_method).

Copyright 2025-2026 Fujitsu Ltd.

"""

from unittest.mock import MagicMock

import pytest

try:
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    from vllm_plugins.dbf.vllm_plugin import DbfConfig

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

_needs_vllm = pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")


@_needs_vllm
class TestDbfConfig:
    """Smoke tests for DBF vLLM quantization config parsing and dispatch."""

    def test_from_config_round_trip(self):
        raw = {
            "quant_method": "dbf",
            "quantization_bits": [
                {
                    "self_attn.q_proj": {"bits": 1.5, "method": "dbf"},
                    "mlp.down_proj": {"bits": 1.0, "method": "dbf"},
                }
            ],
        }

        cfg = DbfConfig.from_config(raw)

        assert cfg.get_name() == "dbf"
        assert cfg.quantization_bits == raw["quantization_bits"]

    def test_get_quant_method_attaches_module_config(self):
        cfg = DbfConfig.from_config(
            {"quantization_bits": [{"self_attn.q_proj": {"bits": 1.5, "method": "dbf"}}]}
        )
        layer = MagicMock(spec=LinearBase)

        method = cfg.get_quant_method(layer, "model.layers.0.self_attn.q_proj")

        assert not isinstance(method, UnquantizedLinearMethod)
        assert layer._dbf_prefix == "model.layers.0.self_attn.q_proj"
        assert layer._dbf_mod_cfg == {"bits": 1.5, "method": "dbf"}

    def test_get_quant_method_returns_unquantized_for_bits_zero(self):
        cfg = DbfConfig.from_config(
            {"quantization_bits": [{"self_attn.q_proj": {"bits": 0, "method": "dbf"}}]}
        )
        layer = MagicMock(spec=LinearBase)

        method = cfg.get_quant_method(layer, "model.layers.0.self_attn.q_proj")

        assert isinstance(method, UnquantizedLinearMethod)
