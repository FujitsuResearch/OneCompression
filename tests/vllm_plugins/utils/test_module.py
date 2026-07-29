"""Unit tests for ``vllm_plugins.utils.module``.

Covers:
  - _resolve_fused_bits / _lookup_module_config / _validate_quant_config_within_shard:
    regression for the fix that substitutes constituents at the fused
    name's own position instead of assuming a hardcoded "self_attn."/
    "mlp." parent path (needed for architectures that fuse qkv/gate_up
    under a different parent, e.g. "linear_attn.qkv_proj").
  - _lookup_moe_config: aggregating one uniform GPTQ config across all
    experts in a FusedMoE layer.

No vLLM dependency is needed for this module.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest

from vllm_plugins.utils.module import (
    _lookup_module_config,
    _lookup_moe_config,
    _parse_layer_and_module,
    _resolve_fused_bits,
    _validate_quant_config_within_shard,
)

# ---------------------------------------------------------------------------
# _parse_layer_and_module
# ---------------------------------------------------------------------------


class TestParseLayerAndModule:
    def test_basic(self):
        assert _parse_layer_and_module("model.layers.3.self_attn.q_proj") == (
            3,
            "self_attn.q_proj",
        )

    def test_no_layer_match(self):
        assert _parse_layer_and_module("model.embed_tokens") == (None, None)

    def test_vision_prefix_excluded(self):
        assert _parse_layer_and_module("vision_tower.layers.0.attn.q_proj") == (None, None)


# ---------------------------------------------------------------------------
# _resolve_fused_bits / _lookup_module_config -- generic parent-path fix
# ---------------------------------------------------------------------------


class TestResolveFusedBitsGenericParent:
    """Regression: fused-name substitution must work at any parent path,

    not just the hardcoded "self_attn."/"mlp." prefixes.
    """

    def test_standard_self_attn_parent(self):
        layer_cfg = {"self_attn.q_proj": {"bits": 4, "method": "gptq"}}
        assert _resolve_fused_bits(layer_cfg, "self_attn.qkv_proj") == {
            "bits": 4,
            "method": "gptq",
        }

    def test_non_standard_parent_path(self):
        """e.g. a Mamba-style linear_attn block fusing qkv outside self_attn."""
        layer_cfg = {"linear_attn.q_proj": {"bits": 8, "method": "gptq"}}
        assert _resolve_fused_bits(layer_cfg, "linear_attn.qkv_proj") == {
            "bits": 8,
            "method": "gptq",
        }

    def test_top_level_fused_name_no_parent(self):
        layer_cfg = {"gate_proj": {"bits": 4, "method": "gptq"}}
        assert _resolve_fused_bits(layer_cfg, "gate_up_proj") == {
            "bits": 4,
            "method": "gptq",
        }

    def test_falls_back_through_constituents(self):
        """k_proj is checked before/after q_proj; whichever constituent

        exists in layer_cfg should be found.
        """
        layer_cfg = {"self_attn.k_proj": {"bits": 2, "method": "gptq"}}
        assert _resolve_fused_bits(layer_cfg, "self_attn.qkv_proj") == {
            "bits": 2,
            "method": "gptq",
        }

    def test_no_constituent_present_returns_none(self):
        layer_cfg = {"mlp.down_proj": {"bits": 4, "method": "gptq"}}
        assert _resolve_fused_bits(layer_cfg, "self_attn.qkv_proj") is None

    def test_not_a_fused_name_returns_none(self):
        layer_cfg = {"self_attn.q_proj": {"bits": 4, "method": "gptq"}}
        assert _resolve_fused_bits(layer_cfg, "self_attn.o_proj") is None


class TestLookupModuleConfig:
    def test_direct_hit(self):
        layer_cfg = {"self_attn.q_proj": {"bits": 4}}
        bits = _lookup_module_config([layer_cfg], 0, "self_attn.q_proj")
        assert bits == {"bits": 4}

    def test_fused_hit_non_standard_parent(self):
        layer_cfg = {"linear_attn.q_proj": {"bits": 4}}
        bits = _lookup_module_config([layer_cfg], 0, "linear_attn.qkv_proj")
        assert bits == {"bits": 4}

    def test_all_fallback(self):
        layer_cfg = {"_all": {"bits": 3}}
        bits = _lookup_module_config([layer_cfg], 0, "self_attn.o_proj")
        assert bits == {"bits": 3}

    def test_layer_idx_out_of_range(self):
        assert _lookup_module_config([{}], 5, "self_attn.q_proj") is None

    def test_no_match_returns_none(self):
        layer_cfg = {"mlp.down_proj": {"bits": 4}}
        assert _lookup_module_config([layer_cfg], 0, "self_attn.q_proj") is None


class TestValidateQuantConfigWithinShard:
    def test_non_standard_parent_all_present_and_identical(self):
        layer_cfg = {
            "linear_attn.q_proj": {"bits": 4, "method": "gptq"},
            "linear_attn.k_proj": {"bits": 4, "method": "gptq"},
            "linear_attn.v_proj": {"bits": 4, "method": "gptq"},
        }
        assert _validate_quant_config_within_shard([layer_cfg], 0, "linear_attn.qkv_proj") is True

    def test_non_standard_parent_partial_present_fails(self):
        layer_cfg = {
            "linear_attn.q_proj": {"bits": 4, "method": "gptq"},
            "linear_attn.k_proj": {"bits": 4, "method": "gptq"},
            # v_proj missing
        }
        assert _validate_quant_config_within_shard([layer_cfg], 0, "linear_attn.qkv_proj") is False

    def test_non_standard_parent_inconsistent_fails(self):
        layer_cfg = {
            "linear_attn.q_proj": {"bits": 4, "method": "gptq"},
            "linear_attn.k_proj": {"bits": 8, "method": "gptq"},
            "linear_attn.v_proj": {"bits": 4, "method": "gptq"},
        }
        assert _validate_quant_config_within_shard([layer_cfg], 0, "linear_attn.qkv_proj") is False

    def test_non_fused_module_always_true(self):
        assert _validate_quant_config_within_shard([{}], 0, "self_attn.o_proj") is True

    def test_layer_idx_out_of_range(self):
        assert _validate_quant_config_within_shard([{}], 5, "self_attn.qkv_proj") is False


# ---------------------------------------------------------------------------
# _lookup_moe_config
# ---------------------------------------------------------------------------


def _expert_cfg(bits=4, method="gptq", group_size=128):
    return {"bits": bits, "method": method, "params": {"group_size": group_size}}


def _uniform_moe_layer_cfg(num_experts=2, bits=4, group_size=128):
    cfg = {}
    for i in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            cfg[f"mlp.experts.{i}.{proj}"] = _expert_cfg(bits=bits, group_size=group_size)
    return cfg


class TestLookupMoeConfig:
    def test_layer_idx_out_of_range_returns_none(self):
        assert _lookup_moe_config([], 0, num_experts=2) is None

    def test_no_expert_keys_returns_none(self):
        layer_cfg = {"self_attn.q_proj": {"bits": 4, "method": "gptq"}}
        assert _lookup_moe_config([layer_cfg], 0, num_experts=2) is None

    def test_uniform_experts_aggregated(self):
        layer_cfg = _uniform_moe_layer_cfg(num_experts=4, bits=4, group_size=64)
        result = _lookup_moe_config([layer_cfg], 0, num_experts=4)
        assert result == {"bits": 4, "method": "gptq", "group_size": 64}

    def test_top_level_experts_no_parent_prefix(self):
        """Some architectures put experts at the top level, not under mlp."""
        layer_cfg = {}
        for i in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                layer_cfg[f"experts.{i}.{proj}"] = _expert_cfg(bits=8, group_size=32)
        result = _lookup_moe_config([layer_cfg], 0, num_experts=2)
        assert result == {"bits": 8, "method": "gptq", "group_size": 32}

    def test_partial_coverage_raises(self):
        layer_cfg = _uniform_moe_layer_cfg(num_experts=2)
        del layer_cfg["mlp.experts.1.down_proj"]
        with pytest.raises(ValueError, match="only"):
            _lookup_moe_config([layer_cfg], 0, num_experts=2)

    def test_inconsistent_bits_raises(self):
        layer_cfg = _uniform_moe_layer_cfg(num_experts=2, bits=4)
        layer_cfg["mlp.experts.1.down_proj"] = _expert_cfg(bits=8)
        with pytest.raises(ValueError, match="inconsistent"):
            _lookup_moe_config([layer_cfg], 0, num_experts=2)

    def test_inconsistent_group_size_raises(self):
        layer_cfg = _uniform_moe_layer_cfg(num_experts=2, group_size=128)
        layer_cfg["mlp.experts.1.down_proj"] = _expert_cfg(group_size=32)
        with pytest.raises(ValueError, match="inconsistent"):
            _lookup_moe_config([layer_cfg], 0, num_experts=2)

    def test_router_key_not_counted_as_expert(self):
        """Router (e.g. 'mlp.router' or 'mlp.gate') is not an expert

        projection and must not confuse expert-count validation.
        """
        layer_cfg = _uniform_moe_layer_cfg(num_experts=2)
        layer_cfg["mlp.gate"] = {"bits": 16, "method": "gptq"}
        result = _lookup_moe_config([layer_cfg], 0, num_experts=2)
        assert result == {"bits": 4, "method": "gptq", "group_size": 128}
