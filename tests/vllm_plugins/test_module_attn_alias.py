"""Tests for vLLM module suffix normalization (GPT-OSS attn alias)."""

from vllm_plugins.utils.module import _lookup_module_config, _normalize_module_suffix


def test_normalize_attn_alias():
    assert _normalize_module_suffix("attn.o_proj") == "self_attn.o_proj"
    assert _normalize_module_suffix("self_attn.o_proj") == "self_attn.o_proj"
    assert _normalize_module_suffix("mlp.gate_up_proj") == "mlp.gate_up_proj"


def test_lookup_o_proj_via_attn_alias():
    qb = [
        {
            "self_attn.o_proj": {
                "bits": 4,
                "method": "gptq",
                "params": {"group_size": -1},
            },
        }
    ]
    cfg = _lookup_module_config(qb, 0, "attn.o_proj")
    assert cfg is not None
    assert cfg["bits"] == 4
