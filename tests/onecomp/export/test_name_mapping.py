"""Tests for the HF -> GGUF tensor name mapping.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import pytest

from onecomp.export import map_tensor_name


@pytest.mark.parametrize(
    "hf_name,gguf_name",
    [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.norm.weight", "output_norm.weight"),
        ("lm_head.weight", "output.weight"),
        ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
        ("model.layers.0.self_attn.k_proj.weight", "blk.0.attn_k.weight"),
        ("model.layers.0.self_attn.v_proj.weight", "blk.0.attn_v.weight"),
        ("model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight"),
        ("model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"),
        ("model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
        ("model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
        ("model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight"),
        ("model.layers.0.post_attention_layernorm.weight", "blk.0.ffn_norm.weight"),
        # Double-digit layer indices.
        ("model.layers.21.self_attn.q_proj.weight", "blk.21.attn_q.weight"),
        ("model.layers.21.mlp.down_proj.weight", "blk.21.ffn_down.weight"),
        # Qwen2-style attention biases.
        ("model.layers.0.self_attn.q_proj.bias", "blk.0.attn_q.bias"),
        ("model.layers.0.self_attn.k_proj.bias", "blk.0.attn_k.bias"),
        ("model.layers.0.self_attn.v_proj.bias", "blk.0.attn_v.bias"),
    ],
)
def test_known_mappings(hf_name, gguf_name):
    assert map_tensor_name(hf_name) == gguf_name


@pytest.mark.parametrize(
    "hf_name",
    [
        "model.layers.0.self_attn.rotary_emb.inv_freq",
        "rotary_emb.inv_freq",
    ],
)
def test_skipped_tensors(hf_name):
    assert map_tensor_name(hf_name) is None


@pytest.mark.parametrize(
    "hf_name",
    [
        "model.layers.0.self_attn.unknown_proj.weight",
        "model.layers.x.self_attn.q_proj.weight",
        "something.completely.different",
    ],
)
def test_unknown_names_rejected(hf_name):
    with pytest.raises(ValueError, match="cannot map tensor name"):
        map_tensor_name(hf_name)
