"""Tests for the GGUF -> Hugging Face load path and runtime metadata.

Covers the reverse tensor-name mapping, ``load_gguf_state_dict`` on a
synthetic checkpoint, and the metadata keys that llama.cpp and the
vLLM GGUF loader require to reconstruct the model config without a
``config.json``.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from onecomp.export import (
    GGUFExportConfig,
    GGUFReader,
    export_gguf,
    load_gguf_state_dict,
    map_tensor_name,
    reverse_map_tensor_name,
)

_VOCAB_SIZE = 8

# Metadata required by consumers that rebuild the model config from the
# GGUF file alone (transformers' GGUF integration, used by vLLM via
# vllm-gguf-plugin, and llama.cpp's llm_load_hparams).
_REQUIRED_MODEL_KEYS = (
    "general.architecture",
    "general.name",
    "general.file_type",
    "{arch}.context_length",
    "{arch}.embedding_length",
    "{arch}.block_count",
    "{arch}.feed_forward_length",
    "{arch}.attention.head_count",
    "{arch}.attention.head_count_kv",
    "{arch}.attention.layer_norm_rms_epsilon",
    "{arch}.rope.freq_base",
    "{arch}.rope.dimension_count",
    "{arch}.vocab_size",
)

_REQUIRED_TOKENIZER_KEYS = (
    "tokenizer.ggml.model",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
)


@pytest.fixture(name="qwen2_model_dir")
def fixture_qwen2_model_dir(tmp_path):
    """Create a tiny synthetic Qwen2-style checkpoint directory."""
    model_dir = tmp_path / "tiny-qwen2"
    model_dir.mkdir()

    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_hidden_layers": 1,
                "intermediate_size": 6,
                "max_position_embeddings": 16,
                "vocab_size": _VOCAB_SIZE,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": True,
                "bos_token_id": 0,
                "eos_token_id": 1,
            }
        )
    )
    (model_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 6, "content": "<|im_start|>", "special": True},
                    {"id": 7, "content": "<|im_end|>", "special": True},
                ],
                "model": {
                    "vocab": {"a": 0, "b": 1, "c": 2, "d": 3, "ab": 4, "cd": 5},
                    "merges": ["a b", "c d"],
                },
            }
        )
    )

    weights = {
        "model.embed_tokens.weight": torch.randn(_VOCAB_SIZE, 4, dtype=torch.float16),
        "model.norm.weight": torch.ones(4, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4, dtype=torch.float16),
        "model.layers.0.self_attn.q_proj.bias": torch.randn(4, dtype=torch.float16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(2, 4, dtype=torch.float16),
        "model.layers.0.self_attn.k_proj.bias": torch.randn(2, dtype=torch.float16),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(2, 4, dtype=torch.float16),
        "model.layers.0.self_attn.v_proj.bias": torch.randn(2, dtype=torch.float16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(4, 4, dtype=torch.float16),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(6, 4, dtype=torch.float16),
        "model.layers.0.mlp.up_proj.weight": torch.randn(6, 4, dtype=torch.float16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(4, 6, dtype=torch.float16),
        "model.layers.0.input_layernorm.weight": torch.ones(4, dtype=torch.float32),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(4, dtype=torch.float32),
    }
    save_file(weights, str(model_dir / "model.safetensors"))
    return str(model_dir), weights


@pytest.mark.parametrize(
    "gguf_name,hf_name",
    [
        ("token_embd.weight", "model.embed_tokens.weight"),
        ("output_norm.weight", "model.norm.weight"),
        ("output.weight", "lm_head.weight"),
        ("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"),
        ("blk.0.attn_output.weight", "model.layers.0.self_attn.o_proj.weight"),
        ("blk.21.ffn_down.weight", "model.layers.21.mlp.down_proj.weight"),
        ("blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight"),
        ("blk.0.ffn_norm.weight", "model.layers.0.post_attention_layernorm.weight"),
        ("blk.0.attn_v.bias", "model.layers.0.self_attn.v_proj.bias"),
    ],
)
def test_reverse_mapping(gguf_name, hf_name):
    assert reverse_map_tensor_name(gguf_name) == hf_name
    assert map_tensor_name(hf_name) == gguf_name


@pytest.mark.parametrize(
    "gguf_name",
    [
        "blk.0.attn_qkv.weight",
        "blk.x.attn_q.weight",
        "rope_freqs.weight",
    ],
)
def test_reverse_mapping_rejects_unknown_names(gguf_name):
    with pytest.raises(ValueError, match="cannot map GGUF tensor name"):
        reverse_map_tensor_name(gguf_name)


def test_state_dict_round_trip(qwen2_model_dir, tmp_path):
    """F16 export followed by load_gguf_state_dict restores every tensor."""
    model_dir, weights = qwen2_model_dir
    out_path = str(tmp_path / "tiny-qwen2.gguf")
    export_gguf(model_dir, GGUFExportConfig(out_path=out_path))

    state_dict = load_gguf_state_dict(out_path)
    # tie_word_embeddings=True: no lm_head.weight in the file.
    assert set(state_dict) == set(weights)
    for name, original in weights.items():
        restored = state_dict[name]
        assert restored.shape == original.shape
        if original.dim() == 1:
            assert restored.dtype == torch.float32
        else:
            assert restored.dtype == torch.float16
        assert torch.equal(restored.to(torch.float32), original.to(torch.float32))


def test_vllm_required_metadata_keys(qwen2_model_dir, tmp_path):
    """All metadata needed to rebuild the config without config.json exists."""
    model_dir, _ = qwen2_model_dir
    out_path = str(tmp_path / "tiny-qwen2.gguf")
    export_gguf(model_dir, GGUFExportConfig(out_path=out_path))
    metadata = GGUFReader(out_path).metadata

    arch = metadata["general.architecture"]
    assert arch == "qwen2"
    for key in _REQUIRED_MODEL_KEYS + _REQUIRED_TOKENIZER_KEYS:
        assert key.format(arch=arch) in metadata, f"missing metadata key: {key.format(arch=arch)}"
    # BPE tokenizers additionally need pre-tokenizer type and merges.
    assert metadata["tokenizer.ggml.pre"] == "qwen2"
    assert metadata["tokenizer.ggml.merges"] == ["a b", "c d"]
    # Token lists must be index-aligned and padded to vocab_size.
    assert len(metadata["tokenizer.ggml.tokens"]) == _VOCAB_SIZE
    assert len(metadata["tokenizer.ggml.token_type"]) == _VOCAB_SIZE


def test_loaded_tensors_match_file_bytes(qwen2_model_dir, tmp_path):
    """load_gguf_state_dict returns the same data as raw GGUFReader reads."""
    model_dir, _ = qwen2_model_dir
    out_path = str(tmp_path / "tiny-qwen2.gguf")
    export_gguf(model_dir, GGUFExportConfig(out_path=out_path))

    reader = GGUFReader(out_path)
    state_dict = load_gguf_state_dict(out_path)
    for info in reader.tensors:
        raw = reader.read_tensor(info.name)
        restored = state_dict[reverse_map_tensor_name(info.name)]
        assert np.array_equal(restored.numpy(), raw)
