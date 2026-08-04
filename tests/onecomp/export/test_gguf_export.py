"""Tests for the HF -> GGUF export pipeline on a synthetic checkpoint.

Builds a tiny Llama-style model directory (config, SentencePiece
tokenizer serialized with a minimal protobuf encoder, safetensors
weights) and verifies the exported GGUF file end to end.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import json
import os
import struct

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from onecomp.export import (
    GGUFExportConfig,
    GGUFReader,
    export_gguf,
    load_gguf_state_dict,
)
from onecomp.export.gguf_export import _load_bpe_tokenizer, permute_rope_rows, unpermute_rope_rows


def _varint(value):
    out = b""
    while True:
        byte, value = value & 0x7F, value >> 7
        out += bytes([byte | (0x80 if value else 0)])
        if not value:
            return out


def _tag(field_number, wire_type):
    return _varint((field_number << 3) | wire_type)


def _sentencepiece_piece(text, score, piece_type):
    raw = text.encode("utf-8")
    sub = _tag(1, 2) + _varint(len(raw)) + raw
    sub += _tag(2, 5) + struct.pack("<f", score)
    sub += _tag(3, 0) + _varint(piece_type)
    return _tag(1, 2) + _varint(len(sub)) + sub


# (text, score, type): unk=2, control=3, byte=6, normal=1.
_SP_PIECES = [
    ("<unk>", 0.0, 2),
    ("<s>", 0.0, 3),
    ("</s>", 0.0, 3),
    ("<0x0A>", 0.0, 6),
    ("\u2581hello", -1.0, 1),
    ("\u2581world", -2.0, 1),
    ("a", -3.0, 1),
    ("b", -4.0, 1),
]

_VOCAB_SIZE = 12  # 8 SentencePiece pieces + 2 added tokens + 2 padding slots


@pytest.fixture(name="model_dir")
def fixture_model_dir(tmp_path):
    """Create a tiny synthetic Llama-style checkpoint directory."""
    model_dir = tmp_path / "tiny-llama"
    model_dir.mkdir()

    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_hidden_layers": 1,
                "intermediate_size": 6,
                "max_position_embeddings": 16,
                "vocab_size": _VOCAB_SIZE,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "tie_word_embeddings": False,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }
        )
    )

    sp_model = b"".join(_sentencepiece_piece(*piece) for piece in _SP_PIECES)
    # Unknown trailing field (trainer_spec) that the parser must skip.
    sp_model += _tag(2, 2) + _varint(4) + b"skip"
    (model_dir / "tokenizer.model").write_bytes(sp_model)

    (model_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 8, "content": "<|extra|>", "special": True},
                    {"id": 9, "content": "custom", "special": False},
                ],
                "model": {"vocab": {}, "merges": []},
            }
        )
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "add_bos_token": True,
                "add_eos_token": False,
                "chat_template": "{{ messages }}",
            }
        )
    )

    weights = {
        "model.embed_tokens.weight": torch.randn(_VOCAB_SIZE, 4, dtype=torch.float16),
        "model.norm.weight": torch.ones(4, dtype=torch.float32),
        # BF16 with a value outside the F16 range to exercise clamping.
        "lm_head.weight": torch.full((_VOCAB_SIZE, 4), 1e5, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4, dtype=torch.float16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(2, 4, dtype=torch.float16),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(2, 4, dtype=torch.float16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(4, 4, dtype=torch.float16),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(6, 4, dtype=torch.float16),
        "model.layers.0.mlp.up_proj.weight": torch.randn(6, 4, dtype=torch.float16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(4, 6, dtype=torch.float16),
        "model.layers.0.input_layernorm.weight": torch.ones(4, dtype=torch.float32),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(4, dtype=torch.float32),
    }
    save_file(weights, str(model_dir / "model.safetensors"))
    return str(model_dir)


def test_export_synthetic_model(model_dir, tmp_path):
    out_path = str(tmp_path / "tiny.gguf")
    assert export_gguf(model_dir, GGUFExportConfig(out_path=out_path)) == out_path

    reader = GGUFReader(out_path)
    metadata = reader.metadata
    assert metadata["general.architecture"] == "llama"
    assert metadata["general.file_type"] == 1
    assert metadata["llama.vocab_size"] == _VOCAB_SIZE
    assert metadata["llama.attention.head_count"] == 2
    assert metadata["llama.attention.head_count_kv"] == 1
    assert metadata["llama.rope.dimension_count"] == 2  # hidden_size // head_count
    assert metadata["tokenizer.ggml.bos_token_id"] == 1
    assert metadata["tokenizer.ggml.eos_token_id"] == 2

    names = {info.name for info in reader.tensors}
    assert "output.weight" in names  # tie_word_embeddings is false
    assert reader.tensor("token_embd.weight").shape == (_VOCAB_SIZE, 4)


def test_added_tokens_merged_beyond_sentencepiece_vocab(model_dir, tmp_path):
    out_path = str(tmp_path / "tiny.gguf")
    export_gguf(model_dir, GGUFExportConfig(out_path=out_path))
    metadata = GGUFReader(out_path).metadata

    tokens = metadata["tokenizer.ggml.tokens"]
    token_types = metadata["tokenizer.ggml.token_type"]
    scores = metadata["tokenizer.ggml.scores"]
    assert len(tokens) == len(token_types) == len(scores) == _VOCAB_SIZE

    assert tokens[:8] == [piece[0] for piece in _SP_PIECES]
    assert tokens[8] == "<|extra|>" and token_types[8] == 3  # CONTROL
    assert tokens[9] == "custom" and token_types[9] == 4  # USER_DEFINED
    assert tokens[10] == "[PAD10]" and token_types[10] == 5  # UNUSED
    assert tokens[11] == "[PAD11]" and token_types[11] == 5
    assert token_types[:8] == [piece[2] for piece in _SP_PIECES]


def test_unknown_token_id_and_chat_template(model_dir, tmp_path):
    out_path = str(tmp_path / "tiny.gguf")
    export_gguf(model_dir, GGUFExportConfig(out_path=out_path))
    metadata = GGUFReader(out_path).metadata

    assert metadata["tokenizer.ggml.unknown_token_id"] == 0
    assert metadata["tokenizer.chat_template"] == "{{ messages }}"
    assert metadata["tokenizer.ggml.add_bos_token"] is True
    assert metadata["tokenizer.ggml.add_eos_token"] is False


def test_bf16_overflow_clamped_to_f16_range(model_dir, tmp_path):
    out_path = str(tmp_path / "tiny.gguf")
    export_gguf(model_dir, GGUFExportConfig(out_path=out_path))

    output = GGUFReader(out_path).read_tensor("output.weight")
    assert output.dtype == np.float16
    assert np.all(np.isfinite(output.astype(np.float32)))
    assert output.max() == np.float16(65504.0)


def test_f32_export_sets_file_type(model_dir, tmp_path):
    out_path = str(tmp_path / "tiny-f32.gguf")
    export_gguf(model_dir, GGUFExportConfig(out_path=out_path, dtype="f32"))

    reader = GGUFReader(out_path)
    assert reader.metadata["general.file_type"] == 0
    assert reader.read_tensor("token_embd.weight").dtype == np.float32


def test_explicit_head_dim_sets_rope_dimension_count(model_dir, tmp_path):
    """An explicit config head_dim overrides hidden_size // num_attention_heads."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    config["head_dim"] = 8
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)

    out_path = str(tmp_path / "tiny-head-dim.gguf")
    export_gguf(model_dir, GGUFExportConfig(out_path=out_path))

    assert GGUFReader(out_path).metadata["llama.rope.dimension_count"] == 8


def test_bpe_merges_accept_both_json_forms(tmp_path):
    """tokenizer.json merges can be "a b" strings or [a, b] pairs."""
    for raw_merges in (["\u0120 a", "a b"], [["\u0120", "a"], ["a", "b"]]):
        path = tmp_path / "tokenizer.json"
        path.write_text(
            json.dumps(
                {
                    "added_tokens": [{"id": 2, "content": "<|eot|>", "special": True}],
                    "model": {"vocab": {"a": 0, "b": 1}, "merges": raw_merges},
                }
            )
        )
        tokens, token_types, merges = _load_bpe_tokenizer(str(path))
        assert tokens == ["a", "b", "<|eot|>"]
        assert token_types == [1, 1, 3]
        assert merges == ["\u0120 a", "a b"]


class TestLlamaRopePermutation:
    """llama.cpp expects interleaved-RoPE Q/K rows for the llama arch."""

    @pytest.mark.parametrize("n_head,head_dim", [(1, 4), (2, 8), (3, 6)])
    def test_unpermute_inverts_permute(self, n_head, head_dim):
        torch.manual_seed(0)
        weight = torch.randn(n_head * head_dim, 5)
        assert torch.equal(unpermute_rope_rows(permute_rope_rows(weight, n_head), n_head), weight)
        bias = torch.randn(n_head * head_dim)
        assert torch.equal(unpermute_rope_rows(permute_rope_rows(bias, n_head), n_head), bias)

    def test_permute_interleaves_half_split_rows(self):
        # One head with head_dim=4: HF rows [0 1 2 3] (first/second RoPE
        # halves) become llama.cpp rows [0 2 1 3] (interleaved pairs).
        weight = torch.arange(4, dtype=torch.float32).unsqueeze(1)
        assert permute_rope_rows(weight, 1).squeeze(1).tolist() == [0.0, 2.0, 1.0, 3.0]

    def test_llama_export_load_round_trip_restores_hf_layout(self, model_dir, tmp_path):
        out_path = str(tmp_path / "tiny-llama.gguf")
        export_gguf(model_dir, GGUFExportConfig(out_path=out_path))

        from safetensors import safe_open

        state_dict = load_gguf_state_dict(out_path)
        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
            for name in (
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
            ):
                original = f.get_tensor(name).to(torch.float16)
                assert torch.equal(state_dict[name].to(torch.float16), original), name
