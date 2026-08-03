"""BPE tokenizer export paths: Llama-3 style checkpoints, multi-EOS, vocab checks.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import json

import pytest

from onecomp.export.gguf_export import _add_tokenizer_metadata


class _RecordingWriter:
    def __init__(self):
        self.metadata = {}

    def add_metadata(self, key, value, element_type=None):
        self.metadata[key] = value


def _write_bpe_model_dir(tmp_path, vocab_size, eos_token_id):
    model_dir = tmp_path / "tiny-llama3"
    model_dir.mkdir()
    vocab = {"hello": 0, "world": 1, "!": 2}
    added = [
        {"id": 3, "content": "<|end_of_text|>", "special": True},
        {"id": 4, "content": "<|eom_id|>", "special": True},
        {"id": 5, "content": "<|eot_id|>", "special": True},
    ]
    (model_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": added,
                "model": {"vocab": vocab, "merges": ["h e", ["w o"][0]]},
            }
        )
    )
    hf_config = {
        "vocab_size": vocab_size,
        "bos_token_id": 0,
        "eos_token_id": eos_token_id,
    }
    return str(model_dir), hf_config


def test_llama_bpe_checkpoint_without_sentencepiece(tmp_path):
    """Llama-3 ships tokenizer.json only; the export must not require tokenizer.model."""
    model_dir, hf_config = _write_bpe_model_dir(tmp_path, vocab_size=8, eos_token_id=3)
    writer = _RecordingWriter()

    _add_tokenizer_metadata(writer, "llama", model_dir, hf_config)

    assert writer.metadata["tokenizer.ggml.model"] == "gpt2"
    assert writer.metadata["tokenizer.ggml.pre"] == "llama-bpe"
    assert len(writer.metadata["tokenizer.ggml.tokens"]) == 8
    assert writer.metadata["tokenizer.ggml.eos_token_id"] == 3


def test_multiple_eos_ids_map_to_eot_and_eom(tmp_path):
    model_dir, hf_config = _write_bpe_model_dir(tmp_path, vocab_size=8, eos_token_id=[3, 4, 5])
    writer = _RecordingWriter()

    _add_tokenizer_metadata(writer, "llama", model_dir, hf_config)

    assert writer.metadata["tokenizer.ggml.eos_token_id"] == 3
    assert writer.metadata["tokenizer.ggml.eom_token_id"] == 4
    assert writer.metadata["tokenizer.ggml.eot_token_id"] == 5


def test_tokenizer_larger_than_embedding_table_is_rejected(tmp_path):
    model_dir, hf_config = _write_bpe_model_dir(tmp_path, vocab_size=4, eos_token_id=3)
    writer = _RecordingWriter()

    with pytest.raises(ValueError, match="vocab_size=4"):
        _add_tokenizer_metadata(writer, "llama", model_dir, hf_config)
