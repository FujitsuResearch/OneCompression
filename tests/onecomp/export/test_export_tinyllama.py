"""Slow integration test: TinyLlama HF checkpoint -> GGUF F16.

Downloads the TinyLlama-1.1B (base) checkpoint from the Hugging Face Hub and
exports it to GGUF.  Requires network access or a pre-populated HF cache.
Run with ``pytest -m slow``.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import pytest
from huggingface_hub import snapshot_download

from onecomp.export import GGMLQuantType, GGUFExportConfig, GGUFReader, export_gguf

pytestmark = pytest.mark.slow

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def test_export_tinyllama_full(tmp_path):
    MODEL_DIR = snapshot_download(MODEL_ID)
    out_path = str(tmp_path / "tinyllama-f16.gguf")
    result = export_gguf(MODEL_DIR, GGUFExportConfig(out_path=out_path))
    assert result == out_path

    reader = GGUFReader(out_path)
    metadata = reader.metadata

    assert metadata["general.architecture"] == "llama"
    assert metadata["llama.block_count"] == 22
    assert metadata["llama.embedding_length"] == 2048
    assert metadata["llama.feed_forward_length"] == 5632
    assert metadata["llama.attention.head_count"] == 32
    assert metadata["llama.attention.head_count_kv"] == 4
    assert metadata["llama.vocab_size"] == 32000
    assert metadata["llama.rope.dimension_count"] == 64
    assert metadata["tokenizer.ggml.model"] == "llama"
    assert len(metadata["tokenizer.ggml.tokens"]) == 32000
    assert len(metadata["tokenizer.ggml.scores"]) == 32000
    assert len(metadata["tokenizer.ggml.token_type"]) == 32000
    assert metadata["tokenizer.ggml.bos_token_id"] == 1
    assert metadata["tokenizer.ggml.eos_token_id"] == 2

    # 22 blocks x 9 tensors + token_embd + output_norm + output.
    assert len(reader.tensors) == 22 * 9 + 3

    embd = reader.tensor("token_embd.weight")
    assert embd.shape == (32000, 2048)
    assert embd.ggml_type == GGMLQuantType.F16

    norm = reader.tensor("blk.0.attn_norm.weight")
    assert norm.shape == (2048,)
    assert norm.ggml_type == GGMLQuantType.F32
