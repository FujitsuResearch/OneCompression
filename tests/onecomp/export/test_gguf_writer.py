"""Tests for the GGUF v3 writer and reader round trip.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import numpy as np
import pytest

from onecomp.export import GGMLQuantType, GGUFReader, GGUFValueType, GGUFWriter


@pytest.fixture(name="gguf_path")
def fixture_gguf_path(tmp_path):
    """Write a small synthetic GGUF file and return its path."""
    writer = GGUFWriter()
    writer.add_metadata("general.architecture", "llama")
    writer.add_metadata("general.name", "synthetic")
    writer.add_metadata("llama.block_count", 2)
    writer.add_metadata("llama.attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_metadata("tokenizer.ggml.tokens", ["<unk>", "<s>", "hello"])
    writer.add_metadata(
        "tokenizer.ggml.scores", [0.0, -1.0, -2.0], element_type=GGUFValueType.FLOAT32
    )
    writer.add_metadata("tokenizer.ggml.token_type", [2, 3, 1], element_type=GGUFValueType.INT32)
    writer.add_metadata("tokenizer.ggml.add_bos_token", True)

    rng = np.random.default_rng(0)
    writer.add_tensor("token_embd.weight", rng.normal(size=(8, 4)).astype(np.float16))
    writer.add_tensor("blk.0.attn_norm.weight", rng.normal(size=(4,)).astype(np.float32))
    writer.add_tensor("blk.0.attn_q.weight", rng.normal(size=(4, 4)).astype(np.float16))

    path = tmp_path / "synthetic.gguf"
    writer.write(str(path))
    return str(path)


def test_magic_and_version(gguf_path):
    reader = GGUFReader(gguf_path)
    assert reader.version == 3


def test_metadata_round_trip(gguf_path):
    metadata = GGUFReader(gguf_path).metadata
    assert metadata["general.architecture"] == "llama"
    assert metadata["general.name"] == "synthetic"
    assert metadata["general.alignment"] == 32
    assert metadata["llama.block_count"] == 2
    assert metadata["llama.attention.layer_norm_rms_epsilon"] == pytest.approx(1e-5)
    assert metadata["tokenizer.ggml.tokens"] == ["<unk>", "<s>", "hello"]
    assert metadata["tokenizer.ggml.scores"] == pytest.approx([0.0, -1.0, -2.0])
    assert metadata["tokenizer.ggml.token_type"] == [2, 3, 1]
    assert metadata["tokenizer.ggml.add_bos_token"] is True


def test_tensor_directory(gguf_path):
    reader = GGUFReader(gguf_path)
    names = [info.name for info in reader.tensors]
    assert names == ["token_embd.weight", "blk.0.attn_norm.weight", "blk.0.attn_q.weight"]

    embd = reader.tensor("token_embd.weight")
    assert embd.shape == (8, 4)
    assert embd.ggml_type == GGMLQuantType.F16

    norm = reader.tensor("blk.0.attn_norm.weight")
    assert norm.shape == (4,)
    assert norm.ggml_type == GGMLQuantType.F32


def test_tensor_alignment(gguf_path):
    reader = GGUFReader(gguf_path)
    assert reader.data_start % reader.alignment == 0
    for info in reader.tensors:
        assert info.offset % reader.alignment == 0


def test_tensor_data_round_trip(tmp_path):
    writer = GGUFWriter()
    writer.add_metadata("general.architecture", "llama")
    original = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    writer.add_tensor("data", original)
    path = tmp_path / "roundtrip.gguf"
    writer.write(str(path))

    restored = GGUFReader(str(path)).read_tensor("data")
    np.testing.assert_array_equal(restored, original)


def test_custom_alignment_via_metadata(tmp_path):
    """general.alignment added as metadata must drive the actual padding."""
    writer = GGUFWriter()
    writer.add_metadata("general.alignment", 64)
    writer.add_tensor("a", np.zeros((3,), np.float32))  # 12 bytes -> pad to 64
    writer.add_tensor("b", np.ones((4,), np.float32))
    path = tmp_path / "aligned.gguf"
    writer.write(str(path))

    reader = GGUFReader(str(path))
    assert reader.alignment == 64
    assert reader.metadata["general.alignment"] == 64
    assert reader.tensor("b").offset == 64
    np.testing.assert_array_equal(reader.read_tensor("b"), np.ones((4,), np.float32))


def test_empty_array_with_explicit_element_type(tmp_path):
    writer = GGUFWriter()
    writer.add_metadata("empty", [], element_type=GGUFValueType.STRING)
    writer.add_tensor("t", np.zeros((2,), np.float32))
    path = tmp_path / "empty-array.gguf"
    writer.write(str(path))
    assert GGUFReader(str(path)).metadata["empty"] == []


def test_empty_array_without_element_type_rejected(tmp_path):
    writer = GGUFWriter()
    writer.add_metadata("empty", [])
    writer.add_tensor("t", np.zeros((2,), np.float32))
    with pytest.raises(ValueError, match="empty GGUF array"):
        writer.write(str(tmp_path / "invalid.gguf"))


def test_duplicate_tensor_name_rejected():
    writer = GGUFWriter()
    writer.add_tensor("t", np.zeros((2, 2), np.float16))
    with pytest.raises(ValueError, match="duplicate tensor name"):
        writer.add_tensor("t", np.zeros((2, 2), np.float16))


def test_unsupported_dtype_rejected():
    writer = GGUFWriter()
    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        writer.add_tensor("t", np.zeros((2, 2), np.int64))
