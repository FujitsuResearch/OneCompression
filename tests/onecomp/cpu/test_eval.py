"""Unit tests for onecomp.cpu.eval (no model / GPU / llama.cpp runtime needed).

Covers the numpy-only parity / perplexity helpers and the GGUF inspector
(exercised against a tiny synthetic GGUF written with gguf.GGUFWriter).

Copyright 2025-2026 Fujitsu Ltd.
"""

import numpy as np
import pytest

try:
    import gguf  # noqa: F401

    _HAS_GGUF = True
except ImportError:
    _HAS_GGUF = False

_needs_gguf = pytest.mark.skipif(not _HAS_GGUF, reason="gguf not installed")


def test_teacher_forced_parity_identical():
    from onecomp.cpu.eval.parity import teacher_forced_parity

    rng = np.random.default_rng(0)
    logits = rng.standard_normal((5, 32)).astype(np.float32)
    r = teacher_forced_parity(logits, logits.copy())
    assert r.n_positions == 5
    assert r.top1_agreement == 1.0
    assert r.last_argmax_match is True
    assert r.mse == pytest.approx(0.0, abs=1e-9)
    assert r.pearson == pytest.approx(1.0, abs=1e-9)


def test_teacher_forced_parity_partial():
    from onecomp.cpu.eval.parity import teacher_forced_parity

    a = np.zeros((3, 4), dtype=np.float32)
    b = np.zeros((3, 4), dtype=np.float32)
    a[0, 0] = 1.0
    b[0, 0] = 1.0  # match
    a[1, 1] = 1.0
    b[1, 2] = 1.0  # mismatch
    a[2, 3] = 1.0
    b[2, 3] = 1.0  # match (last)
    r = teacher_forced_parity(a, b)
    assert r.top1_agreement == pytest.approx(2 / 3)
    assert r.last_argmax_match is True


def test_greedy_parity_divergence():
    from onecomp.cpu.eval.parity import GreedyParity

    gp = GreedyParity(hf_ids=[1, 2, 3, 4], gguf_ids=[1, 2, 9, 4], first_divergence=2, n_new=4)
    assert gp.identical is False
    assert gp.first_divergence == 2

    gp2 = GreedyParity(hf_ids=[1, 2, 3], gguf_ids=[1, 2, 3], first_divergence=3, n_new=3)
    assert gp2.identical is True


def test_log_softmax_gather_matches_reference():
    from onecomp.cpu.eval.perplexity import _log_softmax_gather

    rng = np.random.default_rng(1)
    logits = rng.standard_normal((6, 10)).astype(np.float32)
    targets = rng.integers(0, 10, size=6)
    got = _log_softmax_gather(logits, targets)

    # reference log-softmax
    x = logits.astype(np.float64)
    ref = x[np.arange(6), targets] - np.log(np.exp(x).sum(axis=-1))
    assert np.allclose(got, ref, atol=1e-9)


@_needs_gguf
def test_inspect_gguf_roundtrip(tmp_path):
    import gguf

    from onecomp.cpu.eval.inspect_gguf import inspect_gguf

    path = str(tmp_path / "tiny.gguf")
    w = gguf.GGUFWriter(path, "llama")
    w.add_tensor("token_embd.weight", np.zeros((4, 8), dtype=np.float32))
    w.add_tensor("blk.0.attn_q.weight", np.zeros((8, 8), dtype=np.float16))
    w.add_tensor("blk.1.attn_q.weight", np.zeros((8, 8), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    report = inspect_gguf(path)
    assert report.architecture == "llama"
    assert report.n_tensors == 3
    assert report.type_counts.get("F16") == 2
    assert report.type_counts.get("F32") == 1
    blocks = report.per_block_types()
    assert set(blocks.keys()) == {0, 1}
    assert blocks[0]["attn_q.weight"] == "F16"
    assert report.effective_bits_per_weight > 0
