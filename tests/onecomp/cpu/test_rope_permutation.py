"""Interleaved-RoPE row permutation for stitched llama GGUF exports.

llama.cpp's ``convert_hf_to_gguf.py`` permutes ``attn_q`` / ``attn_k`` rows of
``llama``-architecture models to the interleaved layout, so GPTQ codes stitched
onto such a skeleton must be permuted the same way. These tests check that

  - the permutation of ``(q_int, scales, zeros)`` is *lossless*: dequantizing
    the permuted pack equals row-permuting the dequantized original,
  - only ``attn_q`` / ``attn_k`` tensors are permuted, with the right head
    count each (GQA), and
  - the skeleton metadata probe fires for ``llama`` and stays off for
    NEOX-style architectures (qwen2, gemma, ...).

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import numpy as np
import pytest
import torch

from onecomp.cpu.export.skeleton import (
    permute_gptq_rows,
    rope_heads_for_tensor,
    rope_permutation_heads,
)
from onecomp.export.gguf_export import permute_rope_rows

gguf = pytest.importorskip("gguf")
from gguf import quants as GQ  # noqa: E402


def _dequant_to_logical(packed, qtype, out_f, in_f):
    w = GQ.dequantize(np.ascontiguousarray(packed), qtype).astype(np.float32)
    return w.reshape(out_f, in_f)


def _random_gptq_layer(out_f=64, in_f=64, gs=32, wbits=4, seed=0):
    rng = np.random.default_rng(seed)
    n_groups = in_f // gs
    q = torch.tensor(rng.integers(0, 2**wbits, size=(out_f, in_f)), dtype=torch.int32)
    scales = torch.tensor(rng.uniform(0.01, 0.2, size=(n_groups, out_f)), dtype=torch.float32)
    zeros = torch.full((n_groups, out_f), 2 ** (wbits - 1), dtype=torch.int32)
    return q, scales, zeros


def test_permute_gptq_rows_is_lossless():
    """Dequant(permuted pack) == row-permutation of dequant(original pack)."""
    from onecomp.cpu.export.blocks import pack_gptq_linear

    out_f, in_f, gs, n_head = 64, 64, 32, 4
    q, scales, zeros = _random_gptq_layer(out_f, in_f, gs)

    packed, qtype = pack_gptq_linear(
        q.numpy(), scales.numpy(), zeros.numpy(), wbits=4, sym=True, groupsize=gs
    )
    ref = _dequant_to_logical(packed, qtype, out_f, in_f)

    qp, sp, zp = permute_gptq_rows(q, scales, zeros, n_head)
    packed_p, qtype_p = pack_gptq_linear(
        qp.numpy(), sp.numpy(), zp.numpy(), wbits=4, sym=True, groupsize=gs
    )
    got = _dequant_to_logical(packed_p, qtype_p, out_f, in_f)

    expected = permute_rope_rows(torch.from_numpy(ref), n_head).numpy()
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_permute_gptq_rows_keeps_grouping():
    """The permutation must act on output rows only (input grouping intact)."""
    q, scales, zeros = _random_gptq_layer()
    qp, sp, zp = permute_gptq_rows(q, scales, zeros, n_head=4)
    assert qp.shape == q.shape and sp.shape == scales.shape and zp.shape == zeros.shape
    # Every original row must still exist somewhere (it is a permutation).
    orig = {tuple(r.tolist()) for r in q}
    perm = {tuple(r.tolist()) for r in qp}
    assert orig == perm


def test_rope_heads_for_tensor_routes_q_and_k_only():
    heads = (32, 8)
    assert rope_heads_for_tensor("blk.0.attn_q.weight", heads) == 32
    assert rope_heads_for_tensor("blk.5.attn_k.weight", heads) == 8
    assert rope_heads_for_tensor("blk.5.attn_v.weight", heads) is None
    assert rope_heads_for_tensor("blk.5.ffn_down.weight", heads) is None
    assert rope_heads_for_tensor("blk.0.attn_q.weight", None) is None


class _Field:
    def __init__(self, value):
        self._value = value

    def contents(self):
        return self._value


class _Reader:
    """Minimal stand-in for gguf.GGUFReader metadata access."""

    def __init__(self, fields):
        self._fields = fields

    def get_field(self, name):
        return self._fields.get(name)


def test_rope_permutation_heads_llama_and_neox():
    llama = _Reader(
        {
            "general.architecture": _Field("llama"),
            "llama.attention.head_count": _Field(32),
            "llama.attention.head_count_kv": _Field(8),
        }
    )
    assert rope_permutation_heads(llama) == (32, 8)

    no_kv = _Reader(
        {
            "general.architecture": _Field("llama"),
            "llama.attention.head_count": _Field(16),
        }
    )
    assert rope_permutation_heads(no_kv) == (16, 16)

    qwen2 = _Reader(
        {
            "general.architecture": _Field("qwen2"),
            "qwen2.attention.head_count": _Field(16),
        }
    )
    assert rope_permutation_heads(qwen2) is None
