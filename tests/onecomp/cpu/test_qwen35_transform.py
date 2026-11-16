"""Regression tests for Qwen3.8 dense and Qwen3.6 MoE GGUF transforms.

These tests verify native and composite configuration detection and the
linear-attention V-head permutation required by llama.cpp.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuhki Yano

"""

import json

import pytest
import torch

from onecomp.cpu.export.transforms.qwen35 import (
    Qwen35LinearAttentionLayout,
    load_qwen35_layout,
    transform_qwen35_gptq,
)

_LAYOUT = {
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
}


def _write_config(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


@pytest.mark.parametrize("model_type", ["qwen3_5_text", "qwen3_5_moe_text"])
def test_load_qwen35_native_layout(tmp_path, model_type):
    path = _write_config(tmp_path, {"model_type": model_type, **_LAYOUT})

    assert load_qwen35_layout(path) == Qwen35LinearAttentionLayout(
        num_k_heads=16,
        num_v_heads=32,
        key_head_dim=128,
        value_head_dim=128,
    )


@pytest.mark.parametrize(
    "root_type,text_type",
    [
        ("qwen3_5", "qwen3_5_text"),
        ("qwen3_5_moe", "qwen3_5_moe_text"),
    ],
)
def test_load_qwen35_composite_layout(tmp_path, root_type, text_type):
    path = _write_config(
        tmp_path,
        {
            "model_type": root_type,
            "text_config": {"model_type": text_type, **_LAYOUT},
        },
    )

    assert load_qwen35_layout(path) is not None


def test_load_qwen35_ignores_other_models(tmp_path):
    path = _write_config(tmp_path, {"model_type": "qwen3", **_LAYOUT})
    assert load_qwen35_layout(path) is None


def test_transform_qwen35_qkv_rows():
    layout = Qwen35LinearAttentionLayout(2, 4, 1, 1)
    q_int = torch.arange(8).reshape(8, 1)
    scales = torch.arange(8).reshape(1, 8)
    zeros = scales.clone()

    got_q, got_scales, got_zeros = transform_qwen35_gptq(
        "model.layers.0.linear_attn.in_proj_qkv",
        q_int,
        scales,
        zeros,
        groupsize=1,
        layout=layout,
    )

    expected = torch.tensor([0, 1, 2, 3, 4, 6, 5, 7])
    assert torch.equal(got_q[:, 0], expected)
    assert torch.equal(got_scales[0], expected)
    assert torch.equal(got_zeros[0], expected)
