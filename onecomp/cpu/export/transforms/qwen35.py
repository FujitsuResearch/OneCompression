"""Qwen3.5 linear-attention transformations for direct GGUF export.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuhki Yano

"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from onecomp.cpu.export.blocks import UnsupportedGPTQLayout

_QWEN35_TEXT_MODEL_TYPES = {
    "qwen3_5_text",
    "qwen3_5_moe_text",
}


def get_qwen35_text_config(config: dict) -> dict | None:
    """Return the Qwen3.5 text config, or None for other architectures."""
    text_config = config.get("text_config")
    text_config = text_config if isinstance(text_config, dict) else config

    if text_config.get("model_type") not in _QWEN35_TEXT_MODEL_TYPES:
        return None

    return text_config


def is_qwen35_model(model_dir: str) -> bool:
    """Return whether model_dir contains a Qwen3.5 text configuration."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return False

    try:
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    return get_qwen35_text_config(config) is not None


@dataclass(frozen=True)
class Qwen35LinearAttentionLayout:
    num_k_heads: int
    num_v_heads: int
    key_head_dim: int
    value_head_dim: int


def load_qwen35_layout(
    quantized_dir: str,
) -> Optional[Qwen35LinearAttentionLayout]:
    """Load Qwen3.5 linear-attention layout, or return None for other models."""
    config_path = os.path.join(quantized_dir, "config.json")
    if not os.path.isfile(config_path):
        return None

    with open(config_path, encoding="utf-8") as file:
        root_config = json.load(file)

    config = get_qwen35_text_config(root_config)

    if config is None:
        return None

    required = (
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
    )
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError(f"Qwen3.5 config is missing required fields: {missing}")

    layout = Qwen35LinearAttentionLayout(
        num_k_heads=int(config["linear_num_key_heads"]),
        num_v_heads=int(config["linear_num_value_heads"]),
        key_head_dim=int(config["linear_key_head_dim"]),
        value_head_dim=int(config["linear_value_head_dim"]),
    )

    if layout.num_k_heads <= 0 or layout.num_v_heads <= 0:
        raise ValueError("Qwen3.5 linear-attention head counts must be positive")

    if layout.num_v_heads % layout.num_k_heads != 0:
        raise ValueError("linear_num_value_heads must be divisible by " "linear_num_key_heads")

    # No special reordering is necessary when K and V counts match.
    if layout.num_k_heads == layout.num_v_heads:
        return None

    return layout


def _index_select(
    tensor: Tensor,
    dim: int,
    indices: np.ndarray,
) -> Tensor:
    """Select indices using an int64 tensor on the source tensor's device."""
    index = torch.as_tensor(
        indices,
        dtype=torch.long,
        device=tensor.device,
    )
    return torch.index_select(tensor, dim, index)


def _v_head_permutation(
    layout: Qwen35LinearAttentionLayout,
    head_dim: int,
) -> np.ndarray:
    """Match llama.cpp's Qwen3.5 _reorder_v_heads implementation."""
    num_v_per_k = layout.num_v_heads // layout.num_k_heads

    return (
        np.arange(
            layout.num_v_heads * head_dim,
            dtype=np.int64,
        )
        .reshape(layout.num_k_heads, num_v_per_k, head_dim)
        .transpose(1, 0, 2)
        .reshape(-1)
    )


def _permute_rows(
    q_int: Tensor,
    scales: Tensor,
    zeros: Tensor,
    row_perm: np.ndarray,
) -> Tuple[Tensor, Tensor, Tensor]:
    if row_perm.size != q_int.shape[0]:
        raise UnsupportedGPTQLayout(
            f"Qwen3.5 row permutation has {row_perm.size} entries, "
            f"but tensor has {q_int.shape[0]} rows"
        )

    if scales.shape[1] != q_int.shape[0]:
        raise UnsupportedGPTQLayout(
            f"scale output dimension {scales.shape[1]} does not match "
            f"weight rows {q_int.shape[0]}"
        )

    if zeros.shape[1] != q_int.shape[0]:
        raise UnsupportedGPTQLayout(
            f"zero output dimension {zeros.shape[1]} does not match "
            f"weight rows {q_int.shape[0]}"
        )

    return (
        _index_select(q_int, 0, row_perm),
        _index_select(scales, 1, row_perm),
        _index_select(zeros, 1, row_perm),
    )


def transform_qwen35_gptq(
    layer_name: str,
    q_int: Tensor,
    scales: Tensor,
    zeros: Tensor,
    groupsize: int,
    layout: Optional[Qwen35LinearAttentionLayout],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Reorder Qwen3.5 GPTQ tensors to the layout expected by llama.cpp."""
    if layout is None or ".linear_attn." not in layer_name:
        return q_int, scales, zeros

    # Accept names both with and without the trailing HF ".weight".
    name = layer_name.removesuffix(".weight")

    if name.endswith(".linear_attn.in_proj_qkv"):
        q_dim = layout.num_k_heads * layout.key_head_dim
        k_dim = layout.num_k_heads * layout.key_head_dim
        v_offset = q_dim + k_dim

        v_perm = _v_head_permutation(
            layout,
            layout.value_head_dim,
        )
        row_perm = np.concatenate(
            (
                np.arange(v_offset, dtype=np.int64),
                v_offset + v_perm,
            )
        )
        return _permute_rows(q_int, scales, zeros, row_perm)

    if name.endswith(".linear_attn.in_proj_z"):
        row_perm = _v_head_permutation(
            layout,
            layout.value_head_dim,
        )
        return _permute_rows(q_int, scales, zeros, row_perm)

    if name.endswith(
        (
            ".linear_attn.in_proj_a",
            ".linear_attn.in_proj_b",
        )
    ):
        row_perm = _v_head_permutation(layout, head_dim=1)
        return _permute_rows(q_int, scales, zeros, row_perm)

    if name.endswith(".linear_attn.out_proj"):
        col_perm = _v_head_permutation(
            layout,
            layout.value_head_dim,
        )

        if col_perm.size != q_int.shape[1]:
            raise UnsupportedGPTQLayout(
                f"Qwen3.5 column permutation has {col_perm.size} entries, "
                f"but tensor has {q_int.shape[1]} columns"
            )

        if groupsize <= 0 or col_perm.size % groupsize != 0:
            raise UnsupportedGPTQLayout(f"invalid groupsize={groupsize} for Qwen3.5 out_proj")

        # Direct conversion is lossless only when the permutation moves
        # complete GPTQ groups without rearranging values inside a group.
        grouped_perm = col_perm.reshape(-1, groupsize)
        group_perm = grouped_perm[:, 0] // groupsize

        expected = group_perm[:, None] * groupsize + np.arange(groupsize, dtype=np.int64)[None, :]
        if not np.array_equal(grouped_perm, expected):
            raise UnsupportedGPTQLayout(
                "Qwen3.5 out_proj permutation crosses GPTQ group "
                "boundaries; direct lossless packing is unavailable"
            )

        if scales.shape[0] != group_perm.size:
            raise UnsupportedGPTQLayout(
                f"scale group count {scales.shape[0]} does not match "
                f"permutation group count {group_perm.size}"
            )

        return (
            _index_select(q_int, 1, col_perm),
            _index_select(scales, 0, group_perm),
            _index_select(zeros, 0, group_perm),
        )

    return q_int, scales, zeros
