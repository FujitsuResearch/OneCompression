"""Unit tests for onecomp.utils.blockwise.

Focused on the Qwen3.6-style hybrid linear-attention / full-attention support:
``_get_block_layer_type``'s ``linear_attn`` fallback, the new
``_create_linear_attention_mask`` helper, the mask-creator dispatch added to
``_compute_per_type_attention_masks``, and end-to-end detection of hybrid
layer types in ``get_blocks_and_inputs``.

Copyright 2025-2026 Fujitsu Ltd.
"""

from collections import UserDict
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer

import onecomp.utils.blockwise as blockwise_mod
from onecomp.utils.blockwise import (
    _ATTN_MASK_MAP_KEY,
    _compute_per_type_attention_masks,
    _create_linear_attention_mask,
    _get_block_layer_type,
    expand_kwargs_batch,
    forward_input,
    get_blocks_and_inputs,
    move_kwargs_to_device,
    prepare_block_kwargs,
)

# ---------------------------------------------------------------------------
# _get_block_layer_type
# ---------------------------------------------------------------------------


def test_get_block_layer_type_from_top_level_layer_type():
    block = nn.Module()
    block.layer_type = "full_attention"
    assert _get_block_layer_type(block) == "full_attention"


def test_get_block_layer_type_from_block_type():
    block = nn.Module()
    block.block_type = "sliding_attention"
    assert _get_block_layer_type(block) == "sliding_attention"


def test_get_block_layer_type_from_self_attn():
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.layer_type = "full_attention"
    assert _get_block_layer_type(block) == "full_attention"


def test_get_block_layer_type_from_linear_attn():
    """Qwen3.6-style hybrid blocks expose their linear-attention sub-module as
    ``linear_attn`` rather than ``self_attn``."""
    block = nn.Module()
    block.linear_attn = nn.Module()
    block.linear_attn.layer_type = "linear_attention"
    assert _get_block_layer_type(block) == "linear_attention"


def test_get_block_layer_type_none_when_nothing_set():
    block = nn.Module()
    assert _get_block_layer_type(block) is None


def test_get_block_layer_type_priority_order():
    """layer_type > block_type > self_attn.layer_type > linear_attn.layer_type."""
    block = nn.Module()
    block.linear_attn = nn.Module()
    block.linear_attn.layer_type = "linear_attention"
    block.self_attn = nn.Module()
    block.self_attn.layer_type = "full_attention"
    block.block_type = "block_type_value"
    block.layer_type = "layer_type_value"

    assert _get_block_layer_type(block) == "layer_type_value"

    del block.layer_type
    assert _get_block_layer_type(block) == "block_type_value"

    del block.block_type
    assert _get_block_layer_type(block) == "full_attention"


# ---------------------------------------------------------------------------
# _create_linear_attention_mask
# ---------------------------------------------------------------------------


def test_create_linear_attention_mask_none_attention_mask():
    assert (
        _create_linear_attention_mask(
            config=MagicMock(),
            inputs_embeds=torch.zeros(1, 4, 8),
            attention_mask=None,
        )
        is None
    )


def test_create_linear_attention_mask_with_previous_state_returns_none():
    past_key_values = MagicMock()
    past_key_values.has_previous_state.return_value = True

    result = _create_linear_attention_mask(
        config=MagicMock(),
        inputs_embeds=torch.zeros(1, 4, 8),
        attention_mask=torch.zeros(1, 4, dtype=torch.long),
        past_key_values=past_key_values,
    )
    assert result is None


def test_create_linear_attention_mask_all_attended_returns_none():
    """No padding present (all-ones mask) => no explicit mask needed."""
    result = _create_linear_attention_mask(
        config=MagicMock(),
        inputs_embeds=torch.zeros(1, 4, 8),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
    )
    assert result is None


def test_create_linear_attention_mask_with_padding_returns_bool_mask():
    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    result = _create_linear_attention_mask(
        config=MagicMock(),
        inputs_embeds=torch.zeros(1, 4, 8),
        attention_mask=attention_mask,
    )

    assert result is not None
    assert result.dtype == torch.bool
    assert torch.equal(result, attention_mask.bool())


# ---------------------------------------------------------------------------
# _compute_per_type_attention_masks
# ---------------------------------------------------------------------------


def _blocks_parent(hidden_size=8):
    parent = nn.Module()
    parent.config = MagicMock(hidden_size=hidden_size)
    parent.dummy_param = nn.Parameter(torch.zeros(1))
    return parent


def test_compute_per_type_attention_masks_returns_none_without_position_ids():
    parent = _blocks_parent()
    result = _compute_per_type_attention_masks(parent, {}, {"full_attention", "sliding_attention"})
    assert result is None


def test_compute_per_type_attention_masks_returns_none_for_single_type():
    parent = _blocks_parent()
    kwargs = {"position_ids": torch.arange(4).unsqueeze(0)}
    result = _compute_per_type_attention_masks(parent, kwargs, {"full_attention"})
    assert result is None


def test_compute_per_type_attention_masks_gemma_like_dispatch():
    """Regression guard: full_attention/sliding_attention models keep using
    create_causal_mask / create_sliding_window_causal_mask."""
    parent = _blocks_parent()
    kwargs = {"position_ids": torch.arange(4).unsqueeze(0)}
    full_sentinel = object()
    sliding_sentinel = object()

    with (
        patch(
            "transformers.masking_utils.create_causal_mask", return_value=full_sentinel
        ) as m_full,
        patch(
            "transformers.masking_utils.create_sliding_window_causal_mask",
            return_value=sliding_sentinel,
        ) as m_sliding,
    ):
        result = _compute_per_type_attention_masks(
            parent, kwargs, {"full_attention", "sliding_attention"}
        )

    assert result == {"full_attention": full_sentinel, "sliding_attention": sliding_sentinel}
    m_full.assert_called_once()
    m_sliding.assert_called_once()


def test_compute_per_type_attention_masks_qwen36_hybrid_dispatch():
    """The Qwen3.6-style hybrid branch must route 'linear_attention' through
    _create_linear_attention_mask and 'full_attention' through create_causal_mask,
    instead of the Gemma-style full/sliding pair."""
    parent = _blocks_parent()
    kwargs = {"position_ids": torch.arange(4).unsqueeze(0)}
    full_sentinel = object()
    linear_sentinel = object()

    with (
        patch(
            "transformers.masking_utils.create_causal_mask", return_value=full_sentinel
        ) as m_full,
        patch(
            "onecomp.utils.blockwise._create_linear_attention_mask",
            return_value=linear_sentinel,
        ) as m_linear,
        patch("transformers.masking_utils.create_sliding_window_causal_mask") as m_sliding,
    ):
        result = _compute_per_type_attention_masks(
            parent, kwargs, {"linear_attention", "full_attention"}
        )

    assert result == {"full_attention": full_sentinel, "linear_attention": linear_sentinel}
    m_full.assert_called_once()
    m_linear.assert_called_once()
    m_sliding.assert_not_called()


def test_compute_per_type_attention_masks_unknown_type_is_skipped():
    parent = _blocks_parent()
    kwargs = {"position_ids": torch.arange(4).unsqueeze(0)}
    full_sentinel = object()

    with (
        patch("transformers.masking_utils.create_causal_mask", return_value=full_sentinel),
        patch("transformers.masking_utils.create_sliding_window_causal_mask"),
    ):
        result = _compute_per_type_attention_masks(
            parent, kwargs, {"full_attention", "some_unknown_type"}
        )

    assert result is None  # only one recognized entry => len(mask_map) <= 1


# ---------------------------------------------------------------------------
# get_blocks_and_inputs: end-to-end hybrid-layer-type detection
# ---------------------------------------------------------------------------


class _FakeSubAttn(nn.Module):
    def __init__(self, layer_type: str):
        super().__init__()
        self.layer_type = layer_type


class _FakeDecoderLayer(GradientCheckpointingLayer):
    """Minimal stand-in for a Qwen3.6-style hybrid decoder layer.

    Linear-attention layers expose their sub-module as ``linear_attn``;
    full-attention layers expose it as ``self_attn`` (mirrors
    onecomp/pre_process/quant_models.py's real wrappers).
    """

    def __init__(self, hidden_size: int, layer_type: str):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        if layer_type == "linear_attention":
            self.linear_attn = _FakeSubAttn(layer_type)
        else:
            self.self_attn = _FakeSubAttn(layer_type)

    def forward(self, hidden_states, *args, **kwargs):
        return (self.proj(hidden_states),)


class _FakeInnerModel(nn.Module):
    def __init__(self, config, blocks: nn.ModuleList):
        super().__init__()
        self.config = config
        self.layers = blocks


class _FakeCausalLM(nn.Module):
    def __init__(self, layer_types, hidden_size=8, vocab_size=16):
        super().__init__()
        config = MagicMock(layer_types=layer_types, hidden_size=hidden_size)
        blocks = nn.ModuleList([_FakeDecoderLayer(hidden_size, lt) for lt in layer_types])
        self.model = _FakeInnerModel(config, blocks)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, **kwargs):
        hidden = self.embed_tokens(input_ids)
        for layer in self.model.layers:
            out = layer(hidden, **kwargs)
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden


class _RecordingFakeCausalLM(_FakeCausalLM):
    def __init__(self, layer_types, hidden_size=8, vocab_size=16):
        super().__init__(layer_types, hidden_size=hidden_size, vocab_size=vocab_size)
        self.forward_shapes = []

    def forward(self, input_ids, **kwargs):
        self.forward_shapes.append(
            {
                "input_ids": tuple(input_ids.shape),
                **{k: tuple(v.shape) for k, v in kwargs.items() if isinstance(v, torch.Tensor)},
            }
        )
        return super().forward(input_ids, **kwargs)


def _make_model_and_inputs(layer_types, hidden_size=8, batch=2, seq_len=5, vocab_size=16):
    model = _FakeCausalLM(layer_types, hidden_size=hidden_size, vocab_size=vocab_size)
    model.eval()
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    model_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }
    return model, model_inputs


def test_get_blocks_and_inputs_detects_qwen36_linear_full_hybrid():
    model, model_inputs = _make_model_and_inputs(["full_attention", "linear_attention"])
    full_sentinel = object()

    with patch("transformers.masking_utils.create_causal_mask", return_value=full_sentinel):
        blocks, inps, kwargs = get_blocks_and_inputs(model, model_inputs, batch_size=2)

    assert len(blocks) == 2
    assert inps.shape == (2, 5, 8)

    mask_map = kwargs[_ATTN_MASK_MAP_KEY]
    assert mask_map["full_attention"] is full_sentinel
    # The internal dummy padding mask fed to _create_linear_attention_mask is
    # always all-ones, so this always resolves to "no explicit mask needed"
    # regardless of the caller's real attention_mask.
    assert mask_map["linear_attention"] is None


def test_get_blocks_and_inputs_single_layer_type_has_no_mask_map():
    """Non-hybrid models (single layer type) must be unaffected by the new
    hybrid-detection branch: no attention-mask map should be computed."""
    model, model_inputs = _make_model_and_inputs(["full_attention", "full_attention"])

    blocks, inps, kwargs = get_blocks_and_inputs(model, model_inputs, batch_size=2)

    assert len(blocks) == 2
    assert _ATTN_MASK_MAP_KEY not in kwargs


def test_get_blocks_and_inputs_slices_model_kwargs_for_input_capture():
    model = _RecordingFakeCausalLM(["full_attention", "full_attention"])
    _, model_inputs = _make_model_and_inputs(
        ["full_attention", "full_attention"], batch=5, seq_len=4
    )
    model_inputs["mm_token_type_ids"] = torch.zeros_like(model_inputs["input_ids"])

    _, inps, _ = get_blocks_and_inputs(model, model_inputs, batch_size=2)

    assert inps.shape == (5, 4, 8)
    # call 0 captures batch-independent kwargs with a single sample.
    assert model.forward_shapes[1]["input_ids"] == (2, 4)
    assert model.forward_shapes[1]["attention_mask"] == (2, 4)
    assert model.forward_shapes[1]["position_ids"] == (2, 4)
    assert model.forward_shapes[1]["mm_token_type_ids"] == (2, 4)
    assert model.forward_shapes[3]["input_ids"] == (1, 4)
    assert model.forward_shapes[3]["attention_mask"] == (1, 4)
    assert model.forward_shapes[3]["position_ids"] == (1, 4)
    assert model.forward_shapes[3]["mm_token_type_ids"] == (1, 4)


# ---------------------------------------------------------------------------
# UserDict kwargs
# ---------------------------------------------------------------------------


def test_expand_kwargs_batch_expands_nested_user_dict():
    shared_kv_states = UserDict(
        {
            "key": torch.zeros(1, 2, 4),
            "value": torch.ones(1, 2, 4),
        }
    )

    result = expand_kwargs_batch({"shared_kv_states": shared_kv_states}, batch_size=3)

    assert result["shared_kv_states"] is shared_kv_states
    assert result["shared_kv_states"]["key"].shape == (3, 2, 4)
    assert result["shared_kv_states"]["value"].shape == (3, 2, 4)


def test_expand_kwargs_batch_preserves_user_dict_updates_between_blocks():
    shared_kv_states = UserDict()
    provider_kwargs = expand_kwargs_batch({"shared_kv_states": shared_kv_states}, batch_size=2)

    provider_kwargs["shared_kv_states"]["full_attention"] = torch.zeros(1, 2, 4)
    consumer_kwargs = expand_kwargs_batch({"shared_kv_states": shared_kv_states}, batch_size=2)

    assert provider_kwargs["shared_kv_states"] is shared_kv_states
    assert consumer_kwargs["shared_kv_states"] is shared_kv_states
    assert consumer_kwargs["shared_kv_states"]["full_attention"].shape == (2, 2, 4)


def test_move_kwargs_to_device_moves_nested_user_dict():
    shared_kv_states = UserDict({"key": torch.zeros(1, 2, 4)})

    result = move_kwargs_to_device(shared_kv_states, torch.device("meta"))

    assert result is shared_kv_states
    assert result["key"].device == torch.device("meta")


def test_prepare_block_kwargs_slices_shared_kv_for_consumer_block():
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.layer_type = "full_attention"
    block.self_attn.is_kv_shared_layer = True
    key_states = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    value_states = key_states + 100
    shared_kv_states = UserDict({"full_attention": (key_states, value_states)})

    result = prepare_block_kwargs(
        {"shared_kv_states": shared_kv_states},
        block,
        pli=None,
        offset=1,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert result["shared_kv_states"] is not shared_kv_states
    sliced_key, sliced_value = result["shared_kv_states"]["full_attention"]
    assert torch.equal(sliced_key, key_states[1:3])
    assert torch.equal(sliced_value, value_states[1:3])


def test_prepare_block_kwargs_uses_fresh_shared_kv_for_provider_block():
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.store_full_length_kv = True
    shared_kv_states = UserDict({"full_attention": (torch.zeros(4, 2, 3), torch.ones(4, 2, 3))})

    result = prepare_block_kwargs(
        {"shared_kv_states": shared_kv_states},
        block,
        pli=None,
        offset=0,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert result["shared_kv_states"] is not shared_kv_states
    assert result["shared_kv_states"] == {}


class _SharedKvProviderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.store_full_length_kv = True

    def forward(self, hidden_states, *, shared_kv_states):
        key_states = hidden_states[..., :3].clone()
        value_states = key_states + 100
        shared_kv_states["full_attention"] = (key_states, value_states)
        return hidden_states + 1


class _SharedKvConsumerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.layer_type = "full_attention"
        self.self_attn.is_kv_shared_layer = True
        self.seen_key_batches = []
        self.seen_value_batches = []

    def forward(self, hidden_states, *, shared_kv_states):
        key_states, value_states = shared_kv_states["full_attention"]
        self.seen_key_batches.append(key_states.detach().cpu())
        self.seen_value_batches.append(value_states.detach().cpu())
        return hidden_states + key_states.sum(dim=-1, keepdim=True)


def test_forward_input_carries_shared_kv_from_provider_to_consumer_by_batch_slice():
    inps = torch.arange(5 * 4 * 6, dtype=torch.float32).reshape(5, 4, 6)
    kwargs = {"shared_kv_states": UserDict()}

    provider_out = forward_input(
        inps,
        _SharedKvProviderBlock(),
        kwargs,
        batch_size=2,
        device=torch.device("cpu"),
    )
    consumer = _SharedKvConsumerBlock()
    consumer_out = forward_input(
        provider_out,
        consumer,
        kwargs,
        batch_size=2,
        device=torch.device("cpu"),
    )

    stored_key, stored_value = kwargs["shared_kv_states"]["full_attention"]
    assert torch.equal(stored_key, inps[..., :3])
    assert torch.equal(stored_value, inps[..., :3] + 100)
    assert [tuple(batch.shape) for batch in consumer.seen_key_batches] == [
        (2, 4, 3),
        (2, 4, 3),
        (1, 4, 3),
    ]
    assert torch.equal(consumer.seen_key_batches[0], inps[:2, :, :3])
    assert torch.equal(consumer.seen_key_batches[1], inps[2:4, :, :3])
    assert torch.equal(consumer.seen_key_batches[2], inps[4:5, :, :3])
    assert torch.equal(consumer.seen_value_batches[2], inps[4:5, :, :3] + 100)
    assert consumer_out.shape == provider_out.shape
