"""Tests for router-only post-quantization fine-tuning.

Copyright 2025-2026 Fujitsu Ltd.

"""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset

from onecomp.post_process.router_fine_tuning import (
    RouterFineTuning,
    _compute_next_token_loss,
)


class _Tokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, texts, max_length, **_kwargs):
        input_ids = []
        attention_masks = []
        for index, _text in enumerate(texts):
            tokens = [] if not _text else [1, 2 + index % 2, 3, 4][:max_length]
            padding = max_length - len(tokens)
            input_ids.append(tokens + [0] * padding)
            attention_masks.append([1] * len(tokens) + [0] * padding)
        return {"input_ids": input_ids, "attention_mask": attention_masks}


class _TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.embed_tokens = nn.Embedding(8, 6)
        self.router = nn.Linear(6, 2, bias=False)
        self.experts = nn.ModuleList([nn.Linear(6, 6), nn.Linear(6, 6)])
        self.gate_proj = nn.Linear(6, 6, bias=False)
        self.lm_head = nn.Linear(6, 8, bias=False)
        self.config = SimpleNamespace(
            use_cache=True,
            quantization_config={
                "quant_method": "gptq",
                "modules_in_block_to_quantize": [],
            },
        )

    def forward(self, input_ids, attention_mask=None):  # noqa: ARG002
        hidden = self.embed_tokens(input_ids)
        router_weights = self.router(hidden).softmax(dim=-1)
        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=-2)
        hidden = (expert_outputs * router_weights.unsqueeze(-1)).sum(dim=-2)
        hidden = hidden + 0.1 * self.gate_proj(hidden)
        return SimpleNamespace(logits=self.lm_head(F.silu(hidden)))


def _model_config():
    return SimpleNamespace(device="cpu", load_tokenizer=lambda: _Tokenizer())


def test_next_token_loss_shifts_labels_and_ignores_padding():
    logits = torch.tensor(
        [[[8.0, 0.0], [0.0, 8.0], [8.0, 0.0], [0.0, 8.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[0, 0, 1, -100]])

    loss = _compute_next_token_loss(logits, labels)
    expected = F.cross_entropy(logits[:, :2].reshape(-1, 2), labels[:, 1:3].reshape(-1))

    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None


def test_next_token_loss_is_finite_without_valid_targets():
    logits = torch.randn(1, 4, 8, requires_grad=True)
    labels = torch.full((1, 4), -100)

    loss = _compute_next_token_loss(logits, labels)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_router_selection_uses_exact_path_components():
    model = _TinyMoE()
    process = RouterFineTuning(data_files="unused.jsonl")

    selected = {name for name, _ in process._select_router_parameters(model)}

    assert selected == {"router.weight"}
    assert model.router.weight.requires_grad
    assert not model.gate_proj.weight.requires_grad
    assert all(
        not parameter.requires_grad
        for expert in model.experts
        for parameter in expert.parameters()
    )


def test_run_updates_only_router_and_records_metadata(monkeypatch):
    model = _TinyMoE()
    process = RouterFineTuning(
        data_files="unused.jsonl",
        max_length=4,
        epochs=2,
        batch_size=2,
        lr=0.1,
        logging_steps=0,
    )
    dataset = Dataset.from_dict({"text": ["a", "", "b", ""]})
    monkeypatch.setattr(process, "_load_train_dataset", lambda: dataset)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    process.run(model, _model_config())

    after = dict(model.named_parameters())
    assert not torch.equal(before["router.weight"], after["router.weight"])
    for name, old_parameter in before.items():
        if name != "router.weight":
            torch.testing.assert_close(after[name], old_parameter, rtol=0.0, atol=0.0)

    assert not model.training
    assert model.config.use_cache is True
    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
    metadata = model.config.quantization_config["onecomp_post_processes"][-1]
    assert metadata["class"] == "RouterFineTuning"
    assert metadata["executed"] is True


def test_run_marks_all_skipped_batches_as_not_executed(monkeypatch):
    model = _TinyMoE()
    process = RouterFineTuning(
        data_files="unused.jsonl",
        max_length=4,
        batch_size=2,
        logging_steps=0,
    )
    dataset = Dataset.from_dict({"text": ["", ""]})
    monkeypatch.setattr(process, "_load_train_dataset", lambda: dataset)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    process.run(model, _model_config())

    for name, old_parameter in before.items():
        torch.testing.assert_close(model.get_parameter(name), old_parameter, rtol=0.0, atol=0.0)
    metadata = model.config.quantization_config["onecomp_post_processes"][-1]
    assert metadata["executed"] is False
    assert metadata["reason"] == "no_valid_next_token_targets"


def test_run_restores_state_when_device_move_fails(monkeypatch):
    router_module = importlib.import_module("onecomp.post_process.router_fine_tuning")
    model = _TinyMoE()
    process = RouterFineTuning(data_files="unused.jsonl", max_length=4, logging_steps=0)
    dataset = Dataset.from_dict({"text": ["valid"]})
    monkeypatch.setattr(process, "_load_train_dataset", lambda: dataset)
    monkeypatch.setattr(process, "_resolve_train_device", lambda _config: torch.device("meta"))
    monkeypatch.setattr(
        router_module,
        "_capture_gptq_pack_state",
        lambda _model: {"router": True},
    )
    monkeypatch.setattr(router_module, "_unpack_gptq_linears_in_place", lambda _model: 1)
    restored_states = []
    monkeypatch.setattr(
        router_module,
        "_restore_gptq_pack_state",
        lambda _model, state: restored_states.append(state),
    )
    original_to = model.to

    def fail_on_meta(device):
        if torch.device(device).type == "meta":
            raise RuntimeError("simulated device allocation failure")
        return original_to(device)

    monkeypatch.setattr(model, "to", fail_on_meta)

    with pytest.raises(RuntimeError, match="simulated device allocation failure"):
        process.run(model, _model_config())

    assert model.config.use_cache is True
    assert restored_states == [{"router": True}]
