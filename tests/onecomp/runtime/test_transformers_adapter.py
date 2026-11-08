"""CPU integration tests: real forwards and cache-consistent forced markers."""

from types import SimpleNamespace

import torch

from onecomp.reasoning import ReasoningConfig, ReasoningFormat
from onecomp.runtime.transformers import TransformersReasoningAdapter


class TinyTokenizer:
    eos_token_id = 7
    pieces = {
        0: "",
        1: "Question",
        2: "<think>",
        3: "</think>",
        4: "x",
        5: ".",
        6: "42",
        7: "<eos>",
        12: ">",
    }

    def decode(self, ids, **_):
        return "".join(self.pieces.get(int(i), "?") for i in ids)


class ScriptedLM(torch.nn.Module):
    """Small deterministic forward fixture; synthetic quality is not a benchmark."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(eos_token_id=7)
        self.forward_lengths = []

    def forward(self, input_ids, past_key_values=None, **_):
        incoming = input_ids[0].tolist()
        history = list(past_key_values or []) + incoming
        self.forward_lengths.append(len(incoming))
        if 3 in history:
            next_id = 7 if 6 in history[history.index(3) :] else 6
        else:
            count = max(0, len(history) - 1)
            next_id = 3 if count >= 6 else [4, 5][count % 2]
        logits = torch.full((1, len(incoming), 16), -7.0)
        logits[0, -1, next_id] = 7.0
        return SimpleNamespace(logits=logits, past_key_values=history)


FORMAT = ReasoningFormat((2,), (3,))


def run(config=None, **kwargs):
    model = ScriptedLM()
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    return model, adapter.generate(
        [1], config=config, reasoning_format=FORMAT, max_new_tokens=30, **kwargs
    )


def test_none_matches_disabled_and_streams_identical_tokens():
    _, raw = run()
    events = []
    model, enabled = run(ReasoningConfig(policy="none"), on_token=lambda i, text: events.append(i))
    assert enabled.state.generated_token_ids == raw.state.generated_token_ids == events
    assert enabled.state.signals == []
    assert enabled.state.interventions == []
    assert enabled.metrics["model_evaluated_tokens"] == sum(model.forward_lengths)


def test_fixed_budget_counts_markers_and_continues_final_answer():
    _, result = run(ReasoningConfig(policy="fixed", budget=2))
    assert result.state.reasoning_tokens == 2
    assert result.state.marker_tokens == 1
    assert result.state.answer_tokens == 2
    assert result.final_answer == "42<eos>"
    assert result.state.interventions[0]["kind"] == "stop"
    assert result.metrics["latency_s"] > 0
    assert result.metrics["reasoning_latency_s"] <= result.metrics["latency_s"]


def test_fixed_zero_and_multitoken_end_are_cache_consistent():
    model = ScriptedLM()
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    fmt = ReasoningFormat((2,), (3, 12))
    result = adapter.generate(
        [1],
        config=ReasoningConfig(policy="fixed", budget=0),
        reasoning_format=fmt,
        max_new_tokens=12,
    )
    assert result.state.generated_token_ids[:2] == [3, 12]
    assert result.state.reasoning_tokens == 0
    assert result.state.marker_tokens == 2
    assert result.final_answer == "42<eos>"
    assert sum(model.forward_lengths) == result.metrics["model_evaluated_tokens"]


def test_none_matches_native_greedy_tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(17)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            eos_token_id=31,
            pad_token_id=0,
        )
    ).eval()
    prompt = torch.tensor([[1, 5, 9]])
    with torch.inference_mode():
        native = model.generate(prompt, max_new_tokens=8, do_sample=False, pad_token_id=0)
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    result = adapter.generate(
        prompt[0].tolist(),
        reasoning_format=FORMAT,
        config=ReasoningConfig(policy="none"),
        max_new_tokens=8,
    )
    assert result.state.generated_token_ids == native[0, prompt.shape[1] :].tolist()
