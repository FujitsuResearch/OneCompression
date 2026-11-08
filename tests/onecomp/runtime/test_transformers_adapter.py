"""CPU integration tests: real forwards and cache-consistent forced markers."""

from types import SimpleNamespace

import pytest
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
        8: "{",
        9: "}",
        10: "probe",
        12: ">",
    }

    def decode(self, ids, **_):
        return "".join(self.pieces.get(int(i), "?") for i in ids)

    def encode(self, text, **_):
        return [10] if "Final Answer" in text else [4]


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
        if 10 in history:
            # Answer probe: "{42}" then EOS after the cue.
            after_cue = history[history.index(10) + 1 :]
            next_id = [8, 6, 9, 7][min(len(after_cue), 3)]
        elif 3 in history:
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


class MarkerPrefixLM(ScriptedLM):
    """Keeps emitting the first token of a two-token end marker."""

    def forward(self, input_ids, past_key_values=None, **_):
        output = super().forward(input_ids, past_key_values)
        output.logits.fill_(-7.0)
        output.logits[0, -1, 3] = 7.0
        return output


def test_budget_completes_a_partial_multitoken_end_marker():
    model = MarkerPrefixLM()
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    result = adapter.generate(
        [1],
        config=ReasoningConfig(policy="fixed", budget=1),
        reasoning_format=ReasoningFormat((2,), (3, 12)),
        max_new_tokens=8,
    )
    assert result.state.generated_token_ids[:2] == [3, 12]
    assert result.state.interventions[0]["kind"] == "stop"
    assert result.state.phase == "answer"
    assert (result.state.reasoning_tokens, result.state.marker_tokens) == (0, 2)
    assert sum(model.forward_lengths) == result.metrics["model_evaluated_tokens"]


class ByteTokenizer:
    """One token per UTF-8 byte, so multi-byte characters span several tokens."""

    eos_token_id = 0

    def decode(self, ids, **_):
        return bytes(i for i in ids if i).decode("utf-8", errors="replace")


class ReplayLM(torch.nn.Module):
    """Emits a fixed token script, one token per forward."""

    def __init__(self, script):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(eos_token_id=0)
        self.script = script
        self.steps = 0

    def forward(self, input_ids, past_key_values=None, **_):
        logits = torch.full((1, input_ids.shape[1], 256), -7.0)
        logits[0, -1, self.script[self.steps]] = 7.0
        self.steps += 1
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def test_streamed_text_waits_for_complete_multibyte_characters():
    text = "é\u9b31🦜."  # 2-, 3- and 4-byte UTF-8 characters
    script = list(text.encode()) + [0]
    events = []
    result = TransformersReasoningAdapter(ReplayLM(script), ByteTokenizer()).generate(
        [1],
        reasoning_format=ReasoningFormat((254,), (255,)),
        max_new_tokens=len(script),
        on_token=lambda token_id, token_text: events.append((token_id, token_text)),
    )
    assert [token_id for token_id, _ in events] == script
    assert all("�" not in token_text for _, token_text in events)
    assert "".join(token_text for _, token_text in events) == result.text == text
    # Text is held back until a character's last byte arrives.
    assert [token_text for _, token_text in events][:3] == ["", "é", ""]


@pytest.mark.parametrize("cut", range(1, len("é\u9b31🦜".encode())))
def test_streamed_text_flushes_a_character_cut_by_max_new_tokens(cut):
    script = list("é\u9b31🦜".encode())
    events = []
    result = TransformersReasoningAdapter(ReplayLM(script), ByteTokenizer()).generate(
        [1],
        reasoning_format=ReasoningFormat((254,), (255,)),
        max_new_tokens=cut,
        on_token=lambda token_id, token_text: events.append(token_text),
    )
    assert result.metrics["finish_reason"] == "length"
    assert len(events) == cut
    assert "".join(events) == result.text == bytes(script[:cut]).decode(errors="replace")


def test_streamed_text_flushes_a_character_cut_by_eos():
    script = list("a\u9b31".encode())[:2] + [0]
    events = []
    result = TransformersReasoningAdapter(ReplayLM(script), ByteTokenizer()).generate(
        [1],
        reasoning_format=ReasoningFormat((254,), (255,)),
        max_new_tokens=8,
        on_token=lambda token_id, token_text: events.append(token_text),
    )
    assert result.metrics["finish_reason"] == "eos"
    assert "".join(events) == result.text == "a�"


def test_thinkbrake_adaptive_stop():
    _, result = run(
        ReasoningConfig(policy="thinkbrake", boundary="tokens", interval=1, threshold=20)
    )
    assert result.state.reasoning_tokens == 1
    assert result.state.signals[0]["logit_margin"] == 14
    assert result.state.interventions[0]["reason"] == "end_logit_gap"
    assert result.final_answer == "42<eos>"


def test_thinkbrake_rejected_margins_keep_vanilla_tokens():
    _, raw = run()
    model, result = run(ReasoningConfig(policy="thinkbrake", threshold=-20))
    assert result.state.generated_token_ids == raw.state.generated_token_ids
    # The default sentence boundary fires after every "."; at the last one the
    # model itself is about to emit </think>, so the margin turns negative.
    assert [signal["logit_margin"] for signal in result.state.signals] == [14, 14, -14]
    assert result.state.interventions == []
    assert sum(model.forward_lengths) == result.metrics["model_evaluated_tokens"]


def test_thinkbrake_rejects_multitoken_end():
    adapter = TransformersReasoningAdapter(ScriptedLM(), TinyTokenizer())
    with pytest.raises(ValueError, match="singleton"):
        adapter.generate(
            [1],
            reasoning_format=ReasoningFormat((2,), (3, 12)),
            config=ReasoningConfig(policy="thinkbrake"),
        )


def test_probe_replay_counts_all_tokens_and_does_not_change_rejected_trajectory():
    _, raw = run()
    model, result = run(
        ReasoningConfig(
            policy="deer",
            boundary="tokens",
            interval=2,
            confidence_threshold=1.0,
            probe_max_tokens=6,
        )
    )
    assert result.state.generated_token_ids == raw.state.generated_token_ids
    assert result.metrics["probe_tokens"] == 9
    assert result.metrics["probe_prefill_tokens"] == 4 + 6 + 8
    assert result.metrics["probe_model_evaluated_tokens"] == 24
    assert result.metrics["model_evaluated_tokens"] == sum(model.forward_lengths)
    assert result.metrics["probe_latency_s"] > 0
    assert len(result.metrics["probe_history"]) == 3


def test_incomplete_probe_cannot_stop_and_confidence_excludes_braces():
    _, result = run(
        ReasoningConfig(
            policy="deer",
            boundary="tokens",
            interval=2,
            confidence_threshold=0.5,
            probe_max_tokens=1,
        )
    )
    assert not result.state.interventions
    assert all(s["confidence"] is None and not s["probe_complete"] for s in result.state.signals)
    model, result = run(
        ReasoningConfig(
            policy="deer",
            boundary="tokens",
            interval=2,
            confidence_threshold=0.95,
            probe_max_tokens=6,
        )
    )
    assert result.state.reasoning_tokens == 2
    assert result.state.probe_answers == ["42"]
    assert result.state.interventions[0]["reason"] == "probe_confidence"
    assert result.final_answer == "42<eos>"
    assert result.metrics["model_evaluated_tokens"] == sum(model.forward_lengths)


def test_stability_stops_after_identical_probe_answers():
    _, result = run(
        ReasoningConfig(
            policy="stability",
            boundary="tokens",
            interval=2,
            stability_window=2,
            probe_max_tokens=6,
        )
    )
    assert result.state.reasoning_tokens == 4
    assert result.state.probe_answers == ["42", "42"]
    assert result.state.interventions[0]["reason"] == "answer_stability"


def test_probe_policy_accepts_multitoken_end_without_recording_margin():
    model = ScriptedLM()
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    result = adapter.generate(
        [1],
        reasoning_format=ReasoningFormat((2,), (3, 12)),
        config=ReasoningConfig(policy="deer"),
        max_new_tokens=30,
    )
    assert result.state.signals
    assert all("logit_margin" not in signal for signal in result.state.signals)


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
