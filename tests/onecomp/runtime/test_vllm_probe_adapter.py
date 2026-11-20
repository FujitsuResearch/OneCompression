"""Offline probe tests using vLLM's sampling and output types without a GPU."""

import math
from importlib.metadata import EntryPoint
from types import SimpleNamespace
from weakref import ref

import pytest

pytest.importorskip("vllm")

from vllm import SamplingParams
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput

from onecomp.reasoning import ReasoningConfig, ReasoningFormat
from onecomp.runtime.common import ASAG_JUMP, ASAG_PROBE_CUE, DEER_PROBE_CUE
from onecomp.runtime.transformers import TransformersReasoningAdapter
from onecomp.runtime.vllm import VLLMProbeReasoningAdapter, VLLMReasoningAdapter, probe_adapter
from onecomp.runtime.vllm.logits_processor import ReasoningLogitsProcessor

from .test_transformers_adapter import _tiny_qwen3

FORMAT = ReasoningFormat((2,), (3,))


class ProbeTokenizer:
    eos_token_id = 7
    pieces = {
        3: "</think>",
        4: "x",
        5: ".",
        6: "4",
        7: "<eos>",
        8: "{",
        9: "}",
        12: ">",
        14: "final",
        15: "Wait",
        16: "2",
        17: ASAG_JUMP[:-3],
        18: ASAG_JUMP[-3:],
    }

    def encode(self, text, **_):
        return {DEER_PROBE_CUE: [10], ASAG_PROBE_CUE: [11], ASAG_JUMP: [17, 18]}[text]

    def decode(self, ids, **_):
        return "".join(self.pieces[i] for i in ids)


class ScriptedVLLM:
    """Make every next token a function of the submitted prefix."""

    def __init__(self, script=None, *, context_length=128, max_logprobs=20):
        self.model_config = SimpleNamespace(
            logprobs_mode="raw_logprobs",
            logits_processors=None,
            max_model_len=context_length,
            max_logprobs=max_logprobs,
            get_vocab_size=lambda: 20,
            try_get_generation_config=lambda: {"eos_token_id": [7, 19]},
        )
        self.llm_engine = SimpleNamespace(vllm_config=SimpleNamespace(speculative_config=None))
        self.script = script or self.next_token
        self.calls = []

    @staticmethod
    def next_token(prefix):
        for cue in (10, 11):
            if cue in prefix:
                after_cue = len(prefix) - prefix.index(cue) - 1
                return ([8, 6, 16, 9, 7][after_cue], [0.99, 0.81, 0.49, 0.99, 0.99][after_cue])
        if 3 in prefix:
            return (7 if prefix[-1] == 14 else 14), 0.99
        return (4, 5)[(len(prefix) - 1) % 2], 0.99

    def get_tokenizer(self):
        return ProbeTokenizer()

    def generate(self, prompt, params, *, use_tqdm):
        assert isinstance(params, SamplingParams)
        assert params.temperature == 0 and params.n == 1
        assert params.detokenize == bool(params.stop)
        if params.stop:
            assert params.include_stop_str_in_output
        assert not params.extra_args and not use_tqdm
        prefix = prompt["prompt_token_ids"]
        assert len(prefix) + params.max_tokens <= self.model_config.max_model_len
        self.calls.append((prefix, params))
        ids, rows = [], [] if params.logprobs is not None else None
        finish, stop = "length", None
        for _ in range(params.max_tokens):
            token, probabilities = self.script(prefix + ids)
            if not isinstance(probabilities, dict):
                probabilities = {token: probabilities}
            if rows is not None:
                scores = probabilities
                if params.logprobs:
                    remaining = (1 - sum(scores.values())) / (20 - len(scores))
                    scores = {i: scores.get(i, remaining) for i in range(20)}
                rows.append({i: Logprob(math.log(p)) for i, p in scores.items()})
            ids.append(token)
            if token in (7, 19):
                finish = "stop"
                break
            if params.stop and "}" in self.get_tokenizer().decode([token]):
                assert "}" in params.stop
                finish, stop = "stop", "}"
                break
        completion = CompletionOutput(
            0, "", ids, None, rows, finish_reason=finish, stop_reason=stop
        )
        return [
            RequestOutput(
                "request",
                None,
                prefix,
                None,
                [completion],
                True,
                num_cached_tokens=len(prefix) // 2,
            )
        ]


@pytest.mark.parametrize(
    ("policy", "cue", "expected_confidence", "reason"),
    [
        ("deer", 10, 0.63, "probe_confidence"),
        ("asag_reference", 11, 0.65, "confidence_ablation"),
    ],
)
def test_confidence_probe_stops_before_transition(policy, cue, expected_confidence, reason):
    def script(prefix):
        if prefix == [1, 4]:
            return 15, 0.99
        return ScriptedVLLM.next_token(prefix)

    llm = ScriptedVLLM(script)
    adapter = VLLMProbeReasoningAdapter(llm, lookahead_tokens=2)
    events = []
    result = adapter.generate(
        [1],
        config=ReasoningConfig(
            policy=policy,
            asag_mode="confidence",
            boundary="transition",
            confidence_threshold=0.6,
        ),
        reasoning_format=ReasoningFormat((2,), (3, 12)),
        on_token=lambda token, text: events.append((token, text)),
        max_new_tokens=12,
    )
    assert result.state.probe_confidences == pytest.approx([expected_confidence])
    assert result.state.probe_answers == ["42"]
    assert result.state.interventions[0]["reason"] == reason
    assert result.state.generated_token_ids == [4, 3, 12, 14, 7]
    assert [token for token, _ in events] == result.state.generated_token_ids
    assert "".join(text for _, text in events) == result.text
    assert llm.calls[1][0] == [1, 4, cue]
    assert llm.calls[2][0] == [1, 4, 3, 12]
    assert result.metrics["vllm_discarded_main_tokens"] == 1
    assert result.metrics["probe_tokens"] == 4
    assert result.final_answer == "final<eos>"


@pytest.mark.parametrize("boundary", ["sentence", "tokens"])
def test_stop_at_known_boundary_does_not_generate_an_unused_chunk(boundary):
    llm = ScriptedVLLM()
    result = VLLMProbeReasoningAdapter(llm, lookahead_tokens=2).generate(
        [1],
        config=ReasoningConfig(
            policy="deer", boundary=boundary, interval=2, confidence_threshold=0.0
        ),
        reasoning_format=FORMAT,
        max_new_tokens=12,
    )
    assert [prefix for prefix, _ in llm.calls] == [[1], [1, 4, 5, 10], [1, 4, 5, 3]]
    assert result.metrics["vllm_discarded_main_tokens"] == 0
    assert result.state.generated_token_ids == [4, 5, 3, 14, 7]


def test_stability_requires_consecutive_complete_probes():
    def script(prefix):
        if 10 in prefix:
            boundary = prefix.index(10)
            # The incomplete nested box breaks the consecutive run of answers.
            answer = [8, 8, 6, 9, 7] if boundary == 5 else [8, 8, 6, 9, 9, 7]
            return answer[len(prefix) - boundary - 1], 0.99
        return ScriptedVLLM.next_token(prefix)

    llm = ScriptedVLLM(script)
    result = VLLMProbeReasoningAdapter(llm).generate(
        [1],
        config=ReasoningConfig(policy="stability", stability_window=2),
        reasoning_format=FORMAT,
        max_new_tokens=16,
    )
    assert result.state.probe_answers == ["{4}", "", "{4}", "{4}"]
    assert result.state.boundaries == [2, 4, 6, 8]
    assert result.state.interventions[0]["reason"] == "answer_stability"
    assert result.state.generated_token_ids == [4, 5] * 4 + [3, 14, 7]
    assert [p["sampled_tokens"] for p in result.metrics["probe_history"]] == [5, 5, 5, 5]
    probe_calls = [(prefix, params) for prefix, params in llm.calls if params.logprobs == 0]
    assert len(probe_calls) == 8
    assert probe_calls[1][0] == probe_calls[0][0] + [8, 8, 6, 9]
    assert probe_calls[1][1].max_tokens == probe_calls[0][1].max_tokens - 4


def test_probe_limit_on_inner_closing_brace_does_not_resume():
    def script(prefix):
        if 10 in prefix:
            return [8, 8, 6, 9, 9][len(prefix) - prefix.index(10) - 1], 0.99
        return ScriptedVLLM.next_token(prefix)

    llm = ScriptedVLLM(script)
    result = VLLMProbeReasoningAdapter(llm).generate(
        [1],
        config=ReasoningConfig(policy="deer", probe_max_tokens=4, confidence_threshold=0.0),
        reasoning_format=FORMAT,
        max_new_tokens=3,
    )
    assert result.state.probe_answers == [""]
    assert result.state.interventions == []
    assert result.metrics["vllm_probe_work"]["requests"] == 1
    assert result.state.probe_tokens == 4


def test_asag_attention_only_reuses_the_eager_prefix_without_answer_probes(monkeypatch):
    llm = ScriptedVLLM()
    attention = TransformersReasoningAdapter(_tiny_qwen3(), llm.get_tokenizer())
    calls = []
    forward = attention._forward

    def record_forward(ids, cache=None, *, attention=False):
        calls.append((list(ids), cache.get_seq_length() if cache is not None else 0, attention))
        return forward(ids, cache, attention=attention)

    monkeypatch.setattr(attention, "_forward", record_forward)
    result = VLLMProbeReasoningAdapter(llm, attention_adapter=attention).generate(
        [1],
        config=ReasoningConfig(policy="asag_reference", asag_mode="attention"),
        reasoning_format=FORMAT,
        max_new_tokens=6,
    )
    assert len(llm.calls) == 1
    assert result.state.boundaries == [2, 4]
    assert result.state.probe_tokens == 0
    assert calls == [
        ([1, 4, 5], 0, False),
        ([5, 11], 2, True),
        ([4, 5], 3, False),
        ([5, 11], 4, True),
    ]
    assert result.state.probe_model_evaluated_tokens == 3 + 2 + 2 + 2
    assert [signal["attention_key_tokens"] for signal in result.state.signals] == [4, 6]
    assert all(signal["attention_entropy"] > 0 for signal in result.state.signals)


@pytest.mark.parametrize("force_stop", [False, True])
def test_asag_releases_attention_cache_at_the_end_of_reasoning(monkeypatch, force_stop):
    def script(prefix):
        if not force_stop and prefix == [1, 4, 5, 4, 5]:
            return 3, 0.99
        if prefix[-1] == 3:
            return 12, 0.99
        return ScriptedVLLM.next_token(prefix)

    llm = ScriptedVLLM(script)
    attention = TransformersReasoningAdapter(_tiny_qwen3(), llm.get_tokenizer())
    cache_refs = []
    forward = attention._forward

    def track_cache(ids, cache=None, *, attention=False):
        if attention:
            cache_refs.append(ref(cache))
        return forward(ids, cache, attention=attention)

    monkeypatch.setattr(attention, "_forward", track_cache)

    def on_token(token, _):
        if token == 3:
            # The first half of the marker has not switched the request to answer mode.
            assert cache_refs and cache_refs[-1]() is not None
        elif token == 12:
            assert cache_refs and all(cache_ref() is None for cache_ref in cache_refs)

    result = VLLMProbeReasoningAdapter(llm, attention_adapter=attention).generate(
        [1],
        config=ReasoningConfig(
            policy="asag_reference",
            asag_mode="attention",
            entropy_threshold=1.0 if force_stop else -1.0,
        ),
        reasoning_format=ReasoningFormat((2,), (3, 12)),
        on_token=on_token,
        max_new_tokens=12,
    )
    assert bool(result.state.interventions) == force_stop
    assert result.state.generated_token_ids == [4, 5, 4, 5, 3, 12, 14, 7]


class ScriptedAttention:
    def __init__(self, entropies):
        self.entropies = iter(entropies)
        self.spans = []

    def _encode(self, text):
        return ProbeTokenizer().encode(text)

    def start_attention_probe(self, config):
        assert config.policy == "asag_reference"
        return self.probe

    def probe(self, prefix, *, previous_span, current_span):
        self.spans.append((previous_span, current_span))
        return SimpleNamespace(
            signals={
                "attention_entropy": next(self.entropies),
                "previous_attention": 0.2,
                "current_attention": 0.1,
            },
            model_evaluated_tokens=len(prefix) + 1,
        )


def test_asag_both_can_probe_before_the_first_reasoning_token():
    def script(prefix):
        if prefix in ([1], [1, 15, 5]):
            return 15, 0.99
        if prefix == [1, 15]:
            return 5, 0.99
        if prefix[:4] == [1, 15, 5, 11] and prefix[-1] in (8, 6):
            return (6 if prefix[-1] == 8 else 16), 0.99
        return ScriptedVLLM.next_token(prefix)

    llm = ScriptedVLLM(script, max_logprobs=10)
    attention = ScriptedAttention([10.0, 5.0])
    result = VLLMProbeReasoningAdapter(llm, attention_adapter=attention).generate(
        [1],
        config=ReasoningConfig(policy="asag_reference", asag_mode="both", boundary="transition"),
        reasoning_format=FORMAT,
        max_new_tokens=10,
    )
    assert attention.spans == [(None, None), (None, (1, 3))]
    assert result.state.boundaries == [0, 2]
    assert result.state.probe_confidences == pytest.approx([0.65, 0.99])
    assert result.state.interventions[0]["reason"] == "asag_confidence_and_convergence"
    assert result.state.generated_token_ids == [15, 5, 3, 14, 7]


def test_asag_jump_sentence_suffix_does_not_trigger_jump_limit():
    llm = ScriptedVLLM()
    attention = ScriptedAttention([10.0, 10.0, 10.0])
    result = VLLMProbeReasoningAdapter(llm, attention_adapter=attention).generate(
        [1],
        config=ReasoningConfig(
            policy="asag_reference", confidence_threshold=1.0, enable_injection=False
        ),
        reasoning_format=FORMAT,
        max_new_tokens=16,
    )
    assert result.state.boundaries == [2, 4, 8]
    assert [event["reason"] for event in result.state.interventions] == [
        "asag_backward_attention",
        "asag_jump_limit",
    ]
    assert result.state.generated_token_ids == [4, 5, 4, 5, 17, 18, 4, 5, 3, 14, 7]
    assert attention.spans == [(None, (1, 3)), ((1, 3), (3, 5)), ((3, 5), (5, 9))]
    assert any(prefix == [1, 4, 5, 4, 5, 17, 18] for prefix, _ in llm.calls)


@pytest.mark.parametrize("lookahead_tokens", [4, 32])
def test_asag_mix_replays_the_first_answer_content_distribution(lookahead_tokens):
    def script(prefix):
        if prefix == [1, 4, 5, 4, 5, 11, 8]:
            return 16, {16: 0.90, 4: 0.03}
        if prefix == [1, 4, 5, 4, 5]:
            return 4, {4: 0.45, 16: 0.43}
        if prefix == [1, 4, 5, 4, 5, 16]:
            return 3, 0.99
        return ScriptedVLLM.next_token(prefix)

    llm = ScriptedVLLM(script, max_logprobs=-1)
    adapter = VLLMProbeReasoningAdapter(
        llm, attention_adapter=ScriptedAttention([10.0, 5.0]), lookahead_tokens=lookahead_tokens
    )
    result = adapter.generate(
        [1],
        config=ReasoningConfig(policy="asag_reference", confidence_threshold=1.0),
        reasoning_format=FORMAT,
        max_new_tokens=12,
    )
    assert result.state.interventions[0]["reason"] == "asag_probability_mixture"
    assert result.state.generated_token_ids == [4, 5, 4, 5, 16, 3, 14, 7]
    assert [prefix for prefix, params in llm.calls if params.logprobs == 20] == [
        [1, 4, 5, 4, 5, 11, 8],
        [1, 4, 5, 4, 5],
    ]
    assert [prefix for prefix, params in llm.calls if params.logprobs is None] == [
        [1],
        [1, 4, 5, 4, 5, 16],
    ]


def test_context_limit_shortens_probe_without_stopping_on_incomplete_answer():
    llm = ScriptedVLLM(context_length=6)
    result = VLLMProbeReasoningAdapter(llm).generate(
        [1],
        config=ReasoningConfig(policy="deer", confidence_threshold=0.1),
        reasoning_format=FORMAT,
        max_new_tokens=16,
    )
    assert result.state.boundaries == [2, 4]
    assert result.state.probe_answers == ["", ""]
    assert len(llm.calls) == 2
    assert llm.calls[1][1].max_tokens == 2
    assert result.state.interventions == []
    assert result.state.generated_token_ids == [4, 5, 4, 5, 4]
    assert result.metrics["finish_reason"] == "length"


def test_asag_configuration_requires_attention_and_full_logprobs_before_generation():
    llm = ScriptedVLLM(max_logprobs=10)
    config = ReasoningConfig(policy="asag_reference")
    with pytest.raises(ValueError, match="attention_adapter"):
        VLLMProbeReasoningAdapter(llm).generate([1], config=config, reasoning_format=FORMAT)
    with pytest.raises(ValueError, match="max_logprobs=-1"):
        VLLMProbeReasoningAdapter(llm, attention_adapter=ScriptedAttention([])).generate(
            [1], config=config, reasoning_format=FORMAT
        )
    assert not llm.calls


class ForceTokenProcessor(ReasoningLogitsProcessor):
    def apply(self, logits):
        logits.fill_(-float("inf"))
        logits[:, 4] = 0
        return logits


@pytest.mark.parametrize("processor", [ForceTokenProcessor, f"{__name__}:ForceTokenProcessor"])
def test_probe_adapter_rejects_custom_logits_processors(processor):
    llm = ScriptedVLLM()
    llm.model_config.logits_processors = [VLLMReasoningAdapter.processor_path, processor]
    with pytest.raises(ValueError, match="custom logits processor.*ForceTokenProcessor"):
        VLLMProbeReasoningAdapter(llm)
    assert not llm.calls


def test_probe_adapter_rejects_an_installed_logits_processor(monkeypatch):
    def registered_plugins(*, group):
        assert group == "vllm.logits_processors"
        return [EntryPoint("force", f"{__name__}:ForceTokenProcessor", group)]

    monkeypatch.setattr(probe_adapter, "entry_points", registered_plugins)
    with pytest.raises(ValueError, match="custom logits processor.*ForceTokenProcessor"):
        VLLMProbeReasoningAdapter(ScriptedVLLM())


@pytest.mark.parametrize(
    "processor", [ReasoningLogitsProcessor, VLLMReasoningAdapter.processor_path]
)
def test_probe_adapter_allows_the_onecomp_hook_without_request_metadata(processor):
    llm = ScriptedVLLM()
    llm.model_config.logits_processors = [processor]
    result = VLLMProbeReasoningAdapter(llm).generate(
        [1],
        config=ReasoningConfig(policy="deer", confidence_threshold=0.0),
        reasoning_format=FORMAT,
        max_new_tokens=12,
    )
    assert result.state.generated_token_ids == [4, 5, 3, 14, 7]
    assert all(not params.extra_args for _, params in llm.calls)
