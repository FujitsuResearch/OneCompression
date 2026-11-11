"""CPU contract tests against the installed vLLM 0.21 batching API."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from onecomp.reasoning import ReasoningConfig, ReasoningFormat
from onecomp.runtime.vllm import VLLMReasoningAdapter

# Every test here, not only the fixture, depends on the 0.21 batching API.
try:
    VLLMReasoningAdapter.check_version()
except RuntimeError as error:
    pytest.skip(str(error), allow_module_level=True)

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality

from onecomp.runtime.vllm.logits_processor import ReasoningLogitsProcessor

FORMAT = ReasoningFormat((2,), (3,))


@pytest.fixture
def processor():
    return ReasoningLogitsProcessor(SimpleNamespace(speculative_config=None), "cpu", False)


def params(budget, fmt=FORMAT, policy="fixed", **kwargs):
    config = ReasoningConfig(policy=policy, budget=budget, **kwargs)
    return SamplingParams(temperature=0, extra_args=VLLMReasoningAdapter.extra_args(config, fmt))


def update(processor, *, size=2, added=(), removed=(), moved=()):
    processor.update_state(BatchUpdate(size, removed, added, moved))


def test_disabled_is_exact_identity_and_no_state(processor):
    update(
        processor, added=[(0, SamplingParams(), [1], []), (1, params(1, policy="none"), [1], [])]
    )
    logits = torch.randn(2, 10)
    original = logits.clone()
    assert processor.apply(logits) is logits
    assert torch.equal(logits, original)
    assert processor.requests == {}


def test_different_stop_times_and_live_output_history(processor):
    first, second = [], []
    update(processor, added=[(0, params(1), [1], first), (1, params(2), [1], second)])
    first.append(4)
    second.append(4)
    processor.update_state(None)
    logits = processor.apply(torch.zeros(2, 10))
    assert logits[0].argmax().item() == 3
    assert torch.isneginf(logits[0]).sum() == 9
    assert logits[1].eq(0).all()
    first.append(3)
    second.append(4)
    processor.update_state(None)
    logits = processor.apply(torch.zeros(2, 10))
    assert logits[0].eq(0).all()
    assert logits[1].argmax().item() == 3
    assert processor.requests[0].state.phase == "answer"
    assert processor.requests[1].state.phase == "reasoning"


def test_batch_swap_move_remove_and_same_slot_reuse(processor):
    first, second = [], []
    update(processor, added=[(0, params(1), [1], first), (1, params(5), [1], second)])
    update(processor, moved=[(0, 1, MoveDirectionality.SWAP)])
    assert processor.requests[1].output_ids is first
    assert processor.requests[0].output_ids is second
    update(processor, size=1, removed=[0], moved=[(1, 0, MoveDirectionality.UNIDIRECTIONAL)])
    assert list(processor.requests) == [0]
    assert processor.requests[0].output_ids is first
    replacement = []
    update(processor, size=1, removed=[0], added=[(0, params(9), [1], replacement)])
    assert processor.requests[0].output_ids is replacement
    assert processor.requests[0].controller.config.budget == 9


def test_multitoken_marker_advances_only_after_sampled_output(processor):
    output = []
    update(processor, size=1, added=[(0, params(0, ReasoningFormat((2,), (3, 4))), [1], output)])
    assert processor.apply(torch.zeros(1, 10)).argmax().item() == 3
    assert processor.apply(torch.zeros(1, 10)).argmax().item() == 3
    output.append(3)
    processor.update_state(None)
    assert processor.apply(torch.zeros(1, 10)).argmax().item() == 4
    output.append(4)
    processor.update_state(None)
    assert processor.apply(torch.zeros(1, 10)).eq(0).all()
    assert processor.requests[0].state.marker_tokens == 2


def test_async_placeholders_are_consumed_once_replaced(processor):
    # vLLM's async scheduler shows -1 to update_state and fills it in before apply.
    output = []
    update(processor, size=1, added=[(0, params(2), [1], output)])
    output.append(-1)
    processor.update_state(None)
    assert processor.requests[0].state.reasoning_tokens == 0
    output[-1] = 4
    assert processor.apply(torch.zeros(1, 10)).eq(0).all()
    assert processor.requests[0].state.reasoning_tokens == 1
    output.extend([4, -1])
    processor.update_state(None)
    output[-1] = 4
    assert processor.apply(torch.zeros(1, 10)).argmax().item() == 3
    assert processor.requests[0].seen == 3


def test_async_placeholder_while_marker_is_pending(processor):
    output = []
    update(processor, size=1, added=[(0, params(0), [1], output)])
    assert processor.apply(torch.zeros(1, 10)).argmax().item() == 3
    output.append(-1)
    processor.update_state(None)
    output[-1] = 3
    assert processor.apply(torch.zeros(1, 10)).eq(0).all()
    assert processor.requests[0].state.phase == "answer"


def test_speculative_mode_fails_explicitly():
    with pytest.raises(NotImplementedError, match="speculative"):
        ReasoningLogitsProcessor(SimpleNamespace(speculative_config=object()), "cpu", False)


def test_incompatible_stop_ids_and_multiple_samples_rejected():
    sampling = params(2)
    sampling._all_stop_token_ids.add(3)
    with pytest.raises(ValueError, match="stop tokens"):
        ReasoningLogitsProcessor.validate_params(sampling)
    sampling = params(2)
    sampling.n = 2
    with pytest.raises(ValueError, match="n=1"):
        ReasoningLogitsProcessor.validate_params(sampling)


def test_processor_registration_string_uses_installed_vllm_loader_syntax():
    from vllm.v1.sample.logits_processor import _load_logitsprocs_by_fqcns

    assert _load_logitsprocs_by_fqcns([VLLMReasoningAdapter.processor_path]) == [
        ReasoningLogitsProcessor
    ]
