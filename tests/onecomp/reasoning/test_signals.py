"""CPU checks of paper equations, masking and unavailable observations."""

import math

import pytest
import torch

from onecomp.reasoning.signals import (
    asag_confidence,
    attention_statistics,
    deer_confidence,
    logit_observation,
    relative_entropy_change,
)


def test_paper_confidence_aggregations_are_distinct():
    assert deer_confidence([0.25, 1.0]) == pytest.approx(0.5)
    assert asag_confidence([0.25, 1.0]) == pytest.approx(0.625)
    assert math.isclose(deer_confidence([1e-200] * 50), 1e-200, rel_tol=1e-12)
    assert deer_confidence([0, 1]) == 0
    assert deer_confidence([]) is None
    assert asag_confidence([]) is None
    for invalid in ([math.nan], [math.inf], [-0.1], [1.01]):
        with pytest.raises(ValueError):
            deer_confidence(invalid)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_logit_margin_excludes_end_even_when_end_is_top(as_tensor):
    logits = [1.0, 3.0, 2.0]
    values = torch.tensor(logits) if as_tensor else logits
    observation = logit_observation(values, 1, temperature=0.5)
    assert observation["logit_margin"] == -2.0
    assert observation["end_probability"] == pytest.approx(1 / (1 + math.exp(-2) + math.exp(-4)))
    assert list(values) == logits


def test_masked_logits_remain_json_serializable():
    observation = logit_observation([0.0, -math.inf], 1)
    assert observation == {"end_logit": None, "logit_margin": None, "end_probability": 0.0}
    with pytest.raises(ValueError):
        logit_observation([-math.inf, -math.inf], 0)
    with pytest.raises(ValueError):
        logit_observation([0, 1], 1, temperature=0)


def test_attention_entropy_sum_last_four_layers_and_window():
    # Two uniform query rows, two heads, four used layers: entropy 16, not 1.
    uniform = torch.full((1, 2, 3, 4), 0.25)
    ignored = torch.full_like(uniform, math.nan)
    observation = attention_statistics([ignored] + [uniform] * 4, query_start=1)
    assert observation["attention_entropy"] == pytest.approx(16)
    assert observation["attention_layers"] == 4
    assert observation["attention_query_rows"] == 2
    assert observation["attention_key_tokens"] == 4
    assert observation["backward_ratio"] is None


def test_causal_zero_mask_and_length_normalized_backward_attention():
    attention = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.6, 0.2, 0.1, 0.1]]]])
    observation = attention_statistics(
        [attention], query_start=1, previous_span=(0, 1), current_span=(1, 4)
    )
    expected_entropy = -sum(p * math.log(p) for p in [0.6, 0.2, 0.1, 0.1]) / math.log(4)
    assert observation["attention_entropy"] == pytest.approx(expected_entropy)
    assert observation["previous_attention"] == pytest.approx(0.6)
    assert observation["current_attention"] == pytest.approx(0.4 / 3)
    assert observation["backward_ratio"] == pytest.approx(4.5)
    assert attention_statistics([attention], query_end=1)["attention_entropy"] == 0


def test_attention_rejects_batch_mixing_and_invalid_spans():
    with pytest.raises(ValueError, match="single unpadded request"):
        attention_statistics([torch.ones(2, 1, 1, 1)])
    with pytest.raises(ValueError, match="Thought spans"):
        attention_statistics([torch.ones(1, 1, 1, 1)], previous_span=(0, 0))
    assert attention_statistics([torch.ones(1, 1, 1, 1)])["attention_entropy"] == 0


def test_relative_change_requires_nonzero_initial_entropy():
    assert relative_entropy_change(8, 10) == -0.2
    assert relative_entropy_change(8, None) is None
    assert relative_entropy_change(0, 0) is None
    with pytest.raises(ValueError):
        relative_entropy_change(-1, 1)
