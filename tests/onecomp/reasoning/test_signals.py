"""CPU checks of paper equations, masking and unavailable observations."""

import math

import pytest
import torch

from onecomp.reasoning.signals import deer_confidence, logit_observation


def test_deer_confidence_is_geometric_mean_of_content_probabilities():
    assert deer_confidence([0.25, 1.0]) == pytest.approx(0.5)
    assert math.isclose(deer_confidence([1e-200] * 50), 1e-200, rel_tol=1e-12)
    assert deer_confidence([0, 1]) == 0
    assert deer_confidence([]) is None
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
