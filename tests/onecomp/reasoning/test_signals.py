"""CPU checks of paper equations, masking and unavailable observations."""

import math

import pytest
import torch

from onecomp.reasoning.signals import logit_observation


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
