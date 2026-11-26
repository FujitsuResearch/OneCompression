"""Unit tests for the autonomous compression agent.

The quantize/evaluate/estimate/recommend execution functions are
replaced by a fast simulator with a monotone bit-width → perplexity
error model, so the hill-climbing search can be verified end to end on
CPU in milliseconds.

Copyright 2025-2026 Fujitsu Ltd.
"""

import math

import pytest

from onecomp.mcp.agent import AutoCompressConfig, AutoCompressionAgent, simulate_ppl_increase

BASELINE_PPL = 10.0


class Simulator:
    """Deterministic stand-in for quantisation and evaluation.

    Memory scales linearly with the bit-width (0.1 GB per bit) and the
    perplexity degradation follows :func:`simulate_ppl_increase`
    (monotone: fewer bits → strictly larger degradation).
    """

    def __init__(self, evaluate_fails_at=None):
        self.quantize_calls = []
        self.evaluate_fails_at = evaluate_fails_at

    def quantize(self, _model_id, save_dir, wbits):
        self.quantize_calls.append(wbits)
        return save_dir

    def evaluate(self, save_dir):
        bits = int(save_dir.rsplit("-w", 1)[1])
        if self.evaluate_fails_at is not None and bits == self.evaluate_fails_at:
            raise RuntimeError(f"simulated evaluation failure at {bits}-bit")
        return BASELINE_PPL * (1.0 + simulate_ppl_increase(bits) / 100.0)

    @staticmethod
    def estimate(_model_id, wbits):
        return wbits * 0.1

    @staticmethod
    def recommend(_model_id, target_vram_gb):
        """Largest candidate bit-width fitting the budget (quality rule)."""
        for bits in (8, 6, 5, 4, 3, 2):
            if bits * 0.1 <= target_vram_gb:
                return bits
        return 2


def _make_agent(config, simulator):
    return AutoCompressionAgent(
        "dummy/model",
        config,
        quantize_fn=simulator.quantize,
        evaluate_fn=simulator.evaluate,
        estimate_fn=simulator.estimate,
        recommend_fn=simulator.recommend,
    )


def test_agent_converges_to_smallest_feasible_bits():
    """Budget 0.5 GB (≤5 bits) and 12% PPL tolerance (≥4 bits) → 4-bit.

    With the simulator, 4-bit degrades PPL by 11.25% (within the 12%
    tolerance) while 3-bit degrades by 22.5% (violates it), so the
    hill climb must descend 5 → 4, fail at 3, and settle on 4-bit.
    """
    simulator = Simulator()
    config = AutoCompressConfig(
        target_vram_gb=0.5,
        max_ppl_increase_pct=12.0,
        baseline_ppl=BASELINE_PPL,
    )
    result = _make_agent(config, simulator).run("out")

    assert result.success is True
    assert result.best.wbits == 4
    assert result.best.feasible
    assert result.best.estimated_total_gb <= config.target_vram_gb
    assert result.best.ppl_increase_pct <= config.max_ppl_increase_pct
    # Search path: start at 5 (recommendation), descend to 4, probe 3,
    # then stop because 4 was already visited.
    assert simulator.quantize_calls == [5, 4, 3]
    assert len(result.trials) <= config.max_trials


def test_agent_best_is_minimum_memory_among_feasible():
    """A loose PPL tolerance lets the agent descend to the last candidate."""
    simulator = Simulator()
    config = AutoCompressConfig(
        target_vram_gb=1.0,
        max_ppl_increase_pct=100.0,
        baseline_ppl=BASELINE_PPL,
    )
    result = _make_agent(config, simulator).run("out")

    assert result.success is True
    assert result.best.wbits == 2
    feasible = [t for t in result.trials if t.feasible]
    assert result.best.estimated_total_gb == min(t.estimated_total_gb for t in feasible)


def test_agent_reports_failure_when_constraints_conflict():
    """0.5 GB budget forces ≤5 bits, but 0.5% tolerance needs 8 bits."""
    simulator = Simulator()
    config = AutoCompressConfig(
        target_vram_gb=0.5,
        max_ppl_increase_pct=0.5,
        baseline_ppl=BASELINE_PPL,
    )
    result = _make_agent(config, simulator).run("out")

    assert result.success is False
    assert result.best is None
    assert result.trials, "the agent must have tried at least one recipe"
    assert all(not t.feasible for t in result.trials)


def test_agent_respects_max_trials():
    simulator = Simulator()
    config = AutoCompressConfig(
        target_vram_gb=1.0,
        max_ppl_increase_pct=100.0,
        max_trials=2,
        baseline_ppl=BASELINE_PPL,
    )
    result = _make_agent(config, simulator).run("out")
    assert len(result.trials) == 2
    assert len(simulator.quantize_calls) == 2


def test_agent_measures_baseline_when_not_provided():
    simulator = Simulator()
    config = AutoCompressConfig(
        target_vram_gb=0.5,
        max_ppl_increase_pct=12.0,
        baseline_ppl=None,
    )
    calls = []

    def baseline_fn(model_id):
        calls.append(model_id)
        return BASELINE_PPL

    agent = AutoCompressionAgent(
        "dummy/model",
        config,
        quantize_fn=simulator.quantize,
        evaluate_fn=simulator.evaluate,
        estimate_fn=simulator.estimate,
        recommend_fn=simulator.recommend,
        baseline_fn=baseline_fn,
    )
    result = agent.run("out")
    assert calls == ["dummy/model"]
    assert result.baseline_ppl == BASELINE_PPL
    assert result.success is True


def test_agent_keeps_completed_trials_when_evaluation_fails():
    """A failing trial stops the search but preserves earlier results.

    The search descends 5 → 4 → 3; evaluation raises at 3-bit.  The
    agent must return the 4-bit result found before the failure and
    mention the failure in the message instead of raising.
    """
    simulator = Simulator(evaluate_fails_at=3)
    config = AutoCompressConfig(
        target_vram_gb=0.5,
        max_ppl_increase_pct=12.0,
        baseline_ppl=BASELINE_PPL,
    )
    result = _make_agent(config, simulator).run("out")

    assert result.success is True
    assert result.best.wbits == 4
    assert [t.wbits for t in result.trials] == [5, 4]
    assert "trial failure" in result.message
    assert "simulated evaluation failure at 3-bit" in result.message


def test_agent_reports_failure_when_first_trial_raises():
    simulator = Simulator(evaluate_fails_at=5)
    config = AutoCompressConfig(
        target_vram_gb=0.5,
        max_ppl_increase_pct=12.0,
        baseline_ppl=BASELINE_PPL,
    )
    result = _make_agent(config, simulator).run("out")

    assert result.success is False
    assert result.best is None
    assert result.trials == []
    assert "trial failure" in result.message


def test_dry_run_executes_no_quantization():
    simulator = Simulator()
    config = AutoCompressConfig(
        target_vram_gb=0.5,
        max_ppl_increase_pct=12.0,
        baseline_ppl=BASELINE_PPL,
    )
    result = _make_agent(config, simulator).dry_run()

    assert simulator.quantize_calls == [], "dry_run must not quantize"
    assert result.success is True
    assert result.best.wbits == 4
    assert result.message.startswith("[dry run")


def test_simulated_error_model_is_monotone():
    increases = [simulate_ppl_increase(b) for b in (8, 6, 5, 4, 3, 2)]
    assert increases == sorted(increases)
    assert increases[0] < increases[-1]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"target_vram_gb": 0}, "target_vram_gb"),
        ({"target_vram_gb": math.nan}, "target_vram_gb"),
        ({"target_vram_gb": 1, "max_trials": 0}, "max_trials"),
        ({"target_vram_gb": 1, "context_len": -1}, "context_len"),
        ({"target_vram_gb": 1, "wbits_candidates": ()}, "wbits_candidates"),
        (
            {"target_vram_gb": 1, "wbits_candidates": (4, 8)},
            "strictly descending",
        ),
        ({"target_vram_gb": 1, "baseline_ppl": 0}, "baseline_ppl"),
    ],
)
def test_config_rejects_invalid_search_domains(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AutoCompressConfig(**kwargs)


def test_agent_rejects_empty_model_and_save_dir():
    config = AutoCompressConfig(target_vram_gb=1, baseline_ppl=1)
    simulator = Simulator()
    with pytest.raises(ValueError, match="model_id"):
        AutoCompressionAgent("", config)
    with pytest.raises(ValueError, match="save_dir"):
        _make_agent(config, simulator).run("")
