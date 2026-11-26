"""Autonomous goal-driven compression agent.

Searches for the smallest-memory quantisation recipe that satisfies a
VRAM budget and a perplexity-degradation constraint.  The search is a
hill climb over the candidate bit-widths: start from the recipe
suggested by :func:`onecomp.mcp.tools.recommend_recipe`, quantise,
evaluate, then move to fewer bits while the constraints hold and back
to more bits when perplexity degrades too much.

Classes:
    AutoCompressConfig: Goal definition (VRAM budget, PPL tolerance, trials).
    TrialResult: Outcome of a single quantise-and-evaluate trial.
    AutoCompressResult: Final search outcome with the full trial history.
    AutoCompressionAgent: The search loop with injectable execution functions.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass, field
from logging import getLogger
from math import isfinite
from typing import Callable, List, Optional, Tuple

from . import tools

logger = getLogger(__name__)


@dataclass
class AutoCompressConfig:
    """Goal definition for the autonomous compression search.

    Args:
        target_vram_gb: VRAM budget in GB that the quantised model
            (weights + KV cache) must fit into.
        max_ppl_increase_pct: Maximum tolerated perplexity increase in
            percent relative to the FP16 baseline.
        max_trials: Maximum number of quantise-and-evaluate trials.
        wbits_candidates: Candidate bit-widths, ordered from most to
            fewest bits.
        context_len: Context length used for memory estimation.
        baseline_ppl: FP16 baseline perplexity.  When ``None``, it is
            measured with the injected evaluation function before the
            first trial.
    """

    target_vram_gb: float
    max_ppl_increase_pct: float = 10.0
    max_trials: int = 6
    wbits_candidates: Tuple[int, ...] = (8, 6, 5, 4, 3, 2)
    context_len: int = 4096
    baseline_ppl: Optional[float] = None

    def __post_init__(self) -> None:
        _positive_finite("target_vram_gb", self.target_vram_gb)
        _non_negative_finite("max_ppl_increase_pct", self.max_ppl_increase_pct)
        if not isinstance(self.max_trials, int) or isinstance(self.max_trials, bool):
            raise ValueError("max_trials must be a positive integer")
        if self.max_trials <= 0:
            raise ValueError("max_trials must be a positive integer")
        if not isinstance(self.context_len, int) or isinstance(self.context_len, bool):
            raise ValueError("context_len must be a positive integer")
        if self.context_len <= 0:
            raise ValueError("context_len must be a positive integer")
        if not isinstance(self.wbits_candidates, tuple) or not self.wbits_candidates:
            raise ValueError("wbits_candidates must be a non-empty tuple")
        if any(
            not isinstance(bits, int) or isinstance(bits, bool) or not 1 <= bits <= 16
            for bits in self.wbits_candidates
        ):
            raise ValueError("wbits_candidates must contain integers between 1 and 16")
        if tuple(sorted(set(self.wbits_candidates), reverse=True)) != self.wbits_candidates:
            raise ValueError("wbits_candidates must be unique and strictly descending")
        if self.baseline_ppl is not None:
            _positive_finite("baseline_ppl", self.baseline_ppl)


@dataclass
class TrialResult:
    """Outcome of one quantise-and-evaluate trial."""

    wbits: int
    estimated_total_gb: float
    perplexity: float
    ppl_increase_pct: float
    fits_memory: bool
    meets_ppl: bool
    save_dir: Optional[str] = None

    @property
    def feasible(self) -> bool:
        """Whether the trial satisfies both constraints."""
        return self.fits_memory and self.meets_ppl


@dataclass
class AutoCompressResult:
    """Final outcome of the compression search."""

    success: bool
    best: Optional[TrialResult]
    trials: List[TrialResult] = field(default_factory=list)
    baseline_ppl: Optional[float] = None
    message: str = ""


class AutoCompressionAgent:
    """Goal-driven recipe search over quantisation bit-widths.

    The quantise/evaluate execution functions are injected through the
    constructor so that tests can substitute a fast simulator and real
    runs can delegate to :func:`onecomp.mcp.tools.quantize` /
    :func:`onecomp.mcp.tools.evaluate`.

    Examples:
        >>> config = AutoCompressConfig(target_vram_gb=8.0, max_ppl_increase_pct=5.0)
        >>> agent = AutoCompressionAgent("meta-llama/Llama-2-7b-hf", config)
        >>> plan = agent.dry_run()          # no quantisation executed
        >>> result = agent.run("./compressed")  # full search (long-running)
    """

    def __init__(
        self,
        model_id: str,
        config: AutoCompressConfig,
        quantize_fn: Optional[Callable[[str, str, int], str]] = None,
        evaluate_fn: Optional[Callable[[str], float]] = None,
        estimate_fn: Optional[Callable[[str, int], float]] = None,
        baseline_fn: Optional[Callable[[str], float]] = None,
        recommend_fn: Optional[Callable[[str, float], float]] = None,
    ):
        """
        Args:
            model_id: Hugging Face model ID or local path.
            config: Search goal definition.
            quantize_fn: ``(model_id, save_dir, wbits) -> save_dir`` —
                quantises the model and returns the save location.
                Defaults to :func:`onecomp.mcp.tools.quantize`.
            evaluate_fn: ``(save_dir) -> perplexity`` — evaluates a
                saved quantised model.  Defaults to
                :func:`onecomp.mcp.tools.evaluate`.
            estimate_fn: ``(model_id, wbits) -> total_gb`` — memory
                estimate.  Defaults to
                :func:`onecomp.mcp.tools.estimate_memory`.
            baseline_fn: ``(model_id) -> perplexity`` — FP16 baseline
                evaluation, used only when ``config.baseline_ppl`` is
                ``None``.
            recommend_fn: ``(model_id, target_vram_gb) -> wbits`` —
                provides the initial bit-width for the search.
                Defaults to :func:`onecomp.mcp.tools.recommend_recipe`.
        """
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(config, AutoCompressConfig):
            raise ValueError("config must be an AutoCompressConfig")
        self.model_id = model_id
        self.config = config
        self._quantize_fn = quantize_fn if quantize_fn is not None else self._default_quantize
        self._evaluate_fn = evaluate_fn if evaluate_fn is not None else self._default_evaluate
        self._estimate_fn = estimate_fn if estimate_fn is not None else self._default_estimate
        self._baseline_fn = baseline_fn if baseline_fn is not None else self._default_baseline
        self._recommend_fn = recommend_fn if recommend_fn is not None else self._default_recommend

    # ------------------------------------------------------------------
    # Default execution functions (real runs)
    # ------------------------------------------------------------------
    def _default_quantize(self, model_id: str, save_dir: str, wbits: int) -> str:
        tools.quantize(model_id=model_id, save_dir=save_dir, wbits=wbits, evaluate=False)
        return save_dir

    @staticmethod
    def _default_evaluate(save_dir: str) -> float:
        return tools.evaluate(model_dir=save_dir)["perplexity"]

    def _default_estimate(self, model_id: str, wbits: int) -> float:
        return tools.estimate_memory(
            model_id=model_id, wbits=wbits, context_len=self.config.context_len
        )["total_gb"]

    def _default_recommend(self, model_id: str, target_vram_gb: float) -> float:
        recipe = tools.recommend_recipe(
            model_id=model_id,
            total_vram_gb=target_vram_gb,
            context_len=self.config.context_len,
        )
        return recipe["wbits"]

    @staticmethod
    def _default_baseline(model_id: str) -> float:
        # pylint: disable=import-outside-toplevel
        from ..model_config import ModelConfig
        from ..utils import calculate_perplexity

        model_config = ModelConfig(model_id=model_id)
        ppl = calculate_perplexity(
            model=model_config.load_model(),
            tokenizer=model_config.load_tokenizer(),
            max_samples=16,
        )
        return float(ppl)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _initial_index(self) -> int:
        """Index into ``wbits_candidates`` to start the search from.

        Obtains an initial bit-width from the recommendation function
        (:func:`onecomp.mcp.tools.recommend_recipe` by default) and
        snaps it to the closest candidate not exceeding it.
        """
        candidates = self.config.wbits_candidates
        recommended = _positive_finite(
            "recommended wbits",
            self._recommend_fn(self.model_id, self.config.target_vram_gb),
        )
        for idx, bits in enumerate(candidates):
            if bits <= recommended:
                return idx
        return len(candidates) - 1

    def run(self, save_dir: str) -> AutoCompressResult:
        """Execute the search: quantise, evaluate, and hill-climb.

        A trial that raises (quantisation or evaluation failure) stops
        the search but does not discard the trials already completed:
        the best feasible trial found so far is still returned and the
        failure is reported in the result message.

        Args:
            save_dir: Base directory for trial outputs.  Each trial is
                saved to ``{save_dir}-w{bits}``.

        Returns:
            :class:`AutoCompressResult` — ``best`` is the feasible
            trial with the smallest estimated memory, or ``None`` if no
            trial satisfied the constraints.
        """
        if not isinstance(save_dir, str) or not save_dir.strip():
            raise ValueError("save_dir must be a non-empty string")
        config = self.config
        candidates = config.wbits_candidates

        baseline = config.baseline_ppl
        if baseline is None:
            baseline = self._baseline_fn(self.model_id)
            logger.info("Measured FP16 baseline perplexity: %.4f", baseline)
        baseline = _positive_finite("baseline perplexity", baseline)

        trials: List[TrialResult] = []
        tried = {}
        trial_error: Optional[str] = None
        idx = self._initial_index()

        while len(trials) < config.max_trials and 0 <= idx < len(candidates):
            bits = candidates[idx]
            if bits in tried:
                break

            try:
                est_gb = _non_negative_finite(
                    "estimated memory", self._estimate_fn(self.model_id, bits)
                )
                trial_dir = f"{save_dir}-w{bits}"
                self._quantize_fn(self.model_id, trial_dir, bits)
                ppl = _positive_finite("perplexity", self._evaluate_fn(trial_dir))
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Trial at %d-bit failed; stopping the search", bits)
                trial_error = f"{type(exc).__name__}: {exc}"
                break
            increase_pct = (ppl - baseline) / baseline * 100.0

            trial = TrialResult(
                wbits=bits,
                estimated_total_gb=est_gb,
                perplexity=ppl,
                ppl_increase_pct=increase_pct,
                fits_memory=est_gb <= config.target_vram_gb,
                meets_ppl=increase_pct <= config.max_ppl_increase_pct,
                save_dir=trial_dir,
            )
            trials.append(trial)
            tried[bits] = trial
            logger.info(
                "Trial %d: %d-bit → %.2f GB, PPL %.4f (%+.2f%%), feasible=%s",
                len(trials),
                bits,
                est_gb,
                ppl,
                increase_pct,
                trial.feasible,
            )

            if trial.feasible:
                idx += 1  # constraints hold → try fewer bits (less memory)
            elif not trial.meets_ppl:
                idx -= 1  # perplexity degraded too much → back off to more bits
            else:
                idx += 1  # does not fit in memory → fewer bits needed

        feasible = [t for t in trials if t.feasible]
        if feasible:
            best = min(feasible, key=lambda t: t.estimated_total_gb)
            message = (
                f"Converged to {best.wbits}-bit: {best.estimated_total_gb:.2f} GB "
                f"(budget {config.target_vram_gb:.2f} GB), perplexity "
                f"{best.perplexity:.4f} ({best.ppl_increase_pct:+.2f}% vs baseline, "
                f"tolerance {config.max_ppl_increase_pct:.2f}%)."
            )
            if trial_error is not None:
                message += f" Search stopped early after a trial failure ({trial_error})."
            return AutoCompressResult(
                success=True, best=best, trials=trials, baseline_ppl=baseline, message=message
            )
        message = (
            f"No configuration satisfied the constraints within "
            f"{config.max_trials} trials (budget {config.target_vram_gb:.2f} GB, "
            f"max PPL increase {config.max_ppl_increase_pct:.2f}%)."
        )
        if trial_error is not None:
            message += f" Search stopped early after a trial failure ({trial_error})."
        return AutoCompressResult(
            success=False, best=None, trials=trials, baseline_ppl=baseline, message=message
        )

    def dry_run(self) -> AutoCompressResult:
        """Simulate the search without executing any quantisation.

        Memory estimates are real (from ``config.json``); perplexity is
        simulated with a monotone error model (fewer bits → larger
        degradation), so the returned perplexity values are indicative
        only.  Useful for previewing the trial plan and the expected
        convergence bit-width.
        """
        sim_baseline = 1.0

        def _sim_quantize(_model_id: str, save_dir: str, _wbits: int) -> str:
            return save_dir

        def _sim_evaluate(save_dir: str) -> float:
            bits = int(save_dir.rsplit("-w", 1)[1])
            return sim_baseline * (1.0 + simulate_ppl_increase(bits) / 100.0)

        sim_config = AutoCompressConfig(
            target_vram_gb=self.config.target_vram_gb,
            max_ppl_increase_pct=self.config.max_ppl_increase_pct,
            max_trials=self.config.max_trials,
            wbits_candidates=self.config.wbits_candidates,
            context_len=self.config.context_len,
            baseline_ppl=sim_baseline,
        )
        sim_agent = AutoCompressionAgent(
            self.model_id,
            sim_config,
            quantize_fn=_sim_quantize,
            evaluate_fn=_sim_evaluate,
            estimate_fn=self._estimate_fn,
            recommend_fn=self._recommend_fn,
        )
        result = sim_agent.run("dry-run")
        result.message = "[dry run — simulated perplexity] " + result.message
        return result


def simulate_ppl_increase(wbits: int) -> float:
    """Monotone bit-width → perplexity-increase (%) error model.

    Roughly calibrated to typical GPTQ behaviour: negligible at 8-bit,
    a few percent at 4-bit, and rapidly growing below 3-bit.
    """
    return 100.0 * 0.45 / (2.0 ** (wbits - 2))


def _positive_finite(name: str, value: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _non_negative_finite(name: str, value: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number")
    return float(value)
