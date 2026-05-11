"""End-to-end LPCD integration test via ``Runner``.

Runs GPTQ quantization with LPCD enabled on a single TinyLlama decoder
block and verifies that:

1. ``Runner.run()`` completes without raising when ``lpcd=True``.
2. ``quantizer.results`` is populated with every linear layer in the
   targeted block and all dequantized weights are finite.
3. LPCD with ``enable_residual=True`` actually modifies the residual
   layers (``o_proj`` / ``down_proj``) relative to a plain GPTQ+QEP
   baseline, while leaving non-LPCD layers close to the baseline.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    pytest tests/onecomp/lpcd/test_lpcd_runner.py -v -s --log-cli-level=INFO
"""

import gc

import pytest
import torch

from onecomp import (
    CalibrationConfig,
    LPCDConfig,
    ModelConfig,
    Runner,
    setup_logger,
)
from onecomp.qep import QEPConfig
from onecomp.quantizer.gptq import GPTQ

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# Process exactly one transformer block (7 linear layers per block).
NUM_LAYERS = 7

# Names (relative to the first decoder block) that LPCD's default
# residual-only configuration targets.
RESIDUAL_LAYER_SUFFIXES = ("self_attn.o_proj", "mlp.down_proj")

# Names that LPCD should NOT touch directly (quantized via the plain
# block-wise path).  Note: ``gate_proj`` / ``up_proj`` still see
# LPCD-modified activations from the refined ``o_proj``, so their
# weights may differ from the no-LPCD baseline.  Only the pre-attention
# projections (``q_proj`` / ``k_proj`` / ``v_proj``) are completely
# independent of any LPCD refinement inside the same block.
NON_LPCD_LAYER_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LPCD integration test requires CUDA",
)


def _calib_config() -> CalibrationConfig:
    """Small calibration config (LPCD hard-codes internal batch_size=16)."""
    return CalibrationConfig(
        num_calibration_samples=16,
        max_length=128,
        strategy="drop_rand",
        seed=0,
    )


def _run_gptq(
    *,
    lpcd: bool,
    lpcd_config: LPCDConfig | None = None,
    qep: bool = True,
) -> dict:
    """Run GPTQ (optionally + QEP + LPCD) on the first TinyLlama block.

    Returns ``quantizer.results`` (layer-name → ``QuantResult``).
    """
    setup_logger()

    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
    quantizer = GPTQ(
        wbits=4,
        groupsize=128,
        sym=False,
        num_layers=NUM_LAYERS,
    )

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=_calib_config(),
        qep=qep,
        lpcd=lpcd,
        lpcd_config=lpcd_config,
    )
    runner.run()

    results = quantizer.results

    del runner, quantizer, model_config
    gc.collect()
    torch.cuda.empty_cache()

    return results


@pytest.fixture(scope="module")
def results_lpcd_residual() -> dict:
    """GPTQ + QEP + LPCD (residual-only, default config)."""
    return _run_gptq(lpcd=True, lpcd_config=LPCDConfig(), qep=True)


@pytest.fixture(scope="module")
def results_baseline() -> dict:
    """GPTQ + QEP baseline (no LPCD)."""
    return _run_gptq(lpcd=False, qep=True)


def _relative_error(w_a: torch.Tensor, w_b: torch.Tensor) -> float:
    diff = torch.norm(w_a.float() - w_b.float()).item()
    base = torch.norm(w_a.float()).item()
    return diff / base if base > 0 else float("inf")


class TestLPCDRunnerSmoke:
    """Smoke tests for the LPCD pipeline through ``Runner``."""

    def test_runs_without_error(self, results_lpcd_residual):
        """Runner.run() with ``lpcd=True`` completes without raising."""
        assert isinstance(results_lpcd_residual, dict)
        assert len(results_lpcd_residual) > 0

    def test_all_block_layers_quantized(self, results_lpcd_residual):
        """Every linear layer in the first decoder block is quantized."""
        expected_suffixes = set(RESIDUAL_LAYER_SUFFIXES) | set(NON_LPCD_LAYER_SUFFIXES)
        for suffix in expected_suffixes:
            matching = [name for name in results_lpcd_residual if name.endswith(suffix)]
            assert matching, f"No quantized layer found with suffix {suffix!r}"

    def test_dequantized_weights_are_finite(self, results_lpcd_residual):
        """Dequantized weights produced by LPCD are finite and non-zero."""
        for name, result in results_lpcd_residual.items():
            w = result.compute_dequantized_weight()
            assert torch.isfinite(w).all(), f"{name}: contains NaN/Inf"
            assert w.abs().sum().item() > 0, f"{name}: all zeros"


class TestLPCDRunnerWithQEPConfig:
    """Explicit LPCD + QEP combination should also work."""

    def test_runs_with_explicit_qep_config(self):
        qep_config = QEPConfig(percdamp=0.01, perccorr=0.5)
        setup_logger()

        model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
        quantizer = GPTQ(
            wbits=4,
            groupsize=128,
            sym=False,
            num_layers=NUM_LAYERS,
        )
        runner = Runner(
            model_config=model_config,
            quantizer=quantizer,
            calibration_config=_calib_config(),
            qep=True,
            qep_config=qep_config,
            lpcd=True,
            lpcd_config=LPCDConfig(),
        )
        runner.run()

        assert len(quantizer.results) > 0

        del runner, quantizer, model_config
        gc.collect()
        torch.cuda.empty_cache()


class TestLPCDBehavior:
    """LPCD must actually change the residual layers relative to the baseline."""

    def test_same_layer_set(self, results_lpcd_residual, results_baseline):
        assert set(results_lpcd_residual.keys()) == set(results_baseline.keys())

    def test_residual_layers_modified(self, results_lpcd_residual, results_baseline):
        """``o_proj`` / ``down_proj`` weights differ meaningfully from baseline."""
        # At least one residual layer must differ beyond this threshold.
        # The value is loose because the exact refinement amount depends
        # on calibration data and GPTQ rounding, but a no-op LPCD run
        # would produce bit-identical results (diff = 0).
        MIN_RELATIVE_DIFF = 1e-4

        any_modified = False
        for name in results_lpcd_residual:
            if not name.endswith(RESIDUAL_LAYER_SUFFIXES):
                continue
            w_lpcd = results_lpcd_residual[name].compute_dequantized_weight()
            w_base = results_baseline[name].compute_dequantized_weight()
            rel_err = _relative_error(w_lpcd, w_base)
            if rel_err > MIN_RELATIVE_DIFF:
                any_modified = True
                break
        assert any_modified, (
            "LPCD did not modify any residual layer (o_proj / down_proj) "
            "beyond the baseline GPTQ+QEP output"
        )

    def test_pre_attn_projections_match_baseline(
        self, results_lpcd_residual, results_baseline
    ):
        """Pre-attention projections (q/k/v_proj) match the baseline.

        ``q_proj`` / ``k_proj`` / ``v_proj`` are quantised before any
        LPCD refinement happens within the block, and their input
        (``input_layernorm(residual)``) does not depend on any
        LPCD-refined module.  They therefore must reproduce the
        baseline GPTQ+QEP result bit-for-bit.

        Later layers (``gate_proj``, ``up_proj``) see
        post-attention residual states whose computation includes the
        LPCD-refined ``o_proj``, so they are *not* expected to match
        the baseline — that comparison is deliberately omitted.
        """
        PRE_ATTN_SUFFIXES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
        for name in results_lpcd_residual:
            if not name.endswith(PRE_ATTN_SUFFIXES):
                continue
            w_lpcd = results_lpcd_residual[name].compute_dequantized_weight()
            w_base = results_baseline[name].compute_dequantized_weight()
            rel_err = _relative_error(w_lpcd, w_base)
            assert rel_err <= 1e-4, (
                f"{name}: pre-attention projection diverges from baseline "
                f"(rel_err={rel_err:.2e})"
            )
