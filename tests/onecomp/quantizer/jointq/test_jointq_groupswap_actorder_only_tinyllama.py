"""Isolation test: GPTQ regularized-Hessian vs. post-LS swap on TinyLlama.

This test investigates *why* post-LS swap finds large improvements
under the default JointQ configuration (lambda > 0) but only
negligible improvements under the isolation configuration
(lambda = 0). The hypothesis being tested:

  The benefit comes from an *objective mismatch* between GPTQ
  (which traditionally uses the pre-regularization Hessian
  ``(2/n) * X^T X``) and the downstream LS pass (which optimizes
  the regularized objective). Aligning GPTQ to the regularized
  Hessian should eliminate the mismatch and shrink the swap's
  improvement margin.

Configuration:

- ``actorder=True``                           : GPTQ-style activation ordering
- ``enable_clip_optimize=False``              : GPTQ is the only initial
- ``enable_clip_optimize_ep=False``
- ``regularization_lambda=0.1``               : Tikhonov term *enabled*
                                                (the regime where swap
                                                previously found ~-12%)
- ``gptq_use_regularized_hessian=True``       : feed GPTQ the same
                                                regularized Hessian as
                                                LS

Expected outcomes:

- If the GPTQ pre/post-regularization Hessian mismatch is the
  dominant cause of the previously-observed swap benefit: the
  ``[GroupSwap]`` log should now report close-to-zero F reduction
  even with lambda=0.1, similar to the lambda=0.0 isolation result.
- If the swap benefit persists despite the alignment: the
  mismatch is not the sole cause -- something else (e.g.
  actorder's grouping being inherently sub-optimal for the
  regularized objective) is responsible.

The ``[GroupSwap]`` log lines are the authoritative signal. The
printed per-layer ``output_squared_error`` / ``weight_squared_error``
serves as a secondary cross-check.

Layer choice: same as the main integration test --
``layers.0`` (outlier-heavy) and ``layers.11`` (well-behaved).

The test uses **3-bit** quantization (8 levels) so the grouping
choice has measurable impact on F.

Run with:

    pytest tests/onecomp/quantizer/jointq/test_jointq_groupswap_actorder_only_tinyllama.py \
        -v -s -m slow

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import gc

import pytest
import torch

from onecomp import CalibrationConfig, JointQ, ModelConfig, Runner, setup_logger

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
# Target two representative layers:
#   - ``layers.0``  : early, outlier-heavy, GPTQ struggles here
#   - ``layers.11`` : mid-stack, well-behaved, GPTQ does a good job
# TinyLlama has 22 transformer blocks (layers.0 ... layers.21).
TARGET_LAYERS = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.11.self_attn.q_proj",
]


def _run_jointq() -> dict:
    """Quantize ``TARGET_LAYERS`` with GPTQ (regularized H) + actorder + swap.

    Returns ``quantizer.results``: a dict of layer-name -> JointQResult.
    Uses ``calc_quant_error=True`` so each result carries
    ``output_squared_error`` / ``weight_squared_error`` for the report.

    Configuration designed to test the GPTQ-pre/post-regularization
    mismatch hypothesis:
      - ``actorder=True``                       : GPTQ-style activation ordering
      - ``enable_clip_optimize=False``          : GPTQ is the only initial
      - ``enable_clip_optimize_ep=False``
      - ``regularization_lambda=0.1``           : LS objective is regularized
      - ``gptq_use_regularized_hessian=True``   : GPTQ uses the *same*
                                                  regularized Hessian as LS
    """
    setup_logger()

    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
    quantizer = JointQ(
        bits=3,
        symmetric=True,
        group_size=128,
        include_layer_names=TARGET_LAYERS,
        log_level=1,
        actorder=True,
        enable_group_swap=True,
        enable_clip_optimize=False,
        enable_clip_optimize_ep=False,
        regularization_lambda=0.1,
        gptq_use_regularized_hessian=True,
        calc_quant_error=True,
    )

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=CalibrationConfig(
            max_length=512,
            num_calibration_samples=128,
            strategy="drop_rand",
            seed=0,
        ),
        qep=False,
    )
    runner.run()

    results = quantizer.results

    del runner, quantizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


@pytest.fixture(scope="module")
def results():
    """Quantize TARGET_LAYERS with GPTQ + actorder + post-LS swap."""
    return _run_jointq()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class TestJointQGroupSwapActorderOnlyTinyLlama:
    """Isolation test: probe whether GPTQ-actorder alone is sufficient.

    Configuration is locked to ``actorder=True``, GPTQ-only initial
    solution, and ``lambda=0.0`` so the regularized objective F
    coincides with ``output_squared_error`` and the printed numbers
    can be cross-checked directly against the ``[GroupSwap]`` log.
    """

    def test_target_layers_quantized(self, results):
        """Exactly the targeted layers should be quantized."""
        actual = set(results.keys())
        assert actual == set(TARGET_LAYERS), (
            f"Expected exactly {sorted(TARGET_LAYERS)} to be quantized, " f"got: {sorted(actual)}"
        )

    def test_perm_is_valid_permutation(self, results):
        """``perm`` (from actorder + swap composition) must be a valid
        permutation of ``range(in_features)`` so the dequantizer can
        invert it."""
        for name, result in sorted(results.items()):
            if result.perm is None:
                continue
            perm = result.perm
            in_features = perm.numel()
            sorted_perm, _ = torch.sort(perm)
            expected = torch.arange(in_features, dtype=perm.dtype, device=perm.device)
            assert torch.equal(sorted_perm, expected), (
                f"{name}: perm is not a valid permutation of "
                f"range({in_features}). "
                f"perm.shape={tuple(perm.shape)}, "
                f"unique values={perm.unique().numel()}"
            )

    def test_report_quantization_errors(self, results):
        """Always-pass report: print per-layer output / weight squared
        errors for visual inspection (visible with ``pytest -s``).

        The authoritative signal for this test is the per-layer
        ``[GroupSwap]`` ``F_final / F_init`` ratio: if GPTQ's
        regularized Hessian aligns it with the LS objective, this
        ratio should be close to 1 (no improvement to find).
        """
        print(
            "\n[squared-error summary -- not gating, "
            "lambda=0.1, gptq_use_regularized_hessian=True]"
        )
        print(f"  {'layer':<45s}  " f"{'output_sq_err':>14s}  {'weight_sq_err':>14s}")
        for name in sorted(results):
            r = results[name]
            o = r.output_squared_error
            w = r.weight_squared_error
            print(f"  {name:<45s}  {o:14.4e}  {w:14.4e}")
