"""Integration test: JointQ ``enable_group_swap=True`` (post-LS) on TinyLlama.

Quantize two representative linear layers of TinyLlama with JointQ
**with** ``enable_group_swap=True`` and verify *structural* invariants:

1. Exactly the targeted layers are quantized.
2. Each ``JointQResult.perm`` (when present) is a valid permutation
   of ``range(in_features)`` so the dequantizer can invert it.

The test deliberately does **not** assert anything about the
``output_squared_error`` direction or run a no-swap baseline.
JointQ's actual objective is the *regularized* loss
``||(W - W_hat) X^T||_F^2 + n*lambda*||W - W_hat||^2``, and that is
what post-LS swap is designed to reduce. ``output_squared_error``
measures only the first (non-regularized) term, so it can move in
either direction even when the swap legitimately reduces the
regularized objective. Whether swap reduces the regularized
objective is verified by the per-layer ``[GroupSwap]`` log lines
emitted by ``onecomp.quantizer.jointq.core.group_swap`` during the
run -- that is the authoritative signal.

For convenience we still print the resulting per-layer
``output_squared_error`` / ``weight_squared_error`` so the user can
inspect them with ``pytest -s`` -- but they are not gating the test.

Layer choice: ``layers.0`` (early, outlier-heavy, GPTQ struggles)
and ``layers.11`` (mid-stack, well-behaved) together cover both
regimes where post-LS swap might or might not help.

The test uses **3-bit** quantization (8 levels) rather than 4-bit
(16 levels) so that the choice of column-to-group assignment has a
measurable impact on the regularized objective. At 4 bits the
quantization grid is fine enough that almost any reasonable
grouping reaches a similar local minimum after JointQ's
joint-update phase.

The whole run is parametrized over ``actorder in {False, True}``
so both code paths are exercised; group-swap's column permutation
is composed with the actorder permutation in the latter case.

Run with:

    pytest tests/onecomp/quantizer/jointq/test_jointq_groupswap_tinyllama.py \
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


def _run_jointq(actorder: bool) -> dict:
    """Quantize ``TARGET_LAYERS`` of TinyLlama with JointQ + post-LS swap.

    Returns ``quantizer.results``: a dict of layer-name -> JointQResult.
    Uses ``calc_quant_error=True`` so each result carries
    ``output_squared_error`` / ``weight_squared_error`` for the report.

    Realistic configuration: keeps the default Tikhonov regularization
    and Clip-Optimize candidate enabled. The post-LS swap is designed
    to cooperate with -- not bypass -- these standard knobs, so the
    test exercises the integration the way it will actually be used.
    """
    setup_logger()

    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
    quantizer = JointQ(
        bits=3,
        symmetric=True,
        group_size=128,
        include_layer_names=TARGET_LAYERS,
        log_level=1,
        actorder=actorder,
        enable_group_swap=True,
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


@pytest.fixture(
    scope="module",
    params=[False, True],
    ids=["actorder_off", "actorder_on"],
)
def actorder(request):
    """Run the quantization once per actorder setting."""
    return request.param


@pytest.fixture(scope="module")
def results(actorder):
    """Quantize TARGET_LAYERS with post-LS swap enabled."""
    return _run_jointq(actorder=actorder)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class TestJointQGroupSwapTinyLlama:
    """Structural checks of JointQ + post-LS swap on TinyLlama.

    Parametrized over ``actorder in {False, True}`` so both code
    paths (with / without actorder permutation) are exercised.
    """

    def test_target_layers_quantized(self, results):
        """Exactly the targeted layers should be quantized."""
        actual = set(results.keys())
        assert actual == set(TARGET_LAYERS), (
            f"Expected exactly {sorted(TARGET_LAYERS)} to be quantized, " f"got: {sorted(actual)}"
        )

    def test_perm_is_valid_permutation(self, results):
        """When swap (or actorder) sets ``perm``, it must be a valid
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

        These numbers are *not* gating the test. JointQ's true
        objective is the regularized loss; the per-layer
        ``[GroupSwap]`` log lines emitted by the swap optimizer are
        the authoritative signal for whether swap reduces that
        objective.
        """
        print("\n[squared-error summary -- not gating]")
        print(f"  {'layer':<45s}  " f"{'output_sq_err':>14s}  {'weight_sq_err':>14s}")
        for name in sorted(results):
            r = results[name]
            o = r.output_squared_error
            w = r.weight_squared_error
            print(f"  {name:<45s}  {o:14.4e}  {w:14.4e}")
