"""Unit tests for the lightweight swap-based group-assignment optimizer.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

The tests exercise ``onecomp.quantizer.jointq.core.group_swap`` on
small, deterministic problems. The primary goal is to confirm that the
approximate ``Delta F`` evaluation actually improves the objective on
data where the initial grouping is intentionally sub-optimal.

Each test is CPU-only and small enough to run in a fraction of a second.
"""

import torch

from onecomp.quantizer.jointq.core.clip import clip
from onecomp.quantizer.jointq.core.group_swap import (
    GroupSwapOptimizer,
    make_default_groups,
    optimize_groups_with_swap,
)
from onecomp.quantizer.jointq.core.solution import Solution

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _clip_solution(matrix_W, group_size, symmetric, bits):
    """Build a Solution from per-group clip quantization (no joint refinement).

    Mirrors the very first step of ``initialize_solution_clip_optimize``.
    """
    if symmetric:
        lower_bound = -(2 ** (bits - 1))
        upper_bound = 2 ** (bits - 1) - 1
    else:
        lower_bound = 0
        upper_bound = 2**bits - 1

    scales, assignment, zero_point = clip(
        matrix_W,
        group_size=group_size,
        symmetric=symmetric,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    return Solution(scales, assignment, zero_point)


def _objective(matrix_W, matrix_X, bar_W):
    """Reference objective ``F = ||W X^T - bar_W X^T||_F^2``."""
    diff = (matrix_W - bar_W) @ matrix_X.T
    return torch.sum(diff**2).item()


def _precompute(matrix_W, matrix_X):
    """Compute ``matrix_XX``, ``matrix_YX`` and ``Y_sq_norms`` for classic mode."""
    matrix_XX = matrix_X.T @ matrix_X
    # In classic mode Y = W X^T, so YX = W X^T X = W @ matrix_XX
    matrix_YX = matrix_W @ matrix_XX
    Y_sq_norms = (matrix_YX * matrix_W).sum(dim=1)
    return matrix_XX, matrix_YX, Y_sq_norms


def _build_improvable_problem(seed=0):
    """Build a small problem where lightweight swap is guaranteed to improve.

    The contiguous grouping mixes columns of very different magnitudes,
    so each per-(row, group) scale is dominated by a single large
    column and the smaller column in the same group is wasted (clipped
    to 0). Moving a medium-magnitude column to a small-magnitude group
    that already has a tight scale lets it be quantized accurately,
    which is exactly the kind of improvement the lightweight,
    frozen-scale approximation can detect.

    Layout (per row, ``|.|`` denotes magnitude):

    - Column 0: |big|   (5.0)
    - Column 1: |medium| (1.0)
    - Column 2: |small|  (0.05)
    - Column 3: |medium| (1.0)
    - Column 4: |big|    (5.0)
    - Column 5: |medium| (1.0)
    - Column 6: |small|  (0.05)
    - Column 7: |medium| (1.0)

    Default groups: ``G_0={0,1}, G_1={2,3}, G_2={4,5}, G_3={6,7}``.
    Group scales are dominated by ``big`` in groups 0 and 2 and by
    ``medium`` in groups 1 and 3, so columns 1 and 5 (medium) are
    quantized coarsely under a "big"-sized scale. Swapping col 1
    with col 2 (or col 5 with col 6) places the medium column under a
    smaller "medium"-sized scale where it is quantized accurately.

    Signs are randomized per (row, column) so the X-dependent objective
    is non-trivial.
    """
    torch.manual_seed(seed)

    dim_p = 4
    dim_n = 64
    num_groups = 4
    group_size = 2
    dim_m = num_groups * group_size  # 8

    magnitudes = torch.tensor(
        [5.0, 1.0, 0.05, 1.0, 5.0, 1.0, 0.05, 1.0],
        dtype=DTYPE,
    )
    signs = torch.where(torch.rand(dim_p, dim_m) > 0.5, 1.0, -1.0).to(DTYPE)
    matrix_W = signs * magnitudes.unsqueeze(0)

    matrix_X = torch.randn(dim_n, dim_m, dtype=DTYPE)
    return matrix_W, matrix_X, group_size, dim_n


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_make_default_groups_is_contiguous():
    """Default grouping must equal the contiguous block partition."""
    groups = make_default_groups(num_groups=3, group_size=4, device=DEVICE)
    expected = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        dtype=torch.long,
        device=DEVICE,
    )
    assert torch.equal(groups, expected)


def test_optimizer_initial_F_matches_reference():
    """The optimizer's incremental ``F`` must match a reference recomputation."""
    matrix_W, matrix_X, group_size, dim_n = _build_improvable_problem(seed=3)
    matrix_XX, matrix_YX, Y_sq_norms = _precompute(matrix_W, matrix_X)

    sol = _clip_solution(matrix_W, group_size=group_size, symmetric=False, bits=3)
    opt = GroupSwapOptimizer(
        solution=sol,
        matrix_XX=matrix_XX,
        matrix_YX=matrix_YX,
        Y_sq_norms=Y_sq_norms,
        dim_n=dim_n,
        bits=3,
        symmetric=False,
        log_level=0,
    )

    F_ref = _objective(matrix_W, matrix_X, opt.get_bar_W())
    assert abs(F_ref - float(opt.F.item())) < 1e-8 * max(1.0, F_ref)


def test_optimizer_improves_on_improvable_problem():
    """End-to-end: swap optimization reduces ``F`` for the designed problem."""
    matrix_W, matrix_X, group_size, dim_n = _build_improvable_problem(seed=4)
    matrix_XX, matrix_YX, Y_sq_norms = _precompute(matrix_W, matrix_X)
    bits = 3
    symmetric = False

    sol = _clip_solution(matrix_W, group_size=group_size, symmetric=symmetric, bits=bits)

    # Reference: error of the initial (intentionally bad) grouping.
    initial_bar_W = sol.get_dequantized_weight_matrix()
    F_initial = _objective(matrix_W, matrix_X, initial_bar_W)

    result = optimize_groups_with_swap(
        solution=sol,
        matrix_XX=matrix_XX,
        matrix_YX=matrix_YX,
        Y_sq_norms=Y_sq_norms,
        dim_n=dim_n,
        bits=bits,
        symmetric=symmetric,
        max_rounds=20,
        rho=1.0,
        log_level=0,
    )

    final_bar_W = result["bar_W"]
    F_final_recomputed = _objective(matrix_W, matrix_X, final_bar_W)
    stats = result["stats"]

    # Incremental ``F`` must agree with a fresh recomputation.
    assert abs(stats["F_final"] - F_final_recomputed) < 1e-6 * max(1.0, F_initial)

    # The clustered problem should be improvable by swap.
    assert stats["F_initial"] > stats["F_final"], (
        f"Expected improvement but F_initial = {stats['F_initial']:.4e}, "
        f"F_final = {stats['F_final']:.4e}"
    )
    assert stats["num_accepted"] >= 1

    # Each accepted swap must monotonically reduce ``F`` (subject to FP slack),
    # so the per-round history must be non-increasing.
    history = stats["F_history"]
    for prev_val, next_val in zip(history, history[1:]):
        assert next_val <= prev_val + 1e-6 * max(1.0, abs(prev_val))


def test_optimizer_improves_on_improvable_problem_symmetric():
    """Symmetric-quantization variant of the improvability test."""
    matrix_W, matrix_X, group_size, dim_n = _build_improvable_problem(seed=5)
    matrix_XX, matrix_YX, Y_sq_norms = _precompute(matrix_W, matrix_X)
    bits = 3
    symmetric = True

    sol = _clip_solution(matrix_W, group_size=group_size, symmetric=symmetric, bits=bits)
    F_initial = _objective(matrix_W, matrix_X, sol.get_dequantized_weight_matrix())

    result = optimize_groups_with_swap(
        solution=sol,
        matrix_XX=matrix_XX,
        matrix_YX=matrix_YX,
        Y_sq_norms=Y_sq_norms,
        dim_n=dim_n,
        bits=bits,
        symmetric=symmetric,
        max_rounds=20,
        rho=1.0,
        log_level=1,
    )
    stats = result["stats"]

    assert stats["F_initial"] > stats["F_final"]
    assert stats["num_accepted"] >= 1

    # Sanity: incremental F matches fresh recomputation.
    F_final_recomputed = _objective(matrix_W, matrix_X, result["bar_W"])
    assert abs(stats["F_final"] - F_final_recomputed) < 1e-6 * max(1.0, F_initial)


def test_groups_remain_a_valid_partition_after_optimization():
    """Optimized ``groups`` must still describe a partition of ``[0, m)``."""
    matrix_W, matrix_X, group_size, dim_n = _build_improvable_problem(seed=6)
    matrix_XX, matrix_YX, Y_sq_norms = _precompute(matrix_W, matrix_X)
    bits = 3

    sol = _clip_solution(matrix_W, group_size=group_size, symmetric=False, bits=bits)
    result = optimize_groups_with_swap(
        solution=sol,
        matrix_XX=matrix_XX,
        matrix_YX=matrix_YX,
        Y_sq_norms=Y_sq_norms,
        dim_n=dim_n,
        bits=bits,
        symmetric=False,
        max_rounds=10,
        rho=1.0,
        log_level=0,
    )

    groups = result["groups"]
    flat = groups.reshape(-1)
    sorted_flat, _ = torch.sort(flat)
    expected = torch.arange(matrix_W.shape[1], dtype=sorted_flat.dtype, device=sorted_flat.device)
    assert torch.equal(sorted_flat, expected)


def test_solution_after_swap_matches_permuted_bar_W():
    """Verify the invariant relied on by the JointQ integration.

    After the optimizer mutates ``solution.integers_z`` and tracks the
    new column-to-group mapping in ``groups``, the following holds in
    the permuted column ordering ``swap_perm = groups.reshape(-1)``:

    - ``solution.get_dequantized_weight_matrix() == bar_W[:, swap_perm]``

    i.e., applying the swap permutation to ``matrix_W`` / ``matrix_XX``
    and feeding the (already mutated) Solution into the existing
    contiguous-group pipeline yields exactly the optimized
    dequantized weight matrix.
    """
    matrix_W, matrix_X, group_size, dim_n = _build_improvable_problem(seed=7)
    matrix_XX, matrix_YX, Y_sq_norms = _precompute(matrix_W, matrix_X)
    bits = 3

    sol = _clip_solution(matrix_W, group_size=group_size, symmetric=False, bits=bits)
    result = optimize_groups_with_swap(
        solution=sol,
        matrix_XX=matrix_XX,
        matrix_YX=matrix_YX,
        Y_sq_norms=Y_sq_norms,
        dim_n=dim_n,
        bits=bits,
        symmetric=False,
        max_rounds=10,
        rho=1.0,
        log_level=0,
    )

    swap_perm = result["groups"].reshape(-1)
    bar_W_permuted = result["bar_W"][:, swap_perm]
    bar_W_from_solution = result["solution"].get_dequantized_weight_matrix()
    assert torch.allclose(bar_W_from_solution, bar_W_permuted, atol=1e-10)


def test_smoke_random_problem_does_not_break_objective():
    """Smoke test: on plain random data, optimization may or may not improve,
    but it must never increase ``F``."""
    torch.manual_seed(123)
    dim_p, dim_n, dim_m, group_size = 3, 32, 8, 2
    bits = 3

    matrix_W = torch.randn(dim_p, dim_m, dtype=DTYPE)
    matrix_X = torch.randn(dim_n, dim_m, dtype=DTYPE)
    matrix_XX, matrix_YX, Y_sq_norms = _precompute(matrix_W, matrix_X)

    sol = _clip_solution(matrix_W, group_size=group_size, symmetric=False, bits=bits)
    F_initial = _objective(matrix_W, matrix_X, sol.get_dequantized_weight_matrix())

    result = optimize_groups_with_swap(
        solution=sol,
        matrix_XX=matrix_XX,
        matrix_YX=matrix_YX,
        Y_sq_norms=Y_sq_norms,
        dim_n=dim_n,
        bits=bits,
        symmetric=False,
        max_rounds=5,
        rho=1.0,
        log_level=0,
    )

    F_final_recomputed = _objective(matrix_W, matrix_X, result["bar_W"])
    assert F_final_recomputed <= F_initial + 1e-6 * max(1.0, F_initial)
    assert abs(result["stats"]["F_final"] - F_final_recomputed) < 1e-6 * max(1.0, F_initial)


def test_no_op_when_only_one_group():
    """With ``num_groups = 1`` no swap is possible — the call must be a no-op."""
    torch.manual_seed(9)
    dim_p, dim_n, dim_m = 2, 16, 4
    bits = 3
    group_size = dim_m  # single group

    matrix_W = torch.randn(dim_p, dim_m, dtype=DTYPE)
    matrix_X = torch.randn(dim_n, dim_m, dtype=DTYPE)
    matrix_XX, matrix_YX, Y_sq_norms = _precompute(matrix_W, matrix_X)

    sol = _clip_solution(matrix_W, group_size=group_size, symmetric=False, bits=bits)
    integers_z_before = sol.integers_z.clone()

    result = optimize_groups_with_swap(
        solution=sol,
        matrix_XX=matrix_XX,
        matrix_YX=matrix_YX,
        Y_sq_norms=Y_sq_norms,
        dim_n=dim_n,
        bits=bits,
        symmetric=False,
        max_rounds=3,
        rho=1.0,
        log_level=0,
    )

    assert result["stats"]["num_accepted"] == 0
    assert result["stats"]["num_evaluations"] == 0
    assert torch.equal(sol.integers_z, integers_z_before)
    assert torch.equal(result["groups"], make_default_groups(1, dim_m, DEVICE))
