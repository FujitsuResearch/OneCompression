"""Lightweight swap-based group-assignment optimization for JointQ.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

This module implements the lightweight swap-based local search for
the column-to-group assignment problem in JointQ. See the design note
under the project's ``md/`` directory for the underlying algorithm.

The algorithm refines the column-to-group assignment used by JointQ.
It is intended to be inserted between the initial-solution generation
phase and the joint continuous / integer update phase of the existing
JointQ pipeline.

This file is intentionally self-contained: it does not modify any
existing module. Integration with the main pipeline is handled in a
later step of the implementation plan.

Notation
--------
- ``p`` : number of rows of the weight matrix (output channels)
- ``m`` : number of columns of the weight matrix (input channels)
- ``k`` : number of groups (= ``m / group_size``)
- ``d`` : group size
- ``n`` : number of calibration samples
- ``W^*`` : continuous-relaxation optimal weight matrix (shape ``(p, m)``)
- ``bar_W`` : current dequantized weight matrix (shape ``(p, m)``)
- ``U``  : ``R X`` where ``R = Y - bar_W X^T`` (shape ``(p, m)``)
- ``K``  : ``X^T X`` (shape ``(m, m)``)

"""

# TODO: Describe the algorithm in more detail when the implementation is complete.

import math
import time

import torch

from .solution import Solution


def make_default_groups(num_groups, group_size, device):
    """Return the contiguous default grouping used by JointQ.

    Parameters
    ----------
    num_groups : int
        Number of groups.
    group_size : int
        Size of each group.
    device : torch.device
        Device on which the result is placed.

    Returns
    -------
    torch.Tensor
        ``(num_groups, group_size)`` int64 tensor where row ``t`` lists
        the column indices that belong to group ``t``. Defaults to the
        contiguous block partition
        ``groups[t] = [t*d, t*d + 1, ..., (t+1)*d - 1]``.
    """
    return torch.arange(num_groups * group_size, device=device, dtype=torch.long).view(
        num_groups, group_size
    )


# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements,too-many-branches,invalid-name
class GroupSwapOptimizer:
    """Lightweight swap-based group-assignment optimizer.

    Refines the column-to-group assignment by swapping pairs of columns
    between different groups and accepting swaps that reduce
    ``F = ||Y - bar_W X^T||_F^2``.

    The class maintains the following state and updates it incrementally:

    - ``self.bar_W`` : current dequantized weight matrix, ``(p, m)``
    - ``self.U`` : ``R X``, ``(p, m)``
    - ``self.F`` : current objective value, scalar tensor
    - ``self.groups`` : ``(num_groups, group_size)``, the column-to-group map
    - ``self.column_to_group`` : ``(m,)`` int64, ``self.column_to_group[j] = t``
    - ``self.column_to_slot`` : ``(m,)`` int64, slot index ``d`` of column ``j``
    - ``self.solution.integers_z`` : updated for swapped columns
    - ``self.W_star`` and ``self.diag_K`` : cached for the current round
    """

    def __init__(
        self,
        solution,
        matrix_XX,
        matrix_YX,
        Y_sq_norms,
        dim_n,
        bits,
        symmetric,
        log_level=1,
    ):
        """Initialize the optimizer.

        The optimizer always starts from the contiguous default grouping
        ``G_t = {t*d, t*d + 1, ..., (t+1)*d - 1}``. Accepted swaps then
        update the internal ``groups`` tensor to a non-contiguous
        partition that the caller can use as a column permutation (see
        the module docstring).

        Parameters
        ----------
        solution : Solution
            Current JointQ solution. The optimizer mutates
            ``solution.integers_z`` (and not ``scales`` / ``zero_point``)
            in place when swaps are accepted.
        matrix_XX : torch.Tensor
            ``X^T X``, shape ``(m, m)``, dtype float64, on device.
        matrix_YX : torch.Tensor
            ``Y X`` (or ``W X^T X`` in classic mode), shape ``(p, m)``,
            dtype float64, on device.
        Y_sq_norms : torch.Tensor
            ``||Y[i]||^2`` for each row, shape ``(p,)``, dtype float64,
            on device.
        dim_n : int
            Number of calibration samples (rows of ``X``).
        bits : int
            Number of quantization bits.
        symmetric : bool
            Whether symmetric quantization is used.
        log_level : int, optional
            ``0``: silent, ``1``: minimal, ``2``: detailed.
        """

        if not isinstance(solution, Solution):
            raise TypeError(f"solution must be a Solution, got {type(solution)!r}.")
        if solution.scales is None or solution.integers_z is None:
            raise ValueError("solution must have scales and integers_z set.")
        if solution.zero_point is None:
            raise ValueError("solution must have zero_point set.")
        if not isinstance(bits, int) or bits < 1 or bits > 4:
            raise ValueError("bits must be an integer in [1, 4].")

        self.solution = solution
        self.matrix_XX = matrix_XX
        self.matrix_YX = matrix_YX
        self.Y_sq_norms = Y_sq_norms
        self.dim_n = int(dim_n)
        self.bits = int(bits)
        self.symmetric = bool(symmetric)
        self.log_level = int(log_level)

        self.device = matrix_XX.device
        self.dtype = matrix_XX.dtype

        self.num_groups, self.dim_p = solution.scales.shape
        self.group_size = solution.integers_z.shape[2]
        self.dim_m = self.num_groups * self.group_size

        if self.symmetric:
            self.lower_bound = -(2 ** (self.bits - 1))
            self.upper_bound = 2 ** (self.bits - 1) - 1
        else:
            self.lower_bound = 0
            self.upper_bound = 2**self.bits - 1

        # Group layout: start contiguous. Accepted swaps mutate this tensor.
        self.groups = make_default_groups(self.num_groups, self.group_size, self.device)

        # Column -> (group, slot) lookup. Initially the identity mapping.
        self.column_to_group = (
            torch.arange(self.dim_m, device=self.device, dtype=torch.long) // self.group_size
        )
        self.column_to_slot = (
            torch.arange(self.dim_m, device=self.device, dtype=torch.long) % self.group_size
        )

        # Cached float64 zero_point for clip arithmetic
        self._zero_point_f = solution.zero_point.to(self.dtype)

        # Diagonal of K = X^T X
        self.diag_K = torch.diagonal(matrix_XX).clone()

        # State that depends on the current solution / groups
        self.bar_W = solution.get_dequantized_weight_matrix()
        self.U = matrix_YX - self.bar_W @ matrix_XX
        self.F = self._compute_objective_from_scratch()

        # Caches refreshed each "round"
        self.W_star = None
        # Precomputed quantization tables: ``tilde_W_table[t, :, j]`` is the
        # dequantized value of column ``j`` when assigned to group ``t``
        # (computed from ``W_star`` and the current ``scales`` /
        # ``zero_point``). ``int_z_table`` is the matching shifted-integer
        # tensor. Both are refreshed once per round, right after
        # ``_refresh_w_star``. Shape: ``(k, p, m)``.
        self.tilde_W_table = None
        self.int_z_table = None

        # Statistics
        self.num_evaluations = 0
        self.num_accepted = 0
        self.F_history = [float(self.F.item())]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        max_rounds=10,
        rho=0.3,
        max_accept=None,
        stop_if_no_improvement=True,
        min_round_improvement=5e-3,
        min_swap_improvement=0.0,
    ):
        """Run the lightweight swap-based optimization (Top-K strategy).

        Each round:

        1. Refresh ``W^*`` (one-shot per round).
        2. Score every column with
           ``phi[j] = ||U[:, j]||^2 / K[j, j]`` and keep the top
           ``ceil(rho * m)`` columns as ``J_cand`` (= columns whose
           continuous-relaxation improvement margin is largest).
        3. Sweep ``J_cand`` once. For each ``j``, evaluate the entire
           neighbourhood ``N(j) = {(t', j') : t' != t, j' in G_{t'},
           j' not in used_columns}`` on the GPU and accept the
           **best** swap if its ``Delta F < 0``. Mark ``j`` and the
           chosen ``j'`` as ``used`` so neither is touched again in
           the same round.

        Parameters
        ----------
        max_rounds : int, optional
            Maximum number of rounds. Default is 10.
        rho : float, optional
            Fraction (in ``(0, 1]``) of columns kept in ``J_cand``
            per round (top by ``phi``). Default is 0.3 (= empirically
            the best quality/cost trade-off on TinyLlama 4bit/g128).
            Set lower (e.g. 0.1) to trade quality for speed.
        max_accept : int, optional
            Cap on accepted swaps within a single round. ``None``
            (default) means no cap.
        stop_if_no_improvement : bool, optional
            If ``True``, stop early when a full round produced no
            accepted swap. Default is ``True``.
        min_round_improvement : float, optional
            Minimum *relative* per-round F improvement to keep going
            (``(F_prev - F_curr) / F_prev``). When a round's
            improvement falls below this threshold, the loop exits.
            Empirically swap accept counts collapse after the first
            2-3 rounds, so the diminishing-returns tail is cheap to
            cut. Default is ``5e-3`` (0.5%). Set to 0 to disable
            this check.
        min_swap_improvement : float, optional
            Minimum *relative* per-swap F improvement required to
            accept a swap (``-Delta F / F_curr > min_swap_improvement``).
            Set ``> 0`` to skip noise-level swaps and save the
            per-swap commit cost (two ``outer`` updates of ``U``).
            Also serves as the FP64 safety threshold against FP32
            false-positives in the candidate-evaluation phase.
            Default ``0.0`` (accept any improving swap).

        Returns
        -------
        dict
            Statistics: ``rounds_run``, ``num_evaluations``,
            ``num_accepted``, ``F_initial``, ``F_final``, ``F_history``.

        Notes
        -----
        ``num_evaluations`` counts candidate ``j'`` evaluations (the
        size of ``N(j)`` aggregated across all evaluated ``j``), not
        the number of outer ``j``'s scanned.
        """

        if self.num_groups < 2:
            if self.log_level >= 1:
                print(f"<{self.device}>: [GroupSwap] num_groups < 2, no swaps possible (no-op).")
            return self._build_stats(rounds_run=0)

        if not 0.0 < rho <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {rho!r}.")

        F_initial = float(self.F.item())
        if self.log_level >= 1:
            k_top_initial = int(math.ceil(rho * self.dim_m))
            print(
                f"<{self.device}>: [GroupSwap] start "
                f"(F_init = {F_initial:.4e}, num_groups = {self.num_groups}, "
                f"group_size = {self.group_size}, m = {self.dim_m}, "
                f"rho = {rho:.3f}, |J_cand| = {k_top_initial})"
            )

        begin_time = time.time()
        rounds_run = 0
        for round_idx in range(max_rounds):
            self._refresh_w_star()
            self._precompute_tilde_table()
            J_cand = self._select_top_k_columns(rho)
            used_mask = torch.zeros(self.dim_m, dtype=torch.bool, device=self.device)

            accepted_in_round = 0
            evals_in_round_start = self.num_evaluations
            for j_t in J_cand.tolist():
                if max_accept is not None and accepted_in_round >= max_accept:
                    break
                if bool(used_mask[j_t].item()):
                    continue
                result = self._find_best_swap_for_j(
                    j_t, used_mask, min_swap_improvement=min_swap_improvement
                )
                if result is None:
                    continue

                self._apply_accepted_swap(
                    j=j_t,
                    j_prime=result["j_prime"],
                    t=result["t"],
                    t_prime=result["t_prime"],
                    delta_w_j=result["delta_w_j"],
                    delta_w_jprime=result["delta_w_jprime"],
                    new_z_j=result["new_z_j"],
                    new_z_jprime=result["new_z_jprime"],
                    delta_F=result["delta_F"],
                )
                used_mask[j_t] = True
                used_mask[result["j_prime"]] = True
                accepted_in_round += 1

            rounds_run += 1
            evals_in_round = self.num_evaluations - evals_in_round_start
            F_prev = self.F_history[-1]
            F_curr = float(self.F.item())
            if self.log_level >= 1:
                print(
                    f"<{self.device}>: [GroupSwap] round {round_idx + 1}: "
                    f"candidates = {J_cand.numel()}, "
                    f"evals = {evals_in_round}, "
                    f"accepted = {accepted_in_round}, "
                    f"F = {F_curr:.4e}"
                )
            self.F_history.append(F_curr)

            if stop_if_no_improvement and accepted_in_round == 0:
                break

            # Diminishing-returns early stop: per-round relative
            # improvement collapses quickly after rounds 1-2; cut the
            # tail once the gain per round drops below the threshold.
            if min_round_improvement > 0.0 and F_prev > 0.0:
                rel_round = (F_prev - F_curr) / F_prev
                if rel_round < min_round_improvement:
                    if self.log_level >= 1:
                        print(
                            f"<{self.device}>: [GroupSwap] round "
                            f"improvement {rel_round * 100:.3f}% < "
                            f"{min_round_improvement * 100:.3f}%, "
                            "stopping (diminishing returns)"
                        )
                    break

        F_final = float(self.F.item())
        if self.log_level >= 1:
            elapsed = time.time() - begin_time
            improvement = F_initial - F_final
            rel = improvement / F_initial if F_initial > 0 else 0.0
            print(
                f"<{self.device}>: [GroupSwap] done "
                f"(F_final = {F_final:.4e}, dF = -{improvement:.4e}, "
                f"rel = {rel * 100:.2f}%, "
                f"evals = {self.num_evaluations}, "
                f"accepted = {self.num_accepted}, "
                f"time = {elapsed:.2f}s)"
            )

        return self._build_stats(rounds_run=rounds_run)

    def get_bar_W(self):
        """Return the current dequantized weight matrix, shape ``(p, m)``."""
        return self.bar_W

    def get_groups(self):
        """Return the current column-to-group assignment, shape ``(k, d)``."""
        return self.groups

    def get_objective(self):
        """Recompute and return the objective value from scratch.

        The incrementally tracked ``self.F`` should match this value to
        within numerical precision. Useful as a sanity check.
        """
        return self._compute_objective_from_scratch()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_objective_from_scratch(self):
        """Compute ``F = sum_i ||Y[i]||^2 - 2 (YX*W).sum + ((W XX) * W).sum``."""
        W = self.bar_W
        W_XX = W @ self.matrix_XX
        squared = self.Y_sq_norms.sum() - 2.0 * torch.sum(self.matrix_YX * W) + torch.sum(W_XX * W)
        return squared

    def _refresh_w_star(self):
        """Recompute the continuous-relaxation optimal weight ``W^*``.

        ``W^*_{:, j} = bar_W_{:, j} + U_{:, j} / K_{jj}``.

        Columns with ``K_{jj} <= 0`` (degenerate, all-zero input column)
        are treated as having no continuous improvement available:
        ``W^*_{:, j} = bar_W_{:, j}``.
        """
        safe_diag = self.diag_K.clamp(min=1e-30)
        self.W_star = self.bar_W + self.U / safe_diag.unsqueeze(0)
        # Mask out columns with zero / negative diagonal.
        zero_diag = self.diag_K <= 0.0
        if zero_diag.any():
            self.W_star[:, zero_diag] = self.bar_W[:, zero_diag]

    def _select_top_k_columns(self, rho):
        """Return ``J_cand``: the top ``ceil(rho * m)`` columns by ``phi_j``.

        ``phi_j = ||U[:, j]||^2 / K[j, j]`` (continuous-relaxation
        improvement margin). The result is sorted by ``phi`` in
        descending order so the outer 1-pass examines the most
        promising columns first.
        """
        U_sq = (self.U**2).sum(dim=0)  # (m,)
        safe_diag = self.diag_K.clamp(min=1e-30)
        phi = U_sq / safe_diag  # (m,)
        # Columns with non-positive K_{jj} have no continuous improvement margin.
        phi = torch.where(self.diag_K > 0.0, phi, torch.zeros_like(phi))
        k_top = int(math.ceil(rho * self.dim_m))
        k_top = max(1, min(k_top, self.dim_m))
        # ``torch.topk`` returns indices in descending order by default,
        # which is exactly what we want for the outer 1-pass sweep.
        _, J_cand = torch.topk(phi, k_top, sorted=True)
        return J_cand  # (k_top,) int64

    def _precompute_tilde_table(self):
        """Precompute ``tilde_W`` / ``integer_z`` for every ``(t, j)`` pair.

        ``tilde_W_table[t, :, j]`` is the dequantized value of column ``j``
        when assigned to group ``t`` (i.e. quantize ``W_star[:, j]`` with
        ``scales[t]`` / ``zero_point[t]``):

        .. math::
            \\tilde{W}_{t,i,j}
            = s_{t,i} \\cdot \\mathrm{clip}\\!\\Big(
                \\mathrm{round}\\big(W^\\star_{i,j} / s_{t,i}\\big),
                L - z_{t,i},\\, U - z_{t,i}
            \\Big).

        Since ``W_star`` is fixed within a round and ``scales`` /
        ``zero_point`` are fixed for the entire ``optimize()`` call, this
        table can be built once per round (right after
        :meth:`_refresh_w_star`) and then *only indexed* inside the inner
        ``j`` loop. This replaces 2 quantization kernels per ``j`` with
        2 index ops, eliminating O(|J_cand|) kernel launches per round.

        The table is stored in **FP32** (not FP64) to halve the
        indexing read traffic that dominates the inner ``j`` loop on
        bandwidth-bound layers. This is safe because the table values
        are discrete (``scales[t] * integer``), so FP32 representation
        loses at most one ulp per entry relative to FP64 -- far below
        the quantization grid spacing. The FP64 re-computation in
        :meth:`_find_best_swap_for_j` recomputes the chosen candidate
        from ``scales`` / ``zero_point`` directly, so the table's FP32
        precision never propagates into the state.

        Memory: ``k * p * m * (4 + 1)`` bytes (FP32 ``tilde_W_table`` +
        int8 ``int_z_table``). For TinyLlama's ``down_proj``
        (k=44, p=2048, m=5632) this is ~2.5 GB. To keep peak memory
        bounded we materialise the table one group at a time (the
        intermediate ``(p, m)`` FP64 tensor is the only transient
        overhead).
        """
        # Allocate output tables (lazy: reuse if shape matches).
        if self.tilde_W_table is None or self.tilde_W_table.shape != (
            self.num_groups,
            self.dim_p,
            self.dim_m,
        ):
            self.tilde_W_table = torch.empty(
                (self.num_groups, self.dim_p, self.dim_m),
                dtype=torch.float32,
                device=self.device,
            )
            self.int_z_table = torch.empty(
                (self.num_groups, self.dim_p, self.dim_m),
                dtype=torch.int8,
                device=self.device,
            )

        for t in range(self.num_groups):
            s_t = self.solution.scales[t].unsqueeze(1)  # (p, 1) FP64
            z_t = self._zero_point_f[t].unsqueeze(1)  # (p, 1) FP64
            rounded = torch.round(self.W_star / s_t)  # (p, m) FP64
            integer_zf = torch.clamp(
                rounded,
                self.lower_bound - z_t,
                self.upper_bound - z_t,
            )  # (p, m) FP64
            # Cast to FP32 only when writing to the table.
            self.tilde_W_table[t] = (s_t * integer_zf).to(torch.float32)
            self.int_z_table[t] = integer_zf.to(torch.int8)

    def _find_best_swap_for_j(self, j, used_mask, min_swap_improvement=0.0):
        """Evaluate ``N(j)`` on the GPU and return the best swap if any.

        ``N(j) = {(t', j') : t' != t, j' in G_{t'}, used_mask[j'] is False}``.
        Both quantization directions (``j`` under each ``t'`` and ``j'``
        under the source group ``t``) are read out of the precomputed
        :attr:`tilde_W_table` / :attr:`int_z_table` -- no quantization
        kernels are launched inside this method.

        Numerical strategy (FP32 evaluation + FP64 commit):

        - The candidate-evaluation phase (``(p, q)`` reductions over all
          ``q`` neighbours) is performed in FP32. ``tilde_W_table`` is
          itself FP32 (see :meth:`_precompute_tilde_table`), so both
          the indexing read traffic and the in-kernel arithmetic are
          halved relative to FP64. This is the dominant cost on
          bandwidth-bound layers with large ``m``.
        - Once a best candidate ``j_prime`` has been picked by
          ``argmin``, the ``tilde_w`` column is **recomputed in FP64
          from `scales` / `zero_point` / `W_star`** for that single
          ``(p,)`` column. ``Delta F`` and the two ``delta_w`` columns
          are then computed in FP64. The recomputed FP64 value is
          checked against ``threshold_abs`` so that FP32 false-positives
          (cases where FP32 reports an improvement but FP64 says
          ``>= threshold``) are filtered out.
        - The FP64 ``delta_w`` / ``Delta F`` are used for the state
          update, so ``bar_W`` / ``U`` / ``F`` remain fully FP64 and
          do not accumulate FP32 round-off.

        Per-swap threshold:

        - ``min_swap_improvement > 0`` rejects swaps whose relative
          improvement ``-Delta F / F_curr`` does not exceed the
          threshold. This saves the per-swap commit cost (two
          ``outer`` rank-1 updates of ``U``) for noise-level swaps and
          also serves as the FP64 safety filter described above.

        Returns
        -------
        dict or None
            ``None`` when ``N(j)`` is empty or no candidate beats
            ``threshold_abs = -min_swap_improvement * F_curr``.
            Otherwise returns the FP64 data needed by
            :meth:`_apply_accepted_swap`.
        """
        t = int(self.column_to_group[j].item())

        # Build N(j) = {j' : column_to_group[j'] != t, not used}.
        valid = (self.column_to_group != t) & (~used_mask)
        valid[j] = False  # safety: j is in group t already, but make sure.
        js_prime = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if js_prime.numel() == 0:
            return None

        q = int(js_prime.numel())
        ts_prime = self.column_to_group[js_prime]  # (q,)

        # ------------------------------------------------------------------
        # FP32 evaluation phase for all q candidates.
        # ------------------------------------------------------------------
        # ``tilde_W_table`` is FP32 (see _precompute_tilde_table). Indexing
        # returns FP32 views with no extra cast.
        # j side: tilde_W_table[ts_prime, :, j] is (q, p), transpose -> (p, q).
        tilde_w_j_at_tp_f32 = self.tilde_W_table[ts_prime, :, j].T  # (p, q) FP32
        # j' side: tilde_W_table[t, :, js_prime] is (p, q).
        tilde_w_jp_at_t_f32 = self.tilde_W_table[t, :, js_prime]  # (p, q) FP32

        bar_j_col_f32 = self.bar_W[:, j].unsqueeze(1).to(torch.float32)  # (p, 1)
        bar_jp_cols_f32 = self.bar_W[:, js_prime].to(torch.float32)  # (p, q)

        delta_w_j_f32 = tilde_w_j_at_tp_f32 - bar_j_col_f32
        delta_w_jp_f32 = tilde_w_jp_at_t_f32 - bar_jp_cols_f32

        u_j_f32 = self.U[:, j].unsqueeze(1).to(torch.float32)  # (p, 1)
        u_jp_f32 = self.U[:, js_prime].to(torch.float32)  # (p, q)

        K_jj_f32 = float(self.matrix_XX[j, j].item())  # scalar
        K_jpjp_f32 = self.diag_K[js_prime].to(torch.float32)  # (q,)
        K_jjp_f32 = self.matrix_XX[j, js_prime].to(torch.float32)  # (q,)

        h_j_f32 = -2.0 * (delta_w_j_f32 * u_j_f32).sum(dim=0) + K_jj_f32 * (
            delta_w_j_f32**2
        ).sum(dim=0)
        h_jp_f32 = -2.0 * (delta_w_jp_f32 * u_jp_f32).sum(dim=0) + K_jpjp_f32 * (
            delta_w_jp_f32**2
        ).sum(dim=0)
        interaction_f32 = 2.0 * K_jjp_f32 * (delta_w_j_f32 * delta_w_jp_f32).sum(dim=0)
        delta_F_batch_f32 = h_j_f32 + h_jp_f32 + interaction_f32  # (q,) FP32

        self.num_evaluations += q

        # Threshold: relative -delta_F / F_curr must exceed min_swap_improvement.
        # Equivalently, delta_F must be strictly less than threshold_abs.
        F_curr = float(self.F.item())
        threshold_abs = -float(min_swap_improvement) * max(F_curr, 0.0)
        # When min_swap_improvement == 0, threshold_abs == 0 (= "accept any
        # strictly improving swap"), preserving the historical behaviour.

        idx_min = int(torch.argmin(delta_F_batch_f32).item())
        if float(delta_F_batch_f32[idx_min].item()) >= threshold_abs:
            return None

        # ------------------------------------------------------------------
        # FP64 re-computation for the chosen candidate (safety net + state).
        # ------------------------------------------------------------------
        # The FP32 table cannot give FP64 ``delta_w`` -- recompute the two
        # ``tilde_w`` columns in FP64 directly from W_star / scales /
        # zero_point. Cost is two (p,) round/clamp ops, negligible
        # compared with the j-loop traffic we saved.
        j_prime = int(js_prime[idx_min].item())
        t_prime = int(ts_prime[idx_min].item())

        # tilde_w_j at group t_prime (single column, FP64)
        s_tp = self.solution.scales[t_prime]  # (p,) FP64
        z_tp = self._zero_point_f[t_prime]  # (p,) FP64
        rounded_j_f64 = torch.round(self.W_star[:, j] / s_tp)
        int_zf_j_f64 = torch.clamp(
            rounded_j_f64,
            self.lower_bound - z_tp,
            self.upper_bound - z_tp,
        )
        tilde_w_j_f64 = s_tp * int_zf_j_f64  # (p,) FP64

        # tilde_w_jprime at group t (single column, FP64)
        s_t = self.solution.scales[t]  # (p,) FP64
        z_t = self._zero_point_f[t]  # (p,) FP64
        rounded_jp_f64 = torch.round(self.W_star[:, j_prime] / s_t)
        int_zf_jp_f64 = torch.clamp(
            rounded_jp_f64,
            self.lower_bound - z_t,
            self.upper_bound - z_t,
        )
        tilde_w_jp_f64 = s_t * int_zf_jp_f64  # (p,) FP64

        delta_w_j_f64 = (tilde_w_j_f64 - self.bar_W[:, j]).contiguous()  # (p,)
        delta_w_jp_f64 = (tilde_w_jp_f64 - self.bar_W[:, j_prime]).contiguous()

        u_j_col_f64 = self.U[:, j]
        u_jp_col_f64 = self.U[:, j_prime]
        K_jj_f64 = self.matrix_XX[j, j]
        K_jpjp_f64 = self.diag_K[j_prime]
        K_jjp_f64 = self.matrix_XX[j, j_prime]

        h_j_f64 = -2.0 * (delta_w_j_f64 * u_j_col_f64).sum() + K_jj_f64 * (delta_w_j_f64**2).sum()
        h_jp_f64 = (
            -2.0 * (delta_w_jp_f64 * u_jp_col_f64).sum() + K_jpjp_f64 * (delta_w_jp_f64**2).sum()
        )
        interaction_f64 = 2.0 * K_jjp_f64 * (delta_w_j_f64 * delta_w_jp_f64).sum()
        delta_F_f64 = h_j_f64 + h_jp_f64 + interaction_f64

        # Reject if the FP64 re-evaluation does not actually beat the threshold
        # (catches FP32 false positives near the boundary).
        if float(delta_F_f64.item()) >= threshold_abs:
            return None

        # Fetch int_z only for the chosen candidate (no full (p, q) read).
        new_z_j = self.int_z_table[t_prime, :, j].clone()  # (p,) int8
        new_z_jprime = self.int_z_table[t, :, j_prime].clone()  # (p,) int8

        return {
            "j_prime": j_prime,
            "t": t,
            "t_prime": t_prime,
            "delta_w_j": delta_w_j_f64,
            "delta_w_jprime": delta_w_jp_f64,
            "new_z_j": new_z_j,
            "new_z_jprime": new_z_jprime,
            "delta_F": delta_F_f64,
        }

    def _apply_accepted_swap(
        self,
        j,
        j_prime,
        t,
        t_prime,
        delta_w_j,
        delta_w_jprime,
        new_z_j,
        new_z_jprime,
        delta_F,
    ):
        """Commit an accepted swap and update all incrementally tracked state."""

        # Update U: U <- U - delta_w_j (K[j, :])^T - delta_w_jprime (K[jprime, :])^T
        K_j_row = self.matrix_XX[j]
        K_jprime_row = self.matrix_XX[j_prime]
        self.U -= torch.outer(delta_w_j, K_j_row)
        self.U -= torch.outer(delta_w_jprime, K_jprime_row)

        # Update bar_W
        self.bar_W[:, j] += delta_w_j
        self.bar_W[:, j_prime] += delta_w_jprime

        # Update F
        self.F = self.F + delta_F

        # Update integers_z + group membership
        d = int(self.column_to_slot[j].item())
        d_prime = int(self.column_to_slot[j_prime].item())
        # Slot d in group t previously held j; now holds j_prime
        self.solution.integers_z[t, :, d] = new_z_jprime
        self.groups[t, d] = j_prime
        self.column_to_group[j_prime] = t
        self.column_to_slot[j_prime] = d
        # Slot d_prime in group t_prime previously held j_prime; now holds j
        self.solution.integers_z[t_prime, :, d_prime] = new_z_j
        self.groups[t_prime, d_prime] = j
        self.column_to_group[j] = t_prime
        self.column_to_slot[j] = d_prime

        self.num_accepted += 1

    def _build_stats(self, rounds_run):
        return {
            "rounds_run": int(rounds_run),
            "num_evaluations": int(self.num_evaluations),
            "num_accepted": int(self.num_accepted),
            "F_initial": float(self.F_history[0]),
            "F_final": float(self.F.item()),
            "F_history": list(self.F_history),
        }


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,invalid-name
def optimize_groups_with_swap(
    solution,
    matrix_XX,
    matrix_YX,
    Y_sq_norms,
    dim_n,
    bits,
    symmetric,
    max_rounds=10,
    rho=0.3,
    max_accept=None,
    stop_if_no_improvement=True,
    min_round_improvement=5e-3,
    min_swap_improvement=0.0,
    log_level=1,
):
    """Run lightweight swap-based group optimization (functional wrapper).

    See :class:`GroupSwapOptimizer` and :meth:`GroupSwapOptimizer.optimize`
    for parameter semantics. The ``solution`` is mutated in place; its
    ``integers_z`` reflects the new group assignment after the call.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``solution`` : the mutated :class:`Solution`.
        - ``groups`` : updated column-to-group mapping, shape ``(k, d)``.
          ``groups.reshape(-1)`` is the column permutation to apply to
          ``matrix_W`` / ``matrix_XX`` / ``matrix_YX`` to restore the
          contiguous-group invariant expected by the rest of the
          JointQ pipeline.
        - ``bar_W`` : final dequantized weight matrix, shape ``(p, m)``.
        - ``stats`` : dict of optimization statistics
          (``rounds_run``, ``num_evaluations``, ``num_accepted``,
          ``F_initial``, ``F_final``, ``F_history``).
    """
    optimizer = GroupSwapOptimizer(
        solution=solution,
        matrix_XX=matrix_XX,
        matrix_YX=matrix_YX,
        Y_sq_norms=Y_sq_norms,
        dim_n=dim_n,
        bits=bits,
        symmetric=symmetric,
        log_level=log_level,
    )
    stats = optimizer.optimize(
        max_rounds=max_rounds,
        rho=rho,
        max_accept=max_accept,
        stop_if_no_improvement=stop_if_no_improvement,
        min_round_improvement=min_round_improvement,
        min_swap_improvement=min_swap_improvement,
    )
    return {
        "solution": solution,
        "groups": optimizer.get_groups(),
        "bar_W": optimizer.get_bar_W(),
        "stats": stats,
    }
