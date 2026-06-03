"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import time

import torch

from ..quantizer import Quantizer
from ..solution import Solution
from .local_search_advanced import LocalSearchSolverAdvanced


# pylint: disable=too-many-arguments,too-many-positional-arguments, too-few-public-methods
class QuantizerAdvanced(Quantizer):
    """

    min || Y - W_quant X^T ||_F^2 + || Lambda (W_quant - W_0) ||_F^2
    s.t.  W_quant = S * (A - B)

    where:
    - W_quant: the quantized weight matrix, shape (p, m)
    - W_0: the base weight matrix, shape (p, m)
    - S: the scale matrix
    - A: the assignment matrix
    - B: the bias matrix
    - Lambda:=Diag(sqrt(lambda_1),...,sqrt(lambda_p)) the regularization parameter, shape (p, p)
    - Regularization term to suppress excessive error guarantee
    """

    def __init__(
        self,
        vector_lambda,
        target_weight_matrix,
        **kwargs,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """
        See Quantizer for more details.
        Only additional parameters are described below.

        Parameters
        ----------
        vector_lambda : torch.Tensor
            Regularization weights per row, shape (p,).
            Each lambda_i controls the penalty strength for row i.
        target_weight_matrix : torch.Tensor
            The base weight matrix W_0, shape (p, m), dtype float64.
            Used in the penalty term ||Lambda (W_quant - W_0)||_F^2.
        **kwargs
            Passed to Quantizer.__init__.

        Notes
        -----
        _current_vector_lambda and _current_target_weight_matrix are
        row-selected, device-moved caches used only during scale optimization.
        They allow _compute_GtG_and_Gty to inject the regularization terms
        without changing the base optimization flow.
        """

        super().__init__(**kwargs)
        self.vector_lambda = vector_lambda
        self.target_weight_matrix = target_weight_matrix
        self._current_vector_lambda = None
        self._current_target_weight_matrix = None
        self._local_search_fn = LocalSearchSolverAdvanced.solve
        # matrix_L (shape (num_groups, p, d)):
        # - B := Diag(lambda) tilde_W , shape (p, m), where tilde_W is target_weight_matrix
        # - matrix_L[t] is the sub-matrix of B corresponding to the t-th group
        matrix_B = self.vector_lambda[:, None] * self.target_weight_matrix  # (p, m)
        self._matrix_L = (
            matrix_B.reshape(self.dim_p, self.num_groups, self.group_size)
            .permute(1, 0, 2)
            .contiguous()
        )  # (num_groups, p, d)
        del matrix_B

    def display_log(self, solution, phase, ignore_log_level=False):
        """Display the log."""

        if self.log_level <= 1 and not ignore_log_level:
            return

        spaces = max(0, 15 - len(phase)) * " "
        print(
            f"<{self.device}>: "
            f"[{self._get_time():.2f}s] {phase}: {spaces}"
            f"error = {solution.squared_error:.3e}, "
            f"MSE = {solution.mean_squared_error:.3e}, "
            f"penalty = {solution.penalty:.3e}, "
            f"total = {solution.squared_error + solution.penalty:.3e}"
        )

    def _optimize_scales(self, solution, update_flags=None):
        """Optimize scales with penalty terms.

        This method injects the per-row regularization terms into the normal
        scale optimization flow by caching row-selected tensors and then
        delegating to the base implementation.

        For each row i, the scale optimization solves:
            min_{s_i} ||y_i - G_i s_i||_2^2
                    + lambda_i * sum_t ||tilde_w[i][t] - s_t z[t][i]||_2^2

        The closed-form uses:
            (G_i^T G_i + lambda_i D_i) s_i = (G_i^T y_i + lambda_i u_i),
        where D_i = diag(||z[t][i]||_2^2) and u_i[t] = z[t][i]^T tilde_w[i][t].

        Parameters
        ----------
        solution : SolutionLambda
            Current solution holding integer assignments and scales.
        update_flags : torch.Tensor or None
            Bool mask of rows to update. None updates all rows.
        """
        if update_flags is None:
            vector_lambda = self.vector_lambda
            target_weight_matrix = self.target_weight_matrix
        else:
            vector_lambda = self.vector_lambda[update_flags]
            target_weight_matrix = self.target_weight_matrix[update_flags]

        self._current_vector_lambda = vector_lambda.to(self.device)
        self._current_target_weight_matrix = target_weight_matrix.to(self.device)
        try:
            return super()._optimize_scales(solution, update_flags=update_flags)
        finally:
            self._current_vector_lambda = None
            self._current_target_weight_matrix = None

    def _compute_GtG_and_Gty(self, integers_z, sub_matrices_YX):
        """Compute G^T G and G^T y with regularization terms.

        Extends the base computation by adding:
        - lambda_i D_i to the diagonal of G_i^T G_i
        - lambda_i u_i to G_i^T y_i

        This method expects _current_vector_lambda and
        _current_target_weight_matrix to be set by _optimize_scales.

        Parameters
        ----------
        integers_z : torch.Tensor
            Integer assignments, shape (num_groups, num_rows, group_size).
        sub_matrices_YX : torch.Tensor
            Precomputed yX blocks, shape (num_rows, num_groups, group_size).
        """
        GG_all, GY_all = super()._compute_GtG_and_Gty(integers_z, sub_matrices_YX)

        assert self._current_vector_lambda is not None
        assert self._current_target_weight_matrix is not None

        num_rows = integers_z.shape[1]
        vector_lambda = self._current_vector_lambda
        assert vector_lambda.shape[0] == num_rows

        integers_z_f = integers_z.to(self.dtype)
        # D_i: diag(||z_t||_2^2) for each row i
        z_norm2 = torch.sum(integers_z_f * integers_z_f, dim=2).T  # (num_rows, num_groups)

        diag_indices = torch.arange(self.num_groups, device=GG_all.device)
        # G^T G <- G^T G + lambda_i D_i (diagonal update per row)
        GG_all[:, diag_indices, diag_indices] += vector_lambda[:, None] * z_norm2

        target_weight_grouped = self._current_target_weight_matrix.reshape(
            num_rows, self.num_groups, self.group_size
        )
        # u_i[t] = z[t][i]^T * tilde_w[i][t] (per row/group)
        matrix_U = torch.sum(
            integers_z_f.permute(1, 0, 2) * target_weight_grouped,
            dim=2,
        )
        # G^T y <- G^T y + lambda_i u_i
        GY_all += vector_lambda[:, None] * matrix_U

        return GG_all, GY_all

    def quantize(self, solution):
        """Quantize the weight matrix.

        Perform one round of local search for each group and optimize scales.

        """

        # Do not optimize scale vectors if update count is 0
        updated_counters = torch.zeros(self.dim_p, dtype=torch.int32, device=self.device)

        # Improve each group, one round only
        for t in range(self.num_groups):
            updated = self._group_local_search_all(solution, t)
            if updated.any():
                updated_counters += updated
            self.display_log(solution, f"LS <{t + 1}> (#updated = {updated.sum()})")

        # Optimize scales only for rows with positive update count
        update_flags = updated_counters > 0
        result = self._optimize_scales(solution, update_flags)
        self.display_log(solution, f"OptS (#updated = {result}/{update_flags.sum()})")
        torch.cuda.empty_cache()

        return solution, update_flags

    def update_matrix_L(self, changed_mask):
        """Update _matrix_L for rows where lambda has changed.

        Recompute _matrix_L[t, i, :] = lambda_i * target_weight_matrix[i, t*d:(t+1)*d]
        only for rows where changed_mask is True.

        Parameters
        ----------
        changed_mask : torch.Tensor
            Boolean mask indicating which rows have changed lambda, shape (p,).
        """
        if not changed_mask.any():
            return
        changed_B = (
            self.vector_lambda[changed_mask, None] * self.target_weight_matrix[changed_mask]
        )  # (num_changed, m)
        self._matrix_L[:, changed_mask, :] = changed_B.reshape(
            -1, self.num_groups, self.group_size
        ).permute(1, 0, 2)

    def _build_local_search_kwargs(self, solution, group_index):
        """Build keyword arguments for the local search function.

        Adds QuantizerAdvanced-specific arguments in addition to the parent class arguments.

        Parameters
        ----------
        solution : SolutionLambda
            The current solution.
        group_index : int
            The index of the group to optimize.

        Returns
        -------
        dict
            Keyword arguments for LocalSearchSolverAdvanced.solve.
        """
        kwargs = super()._build_local_search_kwargs(solution, group_index)
        # kwargs["verbose"] = True
        kwargs["vector_lambda"] = self.vector_lambda
        kwargs["matrix_L"] = self._matrix_L[group_index]
        kwargs["matrix_H"] = kwargs["matrix_H"] + self._matrix_L[group_index]
        return kwargs


class SolutionLambda(Solution):
    """Solution class with lambda values for regularization.

    Extends the Solution class to include lambda values used in the
    regularized quantization problem:
    min || Y - W_quant X^T ||_F^2 + || Lambda (W_quant - W_0) ||_F^2

    where:
    - Lambda:=Diag(sqrt(lambda_1),...,sqrt(lambda_p)) the regularization parameter,shape (p, p)
    - Regularization term to suppress excessive error guarantee

    See QuantizerAdvanced for more details.

    """

    def __init__(self, vector_lambda=None, target_weight_matrix=None, **kwargs):
        """Initialize the solution with lambda values.

        Parameters
        ----------
        vector_lambda : torch.Tensor, optional
            The lambda values for regularization, shape (p,).
        **kwargs
            Additional arguments passed to the parent Solution class.
        """
        super().__init__(**kwargs)
        self.vector_lambda = vector_lambda
        self.target_weight_matrix = target_weight_matrix
        self.penalty = None
        self.penalties_per_row = None

    def clean(self):
        """Clean up the solution to free memory."""
        super().clean()
        if self.penalties_per_row is not None:
            del self.penalties_per_row
            self.penalties_per_row = None

    def update_penalties(self, changed_mask):
        """Update penalties for rows where lambda has changed.

        Recompute penalties_per_row and penalty for rows where only lambda
        has changed (weights remain unchanged).

        Parameters
        ----------
        changed_mask : torch.Tensor
            Boolean mask indicating which rows have changed lambda, shape (p,).
        """
        if not changed_mask.any():
            return
        hat_W = self.get_dequantized_weight_matrix()
        diff = hat_W[changed_mask] - self.target_weight_matrix[changed_mask]
        row_sq_norms = torch.sum(diff**2, dim=1)
        del diff
        self.penalties_per_row[changed_mask] = self.vector_lambda[changed_mask] * row_sq_norms
        self.penalty = torch.sum(self.penalties_per_row)
        del hat_W

    def compute_objective_value(
        self, matrix_XX, matrix_YX, Y_sq_norms, dim_n, dequantized_weight_matrix=None
    ):
        """Compute the objective value including the regularization penalty.

        Term 1: || Y - W_quant X^T ||_F^2 (same as parent class)
        Term 2: || Lambda (W_quant - W_0) ||_F^2
             = sum_i lambda_i * || w_quant_i - w_0_i ||_2^2

        Parameters
        ----------
        matrix_XX : torch.Tensor
            X^T X, shape (m, m).
        matrix_YX : torch.Tensor
            Y @ X, shape (p, m).
        Y_sq_norms : torch.Tensor
            ||Y[i]||^2, shape (p,).
        dim_n : int
            Number of samples n.
        dequantized_weight_matrix : torch.Tensor, optional
            Dequantized weight matrix, shape (p, m). If None, computed internally.
        """
        # Compute hat_W once and reuse for both terms
        hat_W = dequantized_weight_matrix
        if hat_W is None:
            hat_W = self.get_dequantized_weight_matrix()

        # Term 1: Call parent class compute_objective_value
        super().compute_objective_value(
            matrix_XX=matrix_XX,
            matrix_YX=matrix_YX,
            Y_sq_norms=Y_sq_norms,
            dim_n=dim_n,
            dequantized_weight_matrix=hat_W,
        )

        # Term 2: || Lambda (W_quant - W_0) ||_F^2
        # W_quant: (p, m), W_0 (target_weight_matrix): (p, m)
        # vector_lambda: (p,)
        # penalty = sum_i lambda_i * || w_quant_i - w_0_i ||_2^2
        diff = hat_W - self.target_weight_matrix
        del hat_W
        diff.pow_(2)
        row_squared_norms = torch.sum(diff, dim=1)  # (p,)
        del diff
        self.penalties_per_row = self.vector_lambda * row_squared_norms  # (p,)
        self.penalty = torch.sum(self.penalties_per_row)

    def try_update_group(
        self,
        group_index,
        new_integers_z,
        new_scales,
        matrix_XX,
        matrix_YX,
        Y_sq_norms,
        dim_n,
    ):
        """Update a group's integers and scales (with penalty term).

        After calling the parent class update, recompute penalty terms for changed rows.
        penalty_i = lambda_i * ||hat_W_i - W_0_i||^2
        """
        # Parent class update (integers_z, scales, errors)
        # Receive hat_W_changed for penalty computation with return_hat_W=True
        changed, hat_W_changed = super().try_update_group(
            group_index,
            new_integers_z,
            new_scales,
            matrix_XX,
            matrix_YX,
            Y_sq_norms,
            dim_n,
            return_hat_W=True,
        )

        if changed.any():
            # Recompute penalties for changed rows (hat_W_changed received from parent)
            diff = hat_W_changed - self.target_weight_matrix[changed]
            row_sq_norms = torch.sum(diff**2, dim=1)
            self.penalties_per_row[changed] = self.vector_lambda[changed] * row_sq_norms
            self.penalty = torch.sum(self.penalties_per_row)

        return changed

    def try_update_scales(
        self,
        new_scales,
        matrix_XX,
        matrix_YX,
        Y_sq_norms,
        dim_n,
        epsilon,
        update_flags=None,
    ):
        """Try to update scales with a new solution (with penalty term).

        Same structure as the parent class try_update_scales, but the improvement
        criterion includes the penalty term lambda_i * ||w_quant_i - w_0_i||^2.

        Parameters
        ----------
        new_scales: torch.Tensor
            The new scales to update.
            update_flags=None: shape (p, num_groups).
            When update_flags is specified: shape (sum(update_flags), num_groups).
        matrix_XX : torch.Tensor
            X^T X, shape (m, m).
        matrix_YX : torch.Tensor
            Y @ X, shape (p, m).
        Y_sq_norms : torch.Tensor
            ||Y[i]||^2, shape (p,).
        dim_n : int
            Number of rows in the input matrix, n.
        epsilon: float
            The epsilon for numerical error.
        update_flags: torch.Tensor or None
            The flags to indicate which rows to optimize, shape (p,).
            If None, all rows are optimized.

        Returns
        -------
        int
            The number of improved solutions.

        """

        new_scales_T = new_scales.T.contiguous()

        # Select target rows
        if update_flags is None:
            integers_z = self.integers_z
            current_YX = matrix_YX
            current_Y_sq_norms = Y_sq_norms
            current_squared_errors = self.squared_errors
            current_penalties = self.penalties_per_row
            target_weight = self.target_weight_matrix
            vec_lambda = self.vector_lambda
        else:
            integers_z = self.integers_z[:, update_flags, :]
            current_YX = matrix_YX[update_flags]
            current_Y_sq_norms = Y_sq_norms[update_flags]
            current_squared_errors = self.squared_errors[update_flags]
            current_penalties = self.penalties_per_row[update_flags]
            target_weight = self.target_weight_matrix[update_flags]
            vec_lambda = self.vector_lambda[update_flags]

        # Compute squared errors with new hat_W
        new_hatW = self.get_dequantized_weight_matrix(new_scales_T, integers_z)
        term1 = current_Y_sq_norms
        term2 = 2.0 * (current_YX * new_hatW).sum(dim=1)
        term3 = ((new_hatW @ matrix_XX) * new_hatW).sum(dim=1)
        new_squared_errors = term1 - term2 + term3
        new_mean_squared_errors = new_squared_errors / dim_n

        # Compute penalty term: lambda_i * ||w_quant_i - w_0_i||^2
        diff = new_hatW - target_weight
        del new_hatW
        diff.pow_(2)
        new_penalties = vec_lambda * torch.sum(diff, dim=1)
        del diff

        # Improvement criterion: compare total of squared error + penalty
        new_total = new_squared_errors + new_penalties
        current_total = current_squared_errors + current_penalties
        improved = new_total + epsilon < current_total
        improved_count = torch.sum(improved)

        # Early return if no improvement in partial-row mode
        if update_flags is not None and improved_count == 0:
            return 0

        # Update (parent class common update + penalty update)
        self._apply_improved_scales(
            update_flags,
            improved,
            improved_count,
            new_scales_T,
            new_squared_errors,
            new_mean_squared_errors,
            new_penalties=new_penalties,
        )

        return improved_count

    def _apply_improved_scales(
        self,
        update_flags,
        improved,
        improved_count,
        new_scales_T,
        new_squared_errors,
        new_mean_squared_errors,
        *,
        new_penalties,
    ):
        """Apply the improved scales including penalty updates.

        Call the parent class common update and then additionally update penalty terms.
        """

        # Parent class common update (scales, errors)
        improved_indices = super()._apply_improved_scales(
            update_flags,
            improved,
            improved_count,
            new_scales_T,
            new_squared_errors,
            new_mean_squared_errors,
        )

        if improved_indices is None:
            # All-row mode
            if improved_count == improved.shape[0]:
                # All improved: bulk assign penalties
                self.penalties_per_row = new_penalties
                self.penalty = torch.sum(new_penalties)
            elif improved_count > 0:
                # Partial improvement: update penalties for improved rows
                self.penalties_per_row[improved] = new_penalties[improved]
                self.penalty = torch.sum(self.penalties_per_row)
            return None

        # Partial-row mode: in-place update penalties for improved rows
        self.penalties_per_row[improved_indices] = new_penalties[improved]
        self.penalty = torch.sum(self.penalties_per_row)
        return improved_indices
