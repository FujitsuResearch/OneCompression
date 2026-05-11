"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import time
import torch

from .clip import clip
from .solution import Solution
from .local_search import LocalSearchSolver


# pylint: disable=too-many-arguments,too-many-positional-arguments, too-few-public-methods
class Quantizer:
    """Class to find hat{W} that minimizes ||target_matrix - hat{W} X^T||_F^2

    - Supports both symmetric and asymmetric quantization
    - Decomposes this optimization problem into $p$ row-wise subproblems and solves them in parallel using GPU
    - Partitions $X = (X_1, .. , X_g)$ ($X_t in R^{n times d}$)

    min_{s, a, b} |y - sum_{t=1}^g s_t X_t (a_t - b_t 1_d)|^2_2,
    s.t.  a_t in Q^d (t=1, .., g),
          b_t in Q (t=1, .., g),
          s_t in R (t=1, .., g),

    where y is the target vector (row of target_matrix),
    Q = {l, l+1, ..., u-1, u},
    and l and u are the lower and upper bounds of the quantization.

    For example, for 4-bit symmetric quantization, l = -8, u = 7.
    For 4-bit asymmetric quantization, l = 0, u = 15.

    Attributes
    ----------
    #TODO: Add attributes

    """

    def __init__(
        self,
        matrix_XX,
        sub_matrices_XX,
        matrix_YX,
        sub_matrices_YX,
        Y_sq_norms,
        dim_n,
        bits=4,
        symmetric=False,
        epsilon=1e-8,
        early_stopping_ratio=0.1,
        log_level=2,  # 0: none, 1: minimal, 2: detailed, 3: debug
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Initialize the Quantizer.

        Parameters
        ----------
        matrix_XX : torch.Tensor
            X^T X, shape (m, m), on device.
        sub_matrices_XX : torch.Tensor
            Block partition of X^T X, shape (num_groups, num_groups, group_size, group_size),
            on device.
        matrix_YX : torch.Tensor
            Y @ X, shape (p, m), on device.
        sub_matrices_YX : torch.Tensor
            Group partition of Y @ X, shape (p, num_groups, group_size), on device.
        Y_sq_norms : torch.Tensor
            Squared norms of each row of the target matrix, shape (p,), on device.
        dim_n : int
            Number of rows n of the input matrix.
        bits : int, optional
            The quantization bits. Default is 4.
        symmetric : bool, optional
            Whether to use symmetric quantization. Default is False.
        epsilon : float, optional
            The epsilon for numerical stability. Default is 1e-8.
        early_stopping_ratio : float, optional
            The ratio for the early stopping. Default is 0.1.
        log_level : int, optional
            The log level. Default is 2.
        """

        # check the arguments
        if not isinstance(bits, int) or bits <= 0 or bits > 4:
            raise ValueError("bits must be a positive integer between 1 and 4.")

        if not isinstance(symmetric, bool):
            raise ValueError("symmetric must be a boolean.")

        self.device = matrix_XX.device

        self.bits = bits
        self.symmetric = symmetric

        self.dim_p = matrix_YX.shape[0]
        self.dim_n = dim_n
        self.num_groups = sub_matrices_XX.shape[0]
        self.group_size = sub_matrices_XX.shape[2]
        self.dim_m = self.group_size * self.num_groups

        if symmetric:
            self.lower_bound = -(2 ** (bits - 1))
            self.upper_bound = 2 ** (bits - 1) - 1
        else:
            self.lower_bound = 0
            self.upper_bound = 2**bits - 1

        self.begin_time = None
        self.dtype = torch.float64
        self.modified_lower_bounds = None
        self.modified_upper_bounds = None
        self.matrix_XX = matrix_XX
        self.sub_matrices_XX = sub_matrices_XX
        self.matrix_YX = matrix_YX
        self.sub_matrices_YX = sub_matrices_YX
        self.Y_sq_norms = Y_sq_norms
        self.objective_args = {
            "matrix_XX": matrix_XX,
            "matrix_YX": matrix_YX,
            "Y_sq_norms": Y_sq_norms,
            "dim_n": dim_n,
        }
        self.epsilon = epsilon
        self.early_stopping_ratio = early_stopping_ratio
        self.log_level = log_level
        self._local_search_fn = LocalSearchSolver.solve

        # check the dimensions and dtype of matrix_XX
        if matrix_XX.shape != (self.dim_m, self.dim_m):
            raise ValueError(f"matrix_XX must have shape ({self.dim_m}, {self.dim_m}).")
        if matrix_XX.dtype != self.dtype:
            raise ValueError(f"matrix_XX must have dtype {self.dtype}.")

        # check the dimensions and dtype of sub_matrices_XX
        dim = (
            self.num_groups,
            self.num_groups,
            self.group_size,
            self.group_size,
        )
        if sub_matrices_XX.shape != dim:
            raise ValueError(f"sub_matrices_XX must have shape {dim}.")
        if sub_matrices_XX.dtype != self.dtype:
            raise ValueError(f"sub_matrices_XX must have dtype {self.dtype}.")

        # check the dimensions and dtype of matrix_YX
        if matrix_YX.shape != (self.dim_p, self.dim_m):
            raise ValueError(f"matrix_YX must have shape ({self.dim_p}, {self.dim_m}).")
        if matrix_YX.dtype != self.dtype:
            raise ValueError(f"matrix_YX must have dtype {self.dtype}.")

        # check the dimensions and dtype of sub_matrices_YX
        if sub_matrices_YX.shape != (self.dim_p, self.num_groups, self.group_size):
            raise ValueError(
                f"sub_matrices_YX must have shape ({self.dim_p}, {self.num_groups}, {self.group_size})."
            )
        if sub_matrices_YX.dtype != self.dtype:
            raise ValueError(f"sub_matrices_YX must have dtype {self.dtype}.")

        # check the dimensions and dtype of Y_sq_norms
        if Y_sq_norms.shape != (self.dim_p,):
            raise ValueError(f"Y_sq_norms must have shape ({self.dim_p},).")
        if Y_sq_norms.dtype != self.dtype:
            raise ValueError(f"Y_sq_norms must have dtype {self.dtype}.")

        # Max safe scale value to avoid FP16 overflow in dequantized weights.
        # dequantized = scale * assignment, so scale must satisfy:
        #   |scale| * max(|assignment|) < FP16_MAX (65504)
        self.scale_threshold = torch.finfo(torch.float16).max / max(abs(self.lower_bound), abs(self.upper_bound))

    def set_begin_time(self):
        """Set the begin time."""

        self.begin_time = time.time()

    def initialize_solution_clip_optimize(self, matrix_W):
        """Initialize the solution.

        Parameters
        ----------
        matrix_W : torch.Tensor
            The initial real-valued weight matrix (tilde_W), shape (p, m).
            - classic mode: matrix_W is the original weight matrix W
            - Y mode: matrix_W is computed via least squares from target_matrix Y

        Algorithm:
        1. Clipping
            1-1. Split W into groups of size group_size along the row vector direction
            1-2. Perform quantization via the clip function for each group
        2. Minimize the quantization error for each group

        """

        if self.log_level >= 2:
            print(f"<{self.device}>: <Quantize (Clipping and Optimization)>")

        if matrix_W.shape != (self.dim_p, self.dim_m):
            raise ValueError(f"matrix_W must have shape ({self.dim_p}, {self.dim_m}).")
        if matrix_W.dtype != self.dtype:
            raise ValueError(f"matrix_W must have dtype {self.dtype}.")
        if matrix_W.device != self.device:
            raise ValueError(f"matrix_W must have device {self.device}.")

        # clipping
        scales, assignment, zero_point = clip(
            matrix_W,
            group_size=self.group_size,
            symmetric=self.symmetric,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )

        # prepare the current solution
        solution = Solution(scales, assignment, zero_point)
        del scales, assignment, zero_point
        solution.compute_objective_value(**self.objective_args)
        self.display_log(solution, "Clip")

        # modified lower and upper bounds
        self.set_modified_lower_and_upper_bounds(solution)

        # execute local search to improve quantization error for each group
        for t in range(self.num_groups):
            sub_matrix_W = matrix_W[:, t * self.group_size : (t + 1) * self.group_size]
            sub_matrix_XX_tt = self.sub_matrices_XX[t, t]
            # H = Y X_t where Y = W_t X_t^T
            # H = W_t (X_t^T X_t) = W_t XX[t,t], shape: (p, group_size)
            matrix_H = sub_matrix_W @ sub_matrix_XX_tt
            solution.integers_z[t], solution.scales[t] = LocalSearchSolver.solve(
                matrix_H=matrix_H,
                lower_bounds=self.modified_lower_bounds[t],
                upper_bounds=self.modified_upper_bounds[t],
                initial_solution=solution.integers_z[t],
                matrix_R=sub_matrix_XX_tt,
                max_iterations=10 * self.group_size,
                epsilon=self.epsilon,
                verbose=False,
            )

        solution.compute_objective_value(**self.objective_args)
        self.display_log(solution, "Init-LS")

        # Reset
        self.modified_lower_bounds = None
        self.modified_upper_bounds = None

        # optimize all scales
        result = self._optimize_scales(solution)
        self.display_log(solution, f"OptS ({result}/{self.dim_p})")
        if self.log_level == 1:
            self.display_log(solution, "Clip-Optimize", ignore_log_level=True)

        return solution

    def initialize_solution_clip_optimize_ep(self, matrix_W):  # pylint: disable=too-many-locals
        """Initialize the solution with error propagation.

        Parameters
        ----------
        matrix_W : torch.Tensor
            The initial real-valued weight matrix (tilde_W), shape (p, m).
            - classic mode: matrix_W is the original weight matrix W
            - Y mode: matrix_W is computed via least squares from target_matrix Y

        Algorithm:
        Initialize with tilde_W = matrix_W
        for t in range(num_groups):
          1. Clipping
          2. Optimization
          3. Error propagation
        Optimize all scales

        """

        if self.log_level >= 2:
            print(
                f"<{self.device}>: <Quantize (Clipping and Optimization with Error Propagation)>"
            )

        if matrix_W.shape != (self.dim_p, self.dim_m):
            raise ValueError(f"matrix_W must have shape ({self.dim_p}, {self.dim_m}).")
        if matrix_W.dtype != self.dtype:
            raise ValueError(f"matrix_W must have dtype {self.dtype}.")
        if matrix_W.device != self.device:
            raise ValueError(f"matrix_W must have device {self.device}.")

        # initialize tilde_W and solution
        tilde_W = matrix_W.clone()
        solution = Solution()
        solution.scales = torch.zeros(
            self.num_groups, self.dim_p, dtype=self.dtype, device=self.device
        )
        solution.zero_point = torch.zeros(
            self.num_groups, self.dim_p, dtype=torch.int8, device=self.device
        )
        solution.integers_z = torch.zeros(
            self.num_groups,
            self.dim_p,
            self.group_size,
            dtype=torch.int8,
            device=self.device,
        )

        for t in range(self.num_groups):
            sub_tilde_W_t = tilde_W[:, t * self.group_size : (t + 1) * self.group_size]
            # step1: Clipping
            scales, assignment, zero_point = clip(
                sub_tilde_W_t,
                group_size=self.group_size,
                symmetric=self.symmetric,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
            )
            # print(scales.T)
            solution.scales[t] = scales.T.contiguous()
            solution.zero_point[t] = zero_point.T.contiguous()
            solution.integers_z[t] = (
                (assignment - zero_point.unsqueeze(2)).transpose(0, 1).contiguous()
            )
            del scales, assignment, zero_point

            # step2: Optimization
            # Y = target_matrix - tilde_W @ X^T + sub_tilde_W_t @ X_t^T
            # H = Y X_t = sub_tilde_W_t @ XX[t,t] + (YX)_t - tilde_W @ XX[:, t_slice]
            t_start = t * self.group_size
            t_end = (t + 1) * self.group_size
            matrix_H = sub_tilde_W_t @ self.sub_matrices_XX[t, t]
            matrix_H += self.sub_matrices_YX[:, t, :]
            matrix_H -= tilde_W @ self.matrix_XX[:, t_start:t_end]
            solution.integers_z[t], solution.scales[t] = LocalSearchSolver.solve(
                matrix_H=matrix_H,
                lower_bounds=self.lower_bound - solution.zero_point[t],
                upper_bounds=self.upper_bound - solution.zero_point[t],
                initial_solution=solution.integers_z[t],
                matrix_R=self.sub_matrices_XX[t, t],
                max_iterations=10 * self.group_size,
                epsilon=self.epsilon,
                verbose=False,
            )
            # Update tilde_W
            tilde_W[:, t * self.group_size : (t + 1) * self.group_size] = (
                solution.scales[t][:, None] * solution.integers_z[t]
            )

            if self.log_level >= 3:
                print(f"(Debug, group {t}) [Opt] error: {self.__current_error(tilde_W):.4e}")

            # Skip step3 on the last iteration
            if t == self.num_groups - 1:
                break
            # step3: Error propagation
            # done_W := tilde_W[:, :t_end] (already determined groups)
            # rem_W := tilde_W[:, t_end:] (remaining groups)
            # Find rem_W that minimizes ||Y - done_W @ done_X^T - rem_W @ rem_X^T||_F^2
            # Normal equation: (rem_X^T rem_X) rem_W^T = rem_X^T Y^T
            #   rem_X^T rem_X = XX[t_end:, t_end:]
            #   rem_X^T Y^T = (YX)[:, t_end:]^T - XX[t_end:, :t_end] @ done_W^T
            t_end = (t + 1) * self.group_size
            done_W = tilde_W[:, :t_end]
            XtX = self.matrix_XX[t_end:, t_end:]
            XtY = self.matrix_YX[:, t_end:].T - self.matrix_XX[t_end:, :t_end] @ done_W.T
            del done_W

            try:
                # Normally use solve (fast)
                tilde_W[:, t_end:] = torch.linalg.solve(XtX, XtY).T
            except torch._C._LinAlgError:
                if self.log_level >= 3:
                    print(f"(Debug, group {t}) [EP] Singular matrix detected")

                # Use pseudo-inverse for singular matrices (robust)
                # rem_W^T = pinv(XtX) @ XtY
                if self.log_level >= 3:
                    print("  Using pseudo-inverse (pinv) with rcond=1e-6")

                XtX_pinv = torch.linalg.pinv(XtX, rcond=1e-6)
                tilde_W[:, t_end:] = (XtX_pinv @ XtY).T
                del XtX_pinv

            del XtX, XtY

            if self.log_level >= 3:
                print(f"(Debug, group {t}) [EP] error: {self.__current_error(tilde_W):.4e}")

        del tilde_W

        solution.compute_objective_value(**self.objective_args)
        self.display_log(solution, "Init-EP")

        # optimize all scales
        result = self._optimize_scales(solution)
        self.display_log(solution, f"OptS ({result}/{self.dim_p})")
        if self.log_level == 1:
            self.display_log(solution, "Clip-Optimize-EP", ignore_log_level=True)

        return solution

    def __current_error(self, tilde_W):
        """Compute the current error.
        Compute ||Y - tilde_W @ X^T||_F^2 using precomputed matrices.
        Debug function for initialize_solution_clip_optimize_ep.
        """
        # Sum of ||Y_i||^2 - 2(YX)_i · W_i + W_i (X^TX) W_i^T
        return (
            self.Y_sq_norms.sum()
            - 2 * (self.matrix_YX * tilde_W).sum()
            + ((tilde_W @ self.matrix_XX) * tilde_W).sum()
        )

    def set_modified_lower_and_upper_bounds(self, solution):
        """Set the modified lower and upper bounds."""
        self.modified_lower_bounds = self.lower_bound - solution.zero_point
        self.modified_upper_bounds = self.upper_bound - solution.zero_point

    def quantize(self, solution):
        """Quantize the weight matrix.

        Algorithm:
        TODO: Write the algorithm

        """

        if self.log_level >= 2:
            print(f"<{self.device}>: <Quantize (Local Search)>")

        if self.log_level >= 1:
            self.display_log(solution, "Given Solution", ignore_log_level=True)

        # Flags to manage active optimization problems; updates stop when False
        active_flags = torch.ones(self.dim_p, dtype=torch.bool, device=self.device)
        active_count = self.dim_p

        iteration_count = 0
        # Improve via local search
        while True:
            if self.log_level >= 2:
                print(
                    f"<{self.device}>: "
                    f"=== Iteration {iteration_count} <#active = {active_count}> ==="
                )
            counters = torch.zeros(self.dim_p).to(self.device)
            pre_mean_squared_errors = solution.mean_squared_errors.clone()

            for t in range(self.num_groups):
                # Improve for each group
                updated = self._group_local_search_all(solution, t)
                if updated.any():
                    counters += updated
                self.display_log(solution, f"LS <{t + 1}> (#updated = {updated.sum()})")
                # TODO: Consider improving only active optimization problems. Implementation is complex.

            # For each optimization problem,
            # optimize the scale vector if the update count is >60% or <20%
            update_flags = (
                (counters > self.num_groups * 0.6) + (counters < self.num_groups * 0.2)
            ) & active_flags
            result = self._optimize_scales(solution, update_flags)
            self.display_log(solution, f"OptS (#updated = {result}/{update_flags.sum()})")

            if self.log_level == 1:
                self.display_log(solution, f"Iter={iteration_count}", ignore_log_level=True)

            # Set active_flags[i] to True if mean_squared_errors[i] decreased
            active_flags = solution.mean_squared_errors + self.epsilon < pre_mean_squared_errors
            active_count = active_flags.sum()

            iteration_count += 1

            # Terminate if fewer than 10% of optimization problems are active
            if active_count < self.dim_p * self.early_stopping_ratio:
                break

            torch.cuda.empty_cache()

        self.display_log(solution, "Results")

        return solution

    def _optimize_scales(self, solution, update_flags=None):
        """Optimize scales.

        When update_flags is None, optimize scales for all rows.
        When update_flags is specified, optimize only rows where the flag is True.

        min_{s_i} |y_i - G_i s_i|^2_2,
        s.t. s_i in R^k,

        where y_i = X w_i, G_i = (X_1 z[1][i], ..., X_k z[k][i]),
        z[t][i] is the i-th row of the t-th group of the solution,
        k = num_groups.

        Computing G_i (shape: (n, k)) and y_i (shape: (n)) and batch processing
        would require a large amount of memory.
        Therefore, we compute G_i^T G_i and G_i^T y_i and batch-solve the linear equations.

        step1. Compute each G_i^T G_i
        step2. Compute each G_i^T y_i
        step3. Solve the linear equations in batch (G_i^T G_i) s_i = G_i^T y_i
        step4. Recompute objective values and update with new scales if improved for each i.

        Parameters
        ----------
        solution: Solution
            The solution to optimize.
        update_flags: torch.Tensor or None
            The flags to indicate which rows to optimize, shape (p,).
            When None, all rows are optimized.

        Returns
        -------
        int
            The number of improved solutions.

        """

        # Early return when update_flags is specified
        if update_flags is not None:
            # If all True, recursively call for all rows
            if update_flags.all():
                return self._optimize_scales(solution)
            # If all False, do nothing
            if not update_flags.any():
                return 0

        # Row selection
        if update_flags is None:
            integers_z = solution.integers_z
            sub_matrices_YX = self.sub_matrices_YX
        else:
            integers_z = solution.integers_z[:, update_flags, :]
            sub_matrices_YX = self.sub_matrices_YX[update_flags]

        # step1-3: Compute optimal scales
        scales_new = self._compute_optimal_scales(integers_z, sub_matrices_YX)

        # step4: Recompute objective values and update with new scales if improved
        result = solution.try_update_scales(
            scales_new,
            **self.objective_args,
            epsilon=self.epsilon,
            update_flags=update_flags,
        )
        return result

    def _compute_optimal_scales(self, integers_z, sub_matrices_YX):
        """Compute optimal scales by solving G^T G s = G^T y.

        step1. Compute each G_i^T G_i and G_i^T y_i
        step2. Solve the linear equations in batch (G_i^T G_i) s_i = G_i^T y_i

        Parameters
        ----------
        integers_z : torch.Tensor
            Selected integer assignments, shape (num_groups, num_rows, group_size).
        sub_matrices_YX : torch.Tensor
            Selected precomputed matrices, shape (num_rows, num_groups, group_size).

        Returns
        -------
        scales_new : torch.Tensor
            The new scales, shape (num_rows, num_groups).

        """

        # step1. Compute the coefficient matrix G_i^T G_i and vector G_i^T y_i
        GG_all, GY_all = self._compute_GtG_and_Gty(integers_z, sub_matrices_YX)

        # step2. Solve the linear equations in batch (G_i^T G_i) s_i = G_i^T y_i
        scales_new = self._solve_batch_linear(GG_all, GY_all)

        del GG_all, GY_all

        return scales_new

    def _compute_GtG_and_Gty(self, integers_z, sub_matrices_YX):
        """Compute the coefficient matrix G^T G and the vector G^T y.

        Compute G_i^T G_i (shape: (k, k)) and G_i^T y_i (shape: (k,)) for each i.

        (u, v)-th entry of G_i^T G_i: z[u][i]^T @ (X_u^T X_v) @ z[v][i]
        t-th entry of G_i^T y_i:      z[t][i]^T @ (X_t^T y_i)

        Parameters
        ----------
        integers_z : torch.Tensor
            Selected integer assignments, shape (num_groups, num_rows, group_size).
        sub_matrices_YX : torch.Tensor
            Selected precomputed matrices, shape (num_rows, num_groups, group_size).

        Returns
        -------
        GG_all : torch.Tensor
            Batch of G^T G matrices, shape (num_rows, num_groups, num_groups).
        GY_all : torch.Tensor
            Batch of G^T y vectors, shape (num_rows, num_groups).

        """

        num_rows = integers_z.shape[1]
        integers_z_f = integers_z.to(self.dtype)

        # Compute G_i^T G_i
        # Vectorize inner loop: for each u, compute all v >= u at once
        XX_device = self.sub_matrices_XX  # (G, G, D, D), already on device

        GG_all = torch.zeros(
            num_rows,
            self.num_groups,
            self.num_groups,
            dtype=self.dtype,
            device=self.device,
        )

        for u in range(self.num_groups):
            Z_u = integers_z_f[u]  # shape: (num_rows, group_size)

            # Compute Z_u @ XX[u, u:] via broadcasting at once
            # (num_rows, D) @ (G-u, D, D) -> (G-u, num_rows, D)
            ZXX_u = Z_u @ XX_device[u, u:]

            # Compute upper triangle of GG via element-wise product + sum at once
            # (G-u, num_rows, D) * (G-u, num_rows, D) -> sum(dim=2) -> (G-u, num_rows)
            GG_all[:, u, u:] = (ZXX_u * integers_z_f[u:]).sum(dim=2).T

        del XX_device

        # Fill lower triangle using symmetry
        indices = torch.triu_indices(self.num_groups, self.num_groups, offset=1)
        GG_all[:, indices[1], indices[0]] = GG_all[:, indices[0], indices[1]]

        # Compute G_i^T y_i (vectorized without loops)
        # integers_z_f: (G, num_rows, D), sub_matrices_YX: (num_rows, G, D)
        # GY_all[i, t] = z[t][i]^T @ sub_matrices_YX[i, t, :] = sum_d z[t][i][d] * sub_matrices_YX[i, t, d]
        GY_all = torch.sum(integers_z_f * sub_matrices_YX.permute(1, 0, 2), dim=2).T

        del integers_z_f

        return GG_all, GY_all

    def _solve_batch_linear(self, GG_all, GY_all):
        """Solve batch linear equations (G^T G) s = G^T y with singular matrix fallback.

        Parameters
        ----------
        GG_all : torch.Tensor
            Batch of G^T G matrices, shape (num_rows, num_groups, num_groups).
        GY_all : torch.Tensor
            Batch of G^T y vectors, shape (num_rows, num_groups).

        Returns
        -------
        scales_new : torch.Tensor
            Solutions, shape (num_rows, num_groups).

        """

        try:
            scales_new = torch.linalg.solve(GG_all, GY_all.unsqueeze(-1)).squeeze(-1)
        except torch._C._LinAlgError as e:
            # Singular matrices detected — re-solve all rows with damping
            print(f"Warning: Singular matrix detected: {e}")
            print("Re-solving all rows with diagonal damping...")
            diag_max = GG_all.diagonal(dim1=-2, dim2=-1).max(dim=-1).values
            damping = (diag_max * 0.01).clamp(min=1e-6)
            GG_damped = GG_all.clone()
            GG_damped.diagonal(dim1=-2, dim2=-1).add_(damping.unsqueeze(-1))
            scales_new = torch.linalg.solve(GG_damped, GY_all.unsqueeze(-1)).squeeze(-1)

        # Re-solve rows where scales exploded (ill-conditioned but not singular)
        exploded = scales_new.abs().max(dim=-1).values > self.scale_threshold
        if exploded.any():
            n_exploded = int(exploded.sum().item())
            print(f"Warning: {n_exploded} rows have exploded scales (>{self.scale_threshold:.0f}), re-solving with damping")
            diag_max = GG_all[exploded].diagonal(dim1=-2, dim2=-1).max(dim=-1).values
            damping = (diag_max * 0.01).clamp(min=1e-6)
            GG_damped = GG_all[exploded].clone()
            GG_damped.diagonal(dim1=-2, dim2=-1).add_(damping.unsqueeze(-1))
            scales_new[exploded] = torch.linalg.solve(
                GG_damped, GY_all[exploded].unsqueeze(-1)
            ).squeeze(-1)

        return scales_new

    def _group_local_search_all(self, solution, group_index):
        """Perform local search for a specific group in the solution.

        - t := group_index
        - Update the scales and assignment variables for the t-th group of each i-th optimization problem
        - Other variables are kept fixed
        - Only reflect updates for rows where the objective value improved

        Returns
        -------
        torch.Tensor
            Boolean tensor, True if updated, shape (p,)

        """

        # Local search
        integers_z, scales = self._local_search_fn(
            **self._build_local_search_kwargs(solution, group_index)
        )

        # Update changed rows and partially recompute errors
        updated = solution.try_update_group(
            group_index,
            integers_z,
            scales,
            **self.objective_args,
        )

        return updated

    def _build_local_search_kwargs(self, solution, group_index):
        """Build keyword arguments for the local search function.

        Generate matrix_H and build keyword arguments for the local search function.
        Can be overridden in subclasses to change the matrix_H computation method
        or to pass additional arguments to the local search function.

        Parameters
        ----------
        solution : Solution
            The current solution.
        group_index : int
            The index of the group to optimize.

        Returns
        -------
        dict
            Keyword arguments for the local search function.
        """
        t = group_index
        t_start = t * self.group_size
        t_end = (t + 1) * self.group_size

        # Y = target_matrix - hat{W} X^T + hat{W}_t X_t^T
        # H = Y X_t = hat_W_t @ XX[t,t] + (YX)_t - hat_W @ XX[:, t_slice]
        hat_W_t = solution.scales[t].unsqueeze(1) * solution.integers_z[t]
        hat_W = solution.get_dequantized_weight_matrix()
        matrix_H = hat_W_t @ self.sub_matrices_XX[t, t]
        matrix_H += self.sub_matrices_YX[:, t, :]
        matrix_H -= hat_W @ self.matrix_XX[:, t_start:t_end]

        return dict(
            matrix_H=matrix_H,
            lower_bounds=self.modified_lower_bounds[t],
            upper_bounds=self.modified_upper_bounds[t],
            initial_solution=solution.integers_z[t],
            matrix_R=self.sub_matrices_XX[t, t],
            max_iterations=10 * self.group_size,
            epsilon=self.epsilon,
            verbose=False,
        )

    def _get_time(self):
        """Get the time."""
        return time.time() - self.begin_time

    def display_log(self, solution, phase, ignore_log_level=False):
        """Display the log."""

        if self.log_level <= 1 and not ignore_log_level:
            return

        spaces = max(0, 15 - len(phase)) * " "
        print(
            f"<{self.device}>: "
            f"[{self._get_time():.2f}s] {phase}: {spaces}"
            f"error = {solution.squared_error:.3e}, "
            f"MSE = {solution.mean_squared_error:.3e}"
        )

    def display_gpu_memory_usage(self):
        """Display the GPU memory usage."""

        print(
            f"<{self.device}>: GPU memory usage: "
            f"{torch.cuda.memory_allocated(self.device) / 1024 ** 3:.2f} GB "
            f"/ {torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 3:.2f} GB"
        )

    def display_information(self):
        """Display the information."""
        print(
            f"<{self.device}>: Parameters:\n"
            f"<{self.device}>:   - bits: {self.bits}\n"
            f"<{self.device}>:   - symmetric: {self.symmetric}\n"
            f"<{self.device}>:   - number of groups: {self.num_groups}\n"
            f"<{self.device}>:   - group size: {self.group_size}\n"
            f"<{self.device}>:   - lower bound: {self.lower_bound}\n"
            f"<{self.device}>:   - upper bound: {self.upper_bound}\n"
            f"<{self.device}>:   - dim_p: {self.dim_p}\n"
            f"<{self.device}>:   - dim_n: {self.dim_n}\n"
            f"<{self.device}>:   - dim_m: {self.dim_m}\n"
            f"<{self.device}>:   - dtype: {self.dtype}\n"
            f"<{self.device}>:   - epsilon: {self.epsilon}\n"
            f"<{self.device}>:   - log_level: {self.log_level}"
        )
