"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

# pylint: disable=too-many-arguments,too-many-positional-arguments, too-many-locals

import torch


class LocalSearchSolver:
    """Solve the quantization problem using a local search algorithm.

    Let y_i (i=1, ..., p) be the row vectors of matrix Y.
    Let alpha = (alpha_1, ..., alpha_p).
    Let z_i = (z_i1, ..., z_id) in {0, 1}^d (i=1, ..., p) be the row vectors of matrix Z.

    Solve p optimization problems. The i-th optimization problem is:
    min_{alpha_i, z_i} ||y_i - alpha_i C^T z_i||^2
    s.t. lower_bounds_i <= z_ij <= upper_bounds_i (j=1, ..., d)

    Parameters
    ----------
    matrix_H : torch.Tensor
        H = Y C^T, shape (p, d). The (i, j) element of H is h_ij = y_i C^T e_j.
    lower_bounds : torch.Tensor
        The lower bounds, shape (p,).
    upper_bounds : torch.Tensor
        The upper bounds, shape (p,).
    initial_solution : torch.Tensor
        The initial solution, shape (p, d).
    matrix_R : torch.Tensor
        The matrix R = C C^T, shape (d, d).
    max_iterations : int
        The maximum number of iterations.
        If None, the maximum number of iterations is set to d.
    epsilon : float
        The epsilon for the local search.
    early_stopping_ratio : float
        The ratio for the early stopping.
        Terminate early when the ratio of non-improved rows exceeds this threshold.
    verbose : bool
        Whether to print the verbose output.
    """

    def __init__(
        self,
        matrix_H,
        lower_bounds,
        upper_bounds,
        initial_solution,
        matrix_R,
        max_iterations=None,
        epsilon=1e-8,
        early_stopping_ratio=0.1,
        verbose=True,
    ):
        self.matrix_H = matrix_H
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.initial_solution = initial_solution
        self.matrix_R = matrix_R
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.early_stopping_ratio = early_stopping_ratio
        self.verbose = verbose

        # Variables initialized in setup
        self.current_Z = None
        self.vector_o = None
        self.vector_q = None
        self.matrix_V = None
        self.current_theta = None

    @classmethod
    def solve(cls, **kwargs):
        """Create a solver instance and run the local search algorithm."""
        solver = cls(**kwargs)
        return solver._run()

    def _run(self):
        """Run the local search algorithm."""

        self._setup()

        dim_p, dim_d = self.current_Z.shape

        # theta_i =  (q_i^2 / o_i) if o_i > epsilon else 0
        self.current_theta = torch.where(
            self.vector_o > self.epsilon,
            (self.vector_q**2) / self.vector_o,
            torch.zeros_like(self.vector_o),  # 0 when vector_o is too small
        )

        if self.verbose:
            if self.current_theta.shape[0] < 10:
                print(f"initial theta: {self.current_theta}")
            else:
                sum_theta = torch.sum(self.current_theta)
                mean_theta = sum_theta / dim_p
                print(f"initial theta: sum={sum_theta:.6f}, mean={mean_theta:.6f}")

        if self.max_iterations is None:
            self.max_iterations = dim_d

        for iteration in range(self.max_iterations):
            if not self._iteration(iteration):
                break

        return self._compute_alpha()

    def _setup(self):
        """Setup the local search.

        Returns
        -------
        Sets the following instance variables:
        current_Z : torch.Tensor
            Copy of the initial solution Z, shape (p, d).
        vector_o : torch.Tensor
            o_i = ||z_i C||^2 = z_i R z_i^T, shape (p,).
        vector_q : torch.Tensor
            q_i = y_i C^T z_i^T = h_i z_i^T, shape (p,).
        matrix_V : torch.Tensor
            V = Z R, shape (p, d). The (i, j) element of V is v_ij = z_i R e_j.

        Notes
        -----
        matrix_H (= Y C^T) is received in the constructor.
        vector_o and vector_q are computed directly from matrix_V and matrix_H.

        """

        # copy the initial solution, shape: (p, d)
        self.current_Z = self.initial_solution.clone().to(self.matrix_R.device)

        # V = ZR, shape: (p, d)
        self.matrix_V = self.current_Z.to(self.matrix_R.dtype) @ self.matrix_R

        # o = (o_1, ..., o_p), shape: (p,), o_i = z_i R z_i^T = (V * Z).sum
        self.vector_o = (self.matrix_V * self.current_Z).sum(dim=1)

        # q = (q_1, ..., q_p), shape: (p,), q_i = h_i z_i^T = (H * Z).sum
        self.vector_q = (self.matrix_H * self.current_Z.to(self.matrix_H.dtype)).sum(dim=1)

    def _iteration(self, iteration):
        """Local search iteration.

        Returns
        -------
        bool
            True if the iteration should continue, False otherwise.
        """

        dim_p, _ = self.current_Z.shape
        device = self.current_Z.device

        # Evaluate the +1 case
        (
            max_lambda_plus,
            max_lambda_plus_index,
            inner_dividend_plus,
            divisor_plus,
        ) = self._evaluate_plus()

        # Evaluate the -1 case
        (
            max_lambda_minus,
            max_lambda_minus_index,
            inner_dividend_minus,
            divisor_minus,
        ) = self._evaluate_minus()

        # gamma_i = max(max_lambda_plus_i, max_lambda_minus_i) - theta_i
        gamma = torch.maximum(max_lambda_plus, max_lambda_minus) - self.current_theta

        # Terminate early if non-improved rows exceed this ratio
        if torch.sum(gamma < self.epsilon) > self.early_stopping_ratio * dim_p:
            if self.verbose:
                print(
                    f"[{iteration+1}/{self.max_iterations}] "
                    "Early stopping due to no improvement"
                )
            return False

        # Mask for rows to update
        update_mask = gamma > self.epsilon
        # Mask for +1 case
        plus_mask = update_mask & (max_lambda_plus >= max_lambda_minus)
        # Mask for -1 case
        minus_mask = update_mask & (max_lambda_plus < max_lambda_minus)
        # Row indices for +1 case
        plus_row_index = torch.arange(dim_p).to(device)[plus_mask]
        # Row indices for -1 case
        minus_row_index = torch.arange(dim_p).to(device)[minus_mask]
        # Column indices for +1 case
        plus_col_index = max_lambda_plus_index[plus_mask]
        # Column indices for -1 case
        minus_col_index = max_lambda_minus_index[minus_mask]

        # Update
        self.current_Z[plus_row_index, plus_col_index] += 1
        self.current_Z[minus_row_index, minus_col_index] -= 1
        self.vector_o[plus_row_index] = divisor_plus[plus_row_index]
        self.vector_o[minus_row_index] = divisor_minus[minus_row_index]
        self.vector_q[plus_row_index] = inner_dividend_plus[plus_row_index]
        self.vector_q[minus_row_index] = inner_dividend_minus[minus_row_index]
        self.matrix_V[plus_row_index] += self.matrix_R[plus_col_index]
        self.matrix_V[minus_row_index] -= self.matrix_R[minus_col_index]
        self.current_theta[plus_row_index] = max_lambda_plus[plus_row_index]
        self.current_theta[minus_row_index] = max_lambda_minus[minus_row_index]

        if self.verbose:
            if self.current_theta.shape[0] < 10:
                print(f"[{iteration+1}/{self.max_iterations}] " f"theta: {self.current_theta}")
            else:
                sum_theta = torch.sum(self.current_theta)
                mean_theta = sum_theta / dim_p
                print(
                    f"[{iteration+1}/{self.max_iterations}] "
                    f"theta: sum={sum_theta:.6f}, mean={mean_theta:.6f}"
                )

        return True

    def _evaluate_plus(self):
        """Evaluate the plus case.

        Compute the best evaluation value when incrementing each variable by +1 for each row.

        """

        # Compute lambda^+_ij = (q_i + h_ij)^2 / (o_i + 2 v_ij + r_jj)
        inner_dividend = self.vector_q.unsqueeze(1) + self.matrix_H
        dividend = inner_dividend**2
        divisor = self._compute_divisor_plus()
        matrix_lambda_plus = torch.where(
            (divisor > self.epsilon) & (self.current_Z < self.upper_bounds.unsqueeze(1)),
            dividend / divisor,
            torch.zeros_like(dividend),
        )
        # Find the maximum lambda^+_ij for each i
        max_lambda_plus, max_lambda_plus_index = torch.max(matrix_lambda_plus, dim=1)

        # Extract intermediate results for the best values and return them together
        row_inds = torch.arange(matrix_lambda_plus.shape[0]).to(matrix_lambda_plus.device)

        return (
            max_lambda_plus,
            max_lambda_plus_index,
            inner_dividend[row_inds, max_lambda_plus_index],
            divisor[row_inds, max_lambda_plus_index],
        )

    def _compute_divisor_plus(self):
        """Compute the divisor for the plus case.

        divisor_ij = o_i + 2 v_ij + r_jj

        Returns
        -------
        torch.Tensor
            The divisor matrix, shape (p, d).
        """
        return self.vector_o.unsqueeze(1) + 2 * self.matrix_V + self.matrix_R.diag().unsqueeze(0)

    def _evaluate_minus(self):
        """Evaluate the minus case.

        Compute the best evaluation value when decrementing each variable by -1 for each row.

        """

        # Compute lambda^-_ij = (q_i - h_ij)^2 / (o_i - 2 v_ij + r_jj)
        inner_dividend = self.vector_q.unsqueeze(1) - self.matrix_H
        dividend = inner_dividend**2
        divisor = self._compute_divisor_minus()
        matrix_lambda_minus = torch.where(
            (divisor > self.epsilon) & (self.current_Z > self.lower_bounds.unsqueeze(1)),
            dividend / divisor,
            torch.zeros_like(dividend),
        )

        # Find the maximum lambda^-_ij for each i
        max_lambda_minus, max_lambda_minus_index = torch.max(matrix_lambda_minus, dim=1)

        # Extract intermediate results for the best values and return them together
        row_inds = torch.arange(matrix_lambda_minus.shape[0]).to(matrix_lambda_minus.device)

        return (
            max_lambda_minus,
            max_lambda_minus_index,
            inner_dividend[row_inds, max_lambda_minus_index],
            divisor[row_inds, max_lambda_minus_index],
        )

    def _compute_divisor_minus(self):
        """Compute the divisor for the minus case.

        divisor_ij = o_i - 2 v_ij + r_jj

        Returns
        -------
        torch.Tensor
            The divisor matrix, shape (p, d).
        """
        return self.vector_o.unsqueeze(1) - 2 * self.matrix_V + self.matrix_R.diag().unsqueeze(0)

    def _compute_alpha(self):
        """Compute the optimal scale alpha.

        alpha_i = q_i / o_i = (y_i C^T z_i) / ||z_i C||^2

        Since vector_q and vector_o are incrementally updated during iterations,
        floating-point errors may accumulate.
        Therefore, they are recomputed from scratch using the final state of current_Z.

        Returns
        -------
        tuple of (current_Z, alpha)
        """
        Z = self.current_Z.to(self.matrix_R.dtype)
        V = Z @ self.matrix_R  # shape: (p, d)
        o = (V * Z).sum(dim=1)  # shape: (p,)
        q = (self.matrix_H * Z).sum(dim=1)  # shape: (p,)
        # Guard against division by zero / near-zero o (degenerate rows)
        safe = o.abs() > 1e-12
        alpha = torch.where(safe, q / o, torch.zeros_like(q))
        return self.current_Z, alpha
