"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

# pylint: disable=too-many-arguments,too-many-positional-arguments, too-many-locals

import torch

from ..local_search import LocalSearchSolver


class LocalSearchSolverAdvanced(LocalSearchSolver):
    """Solve the quantization problem with penalty term using a local search algorithm.

    Let y_i (i=1, ..., p) be the row vectors of matrix Y.
    Let alpha = (alpha_1, ..., alpha_p).
    Let z_i = (z_i1, ..., z_id) in {0, 1}^d (i=1, ..., p) be the row vectors of matrix Z.

    Solve p optimization problems. The i-th optimization problem is:
    min_{alpha_i, z_i} ||y_i - alpha_i C^T z_i||^2 + lambda_i ||tilde_w_i - alpha_i z_i||^2
    s.t. lower_bounds_i <= z_ij <= upper_bounds_i (j=1, ..., d)

    Parameters
    ----------
    vector_lambda : torch.Tensor
        The penalty coefficients, lambda = (lambda_1, ..., lambda_p), shape (p,).
    matrix_L : torch.Tensor
        The matrix L, shape (p, d). Row vectors are lambda_i tilde_w_i.
    **kwargs
        Passed to LocalSearchSolver.__init__.
    """

    def __init__(self, vector_lambda, matrix_L, **kwargs):
        super().__init__(**kwargs)
        self.vector_lambda = vector_lambda
        self.matrix_L = matrix_L

    def _setup(self):
        """Setup the local search for the advanced version.

        Calls the parent class _setup and then adds the penalty term.

        Sets the following instance variables (inherited from parent class):
        current_Z : torch.Tensor
            Copy of the initial solution Z, shape (p, d).
        vector_o : torch.Tensor
            o_i = ||z_i C||^2 + lambda_i ||z_i||^2
              = z_i R z_i^T + lambda_i z_i z_i^T, shape (p,).
        vector_q : torch.Tensor
            q_i = h_i z_i^T = (y_i C^T + l_i) z_i^T, shape (p,).
            Already computed from matrix_H (= Y C^T + L) in the parent class.
        matrix_V : torch.Tensor
            V = Z R, shape (p, d).

        Notes
        -----
        Since matrix_H (= Y C^T + L) is pre-computed by the caller,
        vector_q is correctly computed in the parent class _setup.
        Here, only the penalty term for vector_o is added.

        """

        # Call parent class setup
        # Since matrix_H (= Y C^T + L) is pre-computed by the caller,
        # the parent _setup computes vector_q = (matrix_H * Z).sum including the L contribution.
        super()._setup()

        # o_i = ||z_i C||^2 + lambda_i ||z_i||^2
        # (Parent's vector_o = (V * Z).sum only contains the R term, so add the penalty term)
        current_Z_float = self.current_Z.to(self.matrix_L.dtype)
        self.vector_o += self.vector_lambda * torch.sum(current_Z_float**2, dim=1)
        del current_Z_float

    def _compute_divisor_plus(self):
        """Compute the divisor for the plus case with penalty term.

        divisor_ij = o_i + 2 v_ij + r_jj + lambda_i (1 + 2 z_ij)

        Returns
        -------
        torch.Tensor
            The divisor matrix, shape (p, d).
        """
        return super()._compute_divisor_plus() + self.vector_lambda.unsqueeze(1) * (
            1 + 2 * self.current_Z.to(self.vector_lambda.dtype)
        )

    def _compute_divisor_minus(self):
        """Compute the divisor for the minus case with penalty term.

        divisor_ij = o_i - 2 v_ij + r_jj + lambda_i (1 - 2 z_ij)

        Returns
        -------
        torch.Tensor
            The divisor matrix, shape (p, d).
        """
        return super()._compute_divisor_minus() + self.vector_lambda.unsqueeze(1) * (
            1 - 2 * self.current_Z.to(self.vector_lambda.dtype)
        )

    def _compute_alpha(self):
        """Compute the optimal scale alpha with penalty term.

        alpha_i = (y_i C^T z_i^T + lambda_i tilde_w_i z_i^T)
                / (||z_i C||^2 + lambda_i ||z_i||^2)

        Since vector_q and vector_o are incrementally updated during iterations,
        floating-point errors may accumulate.
        Therefore, they are recomputed from scratch using the final state of current_Z.

        Notes
        -----
        Since matrix_H (= Y C^T + L) is pre-computed by the caller,
        the numerator can be computed as (matrix_H * Z).sum.

        Returns
        -------
        tuple of (current_Z, alpha)
        """
        Z = self.current_Z.to(self.matrix_R.dtype)

        # numerator: (Y C^T + L) z^T = matrix_H z^T
        q = (self.matrix_H * Z).sum(dim=1)

        # denominator: ||z C||^2 + lambda ||z||^2
        V = Z @ self.matrix_R
        o = (V * Z).sum(dim=1) + self.vector_lambda * (Z**2).sum(dim=1)

        # Guard against division by zero / near-zero o (degenerate rows)
        safe = o.abs() > 1e-12
        alpha = torch.where(safe, q / o, torch.zeros_like(q))
        return self.current_Z, alpha
