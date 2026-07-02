"""
Proof-of-concept test: verification of the H^{1/2} computation bug in lowrank_osvd

Issue (0604.pdf): in lowrank_osvd of initialize.py,
  first half:  W_tilde = W @ Q @ diag(sqrt(λ))          <- Q^T is missing
  second half: V' = Q @ diag(1/sqrt(λ)) @ [Q^T] @ V_r   <- Q^T is the correct H^{-1/2} expansion

Since H = Q Λ Q^T implies H^{1/2} = Q Λ^{1/2} Q^T (Q^T is required),
the first-half W_tilde actually computes W @ H^{1/2} @ Q.

V_r is the set of right singular vectors of W @ H^{1/2} @ Q, which relate to
the right singular vectors V_r_correct of W @ H^{1/2} as
  V_r_buggy = Q^T @ V_r_correct.

Second-half computation of V':
  buggy  : Q @ diag(1/sqrt(λ)) @ Q^T @ V_r_buggy = Q @ diag(1/sqrt(λ)) @ Q^T @ Q^T @ V_r_correct
  correct: Q @ diag(1/sqrt(λ))              @ V_r_buggy = Q @ diag(1/sqrt(λ)) @ Q^T @ V_r_correct

Because Q^T @ Q^T = (Q @ Q)^{-1} != I (unless Q is diagonal),
W_hat_buggy = U' @ V'_buggy^T is not the H-weighted best rank-r approximation of W.
"""

import pytest
import torch


def _hessian_error(W: torch.Tensor, W_hat: torch.Tensor, H: torch.Tensor) -> float:
    """tr((W - W_hat) H (W - W_hat)^T)"""
    E = W - W_hat
    return torch.trace(E @ H @ E.T).item()


def _osvd_correct(W: torch.Tensor, H: torch.Tensor, r: int) -> torch.Tensor:
    """
    Correct OSVD implementation: uses H^{1/2} = Q diag(sqrt(λ)) Q^T

    W_tilde = W @ H^{1/2}  -> SVD -> V' = H^{-1/2} @ V_r @ diag(sqrt(Σ))
    """
    eig_vals, eig_vecs = torch.linalg.eigh(H)
    eig_vals = eig_vals.clamp(min=1e-12)
    sqrt_eig = torch.sqrt(eig_vals)
    inv_sqrt_eig = 1.0 / sqrt_eig

    # Correct H^{1/2}: Q diag(sqrt(λ)) Q^T
    W_tilde = W @ eig_vecs @ torch.diag(sqrt_eig) @ eig_vecs.T

    U_w, S_w, Vh_w = torch.linalg.svd(W_tilde, full_matrices=False)
    r = min(r, S_w.numel())
    U_r, S_r, V_r = U_w[:, :r], S_w[:r], Vh_w[:r, :].T
    sqrt_S = torch.sqrt(S_r.clamp(min=1e-12))

    U_prime = U_r * sqrt_S[None, :]
    # V' = H^{-1/2} @ V_r @ diag(sqrt(Σ))  (H^{-1/2} = Q diag(1/sqrt(λ)) Q^T)
    V_prime = eig_vecs @ torch.diag(inv_sqrt_eig) @ eig_vecs.T @ V_r @ torch.diag(sqrt_S)
    return U_prime @ V_prime.T


def _osvd_buggy(W: torch.Tensor, H: torch.Tensor, r: int) -> torch.Tensor:
    """
    Current implementation (buggy): Q^T is missing from W_tilde

    W_tilde = W @ Q @ diag(sqrt(λ))   <- no Q^T (= W @ H^{1/2} @ Q)
    V' = Q @ diag(1/sqrt(λ)) @ Q^T @ V_r @ diag(sqrt(Σ))  <- second half is the full H^{-1/2}

    In the first half V_r is the set of right singular vectors of W @ H^{1/2} @ Q
    (= Q^T @ V_r_correct), so multiplying by Q^T in the second half introduces
    Q^T @ Q^T = (Q^2)^{-1} != I and produces a discrepancy.
    """
    eig_vals, eig_vecs = torch.linalg.eigh(H)
    eig_vals = eig_vals.clamp(min=1e-12)
    sqrt_eig = torch.sqrt(eig_vals)
    inv_sqrt_eig = 1.0 / sqrt_eig

    # Bug: Q^T is missing -> this computes W @ H^{1/2} @ Q
    W_tilde = W @ eig_vecs @ torch.diag(sqrt_eig)

    U_w, S_w, Vh_w = torch.linalg.svd(W_tilde, full_matrices=False)
    r = min(r, S_w.numel())
    U_r, S_r, V_r = U_w[:, :r], S_w[:r], Vh_w[:r, :].T
    sqrt_S = torch.sqrt(S_r.clamp(min=1e-12))

    U_prime = U_r * sqrt_S[None, :]
    # The second half applies the full H^{-1/2}, but V_r is already Q-rotated, so it is inconsistent
    V_prime = eig_vecs @ torch.diag(inv_sqrt_eig) @ eig_vecs.T @ V_r @ torch.diag(sqrt_S)
    return U_prime @ V_prime.T


def _make_pd_matrix(m: int, seed: int, noise: float = 0.1) -> torch.Tensor:
    """Generate a non-diagonal positive-definite matrix (the bug does not manifest when H is diagonal)"""
    torch.manual_seed(seed)
    A = torch.randn(m, m)
    return A @ A.T + noise * torch.eye(m)


class TestOsvdHessianBug:
    """
    lowrank_osvd H^{1/2} bug: numerical proof
    """

    @pytest.mark.parametrize("n,m,r,seed", [
        (16, 12, 3, 0),
        (32, 24, 4, 1),
        (8,  6,  2, 42),
    ])
    def test_buggy_has_larger_hessian_error(self, n, m, r, seed):
        """
        Prove that the buggy implementation yields a larger H-weighted output
        error than the optimal solution.

        Since the correct OSVD returns the optimal solution of the minimization
        problem, buggy error >= correct error holds. The two are equal only when
        H is diagonal (i.e. Q = I).
        """
        torch.manual_seed(seed)
        W = torch.randn(n, m, dtype=torch.float64)
        H = _make_pd_matrix(m, seed=seed + 100).to(torch.float64)

        W_hat_correct = _osvd_correct(W, H, r)
        W_hat_buggy   = _osvd_buggy(W, H, r)

        err_correct = _hessian_error(W, W_hat_correct, H)
        err_buggy   = _hessian_error(W, W_hat_buggy, H)

        print(
            f"\n[n={n},m={m},r={r},seed={seed}]"
            f"  correct={err_correct:.6e}  buggy={err_buggy:.6e}"
            f"  degradation={100*(err_buggy-err_correct)/max(abs(err_correct),1e-12):.2f}%"
        )

        # The correct version is optimal, so it is always <= the buggy version
        assert err_correct <= err_buggy + 1e-6 * abs(err_correct), (
            f"correct ({err_correct:.6e}) should be <= buggy ({err_buggy:.6e})"
        )
        # For non-diagonal H they must genuinely differ (if equal, the bug is not manifesting)
        assert abs(err_buggy - err_correct) > 1e-8 * abs(err_correct), (
            "Both errors are equal: bug is not manifesting "
            "(check whether H is effectively diagonal)"
        )

    def test_reconstruction_differs(self):
        """
        Directly confirm that the matrices W_hat_correct and W_hat_buggy differ.
        """
        torch.manual_seed(7)
        n, m, r = 10, 8, 3
        W = torch.randn(n, m, dtype=torch.float64)
        H = _make_pd_matrix(m, seed=77).to(torch.float64)

        W_hat_correct = _osvd_correct(W, H, r)
        W_hat_buggy   = _osvd_buggy(W, H, r)

        max_diff = (W_hat_correct - W_hat_buggy).abs().max().item()
        print(f"\nmax |W_hat_correct - W_hat_buggy| = {max_diff:.6e}")

        assert max_diff > 1e-6, (
            f"Expected a visible difference between correct and buggy W_hat, got {max_diff:.2e}"
        )

    def test_diagonal_H_no_bug(self):
        """
        Confirm that the bug does not manifest when H is diagonal with
        ascending-sorted eigenvalues (i.e. Q is the identity matrix).

        torch.linalg.eigh returns eigenvalues in ascending order. Only when H is
        diagonal and its diagonal entries are already sorted ascending does
        eig_vecs = I hold, making the bug's Q-rotation the identity so the bug
        does not appear.

        This corroborates that the bug stems from the off-diagonal (or permutation)
        component of Q.
        """
        torch.manual_seed(5)
        n, m, r = 10, 8, 3
        W = torch.randn(n, m, dtype=torch.float64)
        # Diagonal H: sorting eigenvalues ascending makes the Q returned by eigh the identity
        diag_vals = torch.sort(torch.rand(m, dtype=torch.float64) + 0.5).values
        H = torch.diag(diag_vals)

        # Pre-check that eigh returns the identity matrix
        eig_vals, eig_vecs = torch.linalg.eigh(H)
        assert torch.allclose(eig_vecs.abs(), torch.eye(m, dtype=torch.float64), atol=1e-10), \
            "For sorted diagonal H, eig_vecs should be identity"

        W_hat_correct = _osvd_correct(W, H, r)
        W_hat_buggy   = _osvd_buggy(W, H, r)

        max_diff = (W_hat_correct - W_hat_buggy).abs().max().item()
        err_diff = abs(
            _hessian_error(W, W_hat_correct, H) - _hessian_error(W, W_hat_buggy, H)
        )
        print(f"\nDiagonal H (sorted): max diff = {max_diff:.2e}, error diff = {err_diff:.2e}")

        # When Q = I the two agree (= evidence that the bug stems from the Q rotation)
        assert max_diff < 1e-5, (
            "With sorted diagonal H (Q=I), correct and buggy should agree "
            f"(got max diff {max_diff:.2e})"
        )

    def test_buggy_inconsistency_analytically(self):
        """
        Analytic confirmation of the inconsistency:
          W_tilde_buggy = W @ Q @ diag(sqrt(λ)) equals W @ H^{1/2} @ Q.
          Therefore V_r_buggy = Q^T @ V_r_correct.
          Multiplying by H^{-1/2} = Q diag(1/sqrt(λ)) Q^T in the second half gives
            Q diag(1/sqrt(λ)) Q^T @ Q^T @ V_r_correct
          = Q diag(1/sqrt(λ)) @ (Q @ Q)^{-1} @ V_r_correct  (Q^T @ Q^T != I),
          leaving a spurious (Q^2)^{-1} factor.
        """
        torch.manual_seed(99)
        m = 6
        H = _make_pd_matrix(m, seed=99).to(torch.float64)

        eig_vals, Q = torch.linalg.eigh(H)
        eig_vals = eig_vals.clamp(min=1e-12)
        sqrt_eig = torch.sqrt(eig_vals)

        # Confirm H^{1/2} @ Q = Q @ diag(sqrt(λ))
        H_half = Q @ torch.diag(sqrt_eig) @ Q.T
        assert torch.allclose(H_half @ Q, Q @ torch.diag(sqrt_eig), atol=1e-10), \
            "H^{1/2} @ Q should equal Q @ diag(sqrt_eig)"

        # W_tilde_buggy = W @ Q @ diag(sqrt(λ)) = W @ H^{1/2} @ Q != W @ H^{1/2}
        torch.manual_seed(13)
        W = torch.randn(8, m, dtype=torch.float64)
        W_tilde_buggy   = W @ Q @ torch.diag(sqrt_eig)
        W_tilde_correct = W @ H_half

        assert not torch.allclose(W_tilde_buggy, W_tilde_correct, atol=1e-8), \
            "W_tilde_buggy and W_tilde_correct should differ for non-diagonal H"

        # Confirm Q^T @ Q^T != I (the root cause of the bug)
        QtQt = Q.T @ Q.T
        assert not torch.allclose(QtQt, torch.eye(m, dtype=torch.float64), atol=1e-6), \
            "Q^T @ Q^T should not be identity (confirming the extra rotation is spurious)"
