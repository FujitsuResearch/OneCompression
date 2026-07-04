"""
Regression test for the H^{1/2} whitening in lowrank_osvd (OSVD).

OSVD minimizes the Hessian-weighted output error
  min_{U,V} tr((W - U V^T) H (W - U V^T)^T)
by whitening with H^{1/2} before the SVD and un-whitening with H^{-1/2}
afterwards. For H = Q Λ Q^T the matrix square root is
  H^{1/2} = Q Λ^{1/2} Q^T,
so the whitening step must apply the full Q Λ^{1/2} Q^T; in particular the
trailing Q^T is required.

A formulation that drops the trailing Q^T (whitening with W @ Q Λ^{1/2}
instead of W @ Q Λ^{1/2} Q^T) computes W @ H^{1/2} @ Q. Its right singular
vectors are then Q-rotated relative to those of W @ H^{1/2},
  V_r_no_qt = Q^T @ V_r,
and applying the full H^{-1/2} = Q Λ^{-1/2} Q^T afterwards leaves a spurious
Q^T @ Q^T = (Q @ Q)^{-1} != I factor. Its reconstruction is therefore not the
H-weighted best rank-r approximation of W.

These tests verify that the full-Q^T formulation is optimal for the H-weighted
objective and that dropping Q^T is strictly worse for a non-diagonal H (the two
coincide only when Q = I, i.e. H is diagonal).
"""

import pytest
import torch


def _hessian_error(W: torch.Tensor, W_hat: torch.Tensor, H: torch.Tensor) -> float:
    """tr((W - W_hat) H (W - W_hat)^T)"""
    E = W - W_hat
    return torch.trace(E @ H @ E.T).item()


def _osvd_full(W: torch.Tensor, H: torch.Tensor, r: int) -> torch.Tensor:
    """
    OSVD with the full whitening H^{1/2} = Q diag(sqrt(λ)) Q^T.

    W_tilde = W @ H^{1/2}  -> SVD -> V' = H^{-1/2} @ V_r @ diag(sqrt(Σ))
    """
    eig_vals, eig_vecs = torch.linalg.eigh(H)
    eig_vals = eig_vals.clamp(min=1e-12)
    sqrt_eig = torch.sqrt(eig_vals)
    inv_sqrt_eig = 1.0 / sqrt_eig

    # Full H^{1/2}: Q diag(sqrt(λ)) Q^T
    W_tilde = W @ eig_vecs @ torch.diag(sqrt_eig) @ eig_vecs.T

    U_w, S_w, Vh_w = torch.linalg.svd(W_tilde, full_matrices=False)
    r = min(r, S_w.numel())
    U_r, S_r, V_r = U_w[:, :r], S_w[:r], Vh_w[:r, :].T
    sqrt_S = torch.sqrt(S_r.clamp(min=1e-12))

    U_prime = U_r * sqrt_S[None, :]
    # V' = H^{-1/2} @ V_r @ diag(sqrt(Σ))  (H^{-1/2} = Q diag(1/sqrt(λ)) Q^T)
    V_prime = eig_vecs @ torch.diag(inv_sqrt_eig) @ eig_vecs.T @ V_r @ torch.diag(sqrt_S)
    return U_prime @ V_prime.T


def _osvd_no_qt(W: torch.Tensor, H: torch.Tensor, r: int) -> torch.Tensor:
    """
    Variant that drops the trailing Q^T from the whitening step.

    W_tilde = W @ Q @ diag(sqrt(λ))   (= W @ H^{1/2} @ Q, no trailing Q^T)
    V' = Q @ diag(1/sqrt(λ)) @ Q^T @ V_r @ diag(sqrt(Σ))  (full H^{-1/2})

    Because W_tilde is W @ H^{1/2} @ Q, its right singular vectors V_r equal
    Q^T @ V_r_full, so applying the full H^{-1/2} in the second step leaves a
    spurious Q^T @ Q^T = (Q^2)^{-1} != I factor and does not minimize the
    H-weighted objective.
    """
    eig_vals, eig_vecs = torch.linalg.eigh(H)
    eig_vals = eig_vals.clamp(min=1e-12)
    sqrt_eig = torch.sqrt(eig_vals)
    inv_sqrt_eig = 1.0 / sqrt_eig

    # Drop the trailing Q^T -> this whitens as W @ H^{1/2} @ Q
    W_tilde = W @ eig_vecs @ torch.diag(sqrt_eig)

    U_w, S_w, Vh_w = torch.linalg.svd(W_tilde, full_matrices=False)
    r = min(r, S_w.numel())
    U_r, S_r, V_r = U_w[:, :r], S_w[:r], Vh_w[:r, :].T
    sqrt_S = torch.sqrt(S_r.clamp(min=1e-12))

    U_prime = U_r * sqrt_S[None, :]
    # Full H^{-1/2}, but V_r is already Q-rotated, so the rotations do not cancel
    V_prime = eig_vecs @ torch.diag(inv_sqrt_eig) @ eig_vecs.T @ V_r @ torch.diag(sqrt_S)
    return U_prime @ V_prime.T


def _make_pd_matrix(m: int, seed: int, noise: float = 0.1) -> torch.Tensor:
    """Generate a non-diagonal positive-definite matrix (a diagonal H makes Q = I)"""
    torch.manual_seed(seed)
    A = torch.randn(m, m)
    return A @ A.T + noise * torch.eye(m)


class TestOsvdHessianWhitening:
    """
    OSVD H^{1/2} whitening: numerical verification.
    """

    @pytest.mark.parametrize("n,m,r,seed", [
        (16, 12, 3, 0),
        (32, 24, 4, 1),
        (8,  6,  2, 42),
    ])
    def test_no_qt_has_larger_hessian_error(self, n, m, r, seed):
        """
        The no-Q^T variant yields a larger H-weighted output error than the full
        formulation.

        The full OSVD is the optimal solution of the minimization problem, so
        full error <= no-Q^T error holds. The two are equal only when H is
        diagonal (i.e. Q = I).
        """
        torch.manual_seed(seed)
        W = torch.randn(n, m, dtype=torch.float64)
        H = _make_pd_matrix(m, seed=seed + 100).to(torch.float64)

        W_hat_full  = _osvd_full(W, H, r)
        W_hat_no_qt = _osvd_no_qt(W, H, r)

        err_full  = _hessian_error(W, W_hat_full, H)
        err_no_qt = _hessian_error(W, W_hat_no_qt, H)

        print(
            f"\n[n={n},m={m},r={r},seed={seed}]"
            f"  full={err_full:.6e}  no_qt={err_no_qt:.6e}"
            f"  degradation={100*(err_no_qt-err_full)/max(abs(err_full),1e-12):.2f}%"
        )

        # The full formulation is optimal, so it is always <= the no-Q^T variant
        assert err_full <= err_no_qt + 1e-6 * abs(err_full), (
            f"full ({err_full:.6e}) should be <= no_qt ({err_no_qt:.6e})"
        )
        # For non-diagonal H the two must genuinely differ (else Q^T does not matter here)
        assert abs(err_no_qt - err_full) > 1e-8 * abs(err_full), (
            "Both errors are equal: the trailing Q^T is not being exercised "
            "(check whether H is effectively diagonal)"
        )

    def test_reconstruction_differs(self):
        """
        The full and no-Q^T reconstructions are different matrices.
        """
        torch.manual_seed(7)
        n, m, r = 10, 8, 3
        W = torch.randn(n, m, dtype=torch.float64)
        H = _make_pd_matrix(m, seed=77).to(torch.float64)

        W_hat_full  = _osvd_full(W, H, r)
        W_hat_no_qt = _osvd_no_qt(W, H, r)

        max_diff = (W_hat_full - W_hat_no_qt).abs().max().item()
        print(f"\nmax |W_hat_full - W_hat_no_qt| = {max_diff:.6e}")

        assert max_diff > 1e-6, (
            f"Expected a visible difference between full and no-Q^T W_hat, got {max_diff:.2e}"
        )

    def test_diagonal_H_matches(self):
        """
        With a diagonal H whose eigenvalues are ascending-sorted (i.e. Q is the
        identity matrix) the two formulations coincide.

        torch.linalg.eigh returns eigenvalues in ascending order, so eig_vecs = I
        holds exactly when H is diagonal with already-ascending diagonal entries;
        then the trailing Q^T is the identity and both formulations agree.

        This confirms that the trailing Q^T only matters through the off-diagonal
        (or permutation) component of Q.
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

        W_hat_full  = _osvd_full(W, H, r)
        W_hat_no_qt = _osvd_no_qt(W, H, r)

        max_diff = (W_hat_full - W_hat_no_qt).abs().max().item()
        err_diff = abs(
            _hessian_error(W, W_hat_full, H) - _hessian_error(W, W_hat_no_qt, H)
        )
        print(f"\nDiagonal H (sorted): max diff = {max_diff:.2e}, error diff = {err_diff:.2e}")

        # When Q = I the two agree (the trailing Q^T becomes the identity)
        assert max_diff < 1e-5, (
            "With sorted diagonal H (Q=I), full and no-Q^T should agree "
            f"(got max diff {max_diff:.2e})"
        )

    def test_no_qt_inconsistency_analytically(self):
        """
        Analytic confirmation of the inconsistency:
          W_tilde_no_qt = W @ Q @ diag(sqrt(λ)) equals W @ H^{1/2} @ Q.
          Therefore V_r_no_qt = Q^T @ V_r_full.
          Multiplying by H^{-1/2} = Q diag(1/sqrt(λ)) Q^T in the second step gives
            Q diag(1/sqrt(λ)) Q^T @ Q^T @ V_r_full
          = Q diag(1/sqrt(λ)) @ (Q @ Q)^{-1} @ V_r_full  (Q^T @ Q^T != I),
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

        # W_tilde_no_qt = W @ Q @ diag(sqrt(λ)) = W @ H^{1/2} @ Q != W @ H^{1/2}
        torch.manual_seed(13)
        W = torch.randn(8, m, dtype=torch.float64)
        W_tilde_no_qt = W @ Q @ torch.diag(sqrt_eig)
        W_tilde_full  = W @ H_half

        assert not torch.allclose(W_tilde_no_qt, W_tilde_full, atol=1e-8), \
            "W_tilde_no_qt and W_tilde_full should differ for non-diagonal H"

        # Confirm Q^T @ Q^T != I (the source of the spurious factor)
        QtQt = Q.T @ Q.T
        assert not torch.allclose(QtQt, torch.eye(m, dtype=torch.float64), atol=1e-6), \
            "Q^T @ Q^T should not be identity (confirming the extra rotation is spurious)"
