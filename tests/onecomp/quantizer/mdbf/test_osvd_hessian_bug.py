"""
Proof-of-concept test: lowrank_osvd の H^{1/2} 計算バグの検証

0604.pdf 指摘: initialize.py の lowrank_osvd において
  前半: W_tilde = W @ Q @ diag(√λ)          ← Q^T が欠落
  後半: V' = Q @ diag(1/√λ) @ [Q^T] @ V_r   ← Q^T は正しい H^{-1/2} の展開

H = Q Λ Q^T のとき H^{1/2} = Q Λ^{1/2} Q^T（Q^T が必要）であるため、
前半の W_tilde は実際には W @ H^{1/2} @ Q を計算している。

V_r は W @ H^{1/2} @ Q の右特異ベクトルであり、
W @ H^{1/2} の右特異ベクトル V_r_correct とは
  V_r_buggy = Q^T @ V_r_correct
という関係にある。

後半の V' の計算:
  buggy  : Q @ diag(1/√λ) @ Q^T @ V_r_buggy  = Q @ diag(1/√λ) @ Q^T @ Q^T @ V_r_correct
  correct: Q @ diag(1/√λ)                   @ V_r_buggy  = Q @ diag(1/√λ) @ Q^T @ V_r_correct

Q^T @ Q^T = (Q @ Q)^{-1} ≠ I（Q が対角でない限り）なので、
W_hat_buggy = U' @ V'_buggy^T は W の H 重み付き最良 rank-r 近似にならない。
"""

import pytest
import torch


def _hessian_error(W: torch.Tensor, W_hat: torch.Tensor, H: torch.Tensor) -> float:
    """tr((W - W_hat) H (W - W_hat)^T)"""
    E = W - W_hat
    return torch.trace(E @ H @ E.T).item()


def _osvd_correct(W: torch.Tensor, H: torch.Tensor, r: int) -> torch.Tensor:
    """
    正しい OSVD 実装: H^{1/2} = Q diag(√λ) Q^T を使う

    W_tilde = W @ H^{1/2}  → SVD → V' = H^{-1/2} @ V_r @ diag(√Σ)
    """
    eig_vals, eig_vecs = torch.linalg.eigh(H)
    eig_vals = eig_vals.clamp(min=1e-12)
    sqrt_eig = torch.sqrt(eig_vals)
    inv_sqrt_eig = 1.0 / sqrt_eig

    # 正しい H^{1/2}: Q diag(√λ) Q^T
    W_tilde = W @ eig_vecs @ torch.diag(sqrt_eig) @ eig_vecs.T

    U_w, S_w, Vh_w = torch.linalg.svd(W_tilde, full_matrices=False)
    r = min(r, S_w.numel())
    U_r, S_r, V_r = U_w[:, :r], S_w[:r], Vh_w[:r, :].T
    sqrt_S = torch.sqrt(S_r.clamp(min=1e-12))

    U_prime = U_r * sqrt_S[None, :]
    # V' = H^{-1/2} @ V_r @ diag(√Σ)  （H^{-1/2} = Q diag(1/√λ) Q^T）
    V_prime = eig_vecs @ torch.diag(inv_sqrt_eig) @ eig_vecs.T @ V_r @ torch.diag(sqrt_S)
    return U_prime @ V_prime.T


def _osvd_buggy(W: torch.Tensor, H: torch.Tensor, r: int) -> torch.Tensor:
    """
    現行実装（バグあり）: W_tilde に Q^T が欠落

    W_tilde = W @ Q @ diag(√λ)   ← Q^T なし（= W @ H^{1/2} @ Q）
    V' = Q @ diag(1/√λ) @ Q^T @ V_r @ diag(√Σ)  ← 後半は完全な H^{-1/2}

    前半で V_r は W @ H^{1/2} @ Q の右特異ベクトル (= Q^T @ V_r_correct) なので、
    後半に Q^T を掛けると Q^T @ Q^T = (Q^2)^{-1} ≠ I が入り込みズレが生じる。
    """
    eig_vals, eig_vecs = torch.linalg.eigh(H)
    eig_vals = eig_vals.clamp(min=1e-12)
    sqrt_eig = torch.sqrt(eig_vals)
    inv_sqrt_eig = 1.0 / sqrt_eig

    # バグ: Q^T が欠落 → W @ H^{1/2} @ Q を計算している
    W_tilde = W @ eig_vecs @ torch.diag(sqrt_eig)

    U_w, S_w, Vh_w = torch.linalg.svd(W_tilde, full_matrices=False)
    r = min(r, S_w.numel())
    U_r, S_r, V_r = U_w[:, :r], S_w[:r], Vh_w[:r, :].T
    sqrt_S = torch.sqrt(S_r.clamp(min=1e-12))

    U_prime = U_r * sqrt_S[None, :]
    # 後半は完全な H^{-1/2} を適用しているが、V_r が Q 回転済みなので不整合
    V_prime = eig_vecs @ torch.diag(inv_sqrt_eig) @ eig_vecs.T @ V_r @ torch.diag(sqrt_S)
    return U_prime @ V_prime.T


def _make_pd_matrix(m: int, seed: int, noise: float = 0.1) -> torch.Tensor:
    """非対角の正定値行列を生成（H が対角の場合はバグが顕在化しない）"""
    torch.manual_seed(seed)
    A = torch.randn(m, m)
    return A @ A.T + noise * torch.eye(m)


class TestOsvdHessianBug:
    """
    lowrank_osvd の H^{1/2} バグ: 数値的証明
    """

    @pytest.mark.parametrize("n,m,r,seed", [
        (16, 12, 3, 0),
        (32, 24, 4, 1),
        (8,  6,  2, 42),
    ])
    def test_buggy_has_larger_hessian_error(self, n, m, r, seed):
        """
        バグあり実装は、H 重み付き出力誤差が最適解より大きくなることを証明する。

        正しい OSVD は最小化問題の最適解を返すため、
        バグあり版の誤差 ≥ 正しい版の誤差 が成立する。
        両者が等しいのは H が対角行列の場合のみ（Q = I のとき）。
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

        # 正しい版は最適解なので、バグあり版より常に小さいか等しい
        assert err_correct <= err_buggy + 1e-6 * abs(err_correct), (
            f"correct ({err_correct:.6e}) should be ≤ buggy ({err_buggy:.6e})"
        )
        # 非対角 H では真に異なるはず（等しければバグが顕在化していない）
        assert abs(err_buggy - err_correct) > 1e-8 * abs(err_correct), (
            "Both errors are equal: bug is not manifesting "
            "(check whether H is effectively diagonal)"
        )

    def test_reconstruction_differs(self):
        """
        W_hat_correct と W_hat_buggy の行列が異なることを直接確認する。
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
        対角 H かつ固有値が昇順ソート済み（= Q が単位行列）の場合はバグが顕在化しないことを確認する。

        torch.linalg.eigh は固有値を昇順に返す。
        H が対角行列でかつ対角成分がすでに昇順に並んでいる場合のみ eig_vecs = I となり、
        バグの Q 回転が恒等変換になってバグが出ない。

        これにより、バグが Q の非対角（または並べ替え）成分に起因することが裏付けられる。
        """
        torch.manual_seed(5)
        n, m, r = 10, 8, 3
        W = torch.randn(n, m, dtype=torch.float64)
        # 対角 H: 固有値を昇順に並べることで eigh が返す Q が単位行列になる
        diag_vals = torch.sort(torch.rand(m, dtype=torch.float64) + 0.5).values
        H = torch.diag(diag_vals)

        # eigh が単位行列を返すことを事前確認
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

        # Q = I のときは両者が一致する（= バグが Q の回転に起因する証拠）
        assert max_diff < 1e-5, (
            "With sorted diagonal H (Q=I), correct and buggy should agree "
            f"(got max diff {max_diff:.2e})"
        )

    def test_buggy_inconsistency_analytically(self):
        """
        解析的な不整合の確認:
          W_tilde_buggy = W @ Q @ diag(√λ)  は  W @ H^{1/2} @ Q  に等しい。
          したがって V_r_buggy = Q^T @ V_r_correct。
          後半に H^{-1/2} = Q diag(1/√λ) Q^T を掛けると
            Q diag(1/√λ) Q^T @ Q^T @ V_r_correct
          = Q diag(1/√λ) @ (Q @ Q)^{-1} @ V_r_correct  （Q^T @ Q^T ≠ I）
          となり、余分な (Q^2)^{-1} が残る。
        """
        torch.manual_seed(99)
        m = 6
        H = _make_pd_matrix(m, seed=99).to(torch.float64)

        eig_vals, Q = torch.linalg.eigh(H)
        eig_vals = eig_vals.clamp(min=1e-12)
        sqrt_eig = torch.sqrt(eig_vals)

        # H^{1/2} @ Q = Q @ diag(√λ)  を確認
        H_half = Q @ torch.diag(sqrt_eig) @ Q.T
        assert torch.allclose(H_half @ Q, Q @ torch.diag(sqrt_eig), atol=1e-10), \
            "H^{1/2} @ Q should equal Q @ diag(sqrt_eig)"

        # W_tilde_buggy = W @ Q @ diag(√λ) = W @ H^{1/2} @ Q ≠ W @ H^{1/2}
        torch.manual_seed(13)
        W = torch.randn(8, m, dtype=torch.float64)
        W_tilde_buggy   = W @ Q @ torch.diag(sqrt_eig)
        W_tilde_correct = W @ H_half

        assert not torch.allclose(W_tilde_buggy, W_tilde_correct, atol=1e-8), \
            "W_tilde_buggy and W_tilde_correct should differ for non-diagonal H"

        # Q^T @ Q^T ≠ I を確認（バグの根本原因）
        QtQt = Q.T @ Q.T
        assert not torch.allclose(QtQt, torch.eye(m, dtype=torch.float64), atol=1e-6), \
            "Q^T @ Q^T should not be identity (confirming the extra rotation is spurious)"
