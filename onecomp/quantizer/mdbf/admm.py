# -*- coding: utf-8 -*-
"""
MSVID-ADMM: Multi-Scale SVID を用いた ADMM 最適化（Phase 2）

Phase 1 で初期化された MSVIDParams を初期値として,
各パス p' に対して残差 W - Σ_{p≠p'} W^p を最小化する.

Activation-aware拡張:
- Hessian-based: H = X^T @ X / N を使用してメモリ効率良く出力誤差を最小化
- 目的関数: min tr((W - W_hat) @ H @ (W - W_hat)^T)
"""

from typing import List, Optional, Tuple

import torch

from .initialize import MSVIDParams
from .utils import (
    cleanup_gpu_memory,
    compute_hessian_error,
    ensure_float32,
    ensure_float32_clone,
    reconstruct_weight,
    symmetrize_matrix,
    to_binary_sign,
)


# =============================================================================
# ヘルパー関数
# =============================================================================


def _tsvd_block_power(
    M: torch.Tensor,
    k: int,
    n_iter: int = 5,
    oversample: int = 4,
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Block power iteration (subspace iteration) による上位k特異値分解の近似.

    Args:
        M: (n, m) float32推奨
        k: 取りたいランク
        n_iter: power/subspace iteration回数（1〜3が目安）
        oversample: 過剰サンプリング（2〜8が目安）
        seed: ランダムシード（Noneならグローバルシードを使用）

    Returns:
        U: (n, k)
        S: (k,)
        V: (m, k)
        such that M ≈ (U * S) @ V^T
    """
    n, m = M.shape
    k_eff = min(k, n, m)
    if k_eff <= 0:
        raise ValueError("k must be >= 1")
    k0 = min(k_eff + oversample, n, m)

    # 右部分空間の初期化（ランダム）
    if seed is not None:
        gen = torch.Generator(device=M.device).manual_seed(seed)
        V = torch.randn(m, k0, device=M.device, dtype=M.dtype, generator=gen)
    else:
        V = torch.randn(m, k0, device=M.device, dtype=M.dtype)
    V, _ = torch.linalg.qr(V, mode="reduced")  # (m, k0)

    # subspace iteration
    for _ in range(max(0, n_iter)):
        U = M @ V                              # (n, k0)
        U, _ = torch.linalg.qr(U, mode="reduced")
        V = M.T @ U                            # (m, k0)
        V, _ = torch.linalg.qr(V, mode="reduced")

    # 最後に M@V をQRして小行列 R を作る
    Y = M @ V                                  # (n, k0)
    U, R = torch.linalg.qr(Y, mode="reduced")  # U:(n,k0), R:(k0,k0)
    del Y

    # 小行列だけSVD（デフォルトドライバ使用）
    Ur, S, Vhr = torch.linalg.svd(R, full_matrices=False)
    del R

    U = U @ Ur[:, :k_eff]                      # (n, k)
    V = V @ Vhr[:k_eff, :].T                   # (m, k)
    S = S[:k_eff]

    return U, S, V


def _decompose_abs_matrix(
    M_abs: torch.Tensor,
    l: int,
    for_transpose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    振幅行列 |M| を rank-l で分解

    Args:
        M_abs: 振幅行列 (a, b)
        l: ランク
        for_transpose: True の場合, M^T を分解（G側用）

    Returns:
        (scale_left, scale_right): |M| ≈ scale_left @ scale_right^T
    """
    if for_transpose:
        M_abs = M_abs.T

    a, b = M_abs.shape
    l_eff = min(l, min(a, b))

    # 非負行列に対する正則化（定数加算、乱数は負の値を生むため避ける）
    eps = 1e-6 * M_abs.max().clamp(min=1e-12)
    M_abs_reg = M_abs + eps

    U, S, Vh = torch.linalg.svd(M_abs_reg, full_matrices=False)
    l_eff = min(l_eff, S.numel())
    sqrt_S = torch.sqrt(S[:l_eff].clamp(min=1e-12))
    left = U[:, :l_eff] * sqrt_S[None, :]
    right = Vh[:l_eff, :].T * sqrt_S[None, :]
    del U, S, Vh, M_abs_reg

    return left, right


def _solve_linear_system(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    L_cached: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    線形方程式 lhs @ X = rhs を解く（Cholesky分解をキャッシュ利用）

    Args:
        lhs: 左辺行列 (k, k)
        rhs: 右辺 (k, ...) or (k,)
        L_cached: Cholesky分解済み行列（Noneなら直接解く）

    Returns:
        解 X

    Note:
        フォールバック順序: cholesky_solve → solve → lstsq
    """
    if L_cached is not None:
        if rhs.dim() == 1:
            return torch.cholesky_solve(rhs.unsqueeze(-1), L_cached).squeeze(-1)
        return torch.cholesky_solve(rhs, L_cached)

    # Cholesky分解がキャッシュされていない場合は直接解く
    try:
        return torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        # 特異行列または数値的に不安定な場合は最小二乗法にフォールバック
        if rhs.dim() == 1:
            return torch.linalg.lstsq(lhs, rhs.unsqueeze(-1)).solution.squeeze(-1)
        return torch.linalg.lstsq(lhs, rhs).solution


# =============================================================================
# MSVID射影
# =============================================================================


def svd_abs_rank_l(
    W: torch.Tensor,
    l: int,
    seed: int = None,
) -> torch.Tensor:
    """
    MSVID (rank-l) への射影: Z = sign(W) * TSVD_l(|W|)

    符号を sign(W) に固定し, 振幅に rank-l 制約を適用する.

    Args:
        W: 入力行列
        l: Multi-scaleランク
        seed: ランダムシード（Noneならグローバルシードを使用、整数なら決定的）
    """
    S = torch.sign(W)
    S[S == 0] = 1.0

    if l == 1:
        # パワーイテレーションでrank-1近似（randn初期化）
        # 反復回数は l>=2 の block power iteration (n_iter=5) と統一
        W_abs = W.abs().float()
        n, m_dim = W_abs.shape

        # ランダム初期化（縮退固有値がある場合に頑健）
        # seed指定時は決定的な初期化（再現性のため）
        if seed is not None:
            gen = torch.Generator(device=W.device).manual_seed(seed)
            a = torch.randn(n, device=W.device, dtype=torch.float32, generator=gen)
            m_vec = torch.randn(m_dim, device=W.device, dtype=torch.float32, generator=gen)
        else:
            a = torch.randn(n, device=W.device, dtype=torch.float32)
            m_vec = torch.randn(m_dim, device=W.device, dtype=torch.float32)
        a = a / (torch.norm(a) + 1e-12)
        m_vec = m_vec / (torch.norm(m_vec) + 1e-12)

        for _ in range(5):
            a = W_abs @ m_vec
            a.div_(a.norm().clamp(min=1e-12))
            m_vec = W_abs.T @ a
            m_vec.div_(m_vec.norm().clamp(min=1e-12))

        sigma = torch.dot(a, W_abs @ m_vec).clamp(min=1e-12)
        root_sigma = torch.sqrt(sigma)
        a = a * root_sigma
        m_vec = m_vec * root_sigma

        amp = torch.outer(a, m_vec)
        del W_abs, a, m_vec
    else:
        # Block power iteration で高速化
        W_abs = W.abs().float()
        n_dim, m_dim = W_abs.shape
        kmax = min(n_dim, m_dim)

        if l >= kmax:
            amp = W_abs
        else:
            l_eff = max(1, min(l, kmax - 1))
            # block power iteration で高速化（QRとGEMM主体なので正則化不要）
            U_l, S_l, V_l = _tsvd_block_power(W_abs, k=l_eff, seed=seed)
            amp = (U_l * S_l[None, :]) @ V_l.T
            del U_l, S_l, V_l
        del W_abs

    Z = S * amp.to(W.dtype)
    del S, amp
    return Z


# =============================================================================
# パラメータ変換
# =============================================================================


def _params_to_factor_matrices(
    params: MSVIDParams,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MSVIDParams から因子行列 (F, G) を構成

    F = S_A * (A_amp @ Q_U_amp^T)
    G = S_B * (Q_V_amp @ B_amp^T)
    """
    A_sign = params.A_sign.to(device).float()
    B_sign = params.B_sign.to(device).float()
    A_amp = params.A_amp.to(device).float()
    B_amp = params.B_amp.to(device).float()
    Q_U_amp = params.Q_U_amp.to(device).float()
    Q_V_amp = params.Q_V_amp.to(device).float()

    amp_A = A_amp @ Q_U_amp.T
    F = A_sign * amp_A

    amp_B = Q_V_amp @ B_amp.T
    G = B_sign * amp_B

    del A_sign, B_sign, A_amp, B_amp, Q_U_amp, Q_V_amp, amp_A, amp_B
    return F, G


def _factor_matrices_to_params(
    F: torch.Tensor,
    G: torch.Tensor,
    l: int,
    dtype: torch.dtype,
) -> MSVIDParams:
    """
    因子行列 (F, G) から MSVIDParams を抽出

    Note:
        F, G は ADMM 射影後なので |F|, |G| は既に rank-l.
    """
    A_sign = to_binary_sign(F)
    B_sign = to_binary_sign(G)

    F_abs = F.abs().float()
    G_abs = G.abs().float()

    # |F| ≈ A_amp @ Q_U_amp^T
    A_amp, Q_U_amp = _decompose_abs_matrix(F_abs, l)
    del F_abs

    # |G| ≈ Q_V_amp @ B_amp^T (G^T を分解)
    B_amp, Q_V_amp = _decompose_abs_matrix(G_abs, l, for_transpose=True)
    del G_abs

    return MSVIDParams(
        A_sign=A_sign.to(dtype),
        B_sign=B_sign.to(dtype),
        A_amp=A_amp.to(dtype),
        B_amp=B_amp.to(dtype),
        Q_U_amp=Q_U_amp.to(dtype),
        Q_V_amp=Q_V_amp.to(dtype),
    )


# =============================================================================
# 片側固定ADMM
# =============================================================================


def _admm_optimize_one_side(
    Fixed: torch.Tensor,
    W_target: torch.Tensor,
    Z_init: torch.Tensor,
    U_init: torch.Tensor,
    l: int,
    inner_iters: int = 3,
    reg: float = 0.03,
    rho_start: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    片側固定のADMM最適化（DBF互換の固定rhoモード）

    min ||W_target - Fixed @ Z||_F^2 + λ*||Z||_F^2  s.t. Z in MSVID(l)

    ここで λ = reg * mean(diag(Fixed^T @ Fixed))
    （Fixed^T @ Fixed の対角成分平均に比例する正則化）

    Args:
        Fixed: 固定側行列
        W_target: ターゲット
        Z_init, U_init: 初期値
        l: Multi-scaleランク
        inner_iters: 内側反復回数
        reg: 正則化係数（Fixed^T @ Fixed の対角成分平均に乗じる）
        rho_start: 初期rho

    Returns:
        (Z, U): 最適化後の変数
    """
    device = Fixed.device
    Fixed = ensure_float32(Fixed)
    W_target = ensure_float32(W_target)
    Z = ensure_float32_clone(Z_init)
    U = ensure_float32_clone(U_init)

    k = Fixed.shape[1]
    I = torch.eye(k, device=device, dtype=torch.float32)

    XX = Fixed.T @ Fixed
    if reg > 0:
        XX = XX + I * (XX.diag().mean() * reg)
    XY = Fixed.T @ W_target

    Z, U = _admm_fixed_rho_loop(XX, XY, I, Z, U, l, inner_iters, rho_start)

    del XX, XY, I
    return Z.float(), U.float()


def _admm_fixed_rho_loop(
    XX: torch.Tensor,
    XY: torch.Tensor,
    I: torch.Tensor,
    Z: torch.Tensor,
    U: torch.Tensor,
    l: int,
    inner_iters: int,
    rho_start: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """固定rhoモードのADMM内側ループ"""
    rho = 1.0
    lhs_rho = XX + rho * I
    lhs_rho_start = XX + rho_start * I

    try:
        L_rho = torch.linalg.cholesky(lhs_rho)
    except RuntimeError:
        L_rho = None

    try:
        L_rho_start = torch.linalg.cholesky(lhs_rho_start)
    except RuntimeError:
        L_rho_start = None

    # 最初のB更新 (rho_start)
    rhs = XY + rho_start * (Z - U)
    B = _solve_linear_system(lhs_rho_start, rhs, L_rho_start)

    # 残り反復 (rho=1.0)
    for _ in range(inner_iters - 1):
        Z = svd_abs_rank_l(B + U, l)
        U = U + (B - Z)
        rhs = XY + rho * (Z - U)
        B = _solve_linear_system(lhs_rho, rhs, L_rho)

    # 最後のZ, U更新
    Z = svd_abs_rank_l(B + U, l)
    U = U + (B - Z)

    del lhs_rho, lhs_rho_start
    if L_rho is not None:
        del L_rho
    if L_rho_start is not None:
        del L_rho_start

    return Z, U


# =============================================================================
# 単一パスADMM最適化
# =============================================================================


def _admm_refine_single_path(
    W_target: torch.Tensor,
    F_init: torch.Tensor,
    G_init: torch.Tensor,
    l: int,
    iters: int = 50,
    inner_iters: int = 3,
    reg: float = 0.03,
    verbose: bool = False,
    path_idx: int = 0,
    H: Optional[torch.Tensor] = None,
    nsamples: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    単一パスに対するADMM最適化（F<->G交互最適化、DBF互換の固定rhoモード）

    min ||W_target - F @ G||_F^2  s.t. F,G in MSVID(l)
    """
    device = W_target.device
    n, r = F_init.shape
    _, m = G_init.shape

    # F側は転置形式 (r, n) で保持
    Zf_T = ensure_float32_clone(F_init.T)
    Uf_T = torch.zeros(r, n, device=device, dtype=torch.float32)
    Zg = ensure_float32_clone(G_init)
    Ug = torch.zeros_like(Zg)

    W_float = ensure_float32(W_target)
    orig_norm = torch.norm(W_float, p='fro').item() + 1e-12
    W_float_T = W_float.T

    # 出力誤差表示用（Hは呼び出し元で既に対称化されている前提）
    H_float = None
    orig_output_err = None
    if H is not None:
        H_float = ensure_float32(H, device)
        WH = W_float @ H_float
        orig_output_err = (nsamples * torch.sum(WH * W_float)).item() + 1e-12

    # ===== スケール正規化戦略 =====
    # W ≈ F @ G の交互最適化において、F と G のスケールには自由度がある。
    # (F * c) @ (G / c) = F @ G なので、スケールを一方に集約する必要がある。
    #
    # 戦略:
    # 1. F更新時: G を行ごとに正規化 → スケールは F に吸収される
    # 2. G更新時: F を列ごとに正規化 → スケールは G に吸収される
    # 3. 最終出力: F を列ごとに正規化して返す → スケールは G に集約される
    #
    # これにより、最終的な F @ G は元の W を近似し、かつ F の列ノルムが1に揃う。

    for itt in range(iters):
        rho_start = 0.03 + (1.0 - 0.03) * min(1.0, itt / max(1, iters - 3)) ** 3

        # F更新 (G固定): G の各行を正規化して使用
        # 最適化問題: min ||W^T - G_normalized @ F^T||_F^2 s.t. F in MSVID(l)
        mid_norm_g = torch.norm(Zg, dim=1) + 1e-12
        Zg_normalized = Zg / mid_norm_g[:, None]

        Zf_T, Uf_T = _admm_optimize_one_side(
            Fixed=Zg_normalized.T,
            W_target=W_float_T,
            Z_init=Zf_T,
            U_init=Uf_T,
            l=l,
            inner_iters=inner_iters,
            reg=reg,
            rho_start=rho_start,
        )

        # G更新 (F固定): F の各列を正規化して使用
        # 最適化問題: min ||W - F_normalized @ G||_F^2 s.t. G in MSVID(l)
        mid_norm_f = torch.norm(Zf_T, dim=1) + 1e-12
        Zf_normalized = Zf_T.T / mid_norm_f[None, :]

        Zg, Ug = _admm_optimize_one_side(
            Fixed=Zf_normalized,
            W_target=W_float,
            Z_init=Zg,
            U_init=Ug,
            l=l,
            inner_iters=inner_iters,
            reg=reg,
            rho_start=rho_start,
        )

        # ログ出力
        if verbose and (itt % max(10, iters // 5) == 0 or itt == iters - 1):
            mid_norm_final = torch.norm(Zf_T, dim=1) + 1e-12
            W_recon = (Zf_T.T / mid_norm_final[None, :]) @ Zg
            E = W_float - W_recon

            if H_float is not None:
                output_err = compute_hessian_error(E, H_float, nsamples)
                print(f"    Path {path_idx+1}, Outer Step {itt:3d}: "
                      f"output_error = {output_err:.4e} (rel: {output_err/orig_output_err:.4f}), "
                      f"rho_start={rho_start:.3f}")
            else:
                err = torch.norm(E, p='fro').item()
                print(f"    Path {path_idx+1}, Outer Step {itt:3d}: "
                      f"weight_error = {err:.4e} (rel: {err/orig_norm:.4f}), "
                      f"rho_start={rho_start:.3f}")
            del W_recon, E

    del W_float, W_float_T, Uf_T, Ug
    if H_float is not None:
        del H_float

    # 最終スケール正規化: F の列ノルムを1にして、スケールを G に集約
    # これにより W ≈ F_normalized @ G が成立し、F は正規化された因子行列となる
    mid_norm_final = torch.norm(Zf_T, dim=1) + 1e-12
    Zf_normalized = Zf_T.T / mid_norm_final[None, :]

    return Zf_normalized, Zg


# =============================================================================
# 標準ADMM最適化
# =============================================================================


def optimize_msvid_admm(
    W_original: torch.Tensor,
    params_list: List[MSVIDParams],
    l: int,
    iters: int = 260,
    inner_iters: int = 3,
    reg: float = 0.03,
    verbose: bool = True,
    H: Optional[torch.Tensor] = None,
    nsamples: int = 1,
) -> Tuple[List[MSVIDParams], torch.Tensor]:
    """
    ADMM最適化によりMSVIDパラメータを精緻化 (Phase 2, DBF互換の固定rhoモード)

    各パス p' に対して残差 W - Σ_{p≠p'} W^p を最小化
    """
    device = W_original.device
    dtype = W_original.dtype
    n, m = W_original.shape
    P = len(params_list)

    if verbose:
        print(f"[MSVID-ADMM] outer_iters={iters}, inner_iters={inner_iters}, "
              f"l={l}, P={P}, reg={reg}")

    W_float = ensure_float32(W_original)
    orig_norm = torch.norm(W_float, p='fro').item() + 1e-12

    # Hessianの前処理
    H_float = None
    orig_output_err = None
    if H is not None:
        H_float = symmetrize_matrix(ensure_float32(H, device))
        WH = W_float @ H_float
        orig_output_err = (nsamples * torch.sum(WH * W_float)).item() + 1e-12

    # 初期誤差
    W_init_recon = torch.zeros(n, m, device=device, dtype=torch.float32)
    for p in params_list:
        W_p = reconstruct_weight(
            p.A_sign.to(device), p.B_sign.to(device),
            p.A_amp.to(device), p.B_amp.to(device),
            p.Q_U_amp.to(device), p.Q_V_amp.to(device),
        )
        W_init_recon += W_p
        del W_p

    E_init = W_float - W_init_recon
    if H_float is not None:
        init_error = compute_hessian_error(E_init, H_float, nsamples)
        if verbose:
            print(f"[MSVID-ADMM] Initial output_error: {init_error:.4e} "
                  f"(rel: {init_error/orig_output_err:.4f})")
    else:
        init_error = torch.norm(E_init, p='fro').item()
        if verbose:
            print(f"[MSVID-ADMM] Initial weight_error: {init_error:.4e} "
                  f"(rel: {init_error/orig_norm:.4f})")
    del W_init_recon, E_init

    # 因子行列を初期化
    factor_list = [_params_to_factor_matrices(p, device) for p in params_list]

    # 各パスを最適化
    optimized_factors = []
    for p_idx in range(P):
        if verbose:
            print(f"[MSVID-ADMM] Optimizing path {p_idx+1}/{P}...")

        # 残差
        W_target = W_float.clone()
        for other_idx in range(P):
            if other_idx != p_idx:
                F_other, G_other = (optimized_factors[other_idx]
                                    if other_idx < len(optimized_factors)
                                    else factor_list[other_idx])
                W_target -= F_other @ G_other

        F_init, G_init = factor_list[p_idx]
        F_opt, G_opt = _admm_refine_single_path(
            W_target=W_target,
            F_init=F_init,
            G_init=G_init,
            l=l,
            iters=iters,
            inner_iters=inner_iters,
            reg=reg,
            verbose=verbose,
            path_idx=p_idx,
            H=H_float,
            nsamples=nsamples,
        )
        optimized_factors.append((F_opt, G_opt))
        del W_target, F_init, G_init

    # 最終再構成
    W_recon = torch.zeros(n, m, device=device, dtype=torch.float32)
    for F_opt, G_opt in optimized_factors:
        W_recon += F_opt @ G_opt

    if verbose:
        E_final = W_float - W_recon
        if H_float is not None:
            final_error = compute_hessian_error(E_final, H_float, nsamples)
            improvement = (init_error - final_error) / init_error * 100
            print(f"[MSVID-ADMM] Final output_error: {final_error:.4e} "
                  f"(rel: {final_error/orig_output_err:.4f})")
            print(f"[MSVID-ADMM] Improvement: {improvement:+.2f}%")
        else:
            final_error = torch.norm(E_final, p='fro').item()
            improvement = (init_error - final_error) / init_error * 100
            print(f"[MSVID-ADMM] Final weight_error: {final_error:.4e} "
                  f"(rel: {final_error/orig_norm:.4f})")
            print(f"[MSVID-ADMM] Improvement: {improvement:+.2f}%")
        del E_final

    # MSVIDParamsに変換
    optimized_params = [
        _factor_matrices_to_params(F_opt, G_opt, l, dtype)
        for F_opt, G_opt in optimized_factors
    ]

    if verbose:
        W_recon_params = torch.zeros(n, m, device=device, dtype=torch.float32)
        for p in optimized_params:
            W_p = reconstruct_weight(
                p.A_sign.to(device), p.B_sign.to(device),
                p.A_amp.to(device), p.B_amp.to(device),
                p.Q_U_amp.to(device), p.Q_V_amp.to(device),
            )
            W_recon_params += W_p
            del W_p
        E_params = W_float - W_recon_params
        if H_float is not None:
            params_error = compute_hessian_error(E_params, H_float, nsamples)
            print(f"[MSVID-ADMM] Error (via params): {params_error:.4e} "
                  f"(rel: {params_error/orig_output_err:.4f})")
        else:
            params_error = torch.norm(E_params, p='fro').item()
            print(f"[MSVID-ADMM] Error (via params): {params_error:.4e} "
                  f"(rel: {params_error/orig_norm:.4f})")
        del W_recon_params, E_params

    del W_float
    if H_float is not None:
        del H_float
    for F, G in factor_list:
        del F, G
    for F, G in optimized_factors:
        del F, G
    cleanup_gpu_memory()

    return optimized_params, W_recon.to(dtype)


# =============================================================================
# Hessian-based Activation-Aware ADMM
# =============================================================================


def _admm_refine_single_path_hessian(
    H: torch.Tensor,
    nsamples: int,
    W_target: torch.Tensor,
    F_init: torch.Tensor,
    G_init: torch.Tensor,
    l: int,
    iters: int = 50,
    inner_iters: int = 3,
    reg: float = 0.03,
    verbose: bool = False,
    path_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hessian-based Activation-aware版: 単一パスADMM最適化

    目的関数: min tr((W - VU^T) @ H @ (W - VU^T)^T)  s.t. U,V in MSVID(l)

    安定化のための修正:
    - H を一貫して正則化 (H_use = H + eps * I) - 零空間方向へのドリフトを抑制
    - バランスゲージ (V, U のノルムを均等化) - スケールの極端化を防止
    - 常に小さいridgeをU/V更新に追加 - 発散方向への逃げを抑制
    - 定期的に誤差をチェックし最良結果を保存 - 非単調収束への対応
    """
    device = H.device
    m = H.shape[0]
    n, r = F_init.shape
    N = float(nsamples)

    # V = F (n, r), U = G^T (m, r)
    V = ensure_float32_clone(F_init)
    U = ensure_float32_clone(G_init.T)

    # 最良結果の追跡（問題3 & 5）
    best_V = V.clone()
    best_U = U.clone()
    best_err = float('inf')
    check_interval = max(10, iters // 10)  # 10回または全体の1/10ごとにチェック

    LamU = torch.zeros_like(U)
    GamV = torch.zeros_like(V)
    I_r = torch.eye(r, device=device, dtype=torch.float32)
    I_m = torch.eye(m, device=device, dtype=torch.float32)

    W_float = ensure_float32(W_target)
    H0 = symmetrize_matrix(ensure_float32(H))

    # H を一貫して正則化して使う（eigh と更新式の両方で同じ H_use を使用）
    # これにより H の零空間方向へのドリフトを抑制
    H_diag_mean = H0.diag().mean().clamp(min=1e-12)
    eps_H = 1e-3 * H_diag_mean  # 固定の正則化係数
    H_use = H0 + eps_H * I_m
    del H0

    # Hessianの固有分解を事前計算
    H_eig_vals, H_eig_vecs = torch.linalg.eigh(H_use)
    H_eig_vals = H_eig_vals.clamp(min=1e-12)

    W_norm = torch.norm(W_float, p='fro').item() + 1e-12
    rho_scale = N * H_diag_mean

    for itt in range(iters):
        rho_base = 0.03 + (1.0 - 0.03) * min(1.0, itt / max(1, iters - 3)) ** 3
        rho = rho_base * rho_scale
        rho_over_N = rho / N

        # ===== V更新 (U固定) =====
        # 正規方程式: V (U^T H_use U + (ρ/N + λ_v) I) = W H_use U + ρ/N (Z_V - Γ_V)
        HU = H_use @ U
        UtHU = U.T @ HU
        WHU = W_float @ HU

        # 常に小さいridgeを入れて発散を抑制
        lambda_v = reg * UtHU.diag().mean().clamp(min=1e-12)
        lhs_v = UtHU + (rho_over_N + lambda_v) * I_r

        L_v = None
        try:
            L_v = torch.linalg.cholesky(lhs_v)
        except RuntimeError:
            L_v = None

        for _ in range(inner_iters):
            rhs = WHU + rho_over_N * (V - GamV)
            if L_v is not None:
                Vtilde = torch.cholesky_solve(rhs.T, L_v).T
            else:
                try:
                    Vtilde = torch.linalg.solve(lhs_v, rhs.T).T
                except RuntimeError:
                    Vtilde = torch.linalg.lstsq(lhs_v, rhs.T).solution.T

            V = svd_abs_rank_l(Vtilde + GamV, l)
            GamV = GamV + (Vtilde - V)

        # 中間テンソルの解放（問題10）
        del lhs_v, HU, UtHU, WHU
        if L_v is not None:
            del L_v

        # NaN/Inf 検査: 検出したら最良結果を使用してループを抜ける（問題3）
        if not torch.isfinite(V).all() or not torch.isfinite(U).all():
            bad_v = (~torch.isfinite(V)).sum().item()
            bad_u = (~torch.isfinite(U)).sum().item()
            print(f"    [Hessian-ADMM] WARNING: Diverged at step {itt} "
                  f"(V: {bad_v}, U: {bad_u} NaN/Inf). Using best result.")
            V = best_V
            U = best_U
            break

        # ===== U更新 (V固定) =====
        # V^T V の固有分解が必要
        VtV = V.T @ V
        VtV = (VtV + VtV.T) * 0.5  # 対称性を保証
        r_dim = VtV.size(0)
        diag_mean_vtv = VtV.diag().mean().clamp(min=1e-12)
        eps_damp = 1e-4 * diag_mean_vtv
        I_VtV = torch.eye(r_dim, device=VtV.device, dtype=VtV.dtype)
        VtV_reg = VtV + eps_damp * I_VtV
        del VtV

        eig_success = False
        try:
            sigma, Q = torch.linalg.eigh(VtV_reg)
            eig_success = True
        except RuntimeError:
            pass

        if not eig_success:
            # より強いダンピングで再試行
            eps_damp_strong = 1e-2 * diag_mean_vtv
            VtV_reg_strong = VtV_reg + (eps_damp_strong - eps_damp) * I_VtV
            try:
                sigma, Q = torch.linalg.eigh(VtV_reg_strong)
                eig_success = True
            except RuntimeError:
                pass
            del VtV_reg_strong

        if not eig_success:
            # Fallback: block power method
            k_eff = min(l * 25, r_dim)
            try:
                Q_k, sigma_k, _ = _tsvd_block_power(VtV_reg, k=k_eff, n_iter=5, oversample=4)
                sigma = torch.full((r_dim,), eps_damp, device=V.device, dtype=V.dtype)
                sigma[:k_eff] = sigma_k
                del sigma_k
                if k_eff < r_dim:
                    Q_rest = torch.randn(r_dim, r_dim - k_eff, device=V.device, dtype=V.dtype)
                    Q_rest = Q_rest - Q_k @ (Q_k.T @ Q_rest)
                    Q_rest, _ = torch.linalg.qr(Q_rest, mode='reduced')
                    Q = torch.cat([Q_k, Q_rest], dim=1)
                    del Q_rest
                else:
                    Q = Q_k
                del Q_k
                eig_success = True
            except RuntimeError:
                # 最終フォールバック: identity
                sigma = torch.full((r_dim,), eps_damp, device=V.device, dtype=V.dtype)
                Q = torch.eye(r_dim, device=V.device, dtype=V.dtype)
                eig_success = True

        del VtV_reg, I_VtV
        sigma = sigma.clamp(min=1e-12)

        WtV = W_float.T @ V
        HWtV = H_use @ WtV
        del WtV

        # U側にも正則化を追加
        Hscale = H_eig_vals.mean().clamp(min=1e-12)
        lambda_u = reg * Hscale

        for _ in range(inner_iters):
            RHS = HWtV + rho_over_N * (U - LamU)
            B = RHS @ Q
            QtB = H_eig_vecs.T @ B
            denom = sigma[None, :] * H_eig_vals[:, None] + rho_over_N + lambda_u
            denom = denom.clamp(min=1e-12)
            UQ = H_eig_vecs @ (QtB / denom)
            Utilde = UQ @ Q.T

            U = svd_abs_rank_l(Utilde + LamU, l)
            LamU = LamU + (Utilde - U)

        # 中間テンソルの解放（問題10）
        del HWtV, Q, sigma

        # U更新後のNaN/Inf検査
        if not torch.isfinite(U).all():
            bad_u = (~torch.isfinite(U)).sum().item()
            print(f"    [Hessian-ADMM] WARNING: U diverged at step {itt} "
                  f"({bad_u} NaN/Inf). Using best result.")
            V = best_V
            U = best_U
            break

        # バランスゲージ固定 (外側反復1回につき1回)
        # 単位ノルムではなく、V と U のノルムをバランスさせる
        # d_j = sqrt(||V_j|| / ||U_j||) でスケールを均等配分
        with torch.no_grad():
            v_norm = torch.linalg.norm(V, dim=0).clamp(min=1e-6)
            u_norm = torch.linalg.norm(U, dim=0).clamp(min=1e-6)
            d = torch.sqrt(v_norm / u_norm)
            # 極端なスケール変換を禁止 (発散止めに効く)
            d = d.clamp(min=1e-2, max=1e2)

            V = V / d[None, :]
            U = U * d[None, :]
            GamV = GamV / d[None, :]
            LamU = LamU * d[None, :]

        # 定期的に誤差をチェックし最良結果を保存（問題5）
        if itt % check_interval == 0 or itt == iters - 1:
            with torch.no_grad():
                W_recon = V @ U.T
                E = W_float - W_recon
                current_err = compute_hessian_error(E, H_use, N)

                if current_err < best_err:
                    best_err = current_err
                    best_V = V.clone()
                    best_U = U.clone()

                if verbose:
                    weight_err = torch.norm(E, p='fro').item()
                    print(f"    [Hessian-Act] Path {path_idx+1}, Step {itt:3d}: "
                          f"output_err={current_err:.4e}, weight_err={weight_err:.4e} "
                          f"(rel: {weight_err/W_norm:.4f}), rho_base={rho_base:.4f}"
                          f"{' *' if current_err <= best_err else ''}")

    # 最終結果: 最良結果を使用
    F_opt = best_V
    G_opt = best_U.T

    del H_eig_vals, H_eig_vecs, LamU, GamV, H_use, W_float, I_r, I_m
    del V, U, best_V, best_U
    cleanup_gpu_memory()

    return F_opt, G_opt


def optimize_msvid_admm_hessian(
    W_original: torch.Tensor,
    params_list: List[MSVIDParams],
    l: int,
    H: torch.Tensor,
    nsamples: int,
    iters: int = 260,
    inner_iters: int = 3,
    reg: float = 0.03,
    verbose: bool = True,
) -> Tuple[List[MSVIDParams], torch.Tensor]:
    """
    Hessian-based Activation-aware ADMM最適化 (Phase 2) - P=1専用

    目的関数: N * tr((W - W_hat) @ H @ (W - W_hat)^T)
    """
    device = W_original.device
    dtype = W_original.dtype
    n, m = W_original.shape

    if len(params_list) != 1:
        raise ValueError(f"optimize_msvid_admm_hessian only supports P=1, got P={len(params_list)}")

    if H.shape != (m, m):
        raise ValueError(f"H shape must be ({m}, {m}), got {H.shape}")

    H = symmetrize_matrix(ensure_float32(H, device))

    if verbose:
        print(f"[Hessian-ADMM] iters={iters}, inner={inner_iters}, l={l}, N={nsamples}")

    W_float = ensure_float32(W_original)
    orig_norm = torch.norm(W_float, p='fro').item() + 1e-12

    def compute_hess_err(W_hat):
        return compute_hessian_error(W_float - W_hat, H, nsamples)

    # 初期再構成と誤差
    p = params_list[0]
    W_init = reconstruct_weight(
        p.A_sign.to(device), p.B_sign.to(device),
        p.A_amp.to(device), p.B_amp.to(device),
        p.Q_U_amp.to(device), p.Q_V_amp.to(device),
    )

    init_weight_err = torch.norm(W_float - W_init, p='fro').item()
    init_hess_err = compute_hess_err(W_init)

    if verbose:
        print(f"  Initial: output_err={init_hess_err:.4e}, "
              f"weight_err={init_weight_err:.4e} (rel={init_weight_err/orig_norm:.4f})")
    del W_init

    # ADMM最適化
    F_init, G_init = _params_to_factor_matrices(p, device)
    F_opt, G_opt = _admm_refine_single_path_hessian(
        H=H,
        nsamples=nsamples,
        W_target=W_float,
        F_init=F_init,
        G_init=G_init,
        l=l,
        iters=iters,
        inner_iters=inner_iters,
        reg=reg,
        verbose=verbose,
        path_idx=0,
    )

    W_recon = F_opt @ G_opt
    final_weight_err = torch.norm(W_float - W_recon, p='fro').item()
    final_hess_err = compute_hess_err(W_recon)

    if verbose:
        weight_impr = (init_weight_err - final_weight_err) / init_weight_err * 100
        output_impr = (init_hess_err - final_hess_err) / (init_hess_err + 1e-12) * 100
        print(f"  Final: output_err={final_hess_err:.4e}, weight_err={final_weight_err:.4e}")
        print(f"  Output improvement: {output_impr:+.2f}%, Weight improvement: {weight_impr:+.2f}%")

    del W_float, H, F_init, G_init

    opt_params = _factor_matrices_to_params(F_opt, G_opt, l, dtype)

    del F_opt, G_opt
    cleanup_gpu_memory()

    return [opt_params], W_recon.to(dtype)
