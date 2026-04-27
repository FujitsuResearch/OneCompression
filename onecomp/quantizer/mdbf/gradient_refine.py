# -*- coding: utf-8 -*-
"""
MSVID 振幅パラメータの勾配ベース最適化（Phase 3）

ADMM最適化後に, 符号を固定して振幅パラメータを直接勾配法で最適化する.

Activation-aware拡張:
- activation_aware=True: Hessian重み付き誤差を最小化（出力誤差と等価）
- P=1のみ対応
"""

from typing import List, Optional, Tuple

import torch

from logging import getLogger

logger = getLogger(__name__)

from .initialize import MSVIDParams
from .utils import (
    cleanup_gpu_memory,
    compute_hessian_error,
    ensure_float32,
    reconstruct_weight,
    symmetrize_matrix,
)


def _prepare_hessian(
    H: Optional[torch.Tensor],
    m: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Hessianの前処理（対称化）"""
    if H is None or H.shape != (m, m):
        return None
    H_float = ensure_float32(H, device)
    return symmetrize_matrix(H_float)


def refine_amplitude_gradient(
    W_original: torch.Tensor,
    params_list: List[MSVIDParams],
    l: int,
    lr: float = 0.01,
    iters: int = 1000,
    activation_aware: bool = False,
    H: Optional[torch.Tensor] = None,
    nsamples: int = 1,
) -> Tuple[List[MSVIDParams], torch.Tensor]:
    """
    振幅パラメータを勾配法で最適化（Phase 3）

    符号行列 (A_sign, B_sign) を固定し,
    振幅パラメータ (A_amp, B_amp, Q_U_amp, Q_V_amp) を勾配法で最適化する.

    Args:
        W_original: 元の重み行列 (n, m)
        params_list: MSVIDパラメータのリスト（Pパス分）
        l: Multi-scaleランク
        lr: 学習率
        iters: 最適化反復回数
        activation_aware: Hessian重み付き誤差を最小化
        H: Hessian行列 (m, m) = X^T @ X / N
        nsamples: サンプル数 N

    Returns:
        optimized_params: 最適化されたMSVIDパラメータ
        W_recon: 再構成された重み
    """
    device = W_original.device
    dtype = W_original.dtype
    n, m = W_original.shape
    P = len(params_list)

    # Activation-aware: P=1のみ対応
    if activation_aware and P != 1:
        logger.warning("[Gradient Refine] activation_aware=True but P!=1; fallback to non-aware.")
        activation_aware = False

    # Hessianの準備
    H_float = None
    H_for_display = None

    if activation_aware:
        H_float = _prepare_hessian(H, m, device)
        if H_float is None:
            logger.warning("[Gradient Refine] activation_aware=True but H is None; fallback to non-aware.")
            activation_aware = False

    if not activation_aware:
        H_for_display = _prepare_hessian(H, m, device)

    W_float = ensure_float32(W_original)
    orig_norm = torch.norm(W_float, p='fro').item() + 1e-12

    # 出力誤差の正規化定数
    H_for_err = H_float if H_float is not None else H_for_display
    orig_output_err = None
    if H_for_err is not None:
        WH = W_float @ H_for_err
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

    init_weight_error = torch.norm(W_float - W_init_recon, p='fro').item()
    E_init = W_float - W_init_recon

    if H_for_err is not None:
        init_output_error = compute_hessian_error(E_init, H_for_err, nsamples)
    else:
        init_output_error = None

    if activation_aware:
        init_error = init_output_error
        logger.debug(f"[Gradient Refine] Initial output_error: {init_output_error:.4e} "
                     f"(rel: {init_output_error/orig_output_err:.4f})")
    else:
        init_error = init_weight_error ** 2
        if init_output_error is not None:
            logger.debug(f"[Gradient Refine] Initial output_error: {init_output_error:.4e} "
                         f"(rel: {init_output_error/orig_output_err:.4f})")
        else:
            logger.debug(f"[Gradient Refine] Initial weight_error: {init_weight_error:.4e} "
                         f"(rel: {init_weight_error/orig_norm:.4f})")

    del W_init_recon, E_init

    # パラメータ準備
    amp_params = []
    sign_params = []

    for p in params_list:
        A_sign = p.A_sign.to(device).float()
        B_sign = p.B_sign.to(device).float()
        sign_params.append((A_sign, B_sign))

        A_amp = p.A_amp.to(device).float().clone().requires_grad_(True)
        B_amp = p.B_amp.to(device).float().clone().requires_grad_(True)
        Q_U_amp = p.Q_U_amp.to(device).float().clone().requires_grad_(True)
        Q_V_amp = p.Q_V_amp.to(device).float().clone().requires_grad_(True)
        amp_params.append((A_amp, B_amp, Q_U_amp, Q_V_amp))

    all_params = []
    for A_amp, B_amp, Q_U_amp, Q_V_amp in amp_params:
        all_params.extend([A_amp, B_amp, Q_U_amp, Q_V_amp])

    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters)

    best_error = init_error
    best_amp_params = None

    # 勾配最適化ループ
    with torch.enable_grad():
        for itt in range(iters):
            optimizer.zero_grad()

            # 再構成
            W_parts = []
            for p_idx in range(P):
                A_sign, B_sign = sign_params[p_idx]
                A_amp, B_amp, Q_U_amp, Q_V_amp = amp_params[p_idx]

                amp_A = A_amp @ Q_U_amp.T
                F = A_sign * amp_A
                amp_B = Q_V_amp @ B_amp.T
                G = B_sign * amp_B
                W_parts.append(F @ G)

            W_recon = sum(W_parts)
            E = W_float - W_recon

            # 損失計算
            if activation_aware:
                EH = E @ H_float
                loss = float(nsamples) * (EH * E).sum()
            else:
                loss = (E ** 2).sum()

            current_error = loss.item()

            if current_error < best_error:
                best_error = current_error
                best_amp_params = [
                    (A.detach().clone(), B.detach().clone(),
                     QU.detach().clone(), QV.detach().clone())
                    for A, B, QU, QV in amp_params
                ]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if (itt % max(10, iters // 5) == 0 or itt == iters - 1):
                current_lr = scheduler.get_last_lr()[0]
                if activation_aware:
                    logger.debug(f"[Gradient Refine] Step {itt:3d}: output_error = {current_error:.4e} "
                                 f"(rel: {current_error/orig_output_err:.4f}), lr={current_lr:.2e}")
                elif H_for_display is not None:
                    with torch.no_grad():
                        E_step = W_float - W_recon
                        output_err_step = compute_hessian_error(E_step, H_for_display, nsamples)
                    logger.debug(f"[Gradient Refine] Step {itt:3d}: output_error = {output_err_step:.4e} "
                                 f"(rel: {output_err_step/orig_output_err:.4f}), lr={current_lr:.2e}")
                else:
                    logger.debug(f"[Gradient Refine] Step {itt:3d}: weight_error = {current_error**.5:.4e} "
                                 f"(rel: {current_error**.5/orig_norm:.4f}), lr={current_lr:.2e}")

    # ベストパラメータ復元
    if best_amp_params is not None:
        amp_params = best_amp_params

    # MSVIDParams構築
    optimized_params = []
    for p_idx in range(P):
        A_sign, B_sign = sign_params[p_idx]
        A_amp, B_amp, Q_U_amp, Q_V_amp = amp_params[p_idx]

        params = MSVIDParams(
            A_sign=A_sign.to(dtype),
            B_sign=B_sign.to(dtype),
            A_amp=A_amp.to(dtype),
            B_amp=B_amp.to(dtype),
            Q_U_amp=Q_U_amp.to(dtype),
            Q_V_amp=Q_V_amp.to(dtype),
        )
        optimized_params.append(params)

    # 最終再構成
    W_recon_final = torch.zeros(n, m, device=device, dtype=torch.float32)
    for p in optimized_params:
        W_p = reconstruct_weight(
            p.A_sign.to(device), p.B_sign.to(device),
            p.A_amp.to(device), p.B_amp.to(device),
            p.Q_U_amp.to(device), p.Q_V_amp.to(device),
        )
        W_recon_final += W_p
        del W_p

    final_weight_error = torch.norm(W_float - W_recon_final, p='fro').item()
    E_final = W_float - W_recon_final

    H_for_final = H_float if H_float is not None else H_for_display
    if H_for_final is not None:
        final_output_error = compute_hessian_error(E_final, H_for_final, nsamples)
    else:
        final_output_error = None

    if activation_aware:
        improvement = (init_error - final_output_error) / (init_error + 1e-12) * 100
        logger.debug(f"[Gradient Refine] Final output_error: {final_output_error:.4e} "
                     f"(rel: {final_output_error/orig_output_err:.4f})")
        logger.debug(f"[Gradient Refine] Improvement: {improvement:+.2f}%")
    elif final_output_error is not None:
        improvement = (init_output_error - final_output_error) / (init_output_error + 1e-12) * 100
        logger.debug(f"[Gradient Refine] Final output_error: {final_output_error:.4e} "
                     f"(rel: {final_output_error/orig_output_err:.4f})")
        logger.debug(f"[Gradient Refine] Improvement: {improvement:+.2f}%")
    else:
        # init_error は weight_error^2 なので、final_weight_error^2 と比較
        final_error_sq = final_weight_error ** 2
        improvement = (init_error - final_error_sq) / (init_error + 1e-12) * 100
        logger.debug(f"[Gradient Refine] Final weight_error: {final_weight_error:.4e} "
                     f"(rel: {final_weight_error/orig_norm:.4f})")
        logger.debug(f"[Gradient Refine] Improvement: {improvement:+.2f}%")

    # クリーンアップ
    del W_float, E_final
    if H_float is not None:
        del H_float
    if H_for_display is not None:
        del H_for_display
    for A_sign, B_sign in sign_params:
        del A_sign, B_sign
    cleanup_gpu_memory()

    return optimized_params, W_recon_final.to(dtype)
