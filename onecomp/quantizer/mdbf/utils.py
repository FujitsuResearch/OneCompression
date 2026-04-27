"""
Dual-(M)SVID ユーティリティ関数

- BPW（Bits Per Weight）からランク r の計算
- 重み再構成

パラメータ数の内訳（1パスあたり）:
- 符号行列: S_A (n×r), S_B (r×m) -> 二値 = r(n+m) bits
- スケール: A_amp (n×l), B_amp (m×l), Q_U_amp (r×l), Q_V_amp (r×l)
           -> FP16 = 16 * (ln + lm + 2lr) bits
"""

import gc
import math
from logging import getLogger
from typing import Literal
import torch


logger = getLogger(__name__)


def cleanup_gpu_memory() -> None:
    """GPUメモリを解放"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def ensure_float32(
    tensor: torch.Tensor,
    device: torch.device = None,
    clone: bool = False,
) -> torch.Tensor:
    """
    テンソルをfloat32に変換

    Args:
        tensor: 入力テンソル
        device: 転送先デバイス（Noneなら元のデバイスを維持）
        clone: Trueなら常にクローンを返す

    Returns:
        float32テンソル（clone=Falseかつ変換不要なら同一オブジェクト）
    """
    target_device = device if device is not None else tensor.device
    needs_dtype = tensor.dtype != torch.float32
    needs_device = tensor.device != target_device

    if not needs_dtype and not needs_device:
        return tensor.clone() if clone else tensor
    # .to() は新しいテンソルを返すのでclone不要
    return tensor.to(device=target_device, dtype=torch.float32)


def ensure_float32_clone(tensor: torch.Tensor, device: torch.device = None) -> torch.Tensor:
    """
    テンソルをfloat32に変換してクローン（ensure_float32のラッパー）

    Args:
        tensor: 入力テンソル
        device: 転送先デバイス（Noneなら元のデバイスを維持）

    Returns:
        float32テンソルのクローン
    """
    return ensure_float32(tensor, device, clone=True)


def rank_from_bpw(
    n: int,
    m: int,
    b_target: float,
    l: int = 1,
    P: int = 2,
    min_rank: int = 1,
    rounding: Literal["floor", "ceil", "round"] = "floor"
) -> int:
    """
    目標BPWからランク r を計算

    b_eff = P * [r(n+m) + 16*l*(n+m+2r)] / (nm)
    を r について解くと:
    r = (b_target * nm / P - 16*l*(n+m)) / ((n+m) + 32*l)

    Args:
        n: 行数（出力次元）
        m: 列数（入力次元）
        b_target: 目標BPW
        l: Multi-scaleランク
        P: パス数 (1, 2, ...)
        min_rank: 最小ランク
        rounding: 丸め方法
            - "floor": 切り捨て（b_target を上限として守る）
            - "ceil": 切り上げ（近似精度を優先）
            - "round": 四捨五入（バランス）

    Returns:
        計算されたランク r
    """
    # Note: scale_bits=0 は二値行列のみでBPWを計算するモード
    # FP16スケールを含める場合は scale_bits=16 に変更
    scale_bits = 0
    numerator = (b_target * n * m / P) - scale_bits * l * (n + m)
    denominator = (n + m) + 2 * scale_bits * l

    r_real = numerator / denominator
    max_rank = min(n, m)

    if r_real < min_rank:
        logger.warning(
            f"[MDBF] b_target={b_target} is too small for {n}x{m} matrix. "
            f"r_real={r_real:.2f}, using min_rank={min_rank}"
        )
        return min_rank

    if r_real > max_rank:
        logger.warning(
            f"[MDBF] b_target={b_target} exceeds full-rank for {n}x{m} matrix. "
            f"r_real={r_real:.2f}, clamping to {max_rank}"
        )
        return max_rank

    if rounding == "floor":
        r = int(math.floor(r_real))
    elif rounding == "ceil":
        r = int(math.ceil(r_real))
    else:
        r = int(round(r_real))

    return max(r, min_rank)


def bpw_from_rank(
    n: int,
    m: int,
    r: int,
    l: int = 1,
    P: int = 2
) -> float:
    """
    ランク r から実効BPWを計算

    b_eff = P * [r(n+m) + 16*l*(n + m + 2*r)] / (nm)

    Args:
        n: 行数（出力次元）
        m: 列数（入力次元）
        r: ランク
        l: Multi-scaleランク
        P: パス数

    Returns:
        実効BPW
    """
    scale_bits = 16  # FP16

    bits_binary = r * (n + m)
    bits_scale = scale_bits * l * (n + m + 2 * r)

    total_bits = P * (bits_binary + bits_scale)
    return total_bits / (n * m)


def to_binary_sign(x: torch.Tensor) -> torch.Tensor:
    """符号行列を二値化 {-1, +1}"""
    out = torch.sign(x)
    out[out == 0] = 1.0
    return out


def symmetrize_matrix(H: torch.Tensor) -> torch.Tensor:
    """行列を対称化: (H + H^T) / 2"""
    return (H + H.T) * 0.5


def compute_hessian_error(
    E: torch.Tensor,
    H: torch.Tensor,
    nsamples: int
) -> float:
    """
    Hessian重み付き誤差を計算: N * tr(E @ H @ E^T)

    Args:
        E: 誤差行列 (n, m)
        H: Hessian行列 (m, m)
        nsamples: サンプル数 N

    Returns:
        N * tr(E @ H @ E^T) = N * sum((E @ H) * E)
    """
    EH = E @ H
    return float(nsamples) * torch.sum(EH * E).item()


def reconstruct_weight(
    A_sign: torch.Tensor,
    B_sign: torch.Tensor,
    A_amp: torch.Tensor,
    B_amp: torch.Tensor,
    Q_U_amp: torch.Tensor,
    Q_V_amp: torch.Tensor,
) -> torch.Tensor:
    """
    パラメータから重み行列を再構成

    W = F @ G
    where F = S_A * (A_amp @ Q_U_amp^T)  : (n, r)
          G = S_B * (Q_V_amp @ B_amp^T)  : (r, m)

    計算量: O(nlr + rlm + nrm)（l^2ループ版より高速）

    Args:
        A_sign: 符号行列 S_A (n, r) - {-1, +1}
        B_sign: 符号行列 S_B (r, m) - {-1, +1}
        A_amp: 行スケール (n, l)
        B_amp: 列スケール (m, l)
        Q_U_amp: 潜在スケール行側 (r, l)
        Q_V_amp: 潜在スケール列側 (r, l)

    Returns:
        再構成された重み W (n, m)
    """
    # F = S_A * (A_amp @ Q_U_amp^T)
    amp_A = A_amp @ Q_U_amp.T
    F = A_sign * amp_A
    del amp_A

    # G = S_B * (Q_V_amp @ B_amp^T)
    amp_B = Q_V_amp @ B_amp.T
    G = B_sign * amp_B
    del amp_B

    W_recon = F @ G
    del F, G

    return W_recon
