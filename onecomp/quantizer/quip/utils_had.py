"""
Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import gc
import torch


def clean():
    """Memory cleanup function"""
    gc.collect()
    torch.cuda.empty_cache()


def is_pow2(n):
    """Check if n is a power of 2"""
    return (n & (n - 1) == 0) and (n > 0)


def get_hadK(n, transpose=False):
    """Get Hadamard matrix for dimension n"""
    # Only support power of 2 dimensions for simplicity
    if is_pow2(n):
        return None, 1
    # For non-power-of-2, we need special Hadamard matrices
    # This is a simplified version - full version would need all the had matrices
    raise ValueError(
        f"Dimension {n} is not supported for Hadamard transform. " "Use incoh_mode='kron' instead."
    )


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)

    input_tensor = X.clone().view(-1, n, 1)
    output = input_tensor.clone()

    while input_tensor.shape[1] > K:
        input_tensor = input_tensor.view(
            input_tensor.shape[0], input_tensor.shape[1] // 2, 2, input_tensor.shape[2]
        )
        output = output.view(input_tensor.shape)
        output[:, :, 0, :] = input_tensor[:, :, 0, :] + input_tensor[:, :, 1, :]
        output[:, :, 1, :] = input_tensor[:, :, 0, :] - input_tensor[:, :, 1, :]
        output = output.view(input_tensor.shape[0], input_tensor.shape[1], -1)
        (input_tensor, output) = (output, input_tensor)

    del output
    clean()

    if K > 1 and hadK is not None:
        input_tensor = torch.bmm(
            hadK.repeat(len(input_tensor), 1, 1).to(input_tensor.device).to(input_tensor.dtype),
            input_tensor,
        )

    return input_tensor.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadUt(X):
    """Hadamard transform (transpose)"""
    return matmul_hadU(X, transpose=True)


def RHT_H(H, SV):
    """Randomized Hadamard Transform for Hessian"""
    return matmul_hadUt(matmul_hadUt(H * SV.unsqueeze(0)).T * SV.unsqueeze(0))


def RHT_W(W, SU, SV):
    """Randomized Hadamard Transform for Weight"""
    return matmul_hadUt(matmul_hadUt(W.T * SU.unsqueeze(0)).T * SV.unsqueeze(0))


def REVERSE_RHT_W(hatW, SU, SV):
    """Reverse Randomized Hadamard Transform for Weight"""
    return matmul_hadU((matmul_hadU(hatW.T) * SU.unsqueeze(0)).T) * SV.unsqueeze(0)


def REVERSE_RHT_H(hatH, SV):
    """Reverse Randomized Hadamard Transform for Hessian"""
    return matmul_hadU((matmul_hadU(hatH) * SV.unsqueeze(0)).T) * SV.unsqueeze(0)
