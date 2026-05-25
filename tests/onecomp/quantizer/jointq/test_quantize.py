"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import torch

from onecomp.quantizer.jointq.core.quantize import quantize

from .test_quantizer import make_sample_data


def test_quantize_4bit_asymmetric():
    """Test the quantize function."""

    matrix_X, matrix_W = make_sample_data()
    group_size = 3
    bits = 4
    symmetric = False
    device = torch.device(0)

    # with batch size
    print("========== with batch size ==========")
    solution = quantize(
        matrix_W,
        matrix_X,
        device=device,
        group_size=group_size,
        bits=bits,
        symmetric=symmetric,
        batch_size=1,
    )
    print(solution)
    matrix_W = matrix_W.to(device)
    matrix_X = matrix_X.to(device)
    error, mse = solution.get_error_and_mse(matrix_W @ matrix_X.T, matrix_X)
    print(f"Error: {error}, MSE: {mse}")

    print("matrix_W: ", matrix_W)
    print("matrix_hatW: ", solution.get_dequantized_weight_matrix())
