"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import torch


def clip(matrix, group_size, symmetric, lower_bound, upper_bound):
    """Clip

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix to be clipped, shape (p, m).
    group_size : int
        The size of each group.
    symmetric : bool
        Whether to use symmetric quantization.
    lower_bound : int
        The lower bound of the quantization.
    upper_bound : int
        The upper bound of the quantization.

    Returns
    -------
    scales : torch.Tensor
        The scales of the quantization, shape (p, m/group_size).
    assignment : torch.Tensor
        The assignment of the quantization, shape (p, m/group_size, group_size).
    zero_point : torch.Tensor
        The zero point of the quantization, shape (p, m/group_size).

    """
    if symmetric:
        scales, assignment = clip_symmetric(matrix, group_size, lower_bound, upper_bound)
        zero_point = torch.zeros(scales.shape, dtype=torch.int8).to(matrix.device)
        return scales, assignment, zero_point

    assert lower_bound == 0
    return clip_asymmetric(matrix, group_size, upper_bound)


def clip_symmetric(matrix, group_size, lower_bound, upper_bound):
    """Clip the elements of each vector to the specified bounds.

    w ≈ s a (s in R, a in Q^d),
    where Q = {l, l+1, ..., u-1, u}

    s = max(|w|)/ u
    a = clip(round(w/s), l, u)

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix to be clipped, shape (p, m).
    group_size : int
        The size of each group.
    lower_bound : int
        The lower bound of the quantization.
    upper_bound : int
        The upper bound of the quantization.


    Returns
    -------
    scales : torch.Tensor
        The scales of the quantization, shape (p, m/group_size).
    assignment : torch.Tensor
        The assignment of the quantization, shape (p, m/group_size, group_size).

    """

    dim_p, dim_m = matrix.shape
    num_group = dim_m // group_size
    assert dim_m % group_size == 0

    # (p, m) -> (p * m/group_size, group_size)
    vectors = matrix.view(dim_p * num_group, group_size)

    # s = max(|w|)/ u
    scales = torch.max(torch.abs(vectors), dim=1).values / upper_bound

    # Clamp with a small epsilon to avoid division by zero
    eps = 1e-8
    scales = torch.clamp(scales, min=eps)

    # For each group, divide by scale, round, and clip
    # a = clip(round(w/s), l, u)
    quantized = torch.clamp(
        torch.round(vectors / scales.unsqueeze(1)), lower_bound, upper_bound
    ).to(torch.int8)

    # scales: per group per row -> (dim_p, num_group)
    # assignment: split per row per group -> (dim_p, num_group, group_size)
    return scales.view(dim_p, num_group), quantized.view(dim_p, num_group, group_size)


def clip_asymmetric(matrix, group_size, upper_bound):
    """Clip the elements of each vector to the specified bounds.

    w ≈ s (a - b) (s in R, a in Q^d, b in Q),
    where Q = {0, 1, ..., u-1, u}

    s = (max(w) - min(w))/ u
    b = clip(round(-min(w)/s), 0, u)
    a = clip(round((w/s) + b), 0, u)

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix to be clipped, shape (p, m).
    group_size : int
        The size of each group.
    upper_bound : int
        The upper bound of the quantization.


    Returns
    -------
    scales : torch.Tensor
        The scales of the quantization, shape (p, m/group_size).
    assignment : torch.Tensor
        The assignment of the quantization, shape (p, m/group_size, group_size).
    zero_point : torch.Tensor
        The zero point of the quantization, shape (p, m/group_size).

    """

    dim_p, dim_m = matrix.shape
    num_group = dim_m // group_size
    assert dim_m % group_size == 0

    # (p, m) -> (p * m/group_size, group_size)
    vectors = matrix.view(dim_p * num_group, group_size)

    # s = (max(w) - min(w))/ u
    scales = (torch.max(vectors, dim=1).values - torch.min(vectors, dim=1).values) / upper_bound

    # Clamp with a small epsilon to avoid division by zero
    eps = 1e-8
    scales = torch.clamp(scales, min=eps)

    # b = clip(round(-min(w)/s), 0, u)
    zero_point = torch.clamp(
        torch.round(-torch.min(vectors, dim=1).values / scales), 0, upper_bound
    ).to(torch.int8)

    # a = clip(round((w/s) + b), 0, u)
    quantized = torch.clamp(
        torch.round((vectors / scales.unsqueeze(1)) + zero_point.unsqueeze(1)),
        0,
        upper_bound,
    ).to(torch.int8)

    # scales: per group per row -> (dim_p, num_group)
    # assignment: split per row per group -> (dim_p, num_group, group_size)
    # zero_point: per group per row -> (dim_p, num_group)
    return (
        scales.view(dim_p, num_group),
        quantized.view(dim_p, num_group, group_size),
        zero_point.view(dim_p, num_group),
    )


def calculate_scales_and_zero_point(matrix, symmetric, lower_bound, upper_bound):
    """Calculate the scales and zero point.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix to be clipped, shape (p, m).
    symmetric : bool
        Whether to use symmetric quantization.
    lower_bound : int
        The lower bound of the quantization.
    upper_bound : int
        The upper bound of the quantization.

    Returns
    -------
    scales : torch.Tensor
        The scales of the quantization, shape (p,).
    zero_point : torch.Tensor
        The zero point of the quantization, shape (p,).

    """
    if symmetric:
        # s = max(|w|)/ u
        scales = torch.max(torch.abs(matrix), dim=1).values / upper_bound
        # zero_point = 0
        zero_point = torch.zeros(scales.shape, dtype=torch.int8, device=matrix.device)
    else:
        assert lower_bound == 0
        # s = (max(w) - min(w))/ u
        scales = (torch.max(matrix, dim=1).values - torch.min(matrix, dim=1).values) / upper_bound

        # Clamp with a small epsilon to avoid division by zero
        eps = 1e-8
        scales = torch.clamp(scales, min=eps)

        # b = clip(round(-min(w)/s), 0, u)
        zero_point = torch.clamp(
            torch.round(-torch.min(matrix, dim=1).values / scales),
            0,
            upper_bound,
        ).to(torch.int8)

    return scales, zero_point
