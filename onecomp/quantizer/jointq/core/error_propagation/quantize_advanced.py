"""

min || Y - W_quant X^T ||_F^2 + || Lambda (W_quant - W_0) ||_F^2
s.t.  W_quant = S * (A - B)

where:
- W_quant: the quantized weight matrix, shape (p, m)
- W_0: the base weight matrix, shape (p, m)
- S: the scale matrix
- A: the assignment matrix
- B: the bias matrix
- Lambda:=Diag(sqrt(lambda_1),...,sqrt(lambda_p)) the regularization parameter, shape (p, p)
  - Regularization term to suppress excessive error guarantee
  - Lambda is dynamically adjusted.

The goal is to minimize || Y - W_quant X^T ||_F^2 within the range that satisfies the given
maximum variation rate. The variation rate for the i-th row w_quant_i of W_quant is defined as:

rho_i(w_quant_i) = || w_quant_i - w_0_i ||_F^2 / || w_0_i ||_F^2

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import math
import time

import torch

from ..__version__ import __version__
from ..quantize import compute_batch_precomputations, setup
from ..solution import Solution
from .quantizer_advanced import QuantizerAdvanced, SolutionLambda


# pylint: disable=too-many-arguments,too-many-positional-arguments, too-many-locals
def quantize_advanced(
    matrix_Y,
    matrix_X,
    init_scale,
    init_zero_point,
    init_assignment,
    max_variation_rate,
    device,
    bits=4,
    symmetric=False,
    group_size=128,
    batch_size=None,
    epsilon=1e-8,
    max_iter=10,
    log_level=2,
):
    """Quantize the weight matrix with initial solution.

    This is an advanced version of the quantize function that accepts
    an initial solution (scale, zero_point, assignment) to start from.

    Note: All torch.Tensor arguments are assumed to be on CPU.

    Parameters
    ----------
    matrix_Y : torch.Tensor
        The target matrix, shape (p, n), dtype float64.
    matrix_X : torch.Tensor
        The input matrix, shape (n, m), dtype float64.
    init_scale : torch.Tensor
        The initial scale values, shape (p, num_groups), dtype float64.
    init_zero_point : torch.Tensor
        The initial zero point values, shape (p, num_groups), dtype int8.
    init_assignment : torch.Tensor
        The initial assignment (quantized integers), shape (p, num_groups, group_size), dtype int8.
    max_variation_rate : torch.Tensor
        The maximum variation rate for each row, shape (p,), dtype float64.
        Variation rate is defined as: rho_i = ||w_quant_i - w_0_i||_F^2 / ||w_0_i||_F^2.
    device : torch.device
        The device to use for quantization.
    bits : int, optional
        The quantization bits. Default is 4.
    symmetric : bool, optional
        Whether to use symmetric quantization. Default is False.
    group_size : int, optional
        The size of each group. Default is 128.
    batch_size : int or None, optional
        The batch size to use for quantization.
        If None, all rows are optimized together without splitting. Default is None.
    max_iter : int, optional
        The maximum number of iterations for the optimization loop. Default is 10.
    log_level : int, optional
        The log level. Default is 2.

    Returns
    -------
    solution : Solution
        The quantized solution.
    """

    if log_level >= 1:
        begin_time = time.time()
        print(f"<{device}>: Start quantization [JointQ version: {__version__}]")

    # Check the arguments
    _validate_arguments(
        matrix_Y=matrix_Y,
        matrix_X=matrix_X,
        init_scale=init_scale,
        init_zero_point=init_zero_point,
        init_assignment=init_assignment,
        max_variation_rate=max_variation_rate,
        bits=bits,
        symmetric=symmetric,
        group_size=group_size,
    )

    # If group_size is None, set to m (no group splitting)
    if group_size is None:
        group_size = matrix_X.shape[1]

    # Setup
    num_groups, matrix_XX, sub_matrices_XX = setup(
        matrix_X=matrix_X,
        device=device,
        group_size=group_size,
        log_level=log_level,
    )

    # Initialize the result Solution object on CPU with the initial solution
    result_solution = Solution(
        scales=init_scale,
        zero_point=init_zero_point,
        assignment=init_assignment,
    )

    # Batch processing
    # Collect indices where max_variation_rate[i] > 0
    target_indices = torch.nonzero(max_variation_rate > 0).squeeze(1)
    num_targets = len(target_indices)
    if batch_size is None:
        batch_size = num_targets

    for i in range(0, num_targets, batch_size):
        batch_indices = target_indices[i : i + batch_size]

        if log_level >= 1:
            print(
                f"<{device}>: Processing batch "
                f"{i // batch_size + 1} of {math.ceil(num_targets / batch_size)}:"
            )

        # Y_batch: shape (current_batch_size, n), stored on CPU
        Y_batch = matrix_Y[batch_indices]

        # Generate init_solution
        init_solution = SolutionLambda(
            scales=init_scale[batch_indices].to(device),
            zero_point=init_zero_point[batch_indices].to(device),
            assignment=init_assignment[batch_indices].to(device),
        )

        # max_variation_rate_batch: shape (current_batch_size,), stored on GPU
        max_variation_rate_batch = max_variation_rate[batch_indices].to(device)

        # Quantize
        solution = _quantize(
            Y_batch=Y_batch,
            matrix_X=matrix_X,
            matrix_XX=matrix_XX,
            init_solution=init_solution,
            max_variation_rate_batch=max_variation_rate_batch,
            device=device,
            bits=bits,
            symmetric=symmetric,
            group_size=group_size,
            log_level=log_level,
            num_groups=num_groups,
            sub_matrices_XX=sub_matrices_XX,
            epsilon=epsilon,
            max_iter=max_iter,
        )

        # Update result_solution (GPU -> CPU)
        result_solution.scales[:, batch_indices] = solution.scales.cpu()
        result_solution.zero_point[:, batch_indices] = solution.zero_point.cpu()
        result_solution.integers_z[:, batch_indices, :] = solution.integers_z.cpu()

    # Transfer result_solution to GPU
    result_solution.scales = result_solution.scales.to(device)
    result_solution.zero_point = result_solution.zero_point.to(device)
    result_solution.integers_z = result_solution.integers_z.to(device)

    if log_level >= 1:
        end_time = time.time()
        print(f"<{device}>: Total time: {end_time - begin_time:.2f} seconds")

    return result_solution


def _validate_arguments(
    matrix_Y,
    matrix_X,
    init_scale,
    init_zero_point,
    init_assignment,
    max_variation_rate,
    bits,
    symmetric,
    group_size,
):
    """Validate the arguments for quantize_advanced.

    Checks:
    - All torch.Tensor arguments are on CPU
    - Shape consistency between arguments
    - dtype: matrix_Y, matrix_X, init_scale, max_variation_rate are float64;
             init_zero_point, init_assignment are int8
    - init_assignment values are within the valid range based on bits and symmetric
    """
    # Check all tensors are on CPU
    tensors = {
        "matrix_Y": matrix_Y,
        "matrix_X": matrix_X,
        "init_scale": init_scale,
        "init_zero_point": init_zero_point,
        "init_assignment": init_assignment,
        "max_variation_rate": max_variation_rate,
    }
    for name, tensor in tensors.items():
        if tensor.device != torch.device("cpu"):
            raise ValueError(f"{name} must be on CPU, but got {tensor.device}.")

    # Check dtypes
    if matrix_Y.dtype != torch.float64:
        raise ValueError(f"matrix_Y must have dtype float64, but got {matrix_Y.dtype}.")
    if matrix_X.dtype != torch.float64:
        raise ValueError(f"matrix_X must have dtype float64, but got {matrix_X.dtype}.")
    if init_scale.dtype != torch.float64:
        raise ValueError(f"init_scale must have dtype float64, but got {init_scale.dtype}.")
    if max_variation_rate.dtype != torch.float64:
        raise ValueError(
            f"max_variation_rate must have dtype float64, but got {max_variation_rate.dtype}."
        )
    if init_zero_point.dtype != torch.int8:
        raise ValueError(f"init_zero_point must have dtype int8, but got {init_zero_point.dtype}.")
    if init_assignment.dtype != torch.int8:
        raise ValueError(f"init_assignment must have dtype int8, but got {init_assignment.dtype}.")

    # Check shapes
    # matrix_Y: (p, n), matrix_X: (n, m)
    # init_scale: (p, num_groups), init_zero_point: (p, num_groups)
    # init_assignment: (p, num_groups, group_size)
    dim_p = matrix_Y.shape[0]
    dim_n = matrix_Y.shape[1]
    dim_m = matrix_X.shape[1]
    num_groups = dim_m // group_size

    if matrix_X.shape[0] != dim_n:
        raise ValueError(
            f"matrix_X.shape[0] must be {dim_n} (= matrix_Y.shape[1]), "
            f"but got {matrix_X.shape[0]}."
        )
    if dim_m % group_size != 0:
        raise ValueError(
            f"matrix_X.shape[1] (={dim_m}) must be divisible by group_size (={group_size})."
        )
    if init_scale.shape != (dim_p, num_groups):
        raise ValueError(
            f"init_scale must have shape ({dim_p}, {num_groups}), " f"but got {init_scale.shape}."
        )
    if init_zero_point.shape != (dim_p, num_groups):
        raise ValueError(
            f"init_zero_point must have shape ({dim_p}, {num_groups}), "
            f"but got {init_zero_point.shape}."
        )
    if init_assignment.shape != (dim_p, num_groups, group_size):
        raise ValueError(
            f"init_assignment must have shape ({dim_p}, {num_groups}, {group_size}), "
            f"but got {init_assignment.shape}."
        )
    if max_variation_rate.shape != (dim_p,):
        raise ValueError(
            f"max_variation_rate must have shape ({dim_p},), "
            f"but got {max_variation_rate.shape}."
        )

    # Check init_assignment values are within valid range
    if symmetric:
        lower_bound = -(2 ** (bits - 1))
        upper_bound = 2 ** (bits - 1) - 1
    else:
        lower_bound = 0
        upper_bound = 2**bits - 1

    min_val = init_assignment.min().item()
    max_val = init_assignment.max().item()
    if min_val < lower_bound or max_val > upper_bound:
        raise ValueError(
            f"init_assignment values must be in range [{lower_bound}, {upper_bound}], "
            f"but got values in range [{min_val}, {max_val}]."
        )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _quantize(
    Y_batch,
    matrix_X,
    matrix_XX,
    init_solution,
    max_variation_rate_batch,
    device,
    bits,
    symmetric,
    group_size,
    log_level,
    num_groups,
    sub_matrices_XX,
    epsilon,
    max_iter=10,
):
    """Quantize a batch of rows.

    Parameters
    ----------
    Y_batch : torch.Tensor
        The target matrix batch, shape (batch_size, n), on CPU.
    matrix_X : torch.Tensor
        The input matrix, shape (n, m), on CPU.
    matrix_XX : torch.Tensor
        X^T X, shape (m, m), on GPU.
    init_solution : SolutionLambda
        The initial solution for the batch.
    max_variation_rate_batch : torch.Tensor
        The maximum variation rate for the batch, shape (batch_size,), on GPU.
    device : torch.device
        The device to use for quantization.
    bits : int
        The quantization bits.
    symmetric : bool
        Whether to use symmetric quantization.
    group_size : int
        The size of each group.
    log_level : int
        The log level.
    num_groups : int
        The number of groups.
    sub_matrices_XX : torch.Tensor
        The sub matrices, shape (num_groups, num_groups, group_size, group_size), on GPU.
    max_iter : int, optional
        The maximum number of iterations for the optimization loop. Default is 10.

    Returns
    -------
    solution : Solution
        The quantized solution.
    """

    # Initialization -------------------------------------------------->
    dim_n = matrix_X.shape[0]

    # Precompute: matrix_YX, sub_matrices_YX, Y_sq_norms
    _, matrix_YX, sub_matrices_YX, Y_sq_norms = compute_batch_precomputations(
        matrix_W_batch=None,
        matrix_Y_batch=Y_batch,
        matrix_X=matrix_X,
        matrix_XX=matrix_XX,
        num_groups=num_groups,
        group_size=group_size,
        device=device,
        compute_tilde_W=False,
    )

    # Generate the base W_0 (target_weight_matrix)
    target_weight_matrix = init_solution.get_dequantized_weight_matrix()

    # Initialize lambda
    vector_lambda = _initialize_lambda(
        matrix_XX=matrix_XX,
        matrix_YX=matrix_YX,
        Y_sq_norms=Y_sq_norms,
        target_weight_matrix=target_weight_matrix,
    )
    init_solution.vector_lambda = vector_lambda
    init_solution.target_weight_matrix = target_weight_matrix

    # Compute the objective function value
    init_solution.compute_objective_value(
        matrix_XX=matrix_XX,
        matrix_YX=matrix_YX,
        Y_sq_norms=Y_sq_norms,
        dim_n=dim_n,
    )

    # Create QuantizerAdvanced
    quantizer = QuantizerAdvanced(
        matrix_XX=matrix_XX,
        sub_matrices_XX=sub_matrices_XX,
        matrix_YX=matrix_YX,
        sub_matrices_YX=sub_matrices_YX,
        Y_sq_norms=Y_sq_norms,
        dim_n=dim_n,
        bits=bits,
        symmetric=symmetric,
        epsilon=epsilon,
        log_level=log_level,
        vector_lambda=vector_lambda,
        target_weight_matrix=target_weight_matrix,
    )
    quantizer.set_begin_time()
    quantizer.set_modified_lower_and_upper_bounds(init_solution)
    if log_level >= 1:
        quantizer.display_log(init_solution, "Initial", ignore_log_level=True)
    # <--------------------------------------------------- Initialization

    # Optimize scales
    opt_scale_result = quantizer._optimize_scales(init_solution)
    if log_level >= 1:
        quantizer.display_log(
            init_solution,
            f"OptS ({opt_scale_result}/{len(max_variation_rate_batch)})",
            ignore_log_level=True,
        )

    # Optimization
    #  - Perform one round of local search and optimize scales
    #  - Update lambda
    for iter in range(max_iter):
        if log_level >= 2:
            print(f"<{device}>: Iteration {iter + 1} of {max_iter}")
        solution, update_flags = quantizer.quantize(init_solution)

        # Update lambda
        exceed_mask, below_mask = _update_lambda(
            vector_lambda=vector_lambda,
            solution=solution,
            max_variation_rate_batch=max_variation_rate_batch,
            log_level=log_level,
        )

        # Output log for exceed_mask and below_mask
        num_exceed = exceed_mask.sum().item()
        num_below = below_mask.sum().item()
        num_stable = len(exceed_mask) - num_exceed - num_below
        if log_level >= 1:
            print(
                f"<{device}>: Lambda update: "
                f"exceed={num_exceed}, below={num_below}, stable={num_stable}"
            )

        # Converged if num_stable is at maximum (= all rows are stable) and update_flags is 0
        if num_stable == len(exceed_mask) and not update_flags.any():
            # Note: lambda may not converge
            if log_level >= 1:
                print(f"<{device}>: Converged at iteration {iter + 1}")
            break
        # Partially update matrix_L and penalty terms (only rows where lambda changed)
        changed_mask = exceed_mask | below_mask
        quantizer.update_matrix_L(changed_mask)
        solution.update_penalties(changed_mask)

        # Output objective function log
        if log_level >= 1:
            quantizer.display_log(solution, f"Iter {iter + 1}", ignore_log_level=True)

    return solution


def _initialize_lambda(
    matrix_XX,
    matrix_YX,
    Y_sq_norms,
    target_weight_matrix,
    eps=1e-12,
    lambda_max=None,
):
    """Initialize lambda values for regularization.

    λ_i^(0) := 10 * || Y^{(i)} - X * W_0^{(i)} ||_2^2 / || W_0^{(i)} ||_2^2

    Compute the residual norm using precomputed matrices:
    || Y_i - W_0_i X^T ||^2 = ||Y_i||^2 - 2 (YX)_i · W_0_i + W_0_i (X^TX) W_0_i^T

    Parameters
    ----------
    matrix_XX : torch.Tensor
        X^T X, shape (m, m), on GPU.
    matrix_YX : torch.Tensor
        Y @ X, shape (B, m), on GPU.
    Y_sq_norms : torch.Tensor
        ||Y_i||^2, shape (B,), on GPU.
    target_weight_matrix : torch.Tensor
        W_0 (base weight), shape (B, m), dtype float64, on GPU.
    eps : float, optional
        Small constant to avoid division by zero. Default is 1e-12.
    lambda_max : float, optional
        Upper limit for lambda when target_weight_matrix norm is 0 or very small.

    Returns
    -------
    lambda0 : torch.Tensor
        Initial lambda values, shape (B,).
    """
    # Numerator: || Y_i - W_0_i X^T ||^2
    #          = ||Y_i||^2 - 2 (YX)_i · W_0_i + W_0_i (X^TX) W_0_i^T
    W0_XX = target_weight_matrix @ matrix_XX  # (B, m)
    numerator = (
        Y_sq_norms
        - 2.0 * (matrix_YX * target_weight_matrix).sum(dim=1)
        + (W0_XX * target_weight_matrix).sum(dim=1)
    )  # (B,)

    # Denominator: || target_weight_matrix ||_2^2
    denominator = torch.sum(target_weight_matrix**2, dim=1)  # (B,)

    # lambda0 = 10 * numerator / (denominator + eps)
    lambda0 = 10.0 * numerator / (denominator + eps)

    # Clip by lambda_max
    if lambda_max is not None:
        lambda0 = torch.clamp(lambda0, max=lambda_max)

    # Ensure non-negative
    lambda0 = torch.clamp(lambda0, min=0.0)

    return lambda0


def _update_lambda(
    vector_lambda,
    solution,
    max_variation_rate_batch,
    lambda_up_rate=2.0,
    lambda_down_rate=0.5,
    threshold_scale=0.5,
    eps=1e-12,
    log_level=0,
):
    """Update lambda values based on current variation rates.

    Compute the variation rate rho_i for each row and update lambda_i based on
    comparison with the maximum variation rate.
    Updates vector_lambda in-place.

    Update rules:
    - If rho_i > max_variation_rate_i: lambda_i *= lambda_up_rate  (increase penalty)
    - If rho_i < threshold_scale * max_variation_rate_i: lambda_i *= lambda_down_rate  (relax penalty)
    - Otherwise: lambda_i is unchanged

    Parameters
    ----------
    vector_lambda : torch.Tensor
        The lambda values to update in-place, shape (B,).
    solution : SolutionLambda
        The current solution. Must have target_weight_matrix set.
    max_variation_rate_batch : torch.Tensor
        The maximum variation rate for each row, shape (B,).
    lambda_up_rate : float, optional
        Multiplier for lambda when variation rate exceeds max (>1). Default is 2.0.
    lambda_down_rate : float, optional
        Multiplier for lambda when variation rate is below threshold (<1). Default is 0.5.
    threshold_scale : float, optional
        Scale factor for the lower threshold boundary. Default is 0.5.
    eps : float, optional
        Small constant to avoid division by zero. Default is 1e-12.

    Returns
    -------
    exceed_mask : torch.Tensor
        Boolean mask of rows where variation rate exceeds max, shape (B,).
    below_mask : torch.Tensor
        Boolean mask of rows where variation rate is below threshold, shape (B,).
    """
    # W_quant: (B, m)
    w_quant = solution.get_dequantized_weight_matrix()
    # W_0: (B, m)
    w_0 = solution.target_weight_matrix

    # Variation rate: rho_i = ||w_quant_i - w_0_i||_F^2 / ||w_0_i||_F^2
    diff_norm2 = torch.sum((w_quant - w_0) ** 2, dim=1)  # (B,)
    w0_norm2 = torch.sum(w_0**2, dim=1)  # (B,)
    variation_rate = diff_norm2 / (w0_norm2 + eps)  # (B,)
    if log_level >= 1:
        print(
            f"variation_rate: "
            f"min={variation_rate.min().item() * 100:.3f}%, "
            f"max={variation_rate.max().item() * 100:.3f}%, "
            f"mean={variation_rate.mean().item() * 100:.3f}%"
        )

    # Variation rate > max -> multiply lambda by lambda_up_rate (increase penalty)
    exceed_mask = variation_rate > max_variation_rate_batch
    vector_lambda[exceed_mask] *= lambda_up_rate

    # Variation rate < threshold_scale * max -> multiply lambda by lambda_down_rate (relax penalty)
    below_mask = variation_rate < threshold_scale * max_variation_rate_batch
    vector_lambda[below_mask] *= lambda_down_rate

    return exceed_mask, below_mask
