"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from concurrent.futures import ThreadPoolExecutor

import torch

from .quantize import _validate_quantize_args, merge_solutions, quantize


def _warmup_linalg_on_devices(device_list):
    """Warm up torch.linalg.solve on each device to initialize lazy wrappers.

    This prevents the "lazy wrapper should be called at most once" error
    when running in multi-threaded context.
    """
    for device in device_list:
        dummy_A = torch.eye(2, dtype=torch.float64, device=device)
        dummy_b = torch.ones(2, 1, dtype=torch.float64, device=device)
        _ = torch.linalg.solve(dummy_A, dummy_b)
        if device.type == "cuda":
            torch.cuda.synchronize(device)


# pylint: disable=too-many-arguments,too-many-positional-arguments, too-many-locals
def _quantize_on_multiple_gpus(
    sub_matrices_W,
    sub_matrices_Y,
    matrix_X,
    matrix_XX,
    dim_n,
    device_list,
    bits,
    symmetric,
    group_size,
    batch_size,
    early_stopping_ratio,
    epsilon,
    ils_num_iterations,
    ils_num_clones,
    ils_num_channels,
    log_level,
):
    """Execute quantization on multiple GPUs in parallel.

    Parameters
    ----------
    sub_matrices_W : list of torch.Tensor or None
        List of sub-matrices of W to be quantized.
        Either sub_matrices_W or sub_matrices_Y must be None.
    sub_matrices_Y : list of torch.Tensor or None
        List of sub-matrices of Y (target matrix) to be quantized.
        Either sub_matrices_W or sub_matrices_Y must be None.
    matrix_X : torch.Tensor or None
        The input matrix, shape (n, m), dtype float64.
        Set to None when matrix_XX is specified.
    matrix_XX : torch.Tensor or None
        Precomputed X^T X, shape (m, m), dtype float64.
        Set to None when matrix_X is specified.
    dim_n : int or None
        Number of rows in the input matrix X, n. Required when matrix_XX is specified.
    device_list : list of torch.device
        The list of devices to use for quantization.
    bits : int
        The quantization bits.
    symmetric : bool
        Whether to use symmetric quantization.
    group_size : int
        The size of each group.
    batch_size : int
        The batch size to use for quantization.
    early_stopping_ratio : float
        The ratio for the early stopping.
    epsilon : float
        The epsilon for the quantization.
    ils_num_iterations : int
        The number of iterations for the iterated local search.
    ils_num_clones : int
        The number of clones for the iterated local search.
    ils_num_channels : int
        The number of channels for the iterated local search.
    log_level : int
        The log level (0: none, 1: minimal, 2: detailed).

    Returns
    -------
    list of Solution
        List of quantization solutions for each sub-matrix.
    """

    # Warm up lazy initialization before parallel execution
    _warmup_linalg_on_devices(device_list)

    # Determine which mode we're in
    use_matrix_Y = sub_matrices_Y is not None
    sub_matrices = sub_matrices_Y if use_matrix_Y else sub_matrices_W

    def quantize_worker(sub_matrix, device, gpu_id):
        """Worker function for quantizing a sub-matrix on a specific GPU."""

        if log_level >= 1:
            print(f"Starting quantization on GPU {gpu_id} (device: {device})")

        if use_matrix_Y:
            solution = quantize(
                matrix_W=None,
                matrix_X=matrix_X,
                device=device,
                matrix_Y=sub_matrix,
                bits=bits,
                symmetric=symmetric,
                group_size=group_size,
                batch_size=batch_size,
                early_stopping_ratio=early_stopping_ratio,
                epsilon=epsilon,
                ils_num_iterations=ils_num_iterations,
                ils_num_clones=ils_num_clones,
                ils_num_channels=ils_num_channels,
                log_level=log_level,
            )
        else:
            solution = quantize(
                matrix_W=sub_matrix,
                matrix_X=matrix_X,
                device=device,
                matrix_Y=None,
                matrix_XX=matrix_XX,
                dim_n=dim_n,
                bits=bits,
                symmetric=symmetric,
                group_size=group_size,
                batch_size=batch_size,
                early_stopping_ratio=early_stopping_ratio,
                epsilon=epsilon,
                ils_num_iterations=ils_num_iterations,
                ils_num_clones=ils_num_clones,
                ils_num_channels=ils_num_channels,
                log_level=log_level,
            )

        if log_level >= 1:
            print(f"Completed quantization on GPU {gpu_id}")
        return solution

    # Execute in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(device_list)) as executor:
        futures = [
            executor.submit(quantize_worker, sub_matrix, device, i)
            for i, (sub_matrix, device) in enumerate(zip(sub_matrices, device_list))
        ]
        solutions = [future.result() for future in futures]

    return solutions


# pylint: disable=too-many-arguments,too-many-positional-arguments, too-many-locals
def quantize_multi_gpu(
    matrix_W=None,
    matrix_X=None,
    device_list=None,
    matrix_Y=None,
    matrix_XX=None,
    dim_n=None,
    bits=4,
    symmetric=False,
    group_size=128,
    batch_size=None,
    early_stopping_ratio=0.1,
    epsilon=1e-8,
    ils_num_iterations=None,
    ils_num_clones=8,
    ils_num_channels=None,
    log_level=1,  # 0: none, 1: minimal, 2: detailed
):
    """Quantize the weight matrix using multiple GPUs.

    - (When matrix_Y is None) Find hat_W that minimizes ||W X^T - hat_W X^T||_F^2
    - (When matrix_Y is given) Find hat_W that minimizes ||Y - hat_W X^T||_F^2
    - Split W or Y across devices in device_list and quantize on each device
    - On each device, split matrix W/Y by batch_size rows and quantize each batch
    - matrix_W, matrix_Y, matrix_X are assumed to be on CPU

    Parameters
    ----------
    matrix_W : torch.Tensor, optional
        The weight matrix to be quantized, shape (p, m), dtype float64.
        Set to None when matrix_Y is provided.
    matrix_X : torch.Tensor, optional
        The input matrix, shape (n, m), dtype float64.
        Set to None when matrix_XX is specified.
    device_list : list of torch.device
        The list of devices to use for quantization.
    matrix_Y : torch.Tensor, optional
        The target matrix, shape (p, n), dtype float64.
        If specified, the objective function becomes ||Y - hat_W X^T||_F^2.
    matrix_XX : torch.Tensor, optional
        Precomputed X^T X, shape (m, m), dtype float64.
        Can be placed on either CPU or GPU (transferred to each GPU internally).
        If specified, used instead of matrix_X (skips computation from matrix_X).
        Requires matrix_Y=None, matrix_X=None, dim_n!=None when specified.
        Default is None.
    dim_n : int, optional
        Number of rows in the input matrix X, n. Required when matrix_XX is specified.
        Default is None.
    bits : int, optional
        The quantization bits. Default is 4.
    symmetric : bool, optional
        Whether to use symmetric quantization. Default is False.
    group_size : int, optional
        The size of each group. If None, this is set to m. Default is 128.
    batch_size : int or None, optional
        The batch size to use for quantization.
        If None, all rows are optimized together without splitting. Default is None.
    early_stopping_ratio : float, optional
        The ratio for the early stopping. Default is 0.1.
    epsilon : float, optional
        The epsilon for the quantization. Default is 1e-8.
    ils_num_iterations : int, optional
        The number of iterations for the iterated local search. Default is None.
    ils_num_clones : int, optional
        The number of clones for the iterated local search. Default is 8.
    ils_num_channels : int, optional
        The number of channels for the iterated local search.
        If None, automatically set to min(dim_p, 1024). Default is None.
    log_level : int, optional
        The log level (0: none, 1: minimal, 2: detailed). Default is 1.
    """

    _validate_quantize_args(
        matrix_W=matrix_W,
        matrix_X=matrix_X,
        matrix_Y=matrix_Y,
        matrix_XX=matrix_XX,
        dim_n=dim_n,
    )

    if device_list is None or len(device_list) == 0:
        raise ValueError("device_list must be a non-empty list.")

    # step1: split matrix_W or matrix_Y into sub-matrices
    # Split evenly with torch.tensor_split (automatically adjusted if p is not divisible)
    if matrix_Y is not None:
        sub_matrices_W = None
        sub_matrices_Y = torch.tensor_split(matrix_Y, len(device_list), dim=0)
        if log_level >= 1:
            print(f"Splitting matrix_Y (shape: {matrix_Y.shape}) across {len(device_list)} GPUs")
        for i, sub_matrix in enumerate(sub_matrices_Y):
            if log_level >= 1:
                print(
                    f"  GPU {i} (device: {device_list[i]}): sub_matrix shape = {sub_matrix.shape}"
                )
    else:
        sub_matrices_W = torch.tensor_split(matrix_W, len(device_list), dim=0)
        sub_matrices_Y = None
        if log_level >= 1:
            print(f"Splitting matrix_W (shape: {matrix_W.shape}) across {len(device_list)} GPUs")
        for i, sub_matrix in enumerate(sub_matrices_W):
            if log_level >= 1:
                print(
                    f"  GPU {i} (device: {device_list[i]}): sub_matrix shape = {sub_matrix.shape}"
                )

    # step2: Execute quantize function on multiple GPUs for each sub-matrix and matrix_X (n, m)
    solutions = _quantize_on_multiple_gpus(
        sub_matrices_W=sub_matrices_W,
        sub_matrices_Y=sub_matrices_Y,
        matrix_X=matrix_X,
        matrix_XX=matrix_XX,
        dim_n=dim_n,
        device_list=device_list,
        bits=bits,
        symmetric=symmetric,
        group_size=group_size,
        batch_size=batch_size,
        early_stopping_ratio=early_stopping_ratio,
        epsilon=epsilon,
        ils_num_iterations=ils_num_iterations,
        ils_num_clones=ils_num_clones,
        ils_num_channels=ils_num_channels,
        log_level=log_level,
    )

    # step3: Move each solution to device_list[0] and merge with merge_solutions
    if log_level >= 1:
        print(f"Merging solutions on device: {device_list[0]}")

    # step4: Merge solutions and output results
    # Move each solution to device_list[0]
    target_device = device_list[0]
    for solution in solutions:
        solution.scales = solution.scales.to(target_device)
        solution.zero_point = solution.zero_point.to(target_device)
        solution.integers_z = solution.integers_z.to(target_device)
        solution.squared_errors = solution.squared_errors.to(target_device)
        solution.mean_squared_errors = solution.mean_squared_errors.to(target_device)

    # Merge with merge_solutions
    merged_solution = merge_solutions(solutions)

    if log_level >= 1:
        error = merged_solution.squared_error
        mse = merged_solution.mean_squared_error
        print(f"Multi-GPU quantization completed. Error: {error:.3e}, MSE: {mse:.3e}")

    return merged_solution
