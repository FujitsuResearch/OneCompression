"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import math
import time

import torch

from .quantizer import Quantizer
from .solution import Solution
from .__version__ import __version__


# pylint: disable=too-many-arguments,too-many-positional-arguments, too-many-locals
def quantize(
    matrix_W=None,
    matrix_X=None,
    device=None,
    matrix_Y=None,
    matrix_XX=None,
    dim_n=None,
    bits=4,
    symmetric=False,
    group_size=128,
    batch_size=None,
    early_stopping_ratio=0.1,
    epsilon=1e-12,
    ils_num_iterations=None,
    ils_num_clones=8,
    ils_num_channels=None,
    log_level=2,  # 0: none, 1: minimal, 2: detailed, 3: debug
    enable_clip_optimize=True,
    enable_clip_optimize_ep=True,
    initial_solutions=None,
):
    """Quantize the weight matrix

    - (When matrix_Y is None) Find hat_W that minimizes ||W X^T - hat_W X^T||_F^2
    - (When matrix_Y is given) Find hat_W that minimizes ||Y - hat_W X^T||_F^2
    - Split the weight matrix W into batches of batch_size rows and quantize each batch

    The design assumes that it works as long as X fits on the GPU.
    The batch size needs to be adjusted depending on the remaining memory.

    Parameters
    ----------
    matrix_W : torch.Tensor
        The weight matrix to be quantized, shape (p, m), dtype float64.
        Set to None when matrix_Y is provided.
    matrix_X : torch.Tensor, optional
        The input matrix, shape (n, m), dtype float64.
        Set to None when matrix_XX is provided.
    matrix_Y : torch.Tensor, optional
        The target matrix, shape (p, n), dtype float64.
        When specified, the objective function becomes ||Y - hat_W X^T||_F^2.
    device : torch.device
        The device to use for quantization.
    matrix_XX : torch.Tensor, optional
        Precomputed X^T X, shape (m, m), dtype float64.
        Can be placed on either CPU or GPU (internally transferred to device).
        When specified, used instead of matrix_X (computation from matrix_X is skipped).
        When specified, matrix_Y=None, matrix_X=None, and dim_n!=None are required.
        Default is None.
    dim_n : int, optional
        Number of rows n of input matrix X. Required when matrix_XX is specified.
        Default is None.
    bits : int, optional
        The quantization bits. Default is 4.
    symmetric : bool, optional
        Whether to use symmetric quantization. Default is False.
    group_size : int, optional
        The size of each group. If None, this is set to m. Default is 128.
    batch_size : int or None, optional
        The batch size to use for quantization.
        When None, all rows are optimized together without splitting. Default is None.
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
        When None, automatically set to min(dim_p, 1024). Default is None.
    log_level : int, optional
        The log level. Default is 2.
    enable_clip_optimize : bool, optional
        Whether to enable the Clip-Optimize initialization strategy. Default is True.
    enable_clip_optimize_ep : bool, optional
        Whether to enable the Clip-Optimize with Error Propagation initialization
        strategy. Default is True.
    initial_solutions : dict[str, Solution], optional
        Pre-built Solution objects to include in the candidate pool.
        Keys are names used for logging, values are Solution objects.
        Must be on the correct device with shapes matching the quantization
        configuration. Default is None.

    Returns
    -------
    Solution
        A Solution object holding the quantization results.
        Main attributes:
        - ``scales`` : torch.Tensor, shape (num_groups, p) — scale coefficients
        - ``zero_point`` : torch.Tensor, shape (num_groups, p) — zero points
        - ``integers_z`` : torch.Tensor, shape (num_groups, p, group_size) — integer assignments
        - ``squared_error`` : torch.Tensor — squared error
        - ``mean_squared_error`` : torch.Tensor — mean squared error

        Use ``solution.get_dequantized_weight_matrix()`` to obtain the dequantized
        weight matrix (shape (p, m)).

    Examples
    --------
    Basic usage (minimizing ||WX^T - hat_W X^T||_F^2):

    >>> import torch
    >>> from onecomp.quantizer.jointq.core import quantize
    >>> # Prepare weight matrix W (p, m) and input matrix X (n, m)
    >>> matrix_W = torch.randn(256, 512, dtype=torch.float64)
    >>> matrix_X = torch.randn(1024, 512, dtype=torch.float64)
    >>> device = torch.device(0)  # Use GPU 0
    >>> solution = quantize(
    ...     matrix_W=matrix_W,
    ...     matrix_X=matrix_X,
    ...     device=device,
    ...     bits=4,
    ...     group_size=128,
    ... )
    >>> # Get the dequantized weight matrix
    >>> matrix_W_hat = solution.get_dequantized_weight_matrix()
    >>> # Compute MSE
    >>> mse = torch.mean(
    ...     (matrix_W.to(device) @ matrix_X.to(device).T
    ...      - matrix_W_hat @ matrix_X.to(device).T) ** 2
    ... )

    When specifying target matrix Y (minimizing ||Y - hat_W X^T||_F^2):

    >>> matrix_Y = torch.randn(256, 1024, dtype=torch.float64)
    >>> solution = quantize(
    ...     matrix_Y=matrix_Y,
    ...     matrix_X=matrix_X,
    ...     device=device,
    ...     bits=4,
    ...     group_size=128,
    ... )

    Enable Iterated Local Search (ILS) for further accuracy improvement:

    >>> solution = quantize(
    ...     matrix_W=matrix_W,
    ...     matrix_X=matrix_X,
    ...     device=device,
    ...     bits=4,
    ...     group_size=128,
    ...     ils_num_iterations=10,
    ...     ils_num_clones=8,
    ... )

    Passing precomputed matrix_XX (= X^T X) (matrix_X becomes unnecessary):

    >>> from onecomp.quantizer.jointq.core import compute_matrix_XX
    >>> matrix_XX = compute_matrix_XX(matrix_X, device)
    >>> n = matrix_X.shape[0]
    >>> del matrix_X  # matrix_X is no longer needed
    >>> solution = quantize(
    ...     matrix_W=matrix_W,
    ...     matrix_XX=matrix_XX,
    ...     dim_n=n,
    ...     device=device,
    ...     bits=4,
    ...     group_size=128,
    ... )
    """

    begin_time = time.time()

    dim_n = _validate_quantize_args(
        matrix_W=matrix_W,
        matrix_X=matrix_X,
        matrix_Y=matrix_Y,
        matrix_XX=matrix_XX,
        dim_n=dim_n,
        enable_clip_optimize=enable_clip_optimize,
        enable_clip_optimize_ep=enable_clip_optimize_ep,
        initial_solutions=initial_solutions,
    )

    # When group_size=None, set to m (no group splitting)
    if group_size is None:
        if matrix_W is not None:
            group_size = matrix_W.shape[1]
        elif matrix_XX is not None:
            group_size = matrix_XX.shape[0]
        else:
            group_size = matrix_X.shape[1]

    if log_level >= 1:
        print(f"<{device}>: Start quantization [JointQ version: {__version__}]")

    num_groups, matrix_XX, sub_matrices_XX = setup(
        matrix_X=matrix_X,
        device=device,
        group_size=group_size,
        log_level=log_level,
        matrix_XX=matrix_XX,
    )

    solution = run_init_local_search(
        matrix_W=matrix_W,
        matrix_Y=matrix_Y,
        matrix_X=matrix_X,
        matrix_XX=matrix_XX,
        sub_matrices_XX=sub_matrices_XX,
        dim_n=dim_n,
        bits=bits,
        symmetric=symmetric,
        epsilon=epsilon,
        early_stopping_ratio=early_stopping_ratio,
        batch_size=batch_size,
        num_groups=num_groups,
        device=device,
        group_size=group_size,
        log_level=log_level,
        enable_clip_optimize=enable_clip_optimize,
        enable_clip_optimize_ep=enable_clip_optimize_ep,
        initial_solutions=initial_solutions,
    )

    if ils_num_iterations is not None:
        if ils_num_channels is None:
            dim_p = matrix_W.shape[0] if matrix_W is not None else matrix_Y.shape[0]
            ils_num_channels = min(dim_p, 1024)
        solution = run_iterated_local_search(
            matrix_W=matrix_W,
            matrix_Y=matrix_Y,
            matrix_X=matrix_X,
            matrix_XX=matrix_XX,
            sub_matrices_XX=sub_matrices_XX,
            dim_n=dim_n,
            bits=bits,
            symmetric=symmetric,
            epsilon=epsilon,
            early_stopping_ratio=early_stopping_ratio,
            num_groups=num_groups,
            device=device,
            group_size=group_size,
            solution=solution,
            ils_num_iterations=ils_num_iterations,
            ils_num_clones=ils_num_clones,
            ils_num_channels=ils_num_channels,
            log_level=log_level,
        )

    end_time = time.time()
    if log_level >= 1:
        print(f"<{device}>: Total time: {end_time - begin_time:.2f} seconds")

    return solution


def _validate_quantize_args(
    matrix_W,
    matrix_X,
    matrix_Y,
    matrix_XX,
    dim_n,
    enable_clip_optimize=True,
    enable_clip_optimize_ep=True,
    initial_solutions=None,
):
    """Validate arguments for quantize / quantize_multi_gpu.

    Performs the following validations:

    Exclusivity / required checks:
    - Exactly one of matrix_W and matrix_Y must be specified.
    - In matrix_XX mode: matrix_Y=None, matrix_X=None, dim_n!=None are required.
    - In classic mode: matrix_X must be specified.

    dtype checks:
    - matrix_W, matrix_X, matrix_Y: must be float64 (when specified).
    - matrix_XX: must be float64 (when specified).

    CPU checks:
    - matrix_W, matrix_X, matrix_Y: must be on CPU (when specified).
    - matrix_XX can be on either CPU or GPU (transferred to device within setup).

    Shape / size checks:
    - matrix_XX: must be a square matrix.
    - matrix_XX and matrix_W: matrix_XX.shape[0] == matrix_W.shape[1] must hold.
    - matrix_Y and matrix_X: matrix_Y.shape[1] == matrix_X.shape[0] must hold.

    Initialization strategy checks:
    - At least one initialization strategy must be enabled or initial_solutions
      must be provided.

    Parameters
    ----------
    matrix_W : torch.Tensor or None
    matrix_X : torch.Tensor or None
    matrix_Y : torch.Tensor or None
    matrix_XX : torch.Tensor or None
    dim_n : int or None
    enable_clip_optimize : bool
    enable_clip_optimize_ep : bool
    initial_solutions : dict[str, Solution] or None

    Returns
    -------
    dim_n : int
        Number of rows n of input matrix X.
        In matrix_XX mode, returned as-is.
        In matrix_X mode, returns matrix_X.shape[0].

    Raises
    ------
    ValueError
        When the argument combination is invalid.
    """
    # --- Exclusivity / required checks ---
    if (matrix_W is None) == (matrix_Y is None):
        raise ValueError("Exactly one of matrix_W or matrix_Y must be specified (exclusive).")

    if matrix_XX is not None:
        # matrix_XX mode: requires matrix_Y=None, matrix_X=None, dim_n!=None
        if matrix_Y is not None:
            raise ValueError("matrix_XX cannot be used together with matrix_Y.")
        if matrix_X is not None:
            raise ValueError("matrix_XX cannot be used together with matrix_X.")
        if dim_n is None:
            raise ValueError("dim_n must be specified when matrix_XX is provided.")
    else:
        # Classic mode: matrix_X is required
        if matrix_X is None:
            raise ValueError("Either matrix_X or matrix_XX must be provided.")
        dim_n = matrix_X.shape[0]

    # --- dtype checks ---
    if matrix_W is not None and matrix_W.dtype != torch.float64:
        raise ValueError(f"matrix_W must have dtype torch.float64, got {matrix_W.dtype}.")
    if matrix_X is not None and matrix_X.dtype != torch.float64:
        raise ValueError(f"matrix_X must have dtype torch.float64, got {matrix_X.dtype}.")
    if matrix_Y is not None and matrix_Y.dtype != torch.float64:
        raise ValueError(f"matrix_Y must have dtype torch.float64, got {matrix_Y.dtype}.")
    if matrix_XX is not None and matrix_XX.dtype != torch.float64:
        raise ValueError(f"matrix_XX must have dtype torch.float64, got {matrix_XX.dtype}.")

    # --- CPU checks (matrix_XX can be on either CPU or GPU) ---
    if matrix_W is not None and matrix_W.device != torch.device("cpu"):
        raise ValueError("matrix_W must be on CPU.")
    if matrix_X is not None and matrix_X.device != torch.device("cpu"):
        raise ValueError("matrix_X must be on CPU.")
    if matrix_Y is not None and matrix_Y.device != torch.device("cpu"):
        raise ValueError("matrix_Y must be on CPU.")

    # --- Shape / size checks ---
    if matrix_XX is not None:
        if matrix_XX.ndim != 2 or matrix_XX.shape[0] != matrix_XX.shape[1]:
            raise ValueError(f"matrix_XX must be a square matrix, got shape {matrix_XX.shape}.")
        if matrix_W is not None and matrix_XX.shape[0] != matrix_W.shape[1]:
            raise ValueError(
                f"matrix_XX.shape[0] (= {matrix_XX.shape[0]}) must equal "
                f"matrix_W.shape[1] (= {matrix_W.shape[1]})."
            )
    if matrix_Y is not None:
        if matrix_X is not None and matrix_Y.shape[1] != matrix_X.shape[0]:
            raise ValueError(
                "matrix_Y must have shape (p, n) where n = matrix_X.shape[0]. "
                f"Got matrix_Y.shape={matrix_Y.shape}, matrix_X.shape={matrix_X.shape}."
            )

    # --- Initialization strategy checks ---
    if not (enable_clip_optimize or enable_clip_optimize_ep
            or initial_solutions):
        raise ValueError(
            "At least one initialization strategy must be enabled or "
            "initial_solutions must be provided."
        )

    return dim_n


def setup(
    matrix_X,
    device,
    group_size,
    log_level,
    matrix_XX=None,
):
    """Setup the quantization.

    Compute matrix_XX and sub_matrices_XX.

    When matrix_XX is provided, it is transferred to device and used directly,
    skipping the computation from matrix_X.
    When matrix_XX is None, it is computed from matrix_X via compute_matrix_XX.

    Assumes that argument validation has been done in _validate_quantize_args beforehand.

    Parameters
    ----------
    matrix_X : torch.Tensor or None
        The input matrix, shape (n, m), dtype float64.
        Can be None when matrix_XX is specified.
    device : torch.device
        The device to use for quantization.
    group_size : int
        The size of each group.
    log_level : int
        The log level.
    matrix_XX : torch.Tensor, optional
        Precomputed X^T X, shape (m, m), dtype float64.
        When specified, computation from matrix_X is skipped and transferred to device.
        Default is None.

    Returns
    -------
    num_groups : int
        Number of groups (m / group_size).
    matrix_XX : torch.Tensor
        X^T X, shape (m, m), dtype float64, placed on device.
    sub_matrices_XX : torch.Tensor
        Block partition of X^T X, shape (num_groups, num_groups, group_size, group_size),
        dtype float64, placed on device.
    """

    if log_level >= 1:
        print(f"<{device}>: Setup")

    if matrix_XX is not None:
        matrix_XX = matrix_XX.to(device)
        m = matrix_XX.shape[0]
    else:
        m = matrix_X.shape[1]
        matrix_XX = compute_matrix_XX(matrix_X, device)

    assert m % group_size == 0, f"m (= {m}) must be divisible by group_size (= {group_size})."
    num_groups = m // group_size

    sub_matrices_XX = (
        matrix_XX.view(num_groups, group_size, num_groups, group_size)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    return num_groups, matrix_XX, sub_matrices_XX


def compute_matrix_XX(matrix_X, device, chunk_size=8192):
    """Compute X^T X from matrix_X in a memory-efficient way.

    Compute X^T X by accumulating chunk-by-chunk without loading
    the entire matrix_X onto the GPU. Can be used to pass matrix_XX directly to quantize().

    Parameters
    ----------
    matrix_X : torch.Tensor
        Input matrix, shape (n, m), dtype float64. Must be placed on CPU.
    device : torch.device
        Computation device (device where the result will be placed).
    chunk_size : int, optional
        Chunk size. Default is 8192.

    Returns
    -------
    matrix_XX : torch.Tensor
        X^T X, shape (m, m), dtype float64, placed on device.

    Examples
    --------
    >>> import torch
    >>> from onecomp.quantizer.jointq.core import compute_matrix_XX, quantize
    >>> matrix_X = torch.randn(1024, 512, dtype=torch.float64)
    >>> device = torch.device(0)
    >>> matrix_XX = compute_matrix_XX(matrix_X, device)
    >>> n = matrix_X.shape[0]
    >>> del matrix_X  # matrix_X is no longer needed
    >>> matrix_W = torch.randn(256, 512, dtype=torch.float64)
    >>> solution = quantize(
    ...     matrix_W=matrix_W,
    ...     matrix_XX=matrix_XX,
    ...     dim_n=n,
    ...     device=device,
    ... )
    """
    m = matrix_X.shape[1]
    matrix_XX = torch.zeros(m, m, dtype=torch.float64, device=device)
    chunk_size = min(matrix_X.shape[0], chunk_size)
    for start in range(0, matrix_X.shape[0], chunk_size):
        chunk = matrix_X[start : start + chunk_size].to(dtype=torch.float64, device=device)
        matrix_XX += chunk.T @ chunk
        del chunk
    return matrix_XX


def run_init_local_search(
    matrix_W,
    matrix_Y,
    matrix_X,
    matrix_XX,
    sub_matrices_XX,
    dim_n,
    bits,
    symmetric,
    epsilon,
    early_stopping_ratio,
    batch_size,
    num_groups,
    device,
    group_size,
    log_level,
    enable_clip_optimize=True,
    enable_clip_optimize_ep=True,
    initial_solutions=None,
):
    """Run the initial local search.

    Parameters
    ----------
    matrix_W : torch.Tensor
        The weight matrix to be quantized, shape (p, m), dtype float64.
    matrix_X : torch.Tensor or None
        The input matrix, shape (n, m), dtype float64. Placed on CPU.
        None when matrix_XX is directly provided in classic mode.
    matrix_XX : torch.Tensor
        X^T X, shape (m, m), dtype float64, on device.
    sub_matrices_XX : torch.Tensor
        Block partition of X^T X,
        shape (num_groups, num_groups, group_size, group_size), on device.
    dim_n : int
        Number of rows n of input matrix X.
    bits : int
        The quantization bits.
    symmetric : bool
        Whether to use symmetric quantization.
    epsilon : float
        The epsilon for the quantization.
    early_stopping_ratio : float
        The ratio for the early stopping.
    batch_size : int
        The batch size to use for quantization.
    num_groups : int
        The number of groups.
    device : torch.device
        The device to use for quantization.
    group_size : int
        The size of each group.
    enable_clip_optimize : bool, optional
        Whether to enable the Clip-Optimize initialization strategy. Default is True.
    enable_clip_optimize_ep : bool, optional
        Whether to enable the Clip-Optimize with Error Propagation initialization
        strategy. Default is True.
    initial_solutions : dict[str, Solution], optional
        Pre-built Solution objects to include in the candidate pool.
        Keys are names for logging, values are Solution objects. Default is None.

    Returns
    -------
    solution : Solution
        The solution.
    quantizer : Quantizer
        The quantizer.

    Note
    ----
    Only matrix_XX and sub_matrices_XX are placed on GPU among the arguments.
    matrix_W, matrix_Y, and matrix_X are all on CPU (matrix_X may be None).

    """

    if log_level >= 1:
        print(f"<{device}>: Run initial local search")

    # Split weight matrix W into batches of batch_size rows and quantize each batch
    solutions = []
    dim_p = matrix_W.shape[0] if matrix_W is not None else matrix_Y.shape[0]
    if batch_size is None:
        batch_size = dim_p
    else:
        batch_size = min(batch_size, dim_p)

    for i in range(0, dim_p, batch_size):
        if log_level >= 1:
            print(
                f"<{device}>: Processing batch "
                f"{i // batch_size + 1} of {math.ceil(dim_p / batch_size)}:"
            )
        matrix_W_batch = matrix_W[i : i + batch_size] if matrix_W is not None else None
        matrix_Y_batch = matrix_Y[i : i + batch_size] if matrix_Y is not None else None

        tilde_W, matrix_YX, sub_matrices_YX, Y_sq_norms = compute_batch_precomputations(
            matrix_W_batch=matrix_W_batch,
            matrix_Y_batch=matrix_Y_batch,
            matrix_X=matrix_X,
            matrix_XX=matrix_XX,
            num_groups=num_groups,
            group_size=group_size,
            device=device,
        )

        quantizer = Quantizer(
            matrix_XX=matrix_XX,
            sub_matrices_XX=sub_matrices_XX,
            matrix_YX=matrix_YX,
            sub_matrices_YX=sub_matrices_YX,
            Y_sq_norms=Y_sq_norms,
            dim_n=dim_n,
            bits=bits,
            symmetric=symmetric,
            epsilon=epsilon,
            early_stopping_ratio=early_stopping_ratio,
            log_level=log_level,
        )
        if log_level >= 1:
            quantizer.display_information()
            quantizer.display_gpu_memory_usage()
        quantizer.set_begin_time()

        # initialize solution
        # For each row, adopt the best solution
        cand_solutions = []
        cand_names = []

        if enable_clip_optimize:
            cand_solutions.append(quantizer.initialize_solution_clip_optimize(tilde_W))
            cand_names.append("Clip-Optimize")

        if num_groups > 1 and enable_clip_optimize_ep:
            # When group_size = m, clip_optimize_ep yields the same result as clip_optimize, so skip
            cand_solutions.append(quantizer.initialize_solution_clip_optimize_ep(tilde_W))
            cand_names.append("Clip-Optimize-EP")

        if initial_solutions is not None:
            for name, sol in initial_solutions.items():
                sol.compute_objective_value(**quantizer.objective_args)
                if quantizer.log_level >= 1:
                    quantizer.display_log(sol, name, ignore_log_level=True)
                result = quantizer._optimize_scales(sol)
                quantizer.display_log(sol, f"OptS ({result}/{quantizer.dim_p})")
                if quantizer.log_level >= 1:
                    quantizer.display_log(sol, name, ignore_log_level=True)
                cand_solutions.append(sol)
                cand_names.append(name)

        del tilde_W
        solution = select_best_solution(
            solutions=cand_solutions,
            names=cand_names,
            device=device,
            log_level=log_level,
        )
        del cand_solutions
        solution.compute_objective_value(**quantizer.objective_args)
        # Quantize
        quantizer.set_modified_lower_and_upper_bounds(solution)
        solution = quantizer.quantize(solution)
        solutions.append(solution)

    solution = merge_solutions(solutions)

    return solution


def compute_batch_precomputations(
    matrix_W_batch,
    matrix_Y_batch,
    matrix_X,
    matrix_XX,
    num_groups,
    group_size,
    device,
    compute_tilde_W=True,
):
    """Compute tilde_W, matrix_YX, sub_matrices_YX, Y_sq_norms for a batch.

    Compute without loading the entire matrix_X onto the GPU by leveraging
    the precomputed matrix_XX (= X^T X).

    Classic mode (matrix_Y_batch is None):
        tilde_W = W_batch
        Since Y = tilde_W @ X^T,
        - matrix_YX = tilde_W @ X^T X (= tilde_W @ matrix_XX)
        - Y_sq_norms = (matrix_YX * tilde_W).sum(dim=1)
        Can be computed using only matrix_XX without matrix_X at all.

    Y mode (matrix_Y_batch is given):
        Split matrix_X and matrix_Y_batch along the n dimension into chunks
        to accumulate matrix_YX and Y_sq_norms.
        tilde_W is obtained by solving the normal equation X^T X @ W^T = (matrix_YX)^T
        (skipped and returns None when compute_tilde_W=False).

    Parameters
    ----------
    matrix_W_batch : torch.Tensor or None
        Weight matrix batch, shape (p_batch, m), CPU.
        Used when matrix_Y_batch is None.
    matrix_Y_batch : torch.Tensor or None
        Target matrix batch, shape (p_batch, n), CPU.
        None for classic mode.
    matrix_X : torch.Tensor or None
        Input matrix, shape (n, m), CPU.
        Can be None in classic mode (matrix_Y_batch is None)
        (m is obtained from matrix_XX).
    matrix_XX : torch.Tensor
        X^T X, shape (m, m), on device.
    num_groups : int
        Number of groups.
    group_size : int
        Group size.
    device : torch.device
        Computation device.
    compute_tilde_W : bool, optional
        Whether to compute tilde_W in Y mode. Default is True.
        When False, tilde_W returns None.
        Ignored in classic mode (always tilde_W = W_batch).

    Returns
    -------
    tilde_W : torch.Tensor or None
        Real-valued weight matrix, shape (p_batch, m), on device.
        None when compute_tilde_W=False and in Y mode.
    matrix_YX : torch.Tensor
        Y @ X, shape (p_batch, m), on device.
    sub_matrices_YX : torch.Tensor
        Group partition of Y @ X, shape (p_batch, num_groups, group_size), on device.
    Y_sq_norms : torch.Tensor
        Squared norms of each row of the target matrix, shape (p_batch,), on device.

    """

    m = matrix_XX.shape[0]

    if matrix_Y_batch is None:
        # Classic mode: Y = tilde_W @ X^T
        # Can be computed using only matrix_XX without matrix_X
        tilde_W = matrix_W_batch.to(device)
        matrix_YX = tilde_W @ matrix_XX  # (p_batch, m)
        Y_sq_norms = (matrix_YX * tilde_W).sum(dim=1)  # (p_batch,)
    else:
        # Y mode: split matrix_X and matrix_Y_batch into chunks and accumulate
        p_batch = matrix_Y_batch.shape[0]
        n = matrix_X.shape[0]
        matrix_YX = torch.zeros(p_batch, m, dtype=torch.float64, device=device)
        Y_sq_norms = torch.zeros(p_batch, dtype=torch.float64, device=device)

        chunk_size = min(n, 8192)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            Y_chunk = matrix_Y_batch[:, start:end].to(device)
            X_chunk = matrix_X[start:end].to(device)
            matrix_YX += Y_chunk @ X_chunk
            Y_sq_norms += Y_chunk.pow(2).sum(dim=1)
            del Y_chunk, X_chunk

        # Solve the normal equation X^T X @ W^T = (matrix_YX)^T
        if compute_tilde_W:
            tilde_W = torch.linalg.solve(matrix_XX, matrix_YX.T).T.contiguous()
        else:
            tilde_W = None

    # Reshape matrix_YX to (p_batch, num_groups, group_size)
    sub_matrices_YX = matrix_YX.view(-1, num_groups, group_size)

    return tilde_W, matrix_YX, sub_matrices_YX, Y_sq_norms


def select_best_solution(solutions, names, device, log_level):
    """Select the best solution."""

    assert len(solutions) == len(names)

    stacked_errors = torch.stack([solution.squared_errors for solution in solutions], dim=0)
    best_index = torch.argmin(stacked_errors, dim=0)
    counts = torch.bincount(best_index, minlength=len(solutions))
    for key, value in zip(names, counts):
        if log_level >= 1:
            print(f"<{device}>: {key}: {value}")

    best_solution = Solution()
    best_solution.scales = torch.zeros_like(solutions[0].scales)
    best_solution.zero_point = torch.zeros_like(solutions[0].zero_point)
    best_solution.integers_z = torch.zeros_like(solutions[0].integers_z)

    for i, solution in enumerate(solutions):
        indices = best_index == i
        best_solution.scales[:, indices] = solution.scales[:, indices]
        best_solution.zero_point[:, indices] = solution.zero_point[:, indices]
        best_solution.integers_z[:, indices, :] = solution.integers_z[:, indices, :]

    return best_solution


def merge_solutions(solutions):
    """Merge the solutions.

    Concatenate solutions from each batch along the row dimension into a single solution.

    Parameters
    ----------
    solutions : list of Solution
        The solutions to be merged.

    Returns
    -------
    solution : Solution
        The merged solution.
    """

    solution = Solution()
    solution.scales = torch.cat(
        [s.scales for s in solutions],
        dim=1,
    )
    solution.zero_point = torch.cat(
        [s.zero_point for s in solutions],
        dim=1,
    )
    solution.integers_z = torch.cat(
        [s.integers_z for s in solutions],
        dim=1,
    )
    solution.squared_errors = torch.cat(
        [s.squared_errors for s in solutions],
        dim=0,
    )
    solution.mean_squared_errors = torch.cat(
        [s.mean_squared_errors for s in solutions],
        dim=0,
    )
    solution.squared_error = torch.sum(solution.squared_errors)
    solution.mean_squared_error = solution.mean_squared_errors.mean()

    return solution


def run_iterated_local_search(
    matrix_W,
    matrix_Y,
    matrix_X,
    matrix_XX,
    sub_matrices_XX,
    dim_n,
    bits,
    symmetric,
    epsilon,
    early_stopping_ratio,
    num_groups,
    device,
    group_size,
    solution,
    ils_num_iterations,
    ils_num_clones,
    ils_num_channels,
    log_level,
):
    """Run the iterated local search.

    Parameters
    ----------
    matrix_W : torch.Tensor or None
        Weight matrix, shape (p, m), CPU.
    matrix_Y : torch.Tensor or None
        Target matrix, shape (p, n), CPU. None for classic mode.
    matrix_X : torch.Tensor or None
        Input matrix, shape (n, m), CPU.
        None when directly providing matrix_XX in classic mode.
    matrix_XX : torch.Tensor
        X^T X, shape (m, m), on device.
    sub_matrices_XX : torch.Tensor
        Block partition of X^T X, shape (num_groups, num_groups, group_size, group_size),
        on device.
    dim_n : int
        Number of rows n of input matrix X.
    bits : int
        Number of quantization bits.
    symmetric : bool
        Whether to use symmetric quantization.
    epsilon : float
        Epsilon for numerical stability.
    early_stopping_ratio : float
        Ratio for early stopping.
    num_groups : int
        Number of groups.
    device : torch.device
        Computation device.
    group_size : int
        Group size.
    solution : Solution
        Initial solution.
    ils_num_iterations : int
        Number of ILS iterations.
    ils_num_clones : int
        Number of clones created for each row.
    ils_num_channels : int
        Number of rows targeted in each iteration.
    log_level : int
        Log level.

    """

    dim_p = matrix_W.shape[0] if matrix_W is not None else matrix_Y.shape[0]
    assert ils_num_channels <= dim_p
    # TODO: Handle the case where this condition is violated.
    # When violated, set ils_num_channels = dim_p.
    # Also, in that case, since clones are created for all rows,
    # there is no need to compute the objective function values.

    # solution holds the squared_errors computed after quantize
    squared_errors = solution.squared_errors.clone()
    num_element = dim_p * dim_n

    if log_level >= 1:
        print(
            f"<{device}>: Run iterated local search"
            f" (error = {torch.sum(squared_errors):.3e}, "
            f"MSE = {(torch.sum(squared_errors) / num_element):.3e})"
        )

    # Fix the torch random seed
    torch.manual_seed(0)

    for i in range(ils_num_iterations):
        if log_level >= 2:
            print(f"<{device}>: === ILS: Iteration {i + 1} of {ils_num_iterations} ====")

        # setup
        # - target_indices: Extract ils_num_channels row indices with the worst objective values
        # - target_solution: For the extracted row indices, create ils_num_clones solutions for each.
        target_indices, target_solution = setup_ils(
            solution=solution,
            squared_errors=squared_errors,
            ils_num_clones=ils_num_clones,
            ils_num_channels=ils_num_channels,
        )

        # Compute matrix_YX, sub_matrices_YX, Y_sq_norms for target rows
        # Extract matrix_W / matrix_Y for the target batch
        target_indices_cpu = target_indices.to("cpu")
        if matrix_Y is not None:
            target_Y_batch = torch.cat(
                [matrix_Y[target_indices_cpu] for _ in range(ils_num_clones)], dim=0
            )
            target_W_batch = None
        else:
            target_W_batch = torch.cat(
                [matrix_W[target_indices_cpu] for _ in range(ils_num_clones)], dim=0
            )
            target_Y_batch = None

        _, matrix_YX, sub_matrices_YX, Y_sq_norms = compute_batch_precomputations(
            matrix_W_batch=target_W_batch,
            matrix_Y_batch=target_Y_batch,
            matrix_X=matrix_X,
            matrix_XX=matrix_XX,
            num_groups=num_groups,
            group_size=group_size,
            device=device,
            compute_tilde_W=False,
        )
        del target_W_batch, target_Y_batch

        # Compute the objective values of target_solution
        target_solution.compute_objective_value(
            matrix_XX=matrix_XX,
            matrix_YX=matrix_YX,
            Y_sq_norms=Y_sq_norms,
            dim_n=dim_n,
        )

        if log_level >= 2:
            print(
                f"<{device}>: error: {torch.sum(squared_errors[target_indices]) * ils_num_clones}"
            )
            print(f"<{device}>: After random change:")
            print(f"<{device}>: error: {target_solution.squared_error}")

        # Quantize
        quantizer = Quantizer(
            matrix_XX=matrix_XX,
            sub_matrices_XX=sub_matrices_XX,
            matrix_YX=matrix_YX,
            sub_matrices_YX=sub_matrices_YX,
            Y_sq_norms=Y_sq_norms,
            dim_n=dim_n,
            bits=bits,
            symmetric=symmetric,
            epsilon=epsilon,
            early_stopping_ratio=early_stopping_ratio,
            log_level=2 if log_level >= 2 else 0,
        )
        if log_level >= 2:
            quantizer.display_information()
            quantizer.display_gpu_memory_usage()
        quantizer.set_begin_time()
        quantizer.set_modified_lower_and_upper_bounds(target_solution)
        target_solution = quantizer.quantize(target_solution)

        # Update squared_errors and solution
        update_squared_errors_and_solution(
            target_indices=target_indices,
            target_solution=target_solution,
            squared_errors=squared_errors,
            solution=solution,
            ils_num_clones=ils_num_clones,
            ils_num_channels=ils_num_channels,
            epsilon=epsilon,
        )

        error = torch.sum(squared_errors)
        if log_level >= 1:
            print(f"<{device}>: <ILS {i + 1}> Result: error = {error:.4e}, MSE = {(error / num_element):.4e}")

    return solution


def setup_ils(
    solution,
    squared_errors,
    ils_num_clones,
    ils_num_channels,
    # random_ratio=0.01,
    random_ratio=0.005,
    # random_ratio=0.05,
):
    """Setup for the iterated local search.

    - Extract ils_num_channels row indices with the worst objective values (target_indices)
    - For the extracted row indices, create ils_num_clones solutions for each (target_solution)
    - After cloning, randomly set integers_z values to 0 with probability random_ratio.

    Parameters
    ----------
    solution : Solution
        Current solution.
    squared_errors : torch.Tensor
        Squared error per row, shape (p,).
    ils_num_clones : int
        Number of clones created for each row.
    ils_num_channels : int
        Number of rows targeted in each iteration.
    random_ratio : float
        Probability of randomly setting integers_z to 0.

    Returns
    -------
    target_indices : torch.Tensor
        Indices of target rows, shape (ils_num_channels,), on device.
    target_solution : Solution
        Solution with random perturbations applied.

    """

    # step1. Extract the top ils_num_channels row indices with the highest squared_errors
    target_indices = torch.argsort(squared_errors, descending=True)[:ils_num_channels]

    # step2. For the extracted row indices, create ils_num_clones solutions for each
    target_solution = Solution()
    target_solution.scales = torch.cat(
        [solution.scales[:, target_indices] for _ in range(ils_num_clones)],
        dim=1,
    )
    target_solution.zero_point = torch.cat(
        [solution.zero_point[:, target_indices] for _ in range(ils_num_clones)],
        dim=1,
    )
    target_solution.integers_z = torch.cat(
        [solution.integers_z[:, target_indices, :] for _ in range(ils_num_clones)],
        dim=1,
    )

    # step3. Randomly set integers_z values to 0 with probability random_ratio.
    target_solution.integers_z[torch.rand(target_solution.integers_z.shape) < random_ratio] = 0
    # Note: The objective values of target_solution are computed by the caller

    return target_indices, target_solution


def update_squared_errors_and_solution(
    target_indices,
    target_solution,
    squared_errors,
    solution,
    ils_num_clones,
    ils_num_channels,
    epsilon,
):
    """Update the squared errors and the solution.

    For each target row, select the best among ils_num_clones solutions,
    and update if it improves over the original solution.

    Parameters
    ----------
    target_indices : torch.Tensor
        Indices of target rows, shape (ils_num_channels,), on device.
    target_solution : Solution
        Post-quantization solution (squared_errors already computed).
    squared_errors : torch.Tensor
        Squared error per row, shape (p,), on device. Updated in-place.
    solution : Solution
        Original solution. Updated in-place.
    ils_num_clones : int
        Number of clones created for each row.
    ils_num_channels : int
        Number of target rows.
    epsilon : float
        Threshold for improvement determination.

    """

    device = target_indices.device

    # 1) For each target row, find the index of the best among ils_num_clones solutions
    # target_solution.squared_errors is already computed after quantize
    candidate_squared_errors = target_solution.squared_errors
    # (ils_num_channels * ils_num_clones, ) -> (ils_num_channels, ils_num_clones)
    candidate_squared_errors2 = candidate_squared_errors.view(ils_num_clones, ils_num_channels).T
    best_indices = torch.argmin(candidate_squared_errors2, dim=1)  # (ils_num_channels, )
    # Since 0 <= best_index < ils_num_clones, find the indices in the original target_solution
    best_indices = torch.arange(ils_num_channels).to(device) + best_indices * ils_num_channels
    # From best_indices, extract indices where squared_errors improved
    # i.e., extract indices satisfying candidate_squared_errors[best_indices] < squared_errors[target_indices]
    improved_indices = (candidate_squared_errors[best_indices] + epsilon) < squared_errors[
        target_indices
    ]
    # Update squared_errors for improved_indices
    target_indices = target_indices[improved_indices]
    best_indices = best_indices[improved_indices]
    squared_errors[target_indices] = candidate_squared_errors[best_indices]
    # Update solution for improved_indices
    solution.scales[:, target_indices] = target_solution.scales[:, best_indices]
    solution.zero_point[:, target_indices] = target_solution.zero_point[:, best_indices]
    solution.integers_z[:, target_indices, :] = target_solution.integers_z[:, best_indices, :]


def display_memory_usage(device1, device2):
    """Display the memory usage."""

    if device1 == device2:
        print(f"Using the same device for quantization: {device1}")
        print(
            f"GPU memory usage: {torch.cuda.memory_allocated(device1) / 1024 ** 3:.2f} GB "
            f"/ {torch.cuda.get_device_properties(device1).total_memory / 1024 ** 3:.2f} GB"
        )
    else:
        print(f"Using different devices for quantization: {device1} and {device2}")
        print(
            f"GPU memory usage: {torch.cuda.memory_allocated(device1) / 1024 ** 3:.2f} GB "
            f"/ {torch.cuda.get_device_properties(device1).total_memory / 1024 ** 3:.2f} GB"
        )
        print(
            f"GPU memory usage: {torch.cuda.memory_allocated(device2) / 1024 ** 3:.2f} GB "
            f"/ {torch.cuda.get_device_properties(device2).total_memory / 1024 ** 3:.2f} GB"
        )
