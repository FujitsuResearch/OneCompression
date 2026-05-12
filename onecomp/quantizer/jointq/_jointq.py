"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from dataclasses import dataclass
from typing import List, Optional

import torch

from .core import compute_matrix_XX, quantize
from .core.solution import Solution

from onecomp.quantizer._quantizer import Quantizer, QuantizationResult
from onecomp.quantizer.gptq._gptq import GPTQ

_DEFAULT_LAMBDA_LIST = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


@dataclass
class JointQResult(QuantizationResult):
    """JointQ quantization result class

    Inherits from QuantizationResult and adds JointQ-specific parameters.

    Attributes:

        [Quantization configuration parameters]
        bits: Number of quantization bits
        symmetric: Whether symmetric quantization was used
        group_size: Group size

        [Data for weight reconstruction]
        scale: Scale factor, shape (out_features, num_groups)
        zero_point: Zero point, shape (out_features, num_groups)
        assignment: Integer assignment, shape (out_features, num_groups, group_size)

    Note:
        - The dequantized weight can be reconstructed as follows:
          W_hat[i, g*group_size:(g+1)*group_size]
              = scale[i, g] * (assignment[i, g, :] - zero_point[i, g])
        - When actorder is used, scale/zero_point/assignment are stored in
          the permuted column order. Use ``perm`` to map back to original
          column order (see ``compute_dequantized_weight``).
    """

    # =========================================
    # Quantization configuration parameters
    # =========================================
    bits: int = None
    symmetric: bool = None
    group_size: int = None

    # =========================================
    # Data for weight reconstruction
    # =========================================
    scale: Optional[torch.Tensor] = None  # Scale factor
    zero_point: Optional[torch.Tensor] = None  # Zero point
    assignment: Optional[torch.Tensor] = None  # Integer assignment
    perm: Optional[torch.Tensor] = None  # Column permutation (actorder)

    # =========================================
    # Incremental lambda statistics
    # =========================================
    accepted_lambda: Optional[float] = None

    def compute_dequantized_weight(self, device: torch.device = None) -> torch.Tensor:
        """Compute the dequantized weight from quantization parameters

        Reconstruct the weight using the following formula:
            W_hat[i, g*group_size:(g+1)*group_size]
                = scale[i, g] * (assignment[i, g, :] - zero_point[i, g])

        Args:
            device (torch.device): Device for computation.
                If None, computation is performed on the device where the quantization parameters reside.

        Returns:
            torch.Tensor: Dequantized weight tensor (FP16), shape (out_features, in_features)

        """
        # If a device is specified, compute on that device
        if device is not None:
            scale = self.scale.to(device)
            zero_point = self.zero_point.to(device)
            assignment = self.assignment.to(device)
        else:
            scale = self.scale
            zero_point = self.zero_point
            assignment = self.assignment

        # scale: (out_features, num_groups)
        # zero_point: (out_features, num_groups)
        # assignment: (out_features, num_groups, group_size)
        out_features = scale.shape[0]

        # Expand dimensions for broadcasting
        # scale_expanded: (out_features, num_groups, 1)
        # zero_point_expanded: (out_features, num_groups, 1)
        scale_expanded = scale.unsqueeze(-1)
        zero_point_expanded = zero_point.unsqueeze(-1)

        # W_hat = scale * (assignment - zero_point)
        # dequantized: (out_features, num_groups, group_size)
        dequantized = scale_expanded * (assignment - zero_point_expanded)

        # Reshape to (out_features, num_groups * group_size) = (out_features, in_features)
        dequantized_weight = dequantized.reshape(out_features, -1)

        # Inverse-permute columns when actorder was used
        if self.perm is not None:
            invperm = torch.argsort(self.perm)
            if device is not None:
                invperm = invperm.to(device)
            dequantized_weight = dequantized_weight[:, invperm]

        return dequantized_weight.to(torch.float16).cpu()

    def get_statistics(self) -> dict:
        stats = super().get_statistics()
        if self.accepted_lambda is not None:
            stats["accepted_lambda"] = self.accepted_lambda
        return stats


@dataclass
class JointQ(Quantizer):
    """JointQ quantizer class.

    JointQ is a post-training quantization method that combines multiple
    initialization strategies (Clip-Optimize, Clip-Optimize-EP, GPTQ) with
    local search optimization to find high-quality quantized weights.

    Attributes:
        bits (int): Number of bits for quantization. Default is 4.
        symmetric (bool): Whether to use symmetric quantization. Default is False.
        group_size (int or None): Group size for quantization. Default is 128.
            If None, per-channel quantization is used (group_size = in_features).
        log_level (int): Log level (0: none, 1: minimal, 2: detailed). Default is 0.
        device (torch.device or None): Device for quantization.
            If None, uses the device of the module being quantized.
        regularization_lambda (float or None): Tikhonov regularization strength.
            Default is 0.2. Replaces X^T X with X^T X + n*lambda*R where
            R depends on ``regularization_mode``.
            lambda is relative to the normalized Hessian (1/n)X^T X, so its
            meaning is consistent across different calibration sample sizes.
            Recommended range: 0.1 to 1.0. Set to None or 0.0 to disable.
            Used only in ``fixed_lambda`` mode.
        regularization_mode (str): Shape of the regularization matrix R.
            ``"identity"`` (default): R = I (standard Tikhonov).
            ``"diagonal"``: R = diag(a) where
            a_i = (diag(X^T X)_i / mean(diag(X^T X))) ^ gamma.
            This makes regularization importance-aware: columns with
            larger activations receive stronger regularization.
            Only supported with ``lambda_mode="fixed_lambda"``.
        regularization_gamma (float): Exponent for the diagonal weights
            in ``"diagonal"`` mode. Default is 0.5. Smaller values
            reduce the spread between weak and strong columns.
        lambda_mode (str): Regularization mode. Default is ``"fixed_lambda"``.
            ``"fixed_lambda"``: Use a single fixed ``regularization_lambda``
            for all layers (existing behavior).
            ``"incremental_lambda"``: For each layer, try increasing lambda
            values from ``lambda_list`` and accept the solution as long as it
            improves weight error without substantially degrading output error.
        lambda_list (list of float or None): Ascending list of lambda values
            to try in ``incremental_lambda`` mode.  Ignored in ``fixed_lambda``
            mode.  Default is ``[0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]``.
        incremental_eps_y (float): Maximum tolerated relative output-error
            increase when accepting a candidate in ``incremental_lambda`` mode.
            Default is 0.03 (3%).
        incremental_eps_w (float): Minimum required relative weight-error
            decrease to accept a candidate whose output error worsened in
            ``incremental_lambda`` mode. Default is 0.10 (10%).
        incremental_initial_skip_ew_threshold (float or None): If the first
            incremental candidate uses ``lambda=0.0`` and its relative weight
            error exceeds this threshold, skip that candidate and try the next
            lambda instead of accepting it as the initial solution. This guard
            is only relevant when ``lambda_list`` starts with ``0.0``. Default
            is 0.3 (30%). Set to ``None`` to disable this guard.
        actorder (bool): Whether to reorder columns by activation magnitude
            (Hessian diagonal) before quantization. Default is False.
            When enabled, columns with larger activations are grouped together,
            improving group quantization efficiency and GPTQ initial solution quality.
        ils_enabled (bool): Whether to enable Iterated Local Search. Default is False.
        ils_num_iterations (int): Number of ILS iterations. Default is 10.
        ils_num_clones (int): Number of clones per row in ILS. Default is 8.
        ils_num_channels (int or None): Number of rows targeted per ILS iteration.
            When None, automatically set to min(dim_p, 1024). Default is None.
        enable_clip_optimize (bool): Whether to use Clip-Optimize initialization.
            Default is True.
        enable_clip_optimize_ep (bool): Whether to use Clip-Optimize with Error
            Propagation initialization. Default is False.
        enable_gptq (bool): Whether to use GPTQ initialization. Default is True.
        gptq (GPTQ or None): GPTQ instance for initial solution generation.
            If None, a default GPTQ is created from bits/group_size/symmetric.
            Pass a custom GPTQ instance to control parameters like blocksize,
            percdamp, mse, q_grid, q_norm. The GPTQ instance must have
            wbits/groupsize/sym matching JointQ's bits/group_size/symmetric,
            and actorder must be False.

    Example:
        Basic usage::

            from onecomp.quantizer.jointq import JointQ

            quantizer = JointQ(
                bits=4,
                symmetric=False,
                group_size=128,
            )

        With all initialization strategies enabled::

            quantizer = JointQ(
                bits=4,
                symmetric=False,
                group_size=128,
                enable_clip_optimize=True,
                enable_clip_optimize_ep=True,
                enable_gptq=True,
            )

        With custom GPTQ parameters::

            from onecomp.quantizer.gptq import GPTQ

            quantizer = JointQ(
                bits=4,
                symmetric=False,
                group_size=128,
                gptq=GPTQ(
                    wbits=4, groupsize=128, sym=False, mse=True
                ),
            )

        With incremental lambda mode::

            quantizer = JointQ(
                bits=4,
                symmetric=False,
                group_size=128,
                lambda_mode="incremental_lambda",
            )

    """

    flag_calibration: bool = True
    flag_hessian: bool = False
    flag_xtx: bool = True
    hessian_dtype: torch.dtype = torch.float64

    # Parameters for the JointQ quantizer

    # Basic parameters
    bits: int = 4
    symmetric: bool = False
    group_size: Optional[int] = 128
    log_level: int = 0  # 0: none, 1: minimal, 2: detailed, 3: debug

    # Device settings
    device: Optional[torch.device] = None

    # Tikhonov regularization: X^T X + n*λ*R  (R depends on regularization_mode)
    regularization_lambda: Optional[float] = 0.1
    regularization_mode: str = "diagonal"
    regularization_gamma: float = 0.5

    # Lambda mode: "fixed_lambda" (default) or "incremental_lambda"
    lambda_mode: str = "fixed_lambda"
    lambda_list: Optional[List[float]] = None
    incremental_eps_y: float = 0.03
    incremental_eps_w: float = 0.10
    incremental_initial_skip_ew_threshold: Optional[float] = 0.3

    # Activation ordering
    actorder: bool = False

    # Iterated Local Search (ILS) parameters
    ils_enabled: bool = False
    ils_num_iterations: int = 10
    ils_num_clones: int = 8
    ils_num_channels: Optional[int] = None

    # Initialization strategies
    enable_clip_optimize: bool = True
    enable_clip_optimize_ep: bool = False
    enable_gptq: bool = True

    # GPTQ initial solution
    gptq: Optional[GPTQ] = None

    def __post_init__(self):
        if self.gptq is None:
            gptq_groupsize = self.group_size if self.group_size is not None else -1
            self.gptq = GPTQ(
                wbits=self.bits, groupsize=gptq_groupsize, sym=self.symmetric
            )
        if self.lambda_list is None:
            self.lambda_list = list(_DEFAULT_LAMBDA_LIST)
        super().__post_init__()

    def validate_params(self):
        """Validate JointQ and GPTQ parameters.

        Called once during setup(). Validates:

        JointQ parameters:
            bits: int >= 1
            group_size: int >= 1 or None
            log_level: int in {0, 1, 2}
            ils_num_iterations: int >= 1 (when ils_enabled)
            ils_num_clones: int >= 1 (when ils_enabled)
            ils_num_channels: int >= 1 or None (when ils_enabled)

        GPTQ consistency:
            gptq.wbits == bits
            gptq.groupsize == group_size (or -1 when group_size is None)
            gptq.sym == symmetric
            gptq.actorder == False

        Also delegates to ``self.gptq.validate_params()`` for GPTQ's own
        parameter validation (blocksize, percdamp, etc.).
        """
        bad = []

        if not (isinstance(self.bits, int) and self.bits >= 1):
            bad.append(f"Invalid JointQ parameter 'bits': {self.bits!r} (expected int >= 1).")

        if self.group_size is not None and not (
            isinstance(self.group_size, int) and self.group_size >= 1
        ):
            bad.append(
                f"Invalid JointQ parameter 'group_size': {self.group_size!r} (expected int >= 1 or None)."
            )

        if not (isinstance(self.log_level, int) and 0 <= self.log_level <= 2):
            bad.append(
                f"Invalid JointQ parameter 'log_level': {self.log_level!r} (expected int in 0..2)."
            )

        if self.ils_enabled:
            if not (isinstance(self.ils_num_iterations, int) and self.ils_num_iterations >= 1):
                bad.append(
                    f"Invalid JointQ parameter 'ils_num_iterations': {self.ils_num_iterations!r} "
                    f"(expected int >= 1 when ILS is enabled)."
                )
            if not (isinstance(self.ils_num_clones, int) and self.ils_num_clones >= 1):
                bad.append(
                    f"Invalid JointQ parameter 'ils_num_clones': {self.ils_num_clones!r} "
                    f"(expected int >= 1 when ILS is enabled)."
                )
            if self.ils_num_channels is not None and not (
                isinstance(self.ils_num_channels, int) and self.ils_num_channels >= 1
            ):
                bad.append(
                    f"Invalid JointQ parameter 'ils_num_channels': {self.ils_num_channels!r} "
                    f"(expected int >= 1 or None when ILS is enabled)."
                )

        if self.regularization_mode not in ("identity", "diagonal"):
            bad.append(
                f"Invalid JointQ parameter 'regularization_mode': "
                f"{self.regularization_mode!r} "
                f"(expected 'identity' or 'diagonal')."
            )
        if self.regularization_mode == "diagonal":
            if self.lambda_mode != "fixed_lambda":
                bad.append(
                    "regularization_mode='diagonal' is only supported "
                    "with lambda_mode='fixed_lambda'."
                )
            if not (isinstance(self.regularization_gamma, (int, float))
                    and self.regularization_gamma > 0):
                bad.append(
                    f"Invalid JointQ parameter 'regularization_gamma': "
                    f"{self.regularization_gamma!r} (expected float > 0)."
                )
        if self.lambda_mode not in ("fixed_lambda", "incremental_lambda"):
            bad.append(
                f"Invalid JointQ parameter 'lambda_mode': {self.lambda_mode!r} "
                f"(expected 'fixed_lambda' or 'incremental_lambda')."
            )

        if self.lambda_mode == "incremental_lambda":
            if not isinstance(self.lambda_list, list) or len(self.lambda_list) < 1:
                bad.append(
                    f"Invalid JointQ parameter 'lambda_list': {self.lambda_list!r} "
                    f"(expected non-empty list when lambda_mode='incremental_lambda')."
                )
            elif not all(isinstance(v, (int, float)) and v >= 0 for v in self.lambda_list):
                bad.append(
                    "Invalid JointQ parameter 'lambda_list': "
                    "all elements must be non-negative numbers."
                )
            if not (isinstance(self.incremental_eps_y, (int, float)) and self.incremental_eps_y >= 0):
                bad.append(
                    f"Invalid JointQ parameter 'incremental_eps_y': {self.incremental_eps_y!r} "
                    f"(expected float >= 0)."
                )
            if not (isinstance(self.incremental_eps_w, (int, float)) and self.incremental_eps_w >= 0):
                bad.append(
                    f"Invalid JointQ parameter 'incremental_eps_w': {self.incremental_eps_w!r} "
                    f"(expected float >= 0)."
                )
            if self.incremental_initial_skip_ew_threshold is not None and not (
                isinstance(self.incremental_initial_skip_ew_threshold, (int, float))
                and self.incremental_initial_skip_ew_threshold >= 0
            ):
                bad.append(
                    "Invalid JointQ parameter 'incremental_initial_skip_ew_threshold': "
                    f"{self.incremental_initial_skip_ew_threshold!r} "
                    "(expected float >= 0 or None)."
                )

        if self.gptq.wbits != self.bits:
            bad.append(
                f"GPTQ.wbits (= {self.gptq.wbits}) must match JointQ.bits (= {self.bits})."
            )
        expected_groupsize = self.group_size if self.group_size is not None else -1
        if self.gptq.groupsize != expected_groupsize:
            bad.append(
                f"GPTQ.groupsize (= {self.gptq.groupsize}) must match "
                f"JointQ.group_size (= {self.group_size}); "
                f"expected GPTQ.groupsize = {expected_groupsize}."
            )
        if self.gptq.sym != self.symmetric:
            bad.append(
                f"GPTQ.sym (= {self.gptq.sym}) must match "
                f"JointQ.symmetric (= {self.symmetric})."
            )
        if self.gptq.actorder:
            bad.append("GPTQ.actorder must be False (JointQ handles actorder separately).")

        if bad:
            raise ValueError("; ".join(bad))

        self.gptq.validate_params()

    def quantize_layer(
        self, module, input=None, hessian=None, matrix_XX=None, dim_n=None
    ):  # pylint: disable=redefined-builtin, too-many-arguments, too-many-positional-arguments
        """Quantize a single layer.

        Processing flow:
            1. Extract weight matrix from module
            2. Prepare matrix_XX (= X^T X) from input or use precomputed
            3. Apply activation ordering (actorder) if enabled
            4. Generate GPTQ initial solution (if enable_gptq=True),
               using the pre-regularization hessian
            5. Convert GPTQ result to JointQ Solution format
            6. Prepare ILS parameters
            7. Apply Tikhonov regularization to matrix_XX
            8. Run JointQ quantization with initial solutions
            9. Return quantization result

        When ``lambda_mode="incremental_lambda"``, steps 7-8 are replaced by
        an iterative loop that tries each value in ``lambda_list`` and keeps
        the solution as long as it improves weight error without substantially
        degrading output error.

        Args:
            module (torch.nn.Module): The layer module to quantize.
            input (tuple or torch.Tensor, optional): Input activations.
                Used to compute matrix_XX when matrix_XX is not provided.
            hessian (torch.Tensor, optional): Not used in JointQ (ignored).
            matrix_XX (torch.Tensor, optional): Precomputed X^T X (FP64).
                If provided, this is used instead of input.
            dim_n (int, optional): Number of samples. Required when
                matrix_XX is provided.

        Returns:
            JointQResult: Quantization result containing scale, zero_point,
                assignment, and perm (column permutation when actorder is used).
        """
        ctx = self._prepare_quantization_context(module, input, matrix_XX, dim_n)

        if self.lambda_mode == "incremental_lambda":
            return self._quantize_layer_incremental(ctx)
        return self._quantize_layer_fixed(ctx)

    # ------------------------------------------------------------------
    # Preparation context
    # ------------------------------------------------------------------

    def _prepare_quantization_context(self, module, input, matrix_XX, dim_n):
        """Prepare common data shared by both fixed and incremental modes.

        Returns a dict with: matrix_W, matrix_XX (pre-regularization), dim_n,
        device, perm, initial_solutions, ils_kwargs, module.
        """

        # 1. Get the weight matrix  W: (out_features, in_features)
        matrix_W = module.weight.data.clone().cpu().to(torch.float64)

        # 2. Prepare matrix_XX
        device = self.device
        if device is None:
            device = module.weight.device

        if matrix_XX is not None:
            matrix_XX = matrix_XX.clone()
        else:
            if isinstance(input, tuple):
                matrix_X = input[0].detach().cpu().to(torch.float64)
            else:
                matrix_X = input.detach().cpu().to(torch.float64)
            if matrix_X.ndim == 3:
                matrix_X = matrix_X.reshape(-1, matrix_X.shape[-1])
            elif matrix_X.ndim != 2:
                raise ValueError(f"Unsupported matrix_X shape: {matrix_X.shape}")

            self.logger.debug(
                "matrix_W shape: %s, matrix_X shape: %s",
                str(matrix_W.shape),
                str(matrix_X.shape),
            )

            dim_n = matrix_X.shape[0]
            matrix_XX = compute_matrix_XX(matrix_X, device)
            del matrix_X

        # 3. Activation ordering: sort columns by X^T X diagonal (descending)
        perm = None
        if self.actorder:
            perm = torch.argsort(torch.diag(matrix_XX), descending=True)
            matrix_W = matrix_W[:, perm.to(matrix_W.device)]
            matrix_XX = matrix_XX[perm][:, perm]

        # 4-5. OneComp GPTQ: compute initial solution (before regularization)
        initial_solutions = {}
        if self.enable_gptq:
            gptq_layer = torch.nn.Linear(
                module.in_features, module.out_features, bias=False,
                device=device,
            )
            gptq_layer.weight.data = matrix_W.to(dtype=module.weight.dtype, device=device)

            gptq_hessian = ((2.0 / dim_n) * matrix_XX.to(device)).float()
            gptq_result = self.gptq.quantize_layer(gptq_layer, None, hessian=gptq_hessian)
            del gptq_hessian

            m = matrix_XX.shape[0]
            gs = self.group_size if self.group_size is not None else m
            num_groups = m // gs
            p = matrix_W.shape[0]

            q_int = gptq_result.qweight.reshape(p, num_groups, gs)
            if self.group_size is not None:
                scale = gptq_result.scales.T.to(torch.float64)
                zero_point = gptq_result.qzeros.T
            else:
                scale = gptq_result.scales.to(torch.float64)
                zero_point = gptq_result.qzeros

            if self.symmetric:
                midpoint = 2 ** (self.bits - 1)
                zero_point = zero_point - midpoint
                q_int = q_int - midpoint

            initial_solutions["GPTQ"] = Solution(
                scales=scale.to(device),
                assignment=q_int.to(torch.int8).to(device),
                zero_point=zero_point.to(torch.int8).to(device),
            )
            del gptq_result

        # 6. Prepare ILS parameters
        ils_kwargs = {}
        if self.ils_enabled:
            ils_kwargs = {
                "ils_num_iterations": self.ils_num_iterations,
                "ils_num_clones": self.ils_num_clones,
                "ils_num_channels": (
                    min(self.ils_num_channels, int(matrix_W.shape[0]))
                    if self.ils_num_channels is not None
                    else None
                ),
            }

        return {
            "matrix_W": matrix_W,
            "matrix_XX": matrix_XX,
            "dim_n": dim_n,
            "device": device,
            "perm": perm,
            "initial_solutions": initial_solutions,
            "ils_kwargs": ils_kwargs,
            "module": module,
        }

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def execute_post_processing(self):
        """Log accepted_lambda statistics after all layers are quantized."""
        super().execute_post_processing()

        if self.lambda_mode != "incremental_lambda":
            return

        lambdas = {
            name: r.accepted_lambda
            for name, r in self.results.items()
            if hasattr(r, "accepted_lambda") and r.accepted_lambda is not None
        }
        if not lambdas:
            return

        vals = list(lambdas.values())
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        mean_val = sum(vals_sorted) / n
        median_val = (
            vals_sorted[n // 2]
            if n % 2 == 1
            else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2
        )

        from collections import Counter
        counts = Counter(vals)

        self.logger.info(
            "[incremental_lambda] summary: %d layers, "
            "mean=%.4f, median=%.4f, min=%.4f, max=%.4f",
            n, mean_val, median_val, vals_sorted[0], vals_sorted[-1],
        )
        for lam in sorted(counts):
            self.logger.info(
                "[incremental_lambda]   lambda=%.4f: %d layers (%.1f%%)",
                lam, counts[lam], counts[lam] / n * 100,
            )

    # ------------------------------------------------------------------
    # Fixed lambda mode (existing behaviour, unchanged)
    # ------------------------------------------------------------------

    def _quantize_layer_fixed(self, ctx):
        """Quantize with a single fixed regularization_lambda (original path)."""
        matrix_W = ctx["matrix_W"]
        matrix_XX = ctx["matrix_XX"]
        dim_n = ctx["dim_n"]
        device = ctx["device"]
        perm = ctx["perm"]
        initial_solutions = ctx["initial_solutions"]
        ils_kwargs = ctx["ils_kwargs"]

        # 7. Tikhonov regularization
        if self.regularization_lambda is not None and self.regularization_lambda > 0.0:
            matrix_XX = self._build_regularization_matrix(
                matrix_XX, dim_n, self.regularization_lambda
            )

        # 8. Perform quantization
        solution = quantize(
            matrix_W=matrix_W,
            matrix_XX=matrix_XX,
            dim_n=dim_n,
            bits=self.bits,
            symmetric=self.symmetric,
            group_size=self.group_size,
            device=device,
            log_level=self.log_level,
            enable_clip_optimize=self.enable_clip_optimize,
            enable_clip_optimize_ep=self.enable_clip_optimize_ep,
            initial_solutions=initial_solutions or None,
            **ils_kwargs,
        )

        # 9. Get quantized result (scale, assignment, zero_point)
        scale, assignment, zero_point = solution.get_quantized_result()

        return JointQResult(
            bits=self.bits,
            symmetric=self.symmetric,
            group_size=self.group_size,
            scale=scale.cpu(),
            zero_point=zero_point.cpu(),
            assignment=assignment.cpu(),
            perm=perm.cpu() if perm is not None else None,
        )

    # ------------------------------------------------------------------
    # Incremental lambda mode
    # ------------------------------------------------------------------

    def _quantize_layer_incremental(self, ctx):
        """Quantize with incrementally increasing lambda values.

        For each lambda in ``lambda_list``:
        1. Build regularized matrix_XX with that lambda.
        2. Quantize (warm-starting from the previous accepted solution).
        3. Compute relative weight error (Ew) and relative output error (Ey).
        4. Accept the candidate if it passes ``_accept_candidate``;
           otherwise stop and return the last accepted solution.
        """
        matrix_W = ctx["matrix_W"]
        matrix_XX_raw = ctx["matrix_XX"]
        dim_n = ctx["dim_n"]
        device = ctx["device"]
        perm = ctx["perm"]
        gptq_initial_solutions = ctx["initial_solutions"]
        ils_kwargs = ctx["ils_kwargs"]

        accepted_solution = None
        accepted_ew = None
        accepted_ey = None
        accepted_lambda = None

        for step_idx, lam in enumerate(self.lambda_list):
            # Build regularized matrix_XX for this lambda
            reg_matrix_XX = self._build_regularization_matrix(
                matrix_XX_raw, dim_n, lam
            )

            # Until the first candidate is accepted, keep using the base
            # initializers (GPTQ / Clip-Optimize / etc.) rather than warm start.
            if accepted_solution is None:
                init_sols = gptq_initial_solutions
            else:
                # Warm start from the accepted solution
                prev_scale, prev_assignment, prev_zero_point = (
                    accepted_solution.get_quantized_result()
                )
                warm_solution = Solution(
                    scales=prev_scale.to(device),
                    assignment=prev_assignment.to(torch.int8).to(device),
                    zero_point=prev_zero_point.to(torch.int8).to(device),
                )
                init_sols = {"warm_start": warm_solution}

            solution = quantize(
                matrix_W=matrix_W,
                matrix_XX=reg_matrix_XX,
                dim_n=dim_n,
                bits=self.bits,
                symmetric=self.symmetric,
                group_size=self.group_size,
                device=device,
                log_level=self.log_level,
                enable_clip_optimize=self.enable_clip_optimize,
                enable_clip_optimize_ep=self.enable_clip_optimize_ep,
                initial_solutions=init_sols or None,
                **ils_kwargs,
            )

            ew, ey = self._compute_relative_errors(solution, matrix_W, matrix_XX_raw, device)

            if (
                step_idx == 0
                and lam == 0.0
                and self.incremental_initial_skip_ew_threshold is not None
                and ew > self.incremental_initial_skip_ew_threshold
                and step_idx + 1 < len(self.lambda_list)
            ):
                if self.log_level >= 1:
                    self.logger.info(
                        "[incremental_lambda] step %d: lambda=%.4f  "
                        "Ew=%.4f%%  Ey=%.4f%%  -> skip initial candidate "
                        "(Ew exceeds threshold %.4f%%)",
                        step_idx,
                        lam,
                        ew * 100,
                        ey * 100,
                        self.incremental_initial_skip_ew_threshold * 100,
                    )
                continue

            if accepted_solution is None:
                accepted_solution = solution
                accepted_ew = ew
                accepted_ey = ey
                accepted_lambda = lam
                if self.log_level >= 1:
                    self.logger.info(
                        "[incremental_lambda] step %d: lambda=%.4f  "
                        "Ew=%.4f%%  Ey=%.4f%%  -> initial accept",
                        step_idx, lam, ew * 100, ey * 100,
                    )
                continue

            accepted, reason = self._accept_candidate(
                accepted_ew, accepted_ey, ew, ey,
                eps_y=self.incremental_eps_y,
                eps_w=self.incremental_eps_w,
            )
            if self.log_level >= 1:
                ew_delta = (ew - accepted_ew) / accepted_ew * 100 if accepted_ew > 0 else 0.0
                ey_delta = (ey - accepted_ey) / accepted_ey * 100 if accepted_ey > 0 else 0.0
                self.logger.info(
                    "[incremental_lambda] step %d: lambda=%.4f  "
                    "Ew=%.4f%% (%+.2f%%)  Ey=%.4f%% (%+.2f%%)  "
                    "-> %s (%s)",
                    step_idx, lam,
                    ew * 100, ew_delta,
                    ey * 100, ey_delta,
                    "accept" if accepted else "reject",
                    reason,
                )

            if accepted:
                accepted_solution = solution
                accepted_ew = ew
                accepted_ey = ey
                accepted_lambda = lam
            else:
                break

        if self.log_level >= 1:
            self.logger.info(
                "[incremental_lambda] final: lambda=%.4f  Ew=%.4f%%  Ey=%.4f%%",
                accepted_lambda, accepted_ew * 100, accepted_ey * 100,
            )

        scale, assignment, zero_point = accepted_solution.get_quantized_result()

        return JointQResult(
            bits=self.bits,
            symmetric=self.symmetric,
            group_size=self.group_size,
            scale=scale.cpu(),
            zero_point=zero_point.cpu(),
            assignment=assignment.cpu(),
            perm=perm.cpu() if perm is not None else None,
            accepted_lambda=accepted_lambda,
        )

    # ------------------------------------------------------------------
    # Helper: build regularization matrix
    # ------------------------------------------------------------------

    def _build_regularization_matrix(self, matrix_XX_raw, dim_n, lam):
        """Build the regularized matrix XX for a given lambda.

        ``regularization_mode`` controls the shape of the added term:

        * ``"identity"``:  X^T X + n*lambda*I   (current default)
        * ``"diagonal"``:  X^T X + n*lambda*diag(a)
          where a_i = ( diag(X^T X)_i / mean(diag(X^T X)) ) ^ gamma.

        Returns the regularized matrix (does not modify the input).
        """
        if lam == 0.0:
            return matrix_XX_raw

        m = matrix_XX_raw.shape[0]
        dtype = matrix_XX_raw.dtype
        device = matrix_XX_raw.device

        if self.regularization_mode == "identity":
            return matrix_XX_raw + (dim_n * lam) * torch.eye(m, dtype=dtype, device=device)

        # "diagonal" mode
        diag_xx = torch.diag(matrix_XX_raw)
        mean_diag = diag_xx.mean()
        if mean_diag > 0:
            a = (diag_xx / mean_diag) ** self.regularization_gamma
        else:
            a = torch.ones(m, dtype=dtype, device=device)

        return matrix_XX_raw + (dim_n * lam) * torch.diag(a)

    # ------------------------------------------------------------------
    # Helper: relative error computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_relative_errors(solution, matrix_W, matrix_XX_raw, device):
        """Compute relative weight error (Ew) and relative output error (Ey).

        Ew = ||W_q - W||^2_F / ||W||^2_F
        Ey = ||(W_q - W) X^T||^2_F / ||W X^T||^2_F
           = trace(dW @ XX @ dW^T) / trace(W @ XX @ W^T)

        All computations use the *pre-regularization* matrix_XX so that
        Ey reflects actual output error rather than regularized loss.

        Args:
            solution: Core Solution object (internal layout).
            matrix_W: Original weight, (p, m), FP64, CPU.
            matrix_XX_raw: Pre-regularization X^T X, (m, m), FP64.
            device: Computation device.

        Returns:
            (ew, ey): Relative weight error and relative output error.
        """
        W_hat = solution.get_dequantized_weight_matrix()  # (p, m) on device
        W_dev = matrix_W.to(device)
        dW = W_hat - W_dev  # (p, m)

        # Ew = ||dW||^2_F / ||W||^2_F
        w_norm_sq = torch.sum(W_dev ** 2).item()
        dw_norm_sq = torch.sum(dW ** 2).item()
        ew = dw_norm_sq / w_norm_sq if w_norm_sq > 0 else 0.0

        # Ey = trace(dW @ XX @ dW^T) / trace(W @ XX @ W^T)
        XX = matrix_XX_raw.to(device)
        # trace(A @ B @ A^T) = sum of element-wise (A @ B) * A
        dW_XX = dW @ XX  # (p, m)
        ey_num = torch.sum(dW_XX * dW).item()
        W_XX = W_dev @ XX  # (p, m)
        ey_den = torch.sum(W_XX * W_dev).item()
        ey = ey_num / ey_den if ey_den > 0 else 0.0

        return ew, ey

    # ------------------------------------------------------------------
    # Helper: acceptance criterion
    # ------------------------------------------------------------------

    @staticmethod
    def _accept_candidate(ew_prev, ey_prev, ew_new, ey_new,
                          eps_y=None, eps_w=None):
        """Decide whether to accept a candidate solution.

        Rules (evaluated in order):
        1. Both Ey and Ew decreased -> accept.
        2. Ew increased -> reject.
        3. Ey worsened within eps_y *and* Ew improved by at least eps_w -> accept.
        4. Otherwise -> reject.

        Returns:
            (accepted, reason): bool and a short explanation string.
        """
        if ey_new < ey_prev and ew_new < ew_prev:
            return True, "Ew and Ey both decreased"
        if ew_new > ew_prev:
            return False, "Ew increased"
        if eps_y is not None and eps_w is not None:
            ey_ok = ey_new <= ey_prev * (1.0 + eps_y)
            ew_ok = ew_new <= ew_prev * (1.0 - eps_w)
            if ey_ok and ew_ok:
                return True, "Ey within tolerance, Ew improved enough"
            if not ey_ok:
                return False, "Ey degraded beyond eps_y"
            return False, "Ew improvement below eps_w"
        return False, "Ew did not decrease"
