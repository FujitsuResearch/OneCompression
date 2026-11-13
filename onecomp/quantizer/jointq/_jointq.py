"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import re
from dataclasses import dataclass
from typing import Any, Optional

import torch

from onecomp.quantizer._quantizer import QuantizationResult, Quantizer
from onecomp.quantizer.gptq._gptq import GPTQ
from onecomp.utils.quant_config import get_quant_param

from .core import compute_matrix_XX, quantize
from .core.solution import Solution


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
        objective_value: JointQ's internal objective value when
            calc_quant_error=True. Without regularization this is
            ||WX^T - ŴX^T||²_F; with regularization it is computed using
            the regularized matrix_XX and is not the unregularized
            reconstruction error.

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
    objective_value: Optional[float] = None  # JointQ objective value

    def compute_dequantized_weight(self, device: torch.device = None) -> torch.Tensor:
        """Compute the dequantized weight from quantization parameters

        Reconstruct the weight using the following formula:
            W_hat[i, g*group_size:(g+1)*group_size]
                = scale[i, g] * (assignment[i, g, :] - zero_point[i, g])

        Args:
            device (torch.device): Device for computation.
                If None, computation is performed on the device where the quantization
                parameters reside.

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
        stats["objective_value"] = self.objective_value
        return stats


@dataclass
class JointQ(Quantizer):
    """JointQ quantizer class

    JointQ is a post-training quantization method that combines multiple
    initialization strategies (Clip-Optimize, Clip-Optimize-EP, GPTQ) with
    local search optimization to find high-quality quantized weights.

    Attributes:
        bits (int): Number of bits for quantization (1--4). Default is 4.
        symmetric (bool): Whether to use symmetric quantization. Default is False.
        group_size (int or None): Group size for quantization. Default is 128.
            Must divide the layer's input dimension. If None, per-channel
            quantization is used (group_size = in_features).
        log_level (int): Log level (0: none, 1: minimal, 2: detailed). Default is 0.
        device (torch.device or None): Device for quantization.
            If None, uses the device of the module being quantized.
        regularization_lambda (float or None): Tikhonov regularization strength.
            Default is None (disabled). When set to a positive value, replaces
            X^T X with X^T X + n*lambda*R where R depends on
            ``regularization_mode``.
            lambda is relative to the normalized Hessian (1/n)X^T X, so its
            meaning is consistent across different calibration sample sizes.
            Recommended range: 0.1 to 1.0. Set to None or 0.0 to disable.
        regularization_lambda_by_layer (dict[str, float] or None): Optional
            per-layer override for ``regularization_lambda``. Keys may be full
            module names or suffixes such as ``"self_attn.q_proj"`` and
            ``"mlp.down_proj"``. Full-name matches take priority; otherwise
            the longest matching suffix is used. Default is None (disabled).
        regularization_mode (str): Shape of the regularization matrix R.
            ``"identity"``: R = I (standard Tikhonov).
            ``"diagonal"`` (default): R = diag(a) where
            a_i = (diag(X^T X)_i / mean(diag(X^T X))) ^ gamma.
            This makes regularization importance-aware: columns with
            larger activations receive stronger regularization.
        regularization_gamma (float): Exponent for the diagonal weights
            in ``"diagonal"`` mode. Default is 0.5. Smaller values
            reduce the spread between weak and strong columns.
        regularization_retry_lambda (float or None): Fallback regularization
            strength used only when ``regularization_lambda`` is disabled
            and the unregularized quantized weight has large relative
            weight error. Default is 0.01. Set to None to disable retry.
        regularization_retry_threshold (float or None): Relative weight
            error threshold for triggering the fallback retry:
            ``||W - W_hat||_F^2 / ||W||_F^2``. Default is 0.2.
        actorder (bool): Whether to reorder columns by activation magnitude
            (Hessian diagonal) before quantization. Default is False.
            When enabled, columns with larger activations are grouped together,
            improving group quantization efficiency and GPTQ initial solution quality.
        enable_group_swap (bool): Whether to refine the column-to-group
            assignment via lightweight swap-based local search. Default
            is False. The swap is applied as a **post-LS** refinement:
            JointQ runs its full local-search phase once, then swap
            reassigns columns between groups on the LS-completed
            solution (where scales are already well-optimized), then a
            second LS pass propagates the swap's gain through the
            scale + integer co-optimization. Automatically skipped
            when ``num_groups < 2`` (no swap possible). Accepted
            swaps are applied as a column permutation to ``matrix_W`` /
            ``matrix_XX``, so the rest of the pipeline continues to
            operate on contiguous groups; the permutation is composed
            with ``actorder``'s permutation and stored in
            ``JointQResult.perm`` for inverse-mapping at dequantization
            time.
        swap_rho (float): Fraction (in ``(0, 1]``) of columns to
            consider per swap round, ranked by the continuous-relaxation
            margin ``phi[j] = ||U[:, j]||^2 / K[j, j]``. Larger values
            evaluate more candidates (higher quality but slower).
            Empirically ``0.3`` is the best quality/cost trade-off on
            TinyLlama 4bit/g128; set to ``0.1`` to trade quality for
            speed. Only used when ``enable_group_swap=True``.
            Default is 0.3.
        swap_min_improvement (float): Per-swap relative-improvement
            threshold. A swap is accepted only when
            ``-Delta F / F_curr > swap_min_improvement``. Skipping
            noise-level swaps avoids the per-swap commit cost (two
            ``outer`` updates of ``U``, ~ ``p * m`` ops each), which
            matters most on wide layers where each round accepts
            hundreds of swaps. Also acts as the FP64 safety filter
            against FP32 false-positives in the swap-evaluation phase.
            Set to ``0.0`` to accept any strictly improving swap.
            Only used when ``enable_group_swap=True``. Default is
            ``1e-5`` (= reject swaps whose relative improvement is
            below 0.001%).
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
        gptq_use_regularized_hessian (bool): When True (default), the
            GPTQ initial solution is computed with the *regularized*
            Hessian ``(2/n) * (matrix_XX + n*lambda*R)`` instead of
            the pre-regularization ``(2/n) * matrix_XX``. This keeps
            GPTQ's objective consistent with the downstream LS pass
            (which optimizes the regularized objective); empirically
            this is the dominant configuration. Set to False to
            restore the canonical GPTQ behavior on the raw Hessian.
            Has no effect when ``enable_gptq=False`` or the active
            regularization lambda for the current pass is None or 0.0.
        gptq (GPTQ or None): GPTQ instance for initial solution generation.
            If None, a default GPTQ is created from bits/group_size/symmetric.
            Pass a custom GPTQ instance to control parameters like blocksize,
            percdamp, mse, q_grid, q_norm. The GPTQ instance must have
            wbits/groupsize/sym matching JointQ's bits/group_size/symmetric,
            and actorder must be False. JointQ forces its internal GPTQ
            instance to return unpacked qweight/qzeros.

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

        With per-layer regularization strengths::

            quantizer = JointQ(
                bits=4,
                symmetric=False,
                group_size=128,
                regularization_lambda_by_layer={
                    "self_attn.q_proj": 0.0,
                    "self_attn.k_proj": 0.0,
                    "self_attn.v_proj": 0.0,
                    "self_attn.o_proj": 0.0,
                    "mlp.gate_proj": 0.05,
                    "mlp.up_proj": 0.05,
                    "mlp.down_proj": 0.01,
                },
            )

    """

    flag_calibration: bool = True
    flag_hessian: bool = False
    flag_xtx: bool = True
    # JointQ does not yet support the generic QEP pipeline.
    # Planned for a future release.
    flag_qep_supported: bool = False
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
    regularization_lambda: Optional[float] = None
    regularization_lambda_by_layer: Optional[dict[str, float]] = None
    regularization_mode: str = "diagonal"
    regularization_gamma: float = 0.5
    regularization_retry_lambda: Optional[float] = 0.01
    regularization_retry_threshold: Optional[float] = 0.2

    # Activation ordering
    actorder: bool = False

    # Group-assignment swap optimization (runs after GPTQ initial solution).
    # Automatically skipped when ``enable_gptq=False`` or ``num_groups < 2``.
    enable_group_swap: bool = False
    # Fraction of columns kept as swap candidates per round (top by
    # phi). Only used when enable_group_swap=True. 0.3 is the empirical
    # quality/cost sweet spot on TinyLlama; set to 0.1 for speed.
    swap_rho: float = 0.3
    # Per-swap relative-improvement threshold. Swaps whose relative F
    # improvement is below this value are rejected, avoiding the
    # per-swap commit cost. 1e-5 = "reject < 0.001% improvements".
    # Only used when enable_group_swap=True. Set to 0 for "accept any
    # strictly improving swap" (historical behaviour).
    swap_min_improvement: float = 1e-5

    # Iterated Local Search (ILS) parameters
    ils_enabled: bool = False
    ils_num_iterations: int = 10
    ils_num_clones: int = 8
    ils_num_channels: Optional[int] = None

    # Initialization strategies
    enable_clip_optimize: bool = True
    enable_clip_optimize_ep: bool = False
    enable_gptq: bool = True

    # When True (default), feed GPTQ the regularized Hessian
    # (matrix_XX + n*lambda*R) instead of the raw matrix_XX. Keeps GPTQ
    # aligned with the LS objective. Set to False to restore canonical
    # GPTQ behavior (raw Hessian).
    gptq_use_regularized_hessian: bool = True

    # GPTQ initial solution
    gptq: Optional[GPTQ] = None

    def __post_init__(self):
        super().__post_init__()
        if self.gptq is None:
            gptq_groupsize = self.group_size if self.group_size is not None else -1
            self.gptq = GPTQ(
                wbits=self.bits,
                groupsize=gptq_groupsize,
                sym=self.symmetric,
                bitpack_on_quantize=False,
            )
        else:
            self.logger.info(
                "Overriding GPTQ bitpack_on_quantize=False for JointQ initial solution."
            )
            self.gptq.bitpack_on_quantize = False

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

        Warnings:
            bits == 1: valid, but GPTQLinear weight packing does not support
                1-bit; the inference layer must be built with pack_weights=False.

        """
        bad = []

        if not (isinstance(self.bits, int) and self.bits >= 1):
            bad.append(f"Invalid JointQ parameter 'bits': {self.bits!r} (expected int >= 1).")
        elif self.bits == 1:
            self.logger.warning(
                "JointQ with bits=1 is not supported by GPTQLinear weight packing; "
                "build the inference layer with pack_weights=False."
            )

        if self.group_size is not None and not (
            isinstance(self.group_size, int) and self.group_size >= 1
        ):
            bad.append(
                "Invalid JointQ parameter 'group_size': "
                f"{self.group_size!r} (expected int >= 1 or None)."
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
            if not (
                isinstance(self.regularization_gamma, (int, float))
                and self.regularization_gamma > 0
            ):
                bad.append(
                    f"Invalid JointQ parameter 'regularization_gamma': "
                    f"{self.regularization_gamma!r} (expected float > 0)."
                )

        if self.regularization_lambda is not None and not (
            isinstance(self.regularization_lambda, (int, float))
            and self.regularization_lambda >= 0.0
        ):
            bad.append(
                f"Invalid JointQ parameter 'regularization_lambda': "
                f"{self.regularization_lambda!r} (expected non-negative float or None)."
            )

        if self.regularization_lambda_by_layer is not None:
            if not isinstance(self.regularization_lambda_by_layer, dict):
                bad.append(
                    "Invalid JointQ parameter 'regularization_lambda_by_layer': "
                    f"must be a dict[str, float], got "
                    f"{type(self.regularization_lambda_by_layer).__name__!r}."
                )
            else:
                for layer_name, lam in self.regularization_lambda_by_layer.items():
                    if not isinstance(layer_name, str):
                        bad.append(
                            "Invalid JointQ parameter 'regularization_lambda_by_layer': "
                            "keys must be layer name strings."
                        )
                    elif not (isinstance(lam, (int, float)) and lam >= 0.0):
                        bad.append(
                            "Invalid JointQ parameter "
                            f"'regularization_lambda_by_layer[{layer_name!r}]': "
                            f"{lam!r} (expected non-negative float)."
                        )

        if self.regularization_retry_lambda is not None and not (
            isinstance(self.regularization_retry_lambda, (int, float))
            and self.regularization_retry_lambda > 0.0
        ):
            bad.append(
                f"Invalid JointQ parameter 'regularization_retry_lambda': "
                f"{self.regularization_retry_lambda!r} (expected float > 0 or None)."
            )

        if self.regularization_retry_threshold is not None and not (
            isinstance(self.regularization_retry_threshold, (int, float))
            and self.regularization_retry_threshold >= 0.0
        ):
            bad.append(
                f"Invalid JointQ parameter 'regularization_retry_threshold': "
                f"{self.regularization_retry_threshold!r} "
                f"(expected non-negative float or None)."
            )

        if not isinstance(self.enable_group_swap, bool):
            bad.append(
                f"Invalid JointQ parameter 'enable_group_swap': "
                f"{self.enable_group_swap!r} (expected bool)."
            )

        if not (isinstance(self.swap_rho, (int, float)) and 0.0 < self.swap_rho <= 1.0):
            bad.append(
                f"Invalid JointQ parameter 'swap_rho': "
                f"{self.swap_rho!r} (expected float in (0, 1])."
            )

        if not (
            isinstance(self.swap_min_improvement, (int, float))
            and self.swap_min_improvement >= 0.0
        ):
            bad.append(
                f"Invalid JointQ parameter 'swap_min_improvement': "
                f"{self.swap_min_improvement!r} (expected non-negative float)."
            )

        if not isinstance(self.gptq_use_regularized_hessian, bool):
            bad.append(
                f"Invalid JointQ parameter 'gptq_use_regularized_hessian': "
                f"{self.gptq_use_regularized_hessian!r} (expected bool)."
            )

        if self.gptq.wbits != self.bits:
            bad.append(f"GPTQ.wbits (= {self.gptq.wbits}) must match JointQ.bits (= {self.bits}).")
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
            4. Apply Tikhonov regularization to matrix_XX once
            5. Generate GPTQ initial solution (if enable_gptq=True),
               using either the regularized hessian (default) or the
               pre-regularization hessian (when
               ``gptq_use_regularized_hessian=False``)
            6. Convert GPTQ result to JointQ Solution format
            7. Prepare ILS parameters
            8. Run JointQ quantization with initial solutions (first LS pass)
            8.5. (optional) When ``enable_group_swap=True``: refine the
                column-to-group assignment via lightweight swap-based
                local search on the LS-completed solution, then run a
                second LS pass warm-started from the swap-modified
                state. The swap permutation is applied to
                matrix_W / matrix_XX so the pipeline continues to
                operate on contiguous groups (similar to actorder).
            9. Return quantization result

        If ``regularization_lambda`` is disabled and the unregularized
        quantized weight has relative weight error above
        ``regularization_retry_threshold``, the layer is quantized again
        with ``regularization_retry_lambda``.

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
        layer_name = self.module_to_name.get(module)
        regularization_lambda, _ = self._resolve_regularization_lambda(layer_name)
        active_regularization_lambda = regularization_lambda
        ctx = self._prepare_quantization_context(
            module, input, matrix_XX, dim_n, regularization_lambda
        )
        result = self._quantize_layer_fixed(ctx)

        if self._should_retry_with_regularization(regularization_lambda):
            relative_weight_error = self._compute_relative_weight_error(module, result)
            if (
                relative_weight_error is not None
                and relative_weight_error >= self.regularization_retry_threshold
            ):
                self.logger.info(
                    "[RegularizationRetry] relative weight error %.6f >= %.6f; "
                    "retrying with lambda=%s",
                    relative_weight_error,
                    self.regularization_retry_threshold,
                    self.regularization_retry_lambda,
                )
                active_regularization_lambda = self.regularization_retry_lambda
                retry_ctx = self._prepare_quantization_context(
                    module,
                    input,
                    matrix_XX,
                    dim_n,
                    self.regularization_retry_lambda,
                )
                result = self._quantize_layer_fixed(retry_ctx)

        self.logger.info(
            "[Regularization] layer=%s adopted lambda=%s",
            layer_name if layer_name is not None else "<unknown>",
            active_regularization_lambda,
        )

        return result

    @staticmethod
    def _find_layer_override(
        layer_name: Optional[str],
        overrides: Optional[dict[str, float]],
    ) -> tuple[bool, Optional[float]]:
        """Find a per-layer override by full name or longest suffix match."""
        if not overrides or layer_name is None:
            return False, None

        if layer_name in overrides:
            return True, overrides[layer_name]

        matched_key = None
        for key in overrides:
            if layer_name.endswith(f".{key}") or layer_name == key:
                if matched_key is None or len(key) > len(matched_key):
                    matched_key = key

        if matched_key is None:
            return False, None

        return True, overrides[matched_key]

    def _resolve_regularization_lambda(
        self, layer_name: Optional[str]
    ) -> tuple[Optional[float], bool]:
        """Resolve regularization lambda for one layer.

        Returns ``(lambda, has_layer_override)``. A per-layer ``0.0`` disables
        regularization for the first pass, while keeping the fallback retry
        path available when the weight-error threshold is exceeded.
        """
        found, lam = JointQ._find_layer_override(layer_name, self.regularization_lambda_by_layer)
        if found:
            return lam, True
        return self.regularization_lambda, False

    # ------------------------------------------------------------------
    # Preparation context
    # ------------------------------------------------------------------

    def _prepare_quantization_context(
        self, module, input, matrix_XX, dim_n, regularization_lambda
    ):
        """Prepare common data shared by the quantization pipeline.

        Returns a dict with: matrix_W, matrix_XX (regularized for LS),
        dim_n, device, perm, initial_solutions, ils_kwargs, module.
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

        # 4. Tikhonov regularization. Keep the raw matrix available because
        # GPTQ can optionally use the canonical pre-regularization Hessian.
        matrix_XX_raw = matrix_XX
        if regularization_lambda is not None and regularization_lambda > 0.0:
            matrix_XX = self._build_regularization_matrix(
                matrix_XX_raw, dim_n, regularization_lambda
            )

        # 5-6. OneComp GPTQ: compute initial solution.
        # When ``gptq_use_regularized_hessian=True`` (default), feed GPTQ the
        # same regularized Hessian as the downstream LS. Set it to False to
        # use the canonical pre-regularization GPTQ Hessian.
        initial_solutions = {}
        if self.enable_gptq:
            gptq_layer = torch.nn.Linear(
                module.in_features,
                module.out_features,
                bias=False,
                device=device,
            )
            gptq_layer.weight.data = matrix_W.to(dtype=module.weight.dtype, device=device)

            gptq_matrix_XX = matrix_XX if self.gptq_use_regularized_hessian else matrix_XX_raw
            gptq_hessian = ((2.0 / dim_n) * gptq_matrix_XX.to(device)).float()
            gptq_result = self.gptq.quantize_layer(gptq_layer, None, hessian=gptq_hessian)
            del gptq_hessian, gptq_matrix_XX

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

        # 7. Prepare ILS parameters
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
    # Helper: build regularization matrix
    # ------------------------------------------------------------------

    def _build_regularization_matrix(self, matrix_XX_raw, dim_n, lam):
        """Build the regularized matrix XX for a given lambda.

        ``regularization_mode`` controls the shape of the added term:

        * ``"identity"``:  X^T X + n*lambda*I
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

    def _should_retry_with_regularization(self, regularization_lambda) -> bool:
        """Return True when the unregularized result should be checked for retry."""
        return (
            (regularization_lambda is None or regularization_lambda <= 0.0)
            and self.regularization_retry_lambda is not None
            and self.regularization_retry_threshold is not None
        )

    def _compute_relative_weight_error(self, module, result) -> float:
        """Compute the existing Quantizer relative weight error metric."""
        dequantized_weight = result.compute_dequantized_weight()
        _, _, relative_weight_error = self.calculate_weight_quantization_error(
            module, dequantized_weight
        )
        return relative_weight_error

    # ------------------------------------------------------------------
    # Quantization main loop
    # ------------------------------------------------------------------

    def _quantize_layer_fixed(self, ctx):
        """Quantize one layer with the fixed regularization_lambda pipeline."""
        matrix_W = ctx["matrix_W"]
        matrix_XX = ctx["matrix_XX"]
        dim_n = ctx["dim_n"]
        device = ctx["device"]
        perm = ctx["perm"]
        initial_solutions = ctx["initial_solutions"]
        ils_kwargs = ctx["ils_kwargs"]

        # 8. First LS pass
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

        # 8.5. Post-LS group-assignment swap (fixed_lambda mode only).
        # Apply the swap to the LS-completed solution -- where scales
        # are already well-optimized -- then re-run LS so the swap's
        # column reassignment is propagated through scale + integer
        # co-optimization. This sidesteps the failure mode of the
        # original pre-LS swap design, where the strong LS phase
        # could discard the swap's gain.
        #
        # The swap is evaluated on the *regularized* matrix_XX (same
        # objective as both LS passes), so its gain is preserved by
        # the second LS pass. The Tikhonov term (``λ·I`` or
        # ``λ·diag(...)``) is permutation-equivariant, so the
        # permuted regularized matrix returned here is identical to
        # what we would get by re-regularizing the permuted raw XX.
        if self.enable_group_swap:
            matrix_W, matrix_XX, perm, swap_applied = self._apply_group_swap(
                matrix_W=matrix_W,
                matrix_XX=matrix_XX,
                dim_n=dim_n,
                device=device,
                perm=perm,
                solution=solution,
            )

            if swap_applied:
                # Round-trip the swap-modified solution through the external
                # (scales, assignment, zero_point) layout so the second
                # pass sees a freshly-constructed Solution.
                prev_scale, prev_assignment, prev_zero_point = solution.get_quantized_result()
                warm_solution = Solution(
                    scales=prev_scale.to(device),
                    assignment=prev_assignment.to(torch.int8).to(device),
                    zero_point=prev_zero_point.to(torch.int8).to(device),
                )

                # Second LS pass: warm-start from post-swap state, skip
                # Clip-Optimize candidates (they would just re-introduce
                # alternative initial solutions and dilute the swap signal).
                solution = quantize(
                    matrix_W=matrix_W,
                    matrix_XX=matrix_XX,
                    dim_n=dim_n,
                    bits=self.bits,
                    symmetric=self.symmetric,
                    group_size=self.group_size,
                    device=device,
                    log_level=self.log_level,
                    enable_clip_optimize=False,
                    enable_clip_optimize_ep=False,
                    initial_solutions={"post_ls_swap": warm_solution},
                    **ils_kwargs,
                )
            elif self.log_level >= 1:
                # Swap did not accept any column reassignment; the second
                # LS pass would only re-converge to the same solution, so
                # skip it.
                self.logger.info("[GroupSwap] no swap accepted, skipping the second LS pass")

        # 9. Get quantized result (scale, assignment, zero_point)
        scale, assignment, zero_point = solution.get_quantized_result()

        # Return the JointQ result
        objective_value = (
            solution.squared_error.item()
            if self.calc_quant_error and solution.squared_error is not None
            else None
        )
        return JointQResult(
            bits=self.bits,
            symmetric=self.symmetric,
            group_size=self.group_size,
            scale=scale.cpu(),
            zero_point=zero_point.cpu(),
            assignment=assignment.cpu(),
            perm=perm.cpu() if perm is not None else None,
            objective_value=objective_value,
        )

    # ------------------------------------------------------------------
    # Group-assignment swap
    # ------------------------------------------------------------------

    def _apply_group_swap(
        self, matrix_W, matrix_XX, dim_n, device, perm, solution
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
        """Run the lightweight group-swap optimizer and apply its column permutation.

        Mutates ``solution.integers_z`` in place. Returns a 4-tuple
        ``(matrix_W, matrix_XX, perm, swap_applied)``:

        - ``matrix_W``, ``matrix_XX``: possibly permuted in-place
        - ``perm``: composed permutation (including ``swap_perm``)
        - ``swap_applied (bool)``: True iff at least one swap was
          accepted (i.e. the returned matrices/perm actually differ
          from the inputs). When False, the caller can skip
          downstream work that only makes sense when columns were
          rearranged (e.g. the second LS pass).

        Skips silently (returns inputs unchanged with
        ``swap_applied=False``) when ``num_groups < 2``, since no
        swap is possible.

        ``solution`` may be either a freshly-built initial solution
        (GPTQ etc.) or a post-LS solution; the optimizer treats it
        uniformly as a starting state whose ``integers_z`` is to be
        refined under fixed ``scales``.
        """
        from .core.group_swap import optimize_groups_with_swap

        m = matrix_XX.shape[0]
        gs = self.group_size if self.group_size is not None else m
        num_groups = m // gs
        if num_groups < 2:
            return matrix_W, matrix_XX, perm, False

        # matrix_XX and the solution live on ``device``; matrix_W is on CPU.
        matrix_XX_dev = matrix_XX.to(device)
        W_dev = matrix_W.to(device)
        matrix_YX = W_dev @ matrix_XX_dev
        Y_sq_norms = (matrix_YX * W_dev).sum(dim=1)
        del W_dev

        swap_result = optimize_groups_with_swap(
            solution=solution,
            matrix_XX=matrix_XX_dev,
            matrix_YX=matrix_YX,
            Y_sq_norms=Y_sq_norms,
            dim_n=dim_n,
            bits=self.bits,
            symmetric=self.symmetric,
            rho=self.swap_rho,
            min_swap_improvement=self.swap_min_improvement,
            log_level=self.log_level,
        )
        del matrix_YX, Y_sq_norms, matrix_XX_dev

        swap_perm_dev = swap_result["groups"].reshape(-1)
        identity = torch.arange(m, device=swap_perm_dev.device, dtype=swap_perm_dev.dtype)
        if torch.equal(swap_perm_dev, identity):
            return matrix_W, matrix_XX, perm, False

        # Apply the swap-induced permutation so the new grouping is contiguous
        # in the post-swap column layout. ``matrix_W`` is FP64 / CPU while
        # ``matrix_XX`` is on ``device``.
        swap_perm_cpu = swap_perm_dev.cpu()
        matrix_W = matrix_W[:, swap_perm_cpu]
        matrix_XX = matrix_XX[swap_perm_dev.to(matrix_XX.device)][
            :, swap_perm_dev.to(matrix_XX.device)
        ]

        # Compose with any existing perm (e.g., from actorder).
        # ``perm`` semantics: ``perm[i]`` = original column index that ended
        # up at position ``i`` in the current layout. After swap, position
        # ``i`` holds the column that was at position ``swap_perm[i]`` in
        # the pre-swap layout, i.e. ``perm[swap_perm[i]]`` in the original.
        if perm is None:
            perm = swap_perm_cpu
        else:
            perm = perm[swap_perm_cpu.to(perm.device)]

        return matrix_W, matrix_XX, perm, True

    def get_quant_config(self) -> dict:
        """Return GPTQ-compatible quantization config.

        JointQ uses the same scale/zero/assignment structure as GPTQ,
        so we emit quant_method="gptq" to reuse GPTQLinear and vLLM GPTQ plugin.
        """
        return {
            "quant_method": "gptq",
            "bits": self.bits,
            "groupsize": self.group_size if self.group_size is not None else -1,
            "group_size": self.group_size if self.group_size is not None else -1,
            "actorder": self.actorder,
            "desc_act": self.actorder,
            "sym": self.symmetric,
            "checkpoint_format": "gptq",
        }

    @staticmethod
    def _build_quantization_bits(
        quantized_names: list[str],
        quant_config: dict[str, Any],
        num_layers: int,
    ) -> list[dict[str, Any]]:
        _LAYER_RE = re.compile(r"\.layers\.(\d+)\.(.*)")
        default_bits = quant_config.get("bits", 4)
        default_gs = get_quant_param(quant_config, "group_size", "groupsize", default=-1)

        layer_modules: dict[int, dict[str, Any]] = {}
        for name in quantized_names:
            m = _LAYER_RE.search(name)
            if m is None:
                continue
            layer_idx = int(m.group(1))
            suffix = m.group(2)

            layer_modules.setdefault(layer_idx, {})[suffix] = {
                "bits": default_bits,
                "method": "gptq",
                "params": {"group_size": default_gs},
            }
        if not layer_modules:
            return []

        return [layer_modules.get(i, {}) for i in range(num_layers)]

    def finalize_quant_config_for_save(
        self,
        quant_config: dict[str, Any],
        quantized_layer_names: list[str],
        num_hidden_layers: Optional[int] = None,
    ) -> dict[str, Any]:
        if num_hidden_layers is None:
            raise ValueError("num_hidden_layers is required")
        quant_config["quantization_bits"] = JointQ._build_quantization_bits(
            quantized_layer_names, quant_config, num_hidden_layers
        )
        return quant_config

    def create_inference_layer(self, result, linear_module, **kwargs):
        """Build GPTQLinear from JointQResult.

        Converts JointQ's 3D assignment (out_features, num_groups, group_size)
        to 2D qweight (out_features, in_features), matching GPTQ format.
        JointQ scale/zero_point shape is (out_features, num_groups);
        GPTQLinear expects (num_groups, out_features), so we transpose.
        """
        from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

        qweight = result.assignment.reshape(result.assignment.shape[0], -1)

        # When `actorder` is enabled, `assignment` is stored in the permuted column order.
        # Restore the original column order before passing to GPTQLinear.
        # GPTQLinear constructs `g_idx` assuming `qweight` uses the original column ordering.
        if result.perm is not None:
            invperm = torch.argsort(result.perm)
            qweight = qweight[:, invperm]

        pack_weights = kwargs.get("pack_weights", True)

        quantized_weight = qweight.to(torch.int32)
        zero = result.zero_point.float()

        # Symmetric quantization uses signed integers [-2^(n-1), 2^(n-1)-1];
        # shift to unsigned [0, 2^n - 1] for GPTQLinear bit packing.
        if result.symmetric:
            offset = 2 ** (result.bits - 1)
            quantized_weight = quantized_weight + offset
            zero = zero + offset

        return GPTQLinear(
            in_features=quantized_weight.shape[1],
            out_features=quantized_weight.shape[0],
            wbits=result.bits,
            groupsize=result.group_size if result.group_size is not None else -1,
            actorder=(result.perm is not None),
            quantized_weight=quantized_weight,
            scale=result.scale.T,
            zero=zero.T,
            perm=result.perm,
            bias=(
                linear_module.bias
                if hasattr(linear_module, "bias") and linear_module.bias is not None
                else None
            ),
            device=linear_module.weight.device,
            pack_weights=pack_weights,
            use_gemlite=kwargs.get("use_gemlite"),
        )
