"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import re
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .core import compute_matrix_XX, quantize

from onecomp.quantizer._quantizer import Quantizer, QuantizationResult
from onecomp.utils.quant_config import get_quant_param


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


@dataclass
class JointQ(Quantizer):
    """JointQ quantizer class

    JointQ is a quantization method that uses the jointq package.

    Attributes:
        bits (int): Number of bits for quantization. Default is 4.
        symmetric (bool): Whether to use symmetric quantization. Default is False.
        group_size (int or None): Group size for quantization. Default is 128.
            If None, per-channel quantization is used.
        batch_size (int): Batch size for quantization. Default is None (solve all at once).
        log_level (int): Log level (0: none, 1: minimal, 2: detailed). Default is 0.
        device (torch.device): Device for quantization.
        regularization_lambda (float): Tikhonov regularization strength. Default is 0.2.
            Replaces X^T X with X^T X + n*λ*I, where n = dim_n.
            λ is relative to the normalized Hessian (1/n)X^T X, so its meaning
            is consistent across different calibration sample sizes.
            Recommended range: 0.1 to 1.0.
        actorder (bool): Whether to reorder columns by activation magnitude
            (Hessian diagonal) before quantization. Default is False.
            When enabled, columns with larger activations are grouped together,
            improving group quantization efficiency and GPTQ initial solution quality.
        ils_enabled (bool): Whether to enable Iterated Local Search. Default is False.
        ils_num_iterations (int): Number of ILS iterations. Default is 10.
        ils_num_clones (int): Number of ILS clones. Default is 8.
        ils_num_channels (int): Number of ILS channels. Default is None.

    Example:
        Basic usage::

            from onecomp.quantizer.jointq import JointQ

            quantizer = JointQ(
                bits=4,
                symmetric=False,
                group_size=128,
                device=torch.device(0),
            )

        With batch_size::

            from onecomp.quantizer.jointq import JointQ

            quantizer = JointQ(
                bits=4,
                symmetric=False,
                group_size=128,
                batch_size=4096,
                device=torch.device(0),
            )

        Without Iterated Local Search (ILS)::

            from onecomp.quantizer.jointq import JointQ

            quantizer = JointQ(
                bits=4,
                symmetric=False,
                group_size=128,
                device=torch.device(0),
                ils_enabled=False,
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
    group_size: int = 128
    batch_size: Optional[int] = None
    log_level: int = 0  # 0: none, 1: minimal, 2: detailed, 3: debug

    # Device settings
    device: Optional[torch.device] = None

    # Tikhonov regularization: X^T X + n*λ*I
    regularization_lambda: Optional[float] = 0.2

    # Activation ordering
    actorder: bool = False

    # Iterated Local Search (ILS) parameters
    ils_enabled: bool = False
    ils_num_iterations: int = 10
    ils_num_clones: int = 8
    ils_num_channels: Optional[int] = None

    def validate_params(self):
        """Validate JointQ parameters once in setup().

        Validated ranges:
            bits: int >= 1
            group_size: int >= 1
            batch_size: int >= 1 or None
            log_level: int in {0, 1, 2}
            ils_num_iterations: int >= 1 (when ils_enabled=True)
            ils_num_clones: int >= 1 (when ils_enabled=True)
            ils_num_channels: int >= 1 or None (when ils_enabled=True)
        """
        bad = []

        if not (isinstance(self.bits, int) and self.bits >= 1):
            bad.append(f"Invalid JointQ parameter 'bits': {self.bits!r} (expected int >= 1).")

        if not (isinstance(self.group_size, int) and self.group_size >= 1):
            bad.append(
                f"Invalid JointQ parameter 'group_size': {self.group_size!r} (expected int >= 1)."
            )

        if self.batch_size is not None and not (
            isinstance(self.batch_size, int) and self.batch_size >= 1
        ):
            bad.append(
                f"Invalid JointQ parameter 'batch_size': {self.batch_size!r} (expected int >= 1 or None)."
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

        if bad:
            raise ValueError("; ".join(bad))

    def quantize_layer(
        self, module, input=None, hessian=None, matrix_XX=None, dim_n=None
    ):  # pylint: disable=redefined-builtin, too-many-arguments, too-many-positional-arguments
        """Quantize the layer

        If matrix_XX and dim_n are provided, uses the precomputed X^T X.
        Otherwise, computes matrix_X from input (legacy behavior).

        Args:
            module (torch.nn.Module): The layer module
            input (tuple or torch.Tensor): The input to the layer (input activations)
            hessian (torch.Tensor): The Hessian matrix (not used in JointQ)
            matrix_XX (torch.Tensor): Precomputed X^T X (FP64).
                If provided, this is used instead of input.
            dim_n (int): Number of samples. Required when matrix_XX is provided.

        Returns:
            JointQResult: JointQ quantization result object
        """

        # Get the weight matrix
        # W: (out_features, in_features)
        matrix_W = module.weight.data.clone().cpu().to(torch.float64)

        # Prepare ILS parameters
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

        # Perform quantization
        device = self.device
        if device is None:
            device = module.weight.device

        # Prepare matrix_XX: use as-is if precomputed, otherwise compute from input
        if matrix_XX is None:
            # Get matrix_X from input and compute via compute_matrix_XX
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

        # Activation ordering: sort columns by X^T X diagonal (descending)
        perm = None
        if self.actorder:
            perm = torch.argsort(torch.diag(matrix_XX), descending=True)
            matrix_W = matrix_W[:, perm.to(matrix_W.device)]
            matrix_XX = matrix_XX[perm][:, perm]

        # Tikhonov regularization: X^T X → X^T X + n*λ*I
        if self.regularization_lambda is not None and self.regularization_lambda > 0.0:
            matrix_XX = matrix_XX + (dim_n * self.regularization_lambda) * torch.eye(
                matrix_XX.shape[0], dtype=matrix_XX.dtype, device=matrix_XX.device
            )

        # Perform quantization
        solution = quantize(
            matrix_W=matrix_W,
            matrix_XX=matrix_XX,
            dim_n=dim_n,
            bits=self.bits,
            symmetric=self.symmetric,
            group_size=self.group_size,
            batch_size=self.batch_size,
            device=device,
            log_level=self.log_level,
            **ils_kwargs,
        )

        # Get quantized result (scale, assignment, zero_point)
        scale, assignment, zero_point = solution.get_quantized_result()

        # Create and return JointQResult object
        return JointQResult(
            bits=self.bits,
            symmetric=self.symmetric,
            group_size=self.group_size,
            scale=scale.cpu(),
            zero_point=zero_point.cpu(),
            assignment=assignment.cpu(),
            perm=perm.cpu() if perm is not None else None,
        )


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
