"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from dataclasses import dataclass
from typing import Optional

import torch

from jointq import compute_matrix_XX, quantize

from onecomp.quantizer._quantizer import Quantizer, QuantizationResult


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
        log_level (int): Log level (0: none, 1: minimal, 2: detailed). Default is 1.
        device (torch.device): Device for quantization.
        regularization_lambda (float): Tikhonov regularization strength. Default is 0.2.
            Replaces X^T X with X^T X + n*λ*I, where n = dim_n.
            λ is relative to the normalized Hessian (1/n)X^T X, so its meaning
            is consistent across different calibration sample sizes.
            Recommended range: 0.1 to 1.0.
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
    log_level: int = 1  # 0: none, 1: minimal, 2: detailed, 3: debug

    # Device settings
    device: Optional[torch.device] = None

    # Tikhonov regularization: X^T X + n*λ*I
    regularization_lambda: Optional[float] = 0.2

    # Iterated Local Search (ILS) parameters
    ils_enabled: bool = False
    ils_num_iterations: int = 10
    ils_num_clones: int = 8
    ils_num_channels: Optional[int] = None

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
        )
