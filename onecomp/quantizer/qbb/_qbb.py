"""QBB (Quantized Binary Bases) quantizer classes

This module defines the QBB quantizer class and result class.

Classes:
    QBBResult: Result class for QBB quantization containing quantized weights and parameters.
    QBB: QBB quantizer class that performs quantization using multiple binary bases.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

from dataclasses import dataclass
from typing import Optional

import torch

from onecomp.quantizer._quantizer import Quantizer, QuantizationResult
from onecomp.quantizer.qbb.qbb_impl import run_qbb


@dataclass
class QBBResult(QuantizationResult):
    """Result class for QBB quantization.

    Inherits from QuantizationResult and adds QBB-specific parameters.
    QBB represents weights using multiple binary bases: W ≈ Σ(alpha_i * quantized_weight_list[i]).
    Each quantized_weight_list[i] takes values {±1}.
    When wbits=N, N binary matrices are used.

    Attributes:
        dequantized_weight (torch.Tensor): Dequantized weights (FP16, CPU)
            - inherited from parent class.
        wbits (int): Number of quantization bits (number of binary bases) used.
        iters_per_basis (int): Number of optimization iterations per basis used.
        ste_type (str): Type of Straight-Through Estimator used.
        quantized_weight_list (list[torch.Tensor], optional): List of
            quantized weights (each element is binary matrix {±1}, INT8, CPU).
        alpha_list (list[torch.Tensor], optional): List of scale coefficients
            for each binary basis (FP16, CPU).
    """
    
    # =========================================
    # Quantization configuration parameters
    # =========================================
    wbits: int = None
    iters_per_basis: int = None
    ste_type: str = None
    
    # =========================================
    # Weight reconstruction data
    # =========================================
    quantized_weight_list: Optional[list[torch.Tensor]] = None  # binary (INT8, ±1)
    alpha_list: Optional[list[torch.Tensor]] = None             # List of scale coefficients


@dataclass
class QBB(Quantizer):
    """QBB (Quantized Binary Bases) quantizer.

    QBB is a quantization method using multiple binary bases that represents
    weight matrices as linear combinations of binary matrices with scale coefs.
    Optimizes each binary basis and scale coefficient using gradient-based optimization.

    Quantization method:
    - Decomposes weight matrix W as W ≈ Σ(alpha_i * B_i)
    - B_i: Binary matrices {±1}^{n×m} (each basis)
    - alpha_i: Scale coefficients for each basis ∈ R^m (per column)
    - Uses N binary bases when wbits=N

    QBB does not require calibration data or Hessian matrix.
    When wbits=1, a closed-form solution exists and optimization is not needed.
    Uses gradient-based optimization, which may take longer computation time.

    Attributes:
        flag_calibration (bool): Whether to use calibration data (False for QBB).
        flag_hessian (bool): Whether to use Hessian matrix (False for QBB).
        wbits (int): Number of quantization bits (number of binary bases). Default is 4.
        iters_per_basis (int): Number of optimization iterations per basis. Default is 1000.
        lr (float): Learning rate. Used for gradient-based optimization. Default is 1e-4.
        ste_type (str): Type of Straight-Through Estimator.
            Choose from "clipped", "identity", "tanh". Default is "clipped".
        use_progressive_quantization (bool): Whether to use progressive quantization.
            If True, first quantizes to progressive_bits bits, then converts to binary bases.
            Default is False.
        progressive_bits (int): Starting number of bits for progressive quantization.
            Used when use_progressive_quantization=True. Default is 2.

    Methods:
        quantize_layer(module, input, hessian): Quantize a layer using QBB.
    """

    flag_calibration: bool = False
    flag_hessian: bool = False

    # Parameters for the QBB quantizer
    wbits: int = 4
    iters_per_basis: int = 1000
    lr: float = 1e-4
    ste_type: str = "clipped"
    use_progressive_quantization: bool = False
    progressive_bits: int = 2

    def quantize_layer(self, module, input, hessian=None):
        """Quantize a layer using QBB.

        Args:
            module (torch.nn.Module): The layer module to quantize.
            input (tuple or torch.Tensor, optional): Input tensor (not used
                in QBB). Default is None.
            hessian (torch.Tensor, optional): Hessian matrix (not used in QBB). Default is None.

        Returns:
            QBBResult: QBB quantization result object containing quantized
                weights and parameters.
        """
        result_dict = run_qbb(
            module,
            wbits=self.wbits,
            iters_per_basis=self.iters_per_basis,
            lr=self.lr,
            ste_type=self.ste_type,
            use_progressive_quantization=self.use_progressive_quantization,
            progressive_bits=self.progressive_bits,
        )
        
        return QBBResult(
            dequantized_weight=result_dict["dequantized_weight"],
            wbits=self.wbits,
            iters_per_basis=self.iters_per_basis,
            ste_type=self.ste_type,
            quantized_weight_list=result_dict["quantized_weight_list"],
            alpha_list=result_dict["alpha_list"],
        )
