"""ARB (Alternating Refined Binarization) quantizer classes

This module defines the ARB quantizer class and result class.

Classes:
    ARBResult: Result class for ARB quantization containing quantized weights and parameters.
    ARB: ARB quantizer class that performs 1-bit binary quantization.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

from dataclasses import dataclass
from typing import Optional

import torch

from onecomp.quantizer._quantizer import Quantizer, QuantizationResult
from onecomp.quantizer.arb.arb_impl import run_arb


@dataclass
class ARBResult(QuantizationResult):
    """Result class for ARB quantization.

    Inherits from QuantizationResult and adds ARB-specific parameters.
    ARB uses 1-bit binary quantization, so quantized_weight takes values {±1}.
    Weight reconstruction: W = alpha[:, None] * quantized_weight + mu[:, None]

    Attributes:
        dequantized_weight (torch.Tensor): Dequantized weights (FP16, CPU)
            - inherited from parent class.
        arb_iters (int): Number of ARB iterations used for quantization.
        split_points (int): Number of split points used for grouping non-salient weights.
        quantized_weight (torch.Tensor, optional): Quantized weights
            (binary matrix {±1}, INT8, CPU).
        alpha (torch.Tensor, optional): Row-wise scale coefficients (FP16, CPU).
        mu (torch.Tensor, optional): Row-wise bias (FP16, CPU).
    """
    
    # =========================================
    # Quantization configuration parameters
    # =========================================
    arb_iters: int = None
    split_points: int = None
    
    # =========================================
    # Weight reconstruction data
    # =========================================
    quantized_weight: Optional[torch.Tensor] = None  # Binary matrix (INT8, {±1})
    alpha: Optional[torch.Tensor] = None             # Scale coefficient
    mu: Optional[torch.Tensor] = None                # Bias


@dataclass
class ARB(Quantizer):
    """ARB (Alternating Refined Binarization) quantizer.

    ARB is a 1-bit binary quantization method that decomposes weight matrices
    into binary matrices with scale and bias terms. Based on the BTC-LLM paper,
    it improves quantization accuracy by alternately refining mean, scale, and sign
    to correct distribution shifts.

    Decomposes weight matrix W as W ≈ α⊙B + μ:
    - B: Binary matrix {±1}^{n×m}
    - α: Row-wise scaling coefficients ∈ R^n
    - μ: Row-wise bias (mean) ∈ R^n

    ARB does not require calibration data or Hessian matrix.
    When split_points > 1, accuracy improves by processing non-salient weights progressively.

    Attributes:
        flag_calibration (bool): Whether to use calibration data (False for ARB).
        flag_hessian (bool): Whether to use Hessian matrix (False for ARB).
        arb_iters (int): Number of ARB iterations. Default is 15.
        split_points (int): Number of split points. Number of groups to split
            non-salient weights into. Default is 2. If 1, standard ARB (no splitting) is executed.
        verbose (bool): Whether to output detailed logs. Default is False.

    Methods:
        quantize_layer(module, input, hessian): Quantize a layer using ARB.
    """

    flag_calibration: bool = False
    flag_hessian: bool = False

    # Parameters for the ARB quantizer
    arb_iters: int = 15
    split_points: int = 2
    verbose: bool = False

    def quantize_layer(self, module, input=None, hessian=None):
        """Quantize a layer using ARB.

        Args:
            module (torch.nn.Module): The layer module to quantize.
            input (torch.Tensor, optional): Input tensor (not used in ARB). Default is None.
            hessian (torch.Tensor, optional): Hessian matrix (not used in ARB). Default is None.

        Returns:
            ARBResult: ARB quantization result object containing quantized weights and parameters.
        """
        result_dict = run_arb(
            module,
            arb_iters=self.arb_iters,
            split_points=self.split_points,
            verbose=self.verbose,
        )
        
        return ARBResult(
            dequantized_weight=result_dict["dequantized_weight"],
            arb_iters=self.arb_iters,
            split_points=self.split_points,
            quantized_weight=result_dict["quantized_weight"],
            alpha=result_dict["alpha"],
            mu=result_dict["mu"],
        )
