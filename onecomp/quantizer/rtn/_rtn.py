"""RTN (Round-To-Nearest) quantizer classes

This module defines the RTN quantizer class and result class.

Classes:
    RTNResult: Result class for RTN quantization containing quantized weights and parameters.
    RTN: RTN quantizer class that performs round-to-nearest quantization.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

from dataclasses import dataclass
from typing import Optional

import torch

from onecomp.quantizer._quantizer import Quantizer, QuantizationResult
from onecomp.quantizer.rtn.rtn_impl import run_rtn


@dataclass
class RTNResult(QuantizationResult):
    """Result class for RTN quantization.

    Inherits from QuantizationResult and adds RTN-specific parameters.

    Attributes:
        dequantized_weight (torch.Tensor): Dequantized weights (FP16, CPU)
            - inherited from parent class.
        wbits (int): Number of quantization bits used.
        groupsize (int): Group size used (-1 means no grouping).
        sym (bool): Whether symmetric quantization was used.
        quantized_weight (torch.Tensor, optional): Quantized weights (INT type, CPU).
        scale (torch.Tensor, optional): Scale coefficients (FP16, CPU).
        zero (torch.Tensor, optional): Zero point (FP16, CPU).
    """

    # =========================================
    # Quantization configuration parameters
    # =========================================
    wbits: int = None
    groupsize: int = None
    sym: bool = None

    # =========================================
    # Weight reconstruction data
    # =========================================
    quantized_weight: Optional[torch.Tensor] = None  # Quantized weights (INT type)
    scale: Optional[torch.Tensor] = None             # Scale coefficient
    zero: Optional[torch.Tensor] = None              # Zero point


@dataclass
class RTN(Quantizer):
    """RTN (Round-To-Nearest) quantizer.

    RTN is the simplest quantization method that rounds weights to the nearest quantization level.
    It does not require calibration data or Hessian matrices, performing quantization
    using only weight statistics.

    Quantization method:
    - Computes minimum and maximum values of weights
    - Computes scale and zero point
    - Rounds weights to nearest quantization level (Round-To-Nearest)

    RTN does not require calibration data or Hessian matrix.
    Fastest method but may have lower accuracy compared to other methods.

    Attributes:
        flag_calibration (bool): Whether to use calibration data (False for RTN).
        flag_hessian (bool): Whether to use Hessian matrix (False for RTN).
        wbits (int): Number of quantization bits. Default is 4.
        groupsize (int): Group size. Computes independent scale and zero point for each group.
            -1 means no grouping (single scale and zero point for entire row). Default is -1.
        sym (bool): Whether to use symmetric quantization. If True, zero point is placed at center.
            Default is False.

    Methods:
        quantize_layer(module, input, hessian): Quantize a layer using RTN.
    """

    flag_calibration: bool = False
    flag_hessian: bool = False

    # Parameters for the RTN quantizer
    wbits: int = 4
    groupsize: int = -1
    sym: bool = False

    def quantize_layer(self, module, input, hessian=None):
        """Quantize a layer using RTN.

        Args:
            module (torch.nn.Module): The layer module to quantize.
            input (tuple or torch.Tensor, optional): Input tensor (not used
                in RTN). Default is None.
            hessian (torch.Tensor, optional): Hessian matrix (not used in RTN). Default is None.

        Returns:
            RTNResult: RTN quantization result object containing quantized
                weights and parameters.
        """
        result_dict = run_rtn(
            module,
            wbits=self.wbits,
            groupsize=self.groupsize,
            sym=self.sym,
        )

        return RTNResult(
            dequantized_weight=result_dict["dequantized_weight"],
            wbits=self.wbits,
            groupsize=self.groupsize,
            sym=self.sym,
            quantized_weight=result_dict["quantized_weight"],
            scale=result_dict["scale"],
            zero=result_dict["zero"],
        )
