"""QUIP (Quantization with Incoherence Processing) quantizer classes

This module defines the QUIP quantizer class and result class.

Classes:
    QUIPResult: Result class for QUIP quantization containing quantized weights and parameters.
    QUIP: QUIP quantizer class that performs quantization using incoherence processing.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

from dataclasses import dataclass
from typing import Optional

import torch

from onecomp.quantizer._quantizer import Quantizer, QuantizationResult
from onecomp.quantizer.quip.quip_impl import run_quip


@dataclass
class QUIPResult(QuantizationResult):
    """Result class for QUIP quantization.

    Inherits from QuantizationResult and adds QUIP-specific parameters.

    Attributes:
        dequantized_weight (torch.Tensor): Dequantized weights (FP16, CPU)
            - inherited from parent class.
        wbits (int): Number of quantization bits used.
        percdamp (float): Damping coefficient used.
        incoh_mode (str): Incoherence mode used ("kron" or "had").
        quantized_weight (torch.Tensor, optional): Quantized weights (INT, CPU).
            Currently None.
        scale (torch.Tensor, optional): Scale coefficients (FP16, CPU).
        zero (torch.Tensor, optional): Zero point (FP16, CPU).
        maxq (torch.Tensor, optional): Maximum quantization level.
    """
    
    # =========================================
    # Quantization configuration parameters
    # =========================================
    wbits: int = None
    percdamp: float = None
    incoh_mode: str = None
    
    # =========================================
    # Weight reconstruction data
    # =========================================
    quantized_weight: Optional[torch.Tensor] = None  # Quantized weights (INT type)
    scale: Optional[torch.Tensor] = None             # Scale coefficient
    zero: Optional[torch.Tensor] = None              # Zero point
    maxq: Optional[torch.Tensor] = None              # Maximum quantization level


@dataclass
class QUIP(Quantizer):
    """QUIP (Quantization with Incoherence Processing) quantizer.

    QUIP is a quantization method using incoherence processing that quantizes weights
    using Hessian matrices. Similar to GPTQ, it uses Hessian matrices but improves
    quantization accuracy through incoherence processing.

    Incoherence processing:
    - "kron": Orthogonal transformation based on Kronecker product (uses Butterfly matrices)
    - "had": Hadamard transform-based processing

    QUIP requires both calibration data and Hessian matrix.
    Incoherence processing transforms weights and Hessian before and after quantization.

    Attributes:
        flag_calibration (bool): Whether to use calibration data (True for QUIP).
        flag_hessian (bool): Whether to use Hessian matrix (True for QUIP).
        wbits (int): Number of quantization bits. Default is 4.
        percdamp (float): Damping coefficient. Ratio of damping added to diagonal elements
            of Hessian matrix. Default is 0.01.
        incoh_mode (str): Incoherence mode. Choose from "kron" or "had". Default is "kron".

    Methods:
        quantize_layer(module, input, hessian): Quantize a layer using QUIP.
    """

    flag_calibration: bool = True
    flag_hessian: bool = True

    # Parameters for the QUIP quantizer
    wbits: int = 4
    percdamp: float = 0.01
    incoh_mode: str = "kron"

    def quantize_layer(self, module, input, hessian=None):
        """Quantize a layer using QUIP.

        Args:
            module (torch.nn.Module): The layer module to quantize.
            input (tuple or torch.Tensor): The input to the layer (input activations).
            hessian (torch.Tensor, optional): The Hessian matrix. If None, it will be calculated.
                Default is None.

        Returns:
            QUIPResult: QUIP quantization result object containing quantized weights and parameters.
        """
        if hessian is None:
            hessian = self.calculate_hessian(module, input)
        
        result_dict = run_quip(
            hessian,
            module,
            percdamp=self.percdamp,
            wbits=self.wbits,
            incoh_mode=self.incoh_mode,
        )
        
        return QUIPResult(
            dequantized_weight=result_dict["dequantized_weight"],
            wbits=self.wbits,
            percdamp=self.percdamp,
            incoh_mode=self.incoh_mode,
            quantized_weight=result_dict["quantized_weight"],
            scale=result_dict["scale"],
            zero=result_dict["zero"],
            maxq=result_dict["maxq"],
        )
