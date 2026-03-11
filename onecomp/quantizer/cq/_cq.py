"""CQ (Clustering Quantization) quantizer classes

This module defines the CQ quantizer class and result class.

Classes:
    CQResult: Result class for CQ quantization containing quantized weights and parameters.
    CQ: CQ quantizer class that performs 2-value clustering quantization.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

from dataclasses import dataclass
from typing import Optional

import torch

from onecomp.quantizer._quantizer import Quantizer, QuantizationResult
from onecomp.quantizer.cq.cq_impl import run_cq


@dataclass
class CQResult(QuantizationResult):
    """Result class for CQ quantization.

    Inherits from QuantizationResult and adds CQ-specific parameters.
    CQ uses 2-value quantization, so quantized_weight takes values {0, 1}.
    Weight reconstruction: W = torch.where(quantized_weight == 0, left_mean, right_mean)
    When each_row=True, threshold/left_mean/right_mean have row-wise values.

    Attributes:
        dequantized_weight (torch.Tensor): Dequantized weights (FP16, CPU)
            - inherited from parent class.
        each_row (bool): Whether quantization was performed row-wise.
        quantized_weight (torch.Tensor, optional): Quantized weights (indices {0, 1}, INT8, CPU).
        threshold (torch.Tensor, optional): Clustering threshold (scalar or row-wise vector).
        left_mean (torch.Tensor, optional): Left cluster mean value (scalar or row-wise vector).
        right_mean (torch.Tensor, optional): Right cluster mean value (scalar or row-wise vector).
    """
    
    # =========================================
    # Quantization configuration parameters
    # =========================================
    each_row: bool = None
    
    # =========================================
    # Weight reconstruction data
    # =========================================
    quantized_weight: Optional[torch.Tensor] = None  # Indices (INT8, {0, 1})
    threshold: Optional[torch.Tensor] = None         # Threshold
    left_mean: Optional[torch.Tensor] = None         # Left cluster mean
    right_mean: Optional[torch.Tensor] = None        # Right cluster mean


@dataclass
class CQ(Quantizer):
    """CQ (Clustering Quantization) quantizer.

    CQ is a 2-value clustering quantization method that splits weights into two clusters
    for quantization. It compresses weights by computing mean values for each cluster
    and binarizing based on thresholds.

    Quantization method:
    - Sorts weights and finds the split point that minimizes SSE (Sum of Squared Errors)
    - Uses the split point as a threshold to classify weights into two
      clusters (left_mean, right_mean)
    - quantized_weight takes values {0, 1}, where 0 corresponds to left_mean and 1 to right_mean

    CQ does not require calibration data or Hessian matrix.
    each_row=True generally provides higher accuracy but increases parameter count.

    Attributes:
        flag_calibration (bool): Whether to use calibration data (False for CQ).
        flag_hessian (bool): Whether to use Hessian matrix (False for CQ).
        each_row (bool): Whether to quantize row-wise. If True, computes independent
            threshold and cluster means for each row. If False, processes entire weight
            as a single vector to compute global threshold and cluster means. Default is True.

    Methods:
        quantize_layer(module, input, hessian): Quantize a layer using CQ.
    """

    flag_calibration: bool = False
    flag_hessian: bool = False

    # Parameters for the CQ quantizer
    each_row: bool = True

    def quantize_layer(self, module, input, hessian=None):
        """Quantize a layer using CQ.

        Args:
            module (torch.nn.Module): The layer module to quantize.
            input (tuple or torch.Tensor, optional): Input tensor (not used in CQ). Default is None.
            hessian (torch.Tensor, optional): Hessian matrix (not used in CQ). Default is None.

        Returns:
            CQResult: CQ quantization result object containing quantized weights and parameters.
        """
        result_dict = run_cq(
            module,
            each_row=self.each_row,
        )
        
        return CQResult(
            dequantized_weight=result_dict["dequantized_weight"],
            each_row=self.each_row,
            quantized_weight=result_dict["quantized_weight"],
            threshold=result_dict["threshold"],
            left_mean=result_dict["left_mean"],
            right_mean=result_dict["right_mean"],
        )
