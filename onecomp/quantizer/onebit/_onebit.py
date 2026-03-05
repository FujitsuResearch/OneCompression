"""OneBit quantization module.

Provides layer-wise OneBit quantization
and result data structures for developers.

Classes:
    OnebitResult: Result class for OneBit quantization containing quantized weights and parameters.
    Onebit: OneBit quantizer class that performs OneBit quantization.

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)
"""

import torch
from dataclasses import dataclass
from typing import Optional

from onecomp.quantizer._quantizer import Quantizer, QuantizationResult

from .onebit_impl import run_onebit


@dataclass
class OnebitResult(QuantizationResult):
    """OneBit quantization result.

    Attributes:
        dequantized_weight (torch.Tensor): Dequantized weight (FP16, CPU).
        iters (int): Optimization iterations.
        use_importance_scaling (bool): Whether to use importance scaling.
        use_balancing (bool): Whether to apply weight balancing.
        balance_iters (int): Balancing iterations.
        balance_alpha (float): Balancing alpha.
        a (Optional[torch.Tensor]): Scaling vector a.
        b (Optional[torch.Tensor]): Scaling vector b.
        sign (Optional[torch.Tensor]): Sign matrix sign(W).
    """

    # =========================================
    # Quantization configuration parameters
    # =========================================
    iters: int = None
    use_importance_scaling: bool = None
    use_balancing: bool = None
    balance_iters: int = None
    balance_alpha: float = None

    # =========================================
    # Data for weight reconstruction
    # =========================================
    a: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None
    sign: Optional[torch.Tensor] = None


@dataclass
class Onebit(Quantizer):
    """OneBit quantizer.

    Runs OneBit quantization per layer.

    Attributes:
        iters (int): Optimization iterations.
        use_importance_scaling (bool): Whether to use importance scaling.
        use_balancing (bool): Whether to apply weight balancing.
        balance_iters (int): Balancing iterations.
        balance_alpha (float): Balancing alpha.
    
    Methods:
        quantize_layer(module, input, hessian): Quantizes a given layer and returns OnebitResult.
    """

    iters: int = 10
    use_importance_scaling: bool = True
    use_balancing: bool = True
    balance_iters: int = 40
    balance_alpha: float = 1.0

    def quantize_layer(
        self, module, input, hessian
    ):
        """Quantize the layer.

        Args:
            module (torch.nn.Module): The layer module.
            input (tuple): The input to the layer.
            hessian (torch.Tensor): The Hessian matrix.

        Returns:
            OnebitResult: OneBit quantization result object containing quantized weights and parameters.
        """

        hessian = self.calculate_hessian(module, input)
        weight_results = run_onebit(
            hessian,
            module,
            iters=self.iters,
            use_importance_scaling=self.use_importance_scaling,
            use_balancing=self.use_balancing,
            balance_iters=self.balance_iters,
            balance_alpha=self.balance_alpha,
        )

        return OnebitResult(
            iters=self.iters,
            use_importance_scaling=self.use_importance_scaling,
            use_balancing=self.use_balancing,
            balance_iters=self.balance_iters,
            balance_alpha=self.balance_alpha,
            dequantized_weight=weight_results["dequantized_weight"],
            a=weight_results["a"],
            b=weight_results["b"],
            sign=weight_results["sign"],
        )
