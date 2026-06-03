"""OneBit quantization module.

Provides layer-wise OneBit quantization
and result data structures for developers.

Classes:
    OnebitResult: Result class for OneBit quantization containing quantized weights and parameters.
    Onebit: OneBit quantizer class that performs OneBit quantization.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

from dataclasses import dataclass
from typing import Optional

import torch

from onecomp.quantizer._quantizer import QuantizationResult, Quantizer

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

    flag_calibration: bool = True
    flag_hessian: bool = True

    iters: int = 10
    use_importance_scaling: bool = True
    use_balancing: bool = True
    balance_iters: int = 40
    balance_alpha: float = 1.0

    def validate_params(self):
        """Validate OneBit parameters once in setup().

        Validated ranges:
            iters: int >= 0
            balance_iters: int >= 1 (when use_balancing=True)
            balance_alpha: float > 0 (when use_balancing=True)
        """
        bad = []

        if not (isinstance(self.iters, int) and self.iters >= 0):
            bad.append(f"Invalid OneBit parameter 'iters': {self.iters!r} (expected int >= 0).")

        if self.use_balancing:
            if not (isinstance(self.balance_iters, int) and self.balance_iters >= 1):
                bad.append(
                    f"Invalid OneBit parameter 'balance_iters': {self.balance_iters!r} "
                    f"(expected int >= 1 when use_balancing=True)."
                )

            if not (isinstance(self.balance_alpha, (int, float)) and self.balance_alpha > 0):
                bad.append(
                    f"Invalid OneBit parameter 'balance_alpha': {self.balance_alpha!r} "
                    f"(expected numeric > 0 when use_balancing=True)."
                )

        if bad:
            raise ValueError("; ".join(bad))

    def quantize_layer(self, module, input=None, hessian=None):
        """Quantize the layer.

        Args:
            module (torch.nn.Module): The layer module.
            input (tuple): The input to the layer (not used).
            hessian (torch.Tensor): The Hessian matrix.

        Returns:
            OnebitResult: OneBit quantization result object containing quantized weights and parameters.
        """
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
