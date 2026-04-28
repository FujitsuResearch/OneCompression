"""OneBit quantization module.

Provides layer-wise OneBit quantization
and result data structures for developers.

Classes:
    OnebitResult: Result class for OneBit quantization containing quantized weights and parameters.
    Onebit: OneBit quantizer class that performs OneBit quantization.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import re
import torch
from dataclasses import dataclass
from typing import Any, Optional

from onecomp.quantizer._quantizer import Quantizer, QuantizationResult
from onecomp.utils.quant_config import get_quant_param

from .onebit_impl import run_onebit


@dataclass
class OnebitResult(QuantizationResult):
    """OneBit quantization result.

    Attributes:
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

    def compute_dequantized_weight(self, device=None) -> torch.Tensor:
        """Compute dequantized weight from a, b, and sign.

        W ≈ a[:, None] * sign * b[None, :]

        Args:
            device (str or torch.device, optional): Device to compute on.

        Returns:
            Dequantized weight tensor (FP16, CPU).
        """
        if self.a is None or self.b is None or self.sign is None:
            raise ValueError("OnebitResult is missing required data for dequantization")

        compute_device = torch.device(device) if device is not None else torch.device("cpu")
        a = self.a.to(torch.float32).to(compute_device)
        b = self.b.to(torch.float32).to(compute_device)
        sign = self.sign.to(torch.float32).to(compute_device)
        weight = a[:, None] * sign * b[None, :]
        return weight.to(torch.float16).cpu()


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
            a=weight_results["a"],
            b=weight_results["b"],
            sign=weight_results["sign"],
        )

    def get_quant_config(self) -> dict:
        """Return OneBit quantization config for saving."""
        return {
            "quant_method": "onebit",
            "bits": 1,
            "iters": self.iters,
            "use_importance_scaling": self.use_importance_scaling,
            "use_balancing": self.use_balancing,
            "balance_iters": self.balance_iters,
            "balance_alpha": self.balance_alpha,
        }

    @staticmethod
    def _build_quantization_bits(
        quantized_names: list[str],
        quant_config: dict[str, Any],
        num_layers: int,
    ) -> list[dict[str, Any]]:
        _LAYER_RE = re.compile(r"\.layers\.(\d+)\.(.*)")

        params: dict[str, Any] = {
            "iters":                  get_quant_param(quant_config, "iters", default=10),
            "use_importance_scaling": get_quant_param(
                quant_config, "use_importance_scaling", default=True
            ),
            "use_balancing":          get_quant_param(quant_config, "use_balancing", default=True),
            "balance_iters":          get_quant_param(quant_config, "balance_iters", default=40),
            "balance_alpha":          get_quant_param(quant_config, "balance_alpha", default=1.0),
        }

        layer_modules: dict[int, dict[str, Any]] = {}
        for name in quantized_names:
            m = _LAYER_RE.search(name)
            if m is None:
                continue
            layer_idx = int(m.group(1))
            suffix = m.group(2)
            layer_modules.setdefault(layer_idx, {})[suffix] = {
                "bits": 1,
                "method": "onebit",
                "params": params,
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
        quant_config["quantization_bits"] = Onebit._build_quantization_bits(
            quantized_layer_names, quant_config, num_hidden_layers
        )
        return quant_config

    def create_inference_layer(self, result, linear_module, **kwargs):
        """Build OneBitLinear from OnebitResult."""
        from .onebit_layer import OneBitLinear

        bias = (
            linear_module.bias
            if hasattr(linear_module, "bias") and linear_module.bias is not None
            else None
        )
        device = linear_module.weight.device

        return OneBitLinear.from_quantization_result(
            result,
            bias=bias,
            device=device,
        )
