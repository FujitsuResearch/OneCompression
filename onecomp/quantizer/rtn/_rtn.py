"""RTN (Round-To-Nearest) quantizer classes

This module defines the RTN quantizer class and result class.

Classes:
    RTNResult: Result class for RTN quantization containing quantized weights and parameters.
    RTN: RTN quantizer class that performs round-to-nearest quantization.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import re
from dataclasses import dataclass
from typing import Any, Optional

import torch

from onecomp.quantizer._quantizer import QuantizationResult, Quantizer
from onecomp.quantizer.rtn.rtn_impl import run_rtn
from onecomp.utils.quant_config import get_quant_param


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
    scale: Optional[torch.Tensor] = None  # Scale coefficient
    zero: Optional[torch.Tensor] = None  # Zero point

    def compute_dequantized_weight(self, device=None) -> torch.Tensor:
        """Compute dequantized weight from quantized data.

        Reconstruction formula: W = (quantized_weight - zero) * scale
        (see rtn/quantizer.py dequantize())

        Args:
            device: Device for computation.

        Returns:
            Dequantized weight (FP16, CPU).
        """
        if self.quantized_weight is None or self.scale is None or self.zero is None:
            raise ValueError("quantized_weight, scale, and zero must be provided.")

        compute_device = torch.device(device) if device is not None else torch.device("cpu")
        quantized_weight = self.quantized_weight.to(compute_device, dtype=torch.float32)
        scale = self.scale.to(compute_device, dtype=torch.float32)
        zero = self.zero.to(compute_device, dtype=torch.float32)
        out_features, in_features = quantized_weight.shape

        if self.groupsize == -1:
            # Per-channel path (broadcast along in_features)
            if scale.ndim == 1:
                scale = scale.unsqueeze(1)
            if zero.ndim == 1:
                zero = zero.unsqueeze(1)
            return ((quantized_weight - zero) * scale).to(torch.float16).cpu()

        # scale/zero shape: (out_features, num_groups)
        g_idx = torch.arange(in_features, device=compute_device) // self.groupsize
        scale_expanded = scale[:, g_idx]
        zero_expanded = zero[:, g_idx]
        return ((quantized_weight - zero_expanded) * scale_expanded).to(torch.float16).cpu()


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
        mse (bool): Enable MSE grid search for optimal clipping. Default is False.
        norm (float): Lp norm exponent for MSE search. Default is 2.4.
        grid (int): Number of candidate shrink levels for MSE search. Default is 100.

    Methods:
        quantize_layer(module, input, hessian): Quantize a layer using RTN.
    """

    flag_calibration: bool = False
    flag_hessian: bool = False

    wbits: int = 4
    groupsize: int = -1
    sym: bool = False
    mse: bool = False
    norm: float = 2.4
    grid: int = 100

    def validate_params(self):
        """Validate RTN parameters once in setup().

        Validated ranges:
            wbits: int, 1 <= wbits <= 64
            groupsize: int, -1 or >= 1
            sym: bool (no constraint)
            grid: int >= 1 (when mse=True)
            norm: float > 0 (when mse=True)
        """
        bad = []

        if not (isinstance(self.wbits, int) and 1 <= self.wbits <= 64):
            bad.append(f"Invalid RTN parameter 'wbits': {self.wbits!r} (expected int in 1..64).")

        if not (isinstance(self.groupsize, int) and (self.groupsize == -1 or 1 <= self.groupsize)):
            bad.append(
                f"Invalid RTN parameter 'groupsize': {self.groupsize!r} "
                f"(expected int: -1 for no grouping, or 1<= groupsize)."
            )

        if self.mse:
            if not (isinstance(self.grid, int) and self.grid >= 1):
                bad.append(
                    f"Invalid RTN parameter 'grid': {self.grid!r} "
                    f"(expected int >= 1 when mse=True)."
                )

            if not (isinstance(self.norm, (int, float)) and self.norm > 0):
                bad.append(
                    f"Invalid RTN parameter 'norm': {self.norm!r} "
                    f"(expected numeric > 0 when mse=True)."
                )

        if bad:
            raise ValueError("; ".join(bad))

    def quantize_layer(self, module, input=None, hessian=None):
        """Quantize a layer using RTN.

        Args:
            module (torch.nn.Module): The layer module to quantize.
            input (tuple or torch.Tensor, optional): Input tensor (not used
                in RTN). Default is None.
            hessian (torch.Tensor, optional): Hessian matrix (not used in RTN). Default is None.

        Returns:
            RTNResult: RTN quantization result object containing quantized
                weights and parameters.

        Raises:
            ValueError: If groupsize does not divide in_features.
        """
        if self.groupsize > 0:
            in_features = module.weight.shape[-1]
            if in_features % self.groupsize != 0:
                raise ValueError(
                    f"groupsize={self.groupsize} does not divide " f"in_features={in_features}."
                )

        result_dict = run_rtn(
            module,
            wbits=self.wbits,
            groupsize=self.groupsize,
            sym=self.sym,
            mse=self.mse,
            norm=self.norm,
            grid=self.grid,
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

    def get_quant_config(self) -> dict:
        """Return GPTQ-compatible quantization config.

        RTN uses the same tensor format as GPTQ (qweight/scales/qzeros),
        so we emit quant_method="gptq" to reuse GPTQLinear and vLLM GPTQ plugin.
        """
        return {
            "quant_method": "gptq",
            "bits": self.wbits,
            "groupsize": self.groupsize,
            "group_size": self.groupsize,
            "actorder": False,
            "desc_act": False,
            "sym": self.sym,
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
        quant_config["quantization_bits"] = RTN._build_quantization_bits(
            quantized_layer_names, quant_config, num_hidden_layers
        )
        return quant_config

    def create_inference_layer(self, result, linear_module, **kwargs):
        """Build GPTQLinear from RTNResult.

        RTN scale/zero shape is (out_features, num_groups) from pseudo_quantize_tensor.
        GPTQLinear expects (num_groups, out_features), so we transpose.
        RTN now stores unsigned qweight/zero even for symmetric mode,
        so no additional signed-to-unsigned shift is needed here.
        """
        from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

        pack_weights = kwargs.get("pack_weights", True)

        return GPTQLinear(
            in_features=result.quantized_weight.shape[1],
            out_features=result.quantized_weight.shape[0],
            wbits=result.wbits,
            groupsize=result.groupsize,
            actorder=False,
            quantized_weight=result.quantized_weight.to(torch.int32),
            scale=result.scale.T,
            zero=result.zero.T,
            perm=None,
            bias=(
                linear_module.bias
                if hasattr(linear_module, "bias") and linear_module.bias is not None
                else None
            ),
            device=linear_module.weight.device,
            pack_weights=pack_weights,
            use_gemlite=kwargs.get("use_gemlite"),
        )
