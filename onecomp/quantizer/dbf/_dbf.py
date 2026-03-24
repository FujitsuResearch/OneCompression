"""DBF (Double Binary Factorization) quantization module.

Provides layer-wise DBF (Double Binary Factorization) quantization
and result data structures for developers.

Classes:
    DBFResult: Result class for DBF quantization containing quantized weights and parameters.
    DBF: DBF quantizer class implementing the quantization flow.

Functions:
    None.

Note:
    DBF uses the approximation:
        W ≈ A * diag(d) * B

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from dataclasses import dataclass
import re
from typing import Any, Dict, Literal, Optional

import torch
from onecomp.quantizer._quantizer import Quantizer, QuantizationResult
from onecomp.utils.quant_config import get_quant_param

from .dbf_impl import run_dbf


@dataclass
class DBFResult(QuantizationResult):
    """DBF quantization result.

    Attributes:
        dequantized_weight (torch.Tensor): Dequantized weights (FP16, CPU)
            - inherited from parent class.
        target_bits (float): Target bit-width (e.g., 1.5).
        iters (int): Optimization iterations.
        reg (float): Regularization coefficient.
        use_balancing (bool): Whether to apply weight balancing.
        balance_iters (int): Balancing iterations.
        balance_alpha (float): Balancing alpha.
        balance_mode (str): Balancing mode.
        use_adaptive_rho (bool): Whether to adapt ADMM rho.
        is_dbf_quantized (Optional[bool]): Whether DBF quantization was applied.
        dbf_Da (Optional[torch.Tensor]): Scaling vector paired with A.
        dbf_A (Optional[torch.Tensor]): Binary A matrix.
        dbf_mid (Optional[torch.Tensor]): Middle scaling vector.
        dbf_B (Optional[torch.Tensor]): Binary B matrix.
        dbf_Db (Optional[torch.Tensor]): Scaling vector paired with B.
    """

    # =========================================
    # Quantization configuration parameters
    # =========================================
    target_bits: float = None
    iters: int = None
    reg: float = None
    use_balancing: bool = None
    balance_iters: int = None
    balance_alpha: float = None
    balance_mode: str = None
    use_adaptive_rho: bool = None

    # =========================================
    # Weight reconstruction data
    # =========================================
    is_dbf_quantized: Optional[bool] = None  # Whether DBF quantization was applied
    dbf_Da: Optional[torch.Tensor] = None  # Scaling vector paired with A (out_dim,)
    dbf_A: Optional[torch.Tensor] = None  # Binary A matrix (out_dim, mid_dim)
    dbf_mid: Optional[torch.Tensor] = None  # Middle scaling vector (mid_dim,)
    dbf_B: Optional[torch.Tensor] = None  # Binary B matrix (mid_dim, in_dim)
    dbf_Db: Optional[torch.Tensor] = None  # Scaling vector paired with B (in_dim,)


@dataclass
class DBF(Quantizer):
    """DBF quantizer.

    Runs DBF (Double Binary Factorization) quantization per layer.

    Attributes:
        flag_calibration (bool): Calibration mode flag.
        flag_hessian (bool): Hessian computation flag.
        target_bits (float): Target bit-width (e.g., 1.5).
        iters (int): Optimization iterations.
        reg (float): Regularization coefficient.
        use_balancing (bool): Whether to apply weight balancing.
        balance_iters (int): Balancing iterations.
        balance_alpha (float): Balancing alpha.
        balance_mode (str): Balancing mode (e.g., "l1").
        use_adaptive_rho (bool): Whether to adapt ADMM rho.

    Methods:
        quantize_layer: Quantizes a given layer using DBF.
    """

    flag_calibration: bool = True
    flag_hessian: bool = True

    # Parameters for the DBF quantizer
    target_bits: float = 1.5
    iters: int = 600
    reg: float = 3e-2
    use_balancing: bool = True
    balance_iters: int = 40
    balance_alpha: float = 1.0
    balance_mode: str = "l1"
    use_adaptive_rho: bool = True
    mlp_target_bits: Optional[float] = None
    module_target_bits: Optional[dict[str, float]] = None

    @staticmethod
    def resolve_bits(
        layer_name: Optional[str],
        default_bits: float,
        mlp_bits: Optional[float] = None,
        module_bits: Optional[dict[str, float]] = None,
    ) -> float:
        """Resolve bit-width from overrides (DBF semantics: module > mlp > default).

        Used by the quantizer and by config loader. If layer_name is None, returns default_bits.
        Does not validate range; caller may validate.
        """
        if module_bits and layer_name is not None:
            b = module_bits.get(layer_name)
            if b is not None:
                return b
        if mlp_bits is not None and layer_name is not None and "mlp" in layer_name:
            return mlp_bits
        return default_bits

    def validate_params(self):
        """Validate DBF parameters once at quantizer initialization."""
        bad = []

        if (
            self.target_bits is None
            or not isinstance(self.target_bits, (int, float))
            or not (self.target_bits >= 1.0)
        ):
            bad.append(
                f"Invalid DBF parameter 'target_bits': {self.target_bits!r} (expected numeric >= 1.0)"
            )

        if not (isinstance(self.iters, int) and self.iters >= 1):
            bad.append(f"Invalid DBF parameter 'iters': {self.iters!r} (expected int >= 1)")

        if not (isinstance(self.reg, (int, float)) and self.reg >= 0):
            bad.append(f"Invalid DBF parameter 'reg': {self.reg!r} (expected numeric >= 0.0)")

        if not (isinstance(self.balance_iters, int) and self.balance_iters >= 1):
            bad.append(
                f"Invalid DBF parameter 'balance_iters': {self.balance_iters!r} (expected int >= 1)"
            )

        if not (isinstance(self.balance_alpha, (int, float)) and self.balance_alpha >= 1):
            bad.append(
                f"Invalid DBF parameter 'balance_alpha': {self.balance_alpha!r} (expected numeric >= 1.0)"
            )

        if self.mlp_target_bits is not None:
            if not (
                isinstance(self.mlp_target_bits, (int, float)) and self.mlp_target_bits >= 1.0
            ):
                bad.append(
                    f"Invalid DBF parameter 'mlp_target_bits': {self.mlp_target_bits!r} (expected numeric >= 1.0)"
                )

        if self.module_target_bits is not None:
            if not isinstance(self.module_target_bits, dict):
                bad.append(
                    f"Invalid DBF parameter 'module_target_bits': must be a dict[str, float], got {type(self.module_target_bits).__name__!r}"
                )
            else:
                for layer_name, bits in self.module_target_bits.items():
                    if not isinstance(layer_name, str):
                        bad.append(
                            "Invalid DBF parameter 'module_target_bits': keys must be layer name strings."
                        )
                    elif not (isinstance(bits, (int, float)) and bits >= 1.0):
                        bad.append(
                            f"Invalid DBF parameter 'module_target_bits[{layer_name!r}]': {bits!r} (expected numeric >= 1.0)"
                        )

        if bad:
            raise ValueError("; ".join(bad))

    def quantize_layer(
        self,
        module: torch.nn.modules,
        input=None,
        hessian: torch.Tensor = None,
    ) -> DBFResult:  # pylint: disable=redefined-builtin
        """Quantize the layer.

        Args:
            module (torch.nn.Module): The layer module.
            input (tuple or torch.Tensor): The input to the layer (activations).
            hessian (torch.Tensor, optional): The Hessian matrix.

        Returns:
            DBFResult: DBF quantization result object containing quantized weights and parameters.
        """

        layer_name = self.module_to_name.get(module)
        resolved_target_bits = DBF.resolve_bits(
            layer_name,
            self.target_bits,
            self.mlp_target_bits,
            self.module_target_bits,
        )

        # Quantize the layer
        weight_results = run_dbf(
            hessian,
            module,
            target_bits=resolved_target_bits,
            iters=self.iters,
            reg=self.reg,
            use_balancing=self.use_balancing,
            balance_iters=self.balance_iters,
            balance_alpha=self.balance_alpha,
            balance_mode=self.balance_mode,
            use_adaptive_rho=self.use_adaptive_rho,
        )

        dbf_result = DBFResult(
            # DBF quantization parameters
            target_bits=resolved_target_bits,
            iters=self.iters,
            reg=self.reg,
            use_balancing=self.use_balancing,
            balance_iters=self.balance_iters,
            balance_alpha=self.balance_alpha,
            balance_mode=self.balance_mode,
            use_adaptive_rho=self.use_adaptive_rho,
            # DBF weight reconstruction data
            dequantized_weight=weight_results["dequantized_weight"],
            is_dbf_quantized=weight_results["is_dbf_quantized"],
            dbf_Da=weight_results["dbf_Da"],
            dbf_A=weight_results["dbf_A"],
            dbf_mid=weight_results["dbf_mid"],
            dbf_B=weight_results["dbf_B"],
            dbf_Db=weight_results["dbf_Db"],
        )

        return dbf_result

    def get_quant_config(self) -> dict:
        """Return quantization_config dict for save_quantized_model.

        Structure: all keys at top-level (quant_method, bits, iters, reg, etc.).
        """
        result: dict[str, Any] = {
            "quant_method": "dbf",
            "bits": self.target_bits,
            "iters": self.iters,
            "reg": self.reg,
            "use_balancing": self.use_balancing,
            "balance_iters": self.balance_iters,
            "balance_alpha": self.balance_alpha,
            "balance_mode": self.balance_mode,
            "use_adaptive_rho": self.use_adaptive_rho,
        }
        if self.mlp_target_bits is not None:
            result["mlp_target_bits"] = self.mlp_target_bits
        if self.module_target_bits:
            result["module_target_bits"] = dict(self.module_target_bits)
        return result

    @staticmethod
    def _build_quantization_bits(
        quantized_names: list[str],
        quant_config: dict[str, Any],
        num_layers: int,
    ) -> list[dict[str, Any]]:
        """Build per-layer quantization_bits list; length is num_layers (model's total layer count)."""
        _LAYER_RE = re.compile(r"\.layers\.(\d+)\.(.*)")
        default_bits = quant_config.get("bits", 1.5)
        mlp_target_bits = get_quant_param(quant_config, "mlp_target_bits")
        module_target_bits: dict[str, float] = (
            get_quant_param(quant_config, "module_target_bits") or {}
        )
        params: dict[str, Any] = {
            "iters": get_quant_param(quant_config, "iters", default=600),
            "reg": get_quant_param(quant_config, "reg", default=3e-2),
            "use_balancing": get_quant_param(quant_config, "use_balancing", default=True),
            "balance_iters": get_quant_param(quant_config, "balance_iters", default=40),
            "balance_alpha": get_quant_param(quant_config, "balance_alpha", default=1.0),
            "balance_mode": get_quant_param(quant_config, "balance_mode", default="l1"),
            "use_adaptive_rho": get_quant_param(quant_config, "use_adaptive_rho", default=True),
        }

        layer_modules: dict[int, dict[str, Any]] = {}
        for name in quantized_names:
            m = _LAYER_RE.search(name)
            if m is None:
                continue
            layer_idx = int(m.group(1))
            suffix = m.group(2)

            bits = DBF.resolve_bits(name, default_bits, mlp_target_bits, module_target_bits)

            layer_modules.setdefault(layer_idx, {})[suffix] = {
                "bits": bits,
                "method": "dbf",
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
            raise ValueError(
                "num_hidden_layers is required for DBF quantization_bits "
                "(Runner passes model.config.num_hidden_layers)"
            )
        quant_config["quantization_bits"] = DBF._build_quantization_bits(
            quantized_layer_names, quant_config, num_hidden_layers
        )
        return quant_config

    def create_inference_layer(self, result, linear_module, **kwargs):
        """Build DoubleBinaryLinear from DBFResult."""
        from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear

        bias = (
            linear_module.bias
            if hasattr(linear_module, "bias") and linear_module.bias is not None
            else None
        )
        return DoubleBinaryLinear.from_quantization_result(
            result=result,
            bias=bias,
            device=linear_module.weight.device,
            use_gemlite=kwargs.get("use_gemlite"),
        )
