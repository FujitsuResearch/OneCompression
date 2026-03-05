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

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)

"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
from onecomp.quantizer._quantizer import Quantizer, QuantizationResult

from .dbf_impl import run_dbf


@dataclass
class DBFResult(QuantizationResult):
    """DBF quantization result.

    Attributes:
        dequantized_weight (torch.Tensor): Dequantized weights (FP16, CPU) - inherited from parent class.
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

        # Quantize the layer
        weight_results = run_dbf(
            hessian,
            module,
            target_bits=self.target_bits,
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
            target_bits=self.target_bits,
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
        """Return quantization_config dict for save_quantized_model."""
        return {
            "quant_method": "dbf",
            "target_bits": self.target_bits,
            "iters": self.iters,
            "reg": self.reg,
            "use_balancing": self.use_balancing,
            "balance_iters": self.balance_iters,
            "balance_alpha": self.balance_alpha,
            "balance_mode": self.balance_mode,
            "use_adaptive_rho": self.use_adaptive_rho,
        }

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
