"""MDBF (Matrix-extended Double Binary Factorization) quantization module.

Provides layer-wise MDBF quantization and result data structures.

Classes:
    MDBFResult: Result class for MDBF quantization containing quantized weights and parameters.
    MDBF: MDBF quantizer class implementing the quantization flow.

Note:
    MDBF uses the approximation (P passes):
        W ≈ Σ_{p=1}^{P} F^(p) @ G^(p)
        where F^(p) = S_A^(p) * (A_amp^(p) @ Q_U_amp^(p)^T)
              G^(p) = S_B^(p) * (Q_V_amp^(p) @ B_amp^(p)^T)

Copyright 2025-2026 Fujitsu Ltd.

"""

from dataclasses import dataclass, field
import re
from typing import Any, List, Optional

import torch
from onecomp.quantizer._quantizer import Quantizer, QuantizationResult
from onecomp.utils.quant_config import get_quant_param

from .mdbf_impl import run_mdbf


@dataclass
class MDBFResult(QuantizationResult):
    """MDBF quantization result.

    Attributes:
        target_bits (float): Target BPW (e.g., 1.0).
        l (int): Multi-scale rank.
        P (int): Number of passes.
        svd_mode (str): SVD initialization mode.
        use_admm (bool): Whether ADMM optimization was used.
        admm_iters (int): ADMM outer iterations.
        admm_inner_iters (int): ADMM inner iterations.
        admm_reg (float): ADMM regularization coefficient.
        use_gradient_refine (bool): Whether gradient refinement was used.
        gradient_iters (int): Gradient refinement iterations.
        gradient_lr (float): Gradient refinement learning rate.
        activation_aware (bool): Whether activation-aware mode was used.
        act_init (str): Activation initialization mode.
        actual_bpw (float): Achieved BPW.
        r (int): Rank used.
        is_mdbf_quantized (bool): Whether MDBF quantization was applied.
        mdbf_A_sign (list): Sign matrices S_A per pass [(n, r)] × P.
        mdbf_B_sign (list): Sign matrices S_B per pass [(r, m)] × P.
        mdbf_A_amp (list): Row-scale matrices per pass [(n, l)] × P.
        mdbf_B_amp (list): Column-scale matrices per pass [(m, l)] × P.
        mdbf_Q_U_amp (list): Latent row-scale matrices per pass [(r, l)] × P.
        mdbf_Q_V_amp (list): Latent column-scale matrices per pass [(r, l)] × P.
    """

    # =========================================
    # Quantization configuration parameters
    # =========================================
    target_bits: float = None
    l: int = None
    P: int = None
    svd_mode: str = None
    use_admm: bool = None
    admm_iters: int = None
    admm_inner_iters: int = None
    admm_reg: float = None
    use_gradient_refine: bool = None
    gradient_iters: int = None
    gradient_lr: float = None
    activation_aware: bool = None
    act_init: str = None
    actual_activation_aware: bool = None
    actual_bpw: float = None
    r: int = None

    # =========================================
    # Weight reconstruction data
    # =========================================
    is_mdbf_quantized: Optional[bool] = None
    mdbf_A_sign: list = field(default_factory=list)   # [(n, r)] × P
    mdbf_B_sign: list = field(default_factory=list)   # [(r, m)] × P
    mdbf_A_amp:  list = field(default_factory=list)   # [(n, l)] × P
    mdbf_B_amp:  list = field(default_factory=list)   # [(m, l)] × P
    mdbf_Q_U_amp: list = field(default_factory=list)  # [(r, l)] × P
    mdbf_Q_V_amp: list = field(default_factory=list)  # [(r, l)] × P

    def compute_dequantized_weight(self, device=None) -> torch.Tensor:
        """Compute dequantized weight from quantized parameters.

        Reconstructs W ≈ Σ_p F^(p) @ G^(p) from the stored per-pass tensors.

        Args:
            device (str or torch.device, optional): Device to compute on.

        Returns:
            Dequantized weight tensor (FP16, CPU).
        """
        if not self.mdbf_A_sign:
            raise ValueError("MDBFResult is missing required data for dequantization")

        from .utils import reconstruct_weight

        compute_device = torch.device(device) if device is not None else torch.device("cpu")

        W = None
        for A_sign, B_sign, A_amp, B_amp, Q_U_amp, Q_V_amp in zip(
            self.mdbf_A_sign, self.mdbf_B_sign,
            self.mdbf_A_amp,  self.mdbf_B_amp,
            self.mdbf_Q_U_amp, self.mdbf_Q_V_amp,
        ):
            W_p = reconstruct_weight(
                A_sign.float().to(compute_device),
                B_sign.float().to(compute_device),
                A_amp.float().to(compute_device),
                B_amp.float().to(compute_device),
                Q_U_amp.float().to(compute_device),
                Q_V_amp.float().to(compute_device),
            )
            W = W_p if W is None else W + W_p

        return W.to(torch.float16).cpu()


@dataclass
class MDBF(Quantizer):
    """MDBF quantizer.

    Runs MDBF (Matrix-extended Double Binary Factorization) quantization per layer.

    Attributes:
        flag_calibration (bool): Calibration mode flag.
        flag_hessian (bool): Hessian computation flag.
        target_bits (float): Target BPW (e.g., 1.0).
        l (int): Multi-scale rank.
        P (int): Number of passes (1 or 2).
        svd_mode (str): SVD initialization mode ("svd" or "svd_llm").
        use_admm (bool): Whether to use ADMM optimization.
        admm_iters (int): ADMM outer iterations.
        admm_inner_iters (int): ADMM inner iterations.
        admm_reg (float): ADMM regularization coefficient.
        use_gradient_refine (bool): Whether to use gradient refinement.
        gradient_iters (int): Gradient refinement iterations.
        gradient_lr (float): Gradient refinement learning rate.
        activation_aware (bool): Whether to use activation-aware mode (P=1 only).
        act_init (str): Activation initialization mode.
        mlp_target_bits (float, optional): BPW override for MLP layers.
        module_target_bits (dict, optional): Per-layer BPW override.

    Methods:
        quantize_layer: Quantizes a given layer using MDBF.
    """

    flag_calibration: bool = True
    flag_hessian: bool = True
    flag_nsamples: bool = True

    # Parameters for the MDBF quantizer
    target_bits: float = 1.0
    l: int = 1
    P: int = 2
    svd_mode: str = "svd"
    use_admm: bool = True
    admm_iters: int = 260
    admm_inner_iters: int = 3
    admm_reg: float = 0.03
    use_gradient_refine: bool = False
    gradient_iters: int = 1000
    gradient_lr: float = 0.01
    activation_aware: bool = False
    act_init: str = "osvd"
    mlp_target_bits: Optional[float] = None
    module_target_bits: Optional[dict[str, float]] = None

    @staticmethod
    def resolve_bits(
        layer_name: Optional[str],
        default_bits: float,
        mlp_bits: Optional[float] = None,
        module_bits: Optional[dict[str, float]] = None,
    ) -> float:
        """Resolve bit-width from overrides (module > mlp > default).

        Used by the quantizer and by config loader. If layer_name is None, returns default_bits.
        """
        if module_bits and layer_name is not None:
            b = module_bits.get(layer_name)
            if b is not None:
                return b
        if mlp_bits is not None and layer_name is not None and "mlp" in layer_name:
            return mlp_bits
        return default_bits

    def __post_init__(self):
        if self.name is None:
            self.name = f"MDBF_{self.target_bits:g}bit_l{self.l}_P{self.P}"
        super().__post_init__()

    def validate_params(self):
        """Validate MDBF parameters once in setup().

        Validated ranges:
            target_bits: float > 0
            l: int >= 1
            P: int in {1, 2}
            admm_iters: int >= 1 (when use_admm=True)
            admm_reg: float >= 0
            gradient_iters: int >= 1 (when use_gradient_refine=True)
            gradient_lr: float > 0
        """
        bad = []

        if not (isinstance(self.target_bits, (int, float)) and self.target_bits > 0):
            bad.append(
                f"Invalid MDBF parameter 'target_bits': {self.target_bits!r} (expected numeric > 0)."
            )

        if not (isinstance(self.l, int) and self.l >= 1):
            bad.append(f"Invalid MDBF parameter 'l': {self.l!r} (expected int >= 1).")

        if self.P not in {1, 2}:
            bad.append(f"Invalid MDBF parameter 'P': {self.P!r} (expected 1 or 2).")

        if not (isinstance(self.admm_reg, (int, float)) and self.admm_reg >= 0):
            bad.append(f"Invalid MDBF parameter 'admm_reg': {self.admm_reg!r} (expected numeric >= 0).")

        if self.use_admm:
            if not (isinstance(self.admm_iters, int) and self.admm_iters >= 1):
                bad.append(
                    f"Invalid MDBF parameter 'admm_iters': {self.admm_iters!r} "
                    f"(expected int >= 1 when use_admm=True)."
                )
            if not (isinstance(self.admm_inner_iters, int) and self.admm_inner_iters >= 1):
                bad.append(
                    f"Invalid MDBF parameter 'admm_inner_iters': {self.admm_inner_iters!r} "
                    f"(expected int >= 1 when use_admm=True)."
                )

        if self.use_gradient_refine:
            if not (isinstance(self.gradient_iters, int) and self.gradient_iters >= 1):
                bad.append(
                    f"Invalid MDBF parameter 'gradient_iters': {self.gradient_iters!r} "
                    f"(expected int >= 1 when use_gradient_refine=True)."
                )
            if not (isinstance(self.gradient_lr, (int, float)) and self.gradient_lr > 0):
                bad.append(
                    f"Invalid MDBF parameter 'gradient_lr': {self.gradient_lr!r} "
                    f"(expected numeric > 0 when use_gradient_refine=True)."
                )

        if self.svd_mode not in {"svd", "svd_llm"}:
            bad.append(
                f"Invalid MDBF parameter 'svd_mode': {self.svd_mode!r} "
                f"(expected 'svd' or 'svd_llm')."
            )
        
        if self.act_init not in {"none", "osvd", "svd_llm"}:
            bad.append(
                f"Invalid MDBF parameter 'act_init': {self.act_init!r} "
                f"(expected 'none', 'osvd', or 'svd_llm')."
            )

        if self.mlp_target_bits is not None:
            if not (isinstance(self.mlp_target_bits, (int, float)) and self.mlp_target_bits > 0):
                bad.append(
                    f"Invalid MDBF parameter 'mlp_target_bits': {self.mlp_target_bits!r} (expected numeric > 0)"
                )

        if self.module_target_bits is not None:
            if not isinstance(self.module_target_bits, dict):
                bad.append(
                    f"Invalid MDBF parameter 'module_target_bits': must be a dict[str, float], "
                    f"got {type(self.module_target_bits).__name__!r}"
                )
            else:
                for layer_name, bits in self.module_target_bits.items():
                    if not isinstance(layer_name, str):
                        bad.append(
                            "Invalid MDBF parameter 'module_target_bits': keys must be layer name strings."
                        )
                    elif not (isinstance(bits, (int, float)) and bits > 0):
                        bad.append(
                            f"Invalid MDBF parameter 'module_target_bits[{layer_name!r}]': "
                            f"{bits!r} (expected numeric > 0)"
                        )

        if bad:
            raise ValueError("; ".join(bad))

    def quantize_layer(
        self,
        module: torch.nn.Module,
        input=None,
        hessian: torch.Tensor = None,
        nsamples: Optional[int] = None,
    ) -> MDBFResult:
        """Quantize the layer using MDBF.

        Args:
            module (torch.nn.Module): The layer module.
            input (tuple or torch.Tensor): The input to the layer (activations).
            hessian (torch.Tensor, optional): The Hessian matrix.
            nsamples (int, optional): Number of tokens used to compute the Hessian.

        Returns:
            MDBFResult: MDBF quantization result.
        """
        layer_name = self.module_to_name.get(module)
        resolved_target_bits = MDBF.resolve_bits(
            layer_name,
            self.target_bits,
            self.mlp_target_bits,
            self.module_target_bits,
        )

        weight_results = run_mdbf(
            hessian=hessian,
            module=module,
            input=input,
            target_bits=resolved_target_bits,
            l=self.l,
            P=self.P,
            svd_mode=self.svd_mode,
            use_admm=self.use_admm,
            admm_iters=self.admm_iters,
            admm_inner_iters=self.admm_inner_iters,
            admm_reg=self.admm_reg,
            use_gradient_refine=self.use_gradient_refine,
            gradient_iters=self.gradient_iters,
            gradient_lr=self.gradient_lr,
            activation_aware=self.activation_aware,
            act_init=self.act_init,
            nsamples=nsamples,
        )

        params_list = weight_results["mdbf_params"]
        # 実際に生成されたパス数を result.P に反映する（params_list の長さ）
        actual_P = len(params_list)

        mdbf_result = MDBFResult(
            # Quantization configuration parameters
            target_bits=resolved_target_bits,
            l=self.l,
            P=actual_P,
            svd_mode=self.svd_mode,
            use_admm=self.use_admm,
            admm_iters=self.admm_iters,
            admm_inner_iters=self.admm_inner_iters,
            admm_reg=self.admm_reg,
            use_gradient_refine=self.use_gradient_refine,
            gradient_iters=self.gradient_iters,
            gradient_lr=self.gradient_lr,
            activation_aware=self.activation_aware,
            act_init=self.act_init,
            actual_bpw=weight_results["actual_bpw"],
            r=weight_results["r"],
            # Weight reconstruction data
            is_mdbf_quantized=weight_results["is_mdbf_quantized"],
            mdbf_A_sign=[p.A_sign for p in params_list],
            mdbf_B_sign=[p.B_sign for p in params_list],
            mdbf_A_amp=[p.A_amp for p in params_list],
            mdbf_B_amp=[p.B_amp for p in params_list],
            mdbf_Q_U_amp=[p.Q_U_amp for p in params_list],
            mdbf_Q_V_amp=[p.Q_V_amp for p in params_list],
            # Keep the requested flag (as set on the quantizer) and record the
            # actually-used flag returned by run_mdbf.
            actual_activation_aware=weight_results.get(
                "actual_activation_aware", self.activation_aware
            ),
        )

        return mdbf_result

    def get_quant_config(self) -> dict:
        """Return quantization_config dict for save_quantized_model."""
        result: dict[str, Any] = {
            "quant_method": "mdbf",
            "bits": self.target_bits,
            "l": self.l,
            "P": self.P,
            "svd_mode": self.svd_mode,
            "use_admm": self.use_admm,
            "admm_iters": self.admm_iters,
            "admm_inner_iters": self.admm_inner_iters,
            "admm_reg": self.admm_reg,
            "use_gradient_refine": self.use_gradient_refine,
            "gradient_iters": self.gradient_iters,
            "gradient_lr": self.gradient_lr,
            "activation_aware": self.activation_aware,
            "act_init": self.act_init,
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
        """Build per-layer quantization_bits list; length is num_layers."""
        _LAYER_RE = re.compile(r"\.layers\.(\d+)\.(.*)")
        default_bits = quant_config.get("bits", 1.0)
        mlp_target_bits = get_quant_param(quant_config, "mlp_target_bits")
        module_target_bits: dict[str, float] = (
            get_quant_param(quant_config, "module_target_bits") or {}
        )
        params: dict[str, Any] = {
            "l": get_quant_param(quant_config, "l", default=1),
            "P": get_quant_param(quant_config, "P", default=2),
            "svd_mode": get_quant_param(quant_config, "svd_mode", default="svd"),
            "use_admm": get_quant_param(quant_config, "use_admm", default=True),
            "admm_iters": get_quant_param(quant_config, "admm_iters", default=260),
            "admm_inner_iters": get_quant_param(quant_config, "admm_inner_iters", default=3),
            "admm_reg": get_quant_param(quant_config, "admm_reg", default=0.03),
            "use_gradient_refine": get_quant_param(quant_config, "use_gradient_refine", default=False),
            "gradient_iters": get_quant_param(quant_config, "gradient_iters", default=1000),
            "gradient_lr": get_quant_param(quant_config, "gradient_lr", default=0.01),
            "activation_aware": get_quant_param(quant_config, "activation_aware", default=False),
            "act_init": get_quant_param(quant_config, "act_init", default="osvd"),
        }

        layer_modules: dict[int, dict[str, Any]] = {}
        for name in quantized_names:
            m = _LAYER_RE.search(name)
            if m is None:
                continue
            layer_idx = int(m.group(1))
            suffix = m.group(2)

            bits = MDBF.resolve_bits(name, default_bits, mlp_target_bits, module_target_bits)

            layer_modules.setdefault(layer_idx, {})[suffix] = {
                "bits": bits,
                "method": "mdbf",
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
                "num_hidden_layers is required for MDBF quantization_bits "
                "(Runner passes model.config.num_hidden_layers)"
            )
        quant_config["quantization_bits"] = MDBF._build_quantization_bits(
            quantized_layer_names, quant_config, num_hidden_layers
        )
        return quant_config

    def create_inference_layer(self, result, linear_module, **kwargs):
        """Build MultipathMSVIDLinear from MDBFResult."""
        from onecomp.quantizer.mdbf.mdbf_layer import MultipathMSVIDLinear

        bias = (
            linear_module.bias
            if hasattr(linear_module, "bias") and linear_module.bias is not None
            else None
        )
        return MultipathMSVIDLinear.from_quantization_result(
            result=result,
            bias=bias,
            device=linear_module.weight.device,
        )
