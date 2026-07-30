"""MDBF (Multi-Envelope Double Binary Factorization) quantization module.

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

Author: Keiji Kimura

"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from onecomp.quantizer._quantizer import QuantizationResult, Quantizer
from onecomp.utils.quant_config import get_quant_param

from .initialize import MDBFParams
from .mdbf_impl import run_mdbf
from .utils import DEFAULT_L, DEFAULT_P, DEFAULT_SCALE_BITS

# Upper bound accepted by torch.Generator.manual_seed(); above it torch raises
# "ValueError: Overflow when unpacking long long" from inside the ADMM phase.
# Torch also takes negative seeds, but MDBF restricts them to non-negative values.
MAX_SEED = 2**64 - 1


@dataclass
class MDBFResult(QuantizationResult):
    """MDBF quantization result.

    Attributes:
        target_bits (float): Target BPW (e.g., 1.0).
        l (int): Multi-scale rank.
        P (int): Number of passes.
        svd_mode (str): SVD initialization mode.
        use_admm (bool): Whether ADMM optimization was used.
        admm_outer_iters (int): ADMM outer iterations.
        admm_inner_iters (int): ADMM inner iterations.
        admm_reg (float): ADMM regularization coefficient.
        admm_seed (Optional[int]): Random seed used for the ADMM MDBF projection
            (None = global RNG).
        use_gradient_refine (bool): Whether gradient refinement was used.
        gradient_iters (int): Gradient refinement iterations.
        gradient_lr (float): Gradient refinement learning rate.
        activation_aware (bool): Requested activation-aware setting (as configured
            on the quantizer). See actual_activation_aware for what was used.
        act_init (str): Activation initialization mode.
        actual_activation_aware (bool): Whether activation-aware mode was actually
            used. run_mdbf falls back to non-aware mode when P != 1 or when no
            Hessian is supplied, so this may be False even if activation_aware
            is True. None means unknown (run_mdbf did not report it).
        scale_bits (int): Bit-width used to account for the FP16 amplitude scales
            when sizing the rank and reporting BPW (accounting only; does not
            change the stored dtype). 16 = FP16, 0 = binary-only.
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
    admm_outer_iters: int = None
    admm_inner_iters: int = None
    admm_reg: float = None
    admm_seed: Optional[int] = None
    use_gradient_refine: bool = None
    gradient_iters: int = None
    gradient_lr: float = None
    activation_aware: bool = None
    act_init: str = None
    scale_bits: int = None
    actual_activation_aware: bool = None
    actual_bpw: float = None
    r: int = None

    # =========================================
    # Weight reconstruction data
    # =========================================
    is_mdbf_quantized: Optional[bool] = None
    mdbf_A_sign: list = field(default_factory=list)  # [(n, r)] × P
    mdbf_B_sign: list = field(default_factory=list)  # [(r, m)] × P
    mdbf_A_amp: list = field(default_factory=list)  # [(n, l)] × P
    mdbf_B_amp: list = field(default_factory=list)  # [(m, l)] × P
    mdbf_Q_U_amp: list = field(default_factory=list)  # [(r, l)] × P
    mdbf_Q_V_amp: list = field(default_factory=list)  # [(r, l)] × P

    def _get_path_components(self) -> list[tuple[str, list]]:
        """Return the six per-path MDBF component lists with their field names."""
        return [
            ("mdbf_A_sign", self.mdbf_A_sign),
            ("mdbf_B_sign", self.mdbf_B_sign),
            ("mdbf_A_amp", self.mdbf_A_amp),
            ("mdbf_B_amp", self.mdbf_B_amp),
            ("mdbf_Q_U_amp", self.mdbf_Q_U_amp),
            ("mdbf_Q_V_amp", self.mdbf_Q_V_amp),
        ]

    def get_MDBF_params_list(self) -> list[MDBFParams]:
        """Validate stored per-path tensors and convert them to MDBFParams objects."""
        components = self._get_path_components()
        empty_fields = [name for name, values in components if len(values) == 0]
        if empty_fields:
            raise ValueError(
                "MDBFResult is missing required per-path data: " + ", ".join(empty_fields)
            )

        component_lengths = {name: len(values) for name, values in components}
        unique_lengths = set(component_lengths.values())
        if len(unique_lengths) != 1:
            details = ", ".join(f"{name}={length}" for name, length in component_lengths.items())
            raise ValueError("MDBFResult has inconsistent per-path data lengths: " + details)

        num_paths = next(iter(unique_lengths))
        if self.P != num_paths:
            raise ValueError(
                f"MDBFResult P ({self.P}) does not match per-path data length ({num_paths})"
            )

        return [
            MDBFParams(
                A_sign=self.mdbf_A_sign[p],
                B_sign=self.mdbf_B_sign[p],
                A_amp=self.mdbf_A_amp[p],
                B_amp=self.mdbf_B_amp[p],
                Q_U_amp=self.mdbf_Q_U_amp[p],
                Q_V_amp=self.mdbf_Q_V_amp[p],
            )
            for p in range(num_paths)
        ]

    def compute_dequantized_weight(self, device=None) -> torch.Tensor:
        """Compute dequantized weight from quantized parameters.

        Reconstructs W ≈ Σ_p F^(p) @ G^(p) from the stored per-pass tensors.

        Args:
            device (str or torch.device, optional): Device to compute on.

        Returns:
            Dequantized weight tensor (FP16, CPU).
        """
        from .utils import reconstruct_weight

        compute_device = torch.device(device) if device is not None else torch.device("cpu")
        params_list = self.get_MDBF_params_list()

        W = None
        for params in params_list:
            W_p = reconstruct_weight(
                params.A_sign.float().to(compute_device),
                params.B_sign.float().to(compute_device),
                params.A_amp.float().to(compute_device),
                params.B_amp.float().to(compute_device),
                params.Q_U_amp.float().to(compute_device),
                params.Q_V_amp.float().to(compute_device),
            )
            W = W_p if W is None else W + W_p

        return W.to(torch.float16).cpu()


@dataclass
class MDBF(Quantizer):
    """MDBF quantizer.

    Runs MDBF (Multi-Envelope Double Binary Factorization) quantization per layer.

    Attributes:
        flag_calibration (bool): Calibration mode flag.
        flag_hessian (bool): Hessian computation flag.
        target_bits (float): Target BPW (e.g., 1.0).
        l (int): Multi-scale (envelope) rank. l=1 collapses the envelope to rank
            one; (l, P) = (1, 1) is DBF and (1, 2) is LittleBit.
        P (int): Number of passes (1 or 2).
        svd_mode (str): SVD initialization mode ("svd" or "svd_llm").
        use_admm (bool): Whether to use ADMM optimization.
        admm_outer_iters (int): ADMM outer iterations.
        admm_inner_iters (int): ADMM inner iterations.
        admm_reg (float): ADMM regularization coefficient.
        admm_seed (Optional[int]): Random seed (int in [0, MAX_SEED]) for the
            randomized SVD initialization inside the ADMM MDBF projection. None uses
            the global RNG, so results depend on the ambient RNG state; setting an
            integer makes the ADMM phase reproducible regardless of that state.
        use_gradient_refine (bool): Whether to use gradient refinement.
        gradient_iters (int): Gradient refinement iterations.
        gradient_lr (float): Gradient refinement learning rate.
        activation_aware (bool): Whether to use activation-aware mode (P=1 only).
        act_init (str): Activation initialization mode.
        scale_bits (int): Bit-width used to account for the FP16 amplitude scales
            when sizing the rank and reporting BPW (accounting only; does not
            change the stored dtype). 16 = FP16, 0 = binary-only.
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
    # See DEFAULT_L / DEFAULT_P in utils.py for why the default is (l, P) = (2, 1).
    l: int = DEFAULT_L
    P: int = DEFAULT_P
    svd_mode: str = "svd"
    use_admm: bool = True
    admm_outer_iters: int = 260
    admm_inner_iters: int = 3
    admm_reg: float = 0.03
    admm_seed: Optional[int] = None
    use_gradient_refine: bool = False
    gradient_iters: int = 1000
    gradient_lr: float = 0.01
    activation_aware: bool = False
    act_init: str = "osvd"
    scale_bits: int = DEFAULT_SCALE_BITS
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
            admm_outer_iters: int >= 1 (when use_admm=True)
            admm_reg: float >= 0
            admm_seed: int in [0, MAX_SEED] or None
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

        if not (isinstance(self.scale_bits, int) and self.scale_bits >= 0):
            bad.append(
                f"Invalid MDBF parameter 'scale_bits': {self.scale_bits!r} (expected int >= 0)."
            )

        if not (isinstance(self.admm_reg, (int, float)) and self.admm_reg >= 0):
            bad.append(
                f"Invalid MDBF parameter 'admm_reg': {self.admm_reg!r} (expected numeric >= 0)."
            )

        if self.admm_seed is not None:
            # bool is an int subclass, but torch.Generator.manual_seed() rejects it
            # ("expected a long, but got bool"), so screen it out here.
            if (
                isinstance(self.admm_seed, bool)
                or not isinstance(self.admm_seed, int)
                or not 0 <= self.admm_seed <= MAX_SEED
            ):
                bad.append(
                    f"Invalid MDBF parameter 'admm_seed': {self.admm_seed!r} "
                    f"(expected int in [0, {MAX_SEED}] or None)."
                )

        if self.use_admm:
            if not (isinstance(self.admm_outer_iters, int) and self.admm_outer_iters >= 1):
                bad.append(
                    f"Invalid MDBF parameter 'admm_outer_iters': {self.admm_outer_iters!r} "
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
            admm_outer_iters=self.admm_outer_iters,
            admm_inner_iters=self.admm_inner_iters,
            admm_reg=self.admm_reg,
            admm_seed=self.admm_seed,
            use_gradient_refine=self.use_gradient_refine,
            gradient_iters=self.gradient_iters,
            gradient_lr=self.gradient_lr,
            activation_aware=self.activation_aware,
            act_init=self.act_init,
            nsamples=nsamples,
            scale_bits=self.scale_bits,
        )

        params_list = weight_results["mdbf_params"]
        # Reflect the actual number of paths generated by run_mdbf in mdbf_result.P.
        actual_P = len(params_list)

        mdbf_result = MDBFResult(
            # Quantization configuration parameters
            target_bits=resolved_target_bits,
            l=self.l,
            P=actual_P,
            svd_mode=self.svd_mode,
            use_admm=self.use_admm,
            admm_outer_iters=self.admm_outer_iters,
            admm_inner_iters=self.admm_inner_iters,
            admm_reg=self.admm_reg,
            admm_seed=self.admm_seed,
            use_gradient_refine=self.use_gradient_refine,
            gradient_iters=self.gradient_iters,
            gradient_lr=self.gradient_lr,
            activation_aware=self.activation_aware,
            act_init=self.act_init,
            scale_bits=self.scale_bits,
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
            # actually-used flag returned by run_mdbf. If the key is absent we
            # leave it None ("unknown") rather than assuming the requested value:
            # finalize_quant_config_for_save() skips None layers instead of
            # reporting a flag that was never confirmed.
            actual_activation_aware=weight_results.get("actual_activation_aware"),
        )

        return mdbf_result

    def get_quant_config(self) -> dict:
        """Return quantization_config dict for save_quantized_model.

        All values record the *requested* quantizer configuration.
        finalize_quant_config_for_save() additionally records
        "actual_activation_aware" (what was actually used after any
        per-layer fallback).
        """
        result: dict[str, Any] = {
            "quant_method": "mdbf",
            "bits": self.target_bits,
            "l": self.l,
            "P": self.P,
            "svd_mode": self.svd_mode,
            "use_admm": self.use_admm,
            "admm_outer_iters": self.admm_outer_iters,
            "admm_inner_iters": self.admm_inner_iters,
            "admm_reg": self.admm_reg,
            "admm_seed": self.admm_seed,
            "use_gradient_refine": self.use_gradient_refine,
            "gradient_iters": self.gradient_iters,
            "gradient_lr": self.gradient_lr,
            "activation_aware": self.activation_aware,
            "act_init": self.act_init,
            "scale_bits": self.scale_bits,
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
        """Build per-layer quantization_bits list; length is num_layers.

        The per-layer "params" entries echo the requested quantizer
        configuration from quant_config (e.g. "activation_aware" is the
        requested setting, not the per-layer outcome).
        """
        _LAYER_RE = re.compile(r"\.layers\.(\d+)\.(.*)")
        default_bits = quant_config.get("bits", 1.0)
        mlp_target_bits = get_quant_param(quant_config, "mlp_target_bits")
        module_target_bits: dict[str, float] = (
            get_quant_param(quant_config, "module_target_bits") or {}
        )
        params: dict[str, Any] = {
            # quant_config comes from get_quant_config() on the save path, so every key
            # below is present in practice; the fallbacks only cover a partial config.
            "l": get_quant_param(quant_config, "l", default=DEFAULT_L),
            "P": get_quant_param(quant_config, "P", default=DEFAULT_P),
            "svd_mode": get_quant_param(quant_config, "svd_mode", default="svd"),
            "use_admm": get_quant_param(quant_config, "use_admm", default=True),
            "admm_outer_iters": get_quant_param(quant_config, "admm_outer_iters", default=260),
            "admm_inner_iters": get_quant_param(quant_config, "admm_inner_iters", default=3),
            "admm_reg": get_quant_param(quant_config, "admm_reg", default=0.03),
            "admm_seed": get_quant_param(quant_config, "admm_seed", default=None),
            "use_gradient_refine": get_quant_param(
                quant_config, "use_gradient_refine", default=False
            ),
            "gradient_iters": get_quant_param(quant_config, "gradient_iters", default=1000),
            "gradient_lr": get_quant_param(quant_config, "gradient_lr", default=0.01),
            "activation_aware": get_quant_param(quant_config, "activation_aware", default=False),
            "act_init": get_quant_param(quant_config, "act_init", default="osvd"),
            "scale_bits": get_quant_param(quant_config, "scale_bits", default=DEFAULT_SCALE_BITS),
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
        # "activation_aware" records the *requested* setting; run_mdbf may fall
        # back to non-aware mode per layer (e.g. P != 1, or no Hessian supplied).
        # Record whether every quantized layer actually ran activation-aware so
        # the saved config reflects what was done, not just what was asked.
        layer_results = (self.results.get(name) for name in quantized_layer_names)
        actual_flags = [
            result.actual_activation_aware
            for result in layer_results
            if isinstance(result, MDBFResult) and result.actual_activation_aware is not None
        ]
        if actual_flags:
            quant_config["actual_activation_aware"] = all(actual_flags)
        return quant_config

    def create_inference_layer(self, result, linear_module, **kwargs):
        """Build MultipathMDBFLinear from MDBFResult."""
        from onecomp.quantizer.mdbf.mdbf_layer import MultipathMDBFLinear

        bias = (
            linear_module.bias
            if hasattr(linear_module, "bias") and linear_module.bias is not None
            else None
        )
        return MultipathMDBFLinear.from_quantization_result(
            result=result,
            bias=bias,
            device=linear_module.weight.device,
            use_gemlite=kwargs.get("use_gemlite"),
        )
