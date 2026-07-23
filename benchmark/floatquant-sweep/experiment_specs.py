"""Shared FloatQuant experiment specifications.

The benchmark entrypoints stay intentionally flat so Slurm command files
remain stable. This module is the single place where paper-facing mode
names, command-generator profiles, and collector output order are defined.
"""

from __future__ import annotations

SWEEP_SMOKE_MODES = (
    "rtn_default",
    "rtn_static_mse",
    "gptq_default",
    "gptq_static_mse",
    "gptq_inloop_wmse",
)

SWEEP_DEFAULT_MODES = (
    "rtn_default",
    "rtn_static_mse",
    "gptq_default",
    "gptq_static_mse",
    "gptq_static_wmse",
    "gptq_inloop_wmse",
    "gptq_inloop_conditional",
    "gptq_static_mse_full",
    "gptq_inloop_conditional_adaptive",
)

SWEEP_PROFILE_FORMATS = {
    "smoke": ("nvfp4",),
    "default": ("nvfp4", "mxfp4"),
}

SWEEP_HESSIAN_MODES = tuple(mode for mode in SWEEP_DEFAULT_MODES if mode.startswith("gptq_"))

SWEEP_MODE_TO_PAPER_SUFFIX = {
    "rtn_default": "rtn_absmax",
    "rtn_static_mse": "rtn_sweep",
    "gptq_default": "hessian_absmax",
    "gptq_static_mse": "hessian_static_mse",
    "gptq_static_wmse": "hessian_static_wmse",
    "gptq_inloop_wmse": "hessian_sweep",
    "gptq_inloop_conditional": "hessian_conditional",
    "gptq_static_mse_full": "hessian_static_mse_full",
    "gptq_inloop_conditional_adaptive": "hessian_conditional_adaptive",
}


def sweep_modes_for_profile(profile: str) -> tuple[str, ...]:
    """Return Hydra sweep modes for a command-generator profile."""
    if profile == "smoke":
        return SWEEP_SMOKE_MODES
    if profile == "default":
        return SWEEP_DEFAULT_MODES
    raise ValueError(f"Unknown sweep profile: {profile!r}")


def sweep_formats_for_profile(profile: str) -> tuple[str, ...]:
    """Return FloatQuant formats for a command-generator profile."""
    try:
        return SWEEP_PROFILE_FORMATS[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown sweep profile: {profile!r}") from exc


REAL_KERNEL_MODE_SPECS = {
    "fp16": {"fmt": None, "paper_tag": "fp16"},
    "nvfp4_rtn_absmax": {
        "fmt": "nvfp4",
        "use_hessian": False,
        "scale_timing": "none",
        "scale_objective": "auto",
        "candidate": "local",
        "paper_tag": "nvfp4_w4a16",
    },
    "nvfp4_rtn_sweep": {
        "fmt": "nvfp4",
        "use_hessian": False,
        "scale_timing": "static",
        "scale_objective": "mse",
        "candidate": "local",
        "paper_tag": "nvfp4_sweep",
    },
    "nvfp4_gptq_absmax": {
        "fmt": "nvfp4",
        "use_hessian": True,
        "scale_timing": "none",
        "scale_objective": "auto",
        "candidate": "local",
        "paper_tag": "nvfp4_hessian",
    },
    "nvfp4_gptq_inloop_wmse": {
        "fmt": "nvfp4",
        "use_hessian": True,
        "scale_timing": "in_loop",
        "scale_objective": "diag_wmse",
        "candidate": "local",
        "paper_tag": "nvfp4_qep_hessian_sweep",
    },
    "nvfp4_gptq_conditional": {
        "fmt": "nvfp4",
        "use_hessian": True,
        "scale_timing": "in_loop",
        "scale_objective": "conditional",
        "candidate": "local",
        "paper_tag": "nvfp4_qep_hessian_conditional",
    },
    "nvfp4_gptq_conditional_adaptive": {
        "fmt": "nvfp4",
        "use_hessian": True,
        "scale_timing": "in_loop",
        "scale_objective": "conditional",
        "candidate": "adaptive",
        "paper_tag": "nvfp4_qep_hessian_conditional_adaptive",
    },
    "mxfp4_rtn_ceil": {
        "fmt": "mxfp4",
        "use_hessian": False,
        "scale_timing": "none",
        "scale_objective": "auto",
        "candidate": "local",
        "paper_tag": "mxfp4",
    },
    "mxfp4_rtn_sweep": {
        "fmt": "mxfp4",
        "use_hessian": False,
        "scale_timing": "static",
        "scale_objective": "mse",
        "candidate": "local",
        "paper_tag": "mxfp4_sweep",
    },
    "mxfp4_gptq_inloop_wmse": {
        "fmt": "mxfp4",
        "use_hessian": True,
        "scale_timing": "in_loop",
        "scale_objective": "diag_wmse",
        "candidate": "local",
        "paper_tag": "mxfp4_qep_hessian_sweep",
    },
    "mxfp4_gptq_conditional_adaptive": {
        "fmt": "mxfp4",
        "use_hessian": True,
        "scale_timing": "in_loop",
        "scale_objective": "conditional",
        "candidate": "adaptive",
        "paper_tag": "mxfp4_qep_hessian_conditional_adaptive",
    },
    "fp8_rtn": {
        "fmt": "fp8",
        "use_hessian": False,
        "scale_timing": "none",
        "scale_objective": "auto",
        "candidate": "local",
        "paper_tag": "fp8",
    },
    "fp8_gptq": {
        "fmt": "fp8",
        "use_hessian": True,
        "scale_timing": "none",
        "scale_objective": "auto",
        "candidate": "local",
        "paper_tag": "fp8_hessian",
    },
}

REAL_KERNEL_SMOKE_MODES = ("fp16", "nvfp4_rtn_sweep")
REAL_KERNEL_NATIVE_MODES = (
    "fp16",
    "nvfp4_rtn_absmax",
    "nvfp4_rtn_sweep",
    "mxfp4_rtn_ceil",
    "mxfp4_rtn_sweep",
    "fp8_rtn",
)
REAL_KERNEL_FULL_MODES = tuple(REAL_KERNEL_MODE_SPECS)

REAL_KERNEL_QEP_COMPATIBLE_MODES = frozenset(
    mode for mode, spec in REAL_KERNEL_MODE_SPECS.items() if spec.get("use_hessian")
)
REAL_KERNEL_W4A4_CAPABLE_MODES = frozenset(
    mode for mode, spec in REAL_KERNEL_MODE_SPECS.items() if spec.get("fmt") == "nvfp4"
)

REAL_KERNEL_NATIVE_05_PAPER_KEYS = {
    "fp16": "fp16_baseline",
    "nvfp4_rtn_absmax": "nvfp4_absmax",
    "nvfp4_rtn_sweep": "nvfp4_sweep",
    "nvfp4_gptq_absmax": "nvfp4_hessian",
    "nvfp4_rtn_sweep_w4a4": "nvfp4_w4a4",
    "mxfp4_rtn_ceil": "mxfp4_ceil",
    "mxfp4_rtn_sweep": "mxfp4_sweep",
    "fp8_rtn": "fp8",
}

REAL_KERNEL_NATIVE_7B_PAPER_KEYS = {
    "fp16": "fp16",
    "nvfp4_rtn_absmax": "nvfp4_absmax",
    "nvfp4_rtn_sweep": "nvfp4_w4a16",
    "nvfp4_rtn_sweep_w4a4": "nvfp4_w4a4",
    "mxfp4_rtn_ceil": "mxfp4",
    "fp8_rtn": "fp8",
}


def real_kernel_modes_for_profile(profile: str) -> tuple[str, ...]:
    """Return real-kernel modes for a command-generator profile."""
    if profile == "smoke":
        return REAL_KERNEL_SMOKE_MODES
    if profile == "native":
        return REAL_KERNEL_NATIVE_MODES
    if profile == "full":
        return REAL_KERNEL_FULL_MODES
    raise ValueError(f"Unknown real-kernel profile: {profile!r}")


def real_kernel_record_key(mode: str, qep: bool = False, w4a4: bool = False) -> str:
    """Return the collector key produced by a real-kernel benchmark record."""
    key = mode
    if qep:
        key += "_qep"
    if w4a4:
        key += "_w4a4"
    return key


def _build_real_kernel_full_record_order() -> tuple[str, ...]:
    order: list[str] = []
    for mode in REAL_KERNEL_FULL_MODES:
        qep_values = (False, True) if mode in REAL_KERNEL_QEP_COMPATIBLE_MODES else (False,)
        for qep in qep_values:
            order.append(real_kernel_record_key(mode, qep=qep))
            if mode in REAL_KERNEL_W4A4_CAPABLE_MODES:
                order.append(real_kernel_record_key(mode, qep=qep, w4a4=True))
    return tuple(order)


REAL_KERNEL_FULL_RECORD_ORDER = _build_real_kernel_full_record_order()

__all__ = [
    "REAL_KERNEL_FULL_MODES",
    "REAL_KERNEL_FULL_RECORD_ORDER",
    "REAL_KERNEL_MODE_SPECS",
    "REAL_KERNEL_NATIVE_05_PAPER_KEYS",
    "REAL_KERNEL_NATIVE_7B_PAPER_KEYS",
    "REAL_KERNEL_NATIVE_MODES",
    "REAL_KERNEL_QEP_COMPATIBLE_MODES",
    "REAL_KERNEL_SMOKE_MODES",
    "REAL_KERNEL_W4A4_CAPABLE_MODES",
    "SWEEP_DEFAULT_MODES",
    "SWEEP_HESSIAN_MODES",
    "SWEEP_MODE_TO_PAPER_SUFFIX",
    "SWEEP_PROFILE_FORMATS",
    "SWEEP_SMOKE_MODES",
    "real_kernel_modes_for_profile",
    "real_kernel_record_key",
    "sweep_formats_for_profile",
    "sweep_modes_for_profile",
]
