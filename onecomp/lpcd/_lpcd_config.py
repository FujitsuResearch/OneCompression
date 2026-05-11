"""LPCD configuration dataclass.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yudai Fujimoto

"""

from dataclasses import dataclass


@dataclass
class LPCDConfig:
    """Configuration for LPCD optimisation.

    Attributes:
        enable_qk: Optimise Query/Key projections jointly.
        enable_vo: Optimise Value/Output projections jointly.
        enable_ud: Optimise Up/Down projections jointly.
        enable_residual: Optimise residual connections (o_proj, down_proj).
        alt_steps: Number of alternating coordinate-descent steps.
        perccorr: Correction percentage for weight relaxation.
        percdamp: Damping percentage for Hessian regularisation.
        use_closed_form: Use closed-form solvers when available.
        gd_steps: Number of gradient-descent epochs per sub-problem.
        gd_batch_size: Effective batch size for gradient accumulation.
        gd_base_lr: Base learning rate for gradient-descent solver.
        device: Device to perform LPCD optimisation on.

    Examples:
        Minimal (residual correction only, fast)::

            LPCDConfig()

        All sub-modules enabled (best quality, slower)::

            LPCDConfig(
                enable_qk=True,
                enable_vo=True,
                enable_ud=True,
            )
    """

    enable_qk: bool = False
    enable_vo: bool = False
    enable_ud: bool = False
    enable_residual: bool = True

    alt_steps: int = 1
    perccorr: float = 0.5
    percdamp: float = 0.01
    use_closed_form: bool = True
    gd_steps: int = 20
    gd_batch_size: int = 16
    gd_base_lr: float = 1e-4
    device: str = "cuda:0"
    