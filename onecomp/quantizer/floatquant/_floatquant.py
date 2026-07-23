"""FloatQuant (NVFP4 / MXFP4 / FP8) quantizer classes

This module defines the FloatQuant quantizer class and result class for
microscaling floating-point fake-quantization. The quantizer supports
the NVFP4, MXFP4, and FP8 (E4M3) formats, either with direct
round-to-nearest quantization or with GPTQ-style Hessian-based
column-wise error compensation.

Classes:
    FloatQuantResult: Result class for FloatQuant quantization containing codes and scales.
    FloatQuant: FloatQuant quantizer class supporting NVFP4 / MXFP4 / FP8 formats.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import gc
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn
from transformers import Conv1D

from onecomp.quantizer._quantizer import QuantizationResult, Quantizer
from onecomp.quantizer.floatquant.config import (
    DEFAULT_BLOCK_SIZES,
    SUPPORTED_FORMATS,
    SUPPORTED_SCALE_CANDIDATE_STRATEGIES,
    SUPPORTED_SCALE_OBJECTIVES,
    SUPPORTED_SCALE_TIMINGS,
)
from onecomp.quantizer.floatquant.formats import (
    E2M1_MAX,
    E4M3_MAX,
    E4M3_MIN_SUBNORMAL,
    _e4m3_neighbor_scales,
    dequantize_from_grid,
    e2m1_grid,
    e8m0_block_scale,
    fp8_dequantize,
    fp8_quantize,
    mxfp4_dequantize,
    mxfp4_quantize,
    nvfp4_dequantize,
    nvfp4_quantize,
    quantize_to_grid,
    round_to_e4m3,
)
from onecomp.quantizer.gptq._gptq import _compute_inverse_hessian


@dataclass
class FloatQuantResult(QuantizationResult):
    """Result class for FloatQuant quantization.

    Inherits from QuantizationResult and adds FloatQuant-specific parameters.
    Codes and scales are stored in the layout used for the matrix product,
    i.e. (out_features, in_features).

    Attributes:
        dequantized_weight (torch.Tensor): Dequantized weights (original
            dtype, CPU) - inherited from parent class.
        fmt (str): Format used ("nvfp4", "mxfp4", or "fp8").
        block_size (int): Block size along the input dimension
            (-1 means per-channel, used by fp8).
        codes (torch.Tensor, optional): E2M1 grid indices (torch.int8, CPU).
            None for fp8.
        block_scales (torch.Tensor, optional): Block scales (FP32, CPU).
            For nvfp4: decoded E4M3 scales of shape
            (out_features, in_features // block_size).
            For mxfp4: E8M0 power-of-two scales of the same shape.
            For fp8: per-channel scales of shape (out_features, 1).
        tensor_scale (torch.Tensor, optional): Per-tensor FP32 scale
            (0-dim tensor). Only used by nvfp4.
        weight_transposed (bool): True when the source module stores the
            transposed weight (e.g. ``transformers.Conv1D``), so the
            stored ``dequantized_weight`` is ``(in, out)`` while
            :meth:`compute_dequantized_weight` returns ``(out, in)``.
    """

    # =========================================
    # Quantization configuration parameters
    # =========================================
    fmt: str = None
    block_size: int = None

    # =========================================
    # Weight reconstruction data
    # =========================================
    codes: Optional[torch.Tensor] = None  # E2M1 grid indices (int8)
    block_scales: Optional[torch.Tensor] = None  # Block / per-channel scales
    tensor_scale: Optional[torch.Tensor] = None  # Per-tensor scale (nvfp4)
    weight_transposed: bool = False  # Module stores W^T (e.g. Conv1D)

    def compute_dequantized_weight(self, device: torch.device = None) -> torch.Tensor:
        """Compute dequantized weight from codes and scales.

        Falls back to the stored ``dequantized_weight`` when codes are
        not available (fp8 format).

        Args:
            device (torch.device, optional): Device to compute on.

        Returns:
            torch.Tensor: Dequantized weight tensor (FP16, CPU) in
            quantization layout ``(out_features, in_features)``,
            regardless of the source module's storage layout.
        """
        if self.codes is None:
            weight = super().compute_dequantized_weight(device)
            if self.weight_transposed:
                # dequantized_weight mirrors the module layout (in, out);
                # restore the (out, in) contract of this method.
                weight = weight.t().contiguous()
            return weight

        compute_device = torch.device(device) if device is not None else self.codes.device
        codes = self.codes.to(compute_device)
        block_scales = self.block_scales.to(compute_device)

        if self.fmt == "nvfp4":
            tensor_scale = self.tensor_scale.to(compute_device)
            dequantized = nvfp4_dequantize(codes, block_scales, tensor_scale, self.block_size)
        else:  # mxfp4
            dequantized = mxfp4_dequantize(codes, block_scales, self.block_size)

        return dequantized.to(torch.float16).cpu()


@dataclass
class FloatQuant(Quantizer):
    """FloatQuant (NVFP4 / MXFP4 / FP8) microscaling fake-quantizer.

    Quantizes weights to microscaling floating-point formats targeting
    NVIDIA Blackwell-generation hardware:

    - ``nvfp4``: FP4 (E2M1) elements, per-block (16) FP8 E4M3 scales, and
      one per-tensor FP32 scale (two-level scaling, NVIDIA definition).
    - ``mxfp4``: FP4 (E2M1) elements with per-block (32) E8M0 power-of-two
      scales (OCP Microscaling specification).
    - ``fp8``: FP8 E4M3 elements with per-channel FP32 scales.

    When ``use_hessian=False``, weights are quantized directly with
    round-to-nearest onto the target format (RTN-style, no calibration
    data required). When ``use_hessian=True``, a GPTQ-style column-wise
    error-compensation loop is used with rounding onto the floating-point
    grid, which requires calibration data for the Hessian.

    Attributes:
        fmt (str): Target format, one of "nvfp4", "mxfp4", or "fp8".
            Default is "nvfp4".
        block_size (int): Block size along the input dimension. If None,
            resolved per format: 16 for nvfp4, 32 for mxfp4, and -1
            (per-channel) for fp8. Default is None.
        use_hessian (bool): If True, use GPTQ-style error-compensated
            quantization (requires calibration data). Default is False.
        scale_search (bool): Backward-compatible switch for enabling
            scale sweeps. With direct quantization it resolves to a
            static MSE sweep; with Hessian quantization it resolves to
            the original in-loop diagonal-WMSE sweep. Ignored for fp8.
            Default is False.
        scale_timing (str): When scale sweeps are enabled, controls when
            scales are selected: ``"static"`` selects them once before
            the GPTQ loop, while ``"in_loop"`` reselects them at physical
            block boundaries from compensated weights. ``"auto"``
            preserves the old ``scale_search`` behavior. Default is
            ``"auto"``.
        scale_objective (str): Scale-sweep objective: ``"mse"``,
            ``"diag_wmse"``, or ``"conditional"``. The conditional
            objective uses the Schur-complement block metric
            ``((H^{-1})_BB)^{-1}``, matching GPTQ's future compensation
            cost. ``"auto"`` resolves to MSE for static/direct sweeps
            and diagonal WMSE for Hessian in-loop sweeps. Default is
            ``"auto"``.
        scale_candidate_strategy (str): Candidate scale set for sweeps.
            ``"local"`` preserves the ScaleSweep-style local windows,
            ``"full"`` evaluates the full positive FP8/E8M0 grid, and
            ``"adaptive"`` starts from the local window and expands if
            the best scale lies on a window boundary. Default is
            ``"local"``.
        blocksize (int): Number of columns processed together in the
            error-compensation loop (only used when ``use_hessian=True``).
            Default is 128.
        percdamp (float): Percentage of the Hessian diagonal average added
            for numerical stability (only used when ``use_hessian=True``).
            Default is 0.01.

    Example:
        >>> from onecomp.quantizer.floatquant import FloatQuant
        >>> quantizer = FloatQuant(fmt="nvfp4")
        >>> quantizer = FloatQuant(fmt="mxfp4", use_hessian=True)
        >>> quantizer = FloatQuant(fmt="nvfp4", scale_search=True)
    """

    fmt: str = "nvfp4"
    block_size: Optional[int] = None
    use_hessian: bool = False
    scale_search: bool = False
    scale_timing: str = "auto"
    scale_objective: str = "auto"
    scale_candidate_strategy: str = "local"
    blocksize: int = 128
    percdamp: float = 0.01

    def __post_init__(self):
        if self.name is None:
            self.name = f"FP4_{self.fmt}"
        if self.block_size is None:
            self.block_size = DEFAULT_BLOCK_SIZES.get(self.fmt)
        # Calibration / Hessian flags follow use_hessian
        self.flag_calibration = bool(self.use_hessian)
        self.flag_hessian = bool(self.use_hessian)
        super().__post_init__()

    def validate_params(self):
        """Validate FloatQuant parameters once in setup().

        Validated ranges:
            fmt: one of "nvfp4", "mxfp4", "fp8"
            block_size: int >= 1 (nvfp4 / mxfp4), -1 (fp8)
            use_hessian: bool
            scale_timing: "auto", "none", "static", or "in_loop"
            scale_objective: "auto", "mse", "diag_wmse", or "conditional"
            scale_candidate_strategy: "local", "full", or "adaptive"
            blocksize: int >= 1 (when use_hessian=True)
            percdamp: float >= 3.95e-4 (when use_hessian=True)
        """
        bad = []

        if self.fmt not in SUPPORTED_FORMATS:
            bad.append(
                f"Invalid FloatQuant parameter 'fmt': {self.fmt!r} "
                f"(expected one of {SUPPORTED_FORMATS})."
            )
        elif self.fmt == "fp8":
            if self.block_size != -1:
                bad.append(
                    f"Invalid FloatQuant parameter 'block_size': {self.block_size!r} "
                    f"(fp8 uses per-channel scales; expected -1 or None)."
                )
        else:
            if not (isinstance(self.block_size, int) and self.block_size >= 1):
                bad.append(
                    f"Invalid FloatQuant parameter 'block_size': {self.block_size!r} "
                    f"(expected int >= 1 for fmt={self.fmt!r})."
                )

        if not isinstance(self.use_hessian, bool):
            bad.append(
                f"Invalid FloatQuant parameter 'use_hessian': {self.use_hessian!r} (expected bool)."
            )

        if not isinstance(self.scale_search, bool):
            bad.append(
                f"Invalid FloatQuant parameter 'scale_search': "
                f"{self.scale_search!r} (expected bool)."
            )

        if self.scale_timing not in SUPPORTED_SCALE_TIMINGS:
            bad.append(
                f"Invalid FloatQuant parameter 'scale_timing': {self.scale_timing!r} "
                f"(expected one of {SUPPORTED_SCALE_TIMINGS})."
            )

        if self.scale_objective not in SUPPORTED_SCALE_OBJECTIVES:
            bad.append(
                f"Invalid FloatQuant parameter 'scale_objective': "
                f"{self.scale_objective!r} (expected one of {SUPPORTED_SCALE_OBJECTIVES})."
            )

        if self.scale_candidate_strategy not in SUPPORTED_SCALE_CANDIDATE_STRATEGIES:
            bad.append(
                f"Invalid FloatQuant parameter 'scale_candidate_strategy': "
                f"{self.scale_candidate_strategy!r} "
                f"(expected one of {SUPPORTED_SCALE_CANDIDATE_STRATEGIES})."
            )

        if self.scale_objective in ("diag_wmse", "conditional") and not self.use_hessian:
            bad.append(
                f"Invalid FloatQuant parameter 'scale_objective': "
                f"{self.scale_objective!r} requires use_hessian=True."
            )

        if self.scale_objective == "conditional" and self.scale_timing == "static":
            bad.append(
                "Invalid FloatQuant parameters: scale_objective='conditional' "
                "currently requires scale_timing='in_loop' or 'auto'."
            )

        if self.use_hessian:
            if not (isinstance(self.blocksize, int) and self.blocksize >= 1):
                bad.append(
                    f"Invalid FloatQuant parameter 'blocksize': {self.blocksize!r} "
                    f"(expected int >= 1 when use_hessian=True)."
                )
            if not (isinstance(self.percdamp, (int, float)) and self.percdamp >= 3.95e-4):
                bad.append(
                    f"Invalid FloatQuant parameter 'percdamp': {self.percdamp!r} "
                    f"(expected numeric >= 3.95e-4 when use_hessian=True)."
                )

        if bad:
            raise ValueError("; ".join(bad))

    def quantize_layer(self, module, input=None, hessian=None):
        """Quantize a layer to the configured microscaling format.

        Args:
            module (torch.nn.Module): The layer module to quantize.
            input (tuple or torch.Tensor, optional): Input tensor (not
                used directly; the Hessian is derived from it upstream).
            hessian (torch.Tensor, optional): Hessian matrix (X^T X).
                Required when ``use_hessian=True``.

        Returns:
            FloatQuantResult: FloatQuant quantization result object containing codes,
                scales, and the dequantized weight.

        Raises:
            ValueError: If block_size does not divide in_features, or if
                ``use_hessian=True`` and no Hessian is provided.
        """
        matrix_W = module.weight.data.clone()
        if isinstance(module, nn.Conv2d):
            matrix_W = matrix_W.flatten(1)
        if isinstance(module, Conv1D):
            matrix_W = matrix_W.t()
        matrix_W = matrix_W.float()

        in_features = matrix_W.shape[1]
        if self.fmt in ("nvfp4", "mxfp4") and in_features % self.block_size != 0:
            raise ValueError(
                f"block_size={self.block_size} does not divide in_features={in_features}."
            )

        if self.use_hessian:
            if hessian is None:
                raise ValueError("use_hessian=True requires a Hessian matrix.")
            codes, block_scales, tensor_scale, dequantized = _run_fp4_gptq(
                hessian,
                matrix_W,
                fmt=self.fmt,
                block_size=self.block_size,
                blocksize=self.blocksize,
                percdamp=self.percdamp,
                scale_search=self.scale_search,
                scale_timing=self.scale_timing,
                scale_objective=self.scale_objective,
                scale_candidate_strategy=self.scale_candidate_strategy,
            )
        elif self.fmt == "nvfp4":
            direct_scale_search = _scale_search_enabled(self.scale_search, self.scale_timing)
            if self.scale_candidate_strategy == "local":
                codes, block_scales, tensor_scale = nvfp4_quantize(
                    matrix_W, self.block_size, scale_search=direct_scale_search
                )
                dequantized = nvfp4_dequantize(codes, block_scales, tensor_scale, self.block_size)
            else:
                codes, block_scales, tensor_scale, dequantized = _run_fp4_direct(
                    matrix_W,
                    fmt=self.fmt,
                    block_size=self.block_size,
                    scale_search=self.scale_search,
                    scale_timing=self.scale_timing,
                    scale_candidate_strategy=self.scale_candidate_strategy,
                )
        elif self.fmt == "mxfp4":
            direct_scale_search = _scale_search_enabled(self.scale_search, self.scale_timing)
            if self.scale_candidate_strategy == "local":
                codes, block_scales = mxfp4_quantize(
                    matrix_W, self.block_size, scale_search=direct_scale_search
                )
                tensor_scale = None
                dequantized = mxfp4_dequantize(codes, block_scales, self.block_size)
            else:
                codes, block_scales, tensor_scale, dequantized = _run_fp4_direct(
                    matrix_W,
                    fmt=self.fmt,
                    block_size=self.block_size,
                    scale_search=self.scale_search,
                    scale_timing=self.scale_timing,
                    scale_candidate_strategy=self.scale_candidate_strategy,
                )
        else:  # fp8
            values, block_scales = fp8_quantize(matrix_W, per_channel=True)
            codes = None
            tensor_scale = None
            dequantized = fp8_dequantize(values, block_scales)

        if isinstance(module, Conv1D):
            dequantized = dequantized.t()

        dequantized_weight = (
            dequantized.reshape(module.weight.shape).to(module.weight.data.dtype).cpu()
        )

        del matrix_W
        gc.collect()
        torch.cuda.empty_cache()

        return FloatQuantResult(
            dequantized_weight=dequantized_weight,
            fmt=self.fmt,
            block_size=self.block_size,
            codes=codes.cpu() if codes is not None else None,
            block_scales=block_scales.cpu(),
            tensor_scale=tensor_scale.cpu() if tensor_scale is not None else None,
            weight_transposed=isinstance(module, Conv1D),
        )

    def get_quant_config(self) -> dict:
        """Return quantization_config dict for save_quantized_model.

        ``quant_method`` is the dedicated name ``"onecomp_fake_quant"`` so
        the vLLM plugin never has to override vLLM's built-in ``fp8`` /
        ``mxfp4`` handlers; the microscaling format that produced the
        weights is recorded in ``fmt``. Weights themselves are stored
        fake-quantized (dequantized FP16), hence
        ``checkpoint_format="fake_quant"``.
        """
        config: dict[str, Any] = {
            "quant_method": "onecomp_fake_quant",
            "fmt": self.fmt,
            "use_hessian": self.use_hessian,
            "scale_search": self.scale_search,
            "scale_timing": self.scale_timing,
            "scale_objective": self.scale_objective,
            "scale_candidate_strategy": self.scale_candidate_strategy,
            "checkpoint_format": "fake_quant",
        }
        if self.fmt in ("nvfp4", "mxfp4"):
            config["block_size"] = self.block_size
            config["group_size"] = self.block_size
        return config

    def create_inference_layer(self, result, linear_module, **kwargs):
        """Build a fake-quant inference layer from an FloatQuantResult.

        Returns a plain ``nn.Linear`` carrying the dequantized FP16
        weights, so the standard save path can serialize the model.
        """
        _ = kwargs
        weight = result.compute_dequantized_weight()
        out_features, in_features = weight.shape
        has_bias = getattr(linear_module, "bias", None) is not None
        layer = nn.Linear(in_features, out_features, bias=has_bias, dtype=torch.float16)
        layer.weight.data.copy_(weight)
        if has_bias:
            layer.bias.data.copy_(linear_module.bias.data.to(torch.float16))
        return layer.to(linear_module.weight.device)


def _select_block_scale(
    block: torch.Tensor,
    fmt: str,
    tensor_scale: Optional[torch.Tensor],
    metric: Optional[torch.Tensor],
    sweep: bool = False,
    candidate_strategy: str = "local",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the scale for one scale block inside the GPTQ loop.

    Args:
        block (torch.Tensor): Error-compensated weights of the current
            scale block, shape (out_features, block_size).
        fmt (str): "nvfp4" or "mxfp4".
        tensor_scale (torch.Tensor, optional): Per-tensor scale (nvfp4).
        metric (torch.Tensor, optional): None for MSE/AbsMax, a vector
            of shape ``(block_size,)`` for diagonal WMSE, or a full block
            metric of shape ``(block_size, block_size)`` for the
            conditional Hessian objective.
        sweep (bool): If True, evaluate local scale candidates. If
            False, use the format default AbsMax / ceil scale.
        candidate_strategy (str): ``"local"``, ``"full"``, or
            ``"adaptive"`` candidate set for sweeps.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (stored scale, effective
        scale used for element rounding), both of shape (out_features,).
    """
    if sweep:
        return _sweep_one_block_scales(block, fmt, tensor_scale, metric, candidate_strategy)

    block_amax = block.abs().amax(dim=-1)
    if fmt == "nvfp4":
        scale = round_to_e4m3(block_amax / (E2M1_MAX * tensor_scale))
        scale = torch.clamp(scale, min=E4M3_MIN_SUBNORMAL)
        return scale, scale * tensor_scale
    scale = e8m0_block_scale(block_amax)
    return scale, scale


def _scale_search_enabled(scale_search: bool, scale_timing: str) -> bool:
    """Resolve whether a sweep should run for the current configuration."""
    if scale_timing == "none":
        return False
    return scale_search or scale_timing in ("static", "in_loop")


def _resolve_scale_timing(scale_search: bool, scale_timing: str, use_hessian: bool) -> str:
    """Resolve the user-facing timing knobs to an execution mode."""
    if not _scale_search_enabled(scale_search, scale_timing):
        return "none"
    if scale_timing != "auto":
        return scale_timing
    return "in_loop" if use_hessian else "static"


def _resolve_scale_objective(scale_objective: str, timing: str, use_hessian: bool) -> str:
    """Resolve the user-facing objective knob to an execution objective."""
    if timing == "none":
        return "mse"
    if scale_objective != "auto":
        return scale_objective
    return "diag_wmse" if use_hessian and timing == "in_loop" else "mse"


def _block_scale_error(
    block: torch.Tensor,
    effective_scale: torch.Tensor,
    grid: torch.Tensor,
    metric: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return per-row reconstruction cost for one candidate block scale."""
    codes = quantize_to_grid(block / effective_scale.unsqueeze(-1), grid)
    dequantized = dequantize_from_grid(codes, grid) * effective_scale.unsqueeze(-1)
    err = block - dequantized
    if metric is None:
        return err.square().sum(dim=-1)
    if metric.dim() == 1:
        return (err.square() * metric.reshape(1, -1)).sum(dim=-1)
    return (err.matmul(metric) * err).sum(dim=-1)


def _nvfp4_local_window(metric: Optional[torch.Tensor]) -> tuple[int, int]:
    """Return ScaleSweep-compatible local E4M3 offset bounds."""
    return (8 if metric is not None else 3), 7


def _sweep_nvfp4_offsets(
    block: torch.Tensor,
    tensor_scale: torch.Tensor,
    metric: Optional[torch.Tensor],
    offset_down: int,
    offset_up: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sweep NVFP4 E4M3-neighbor offsets and return best scale/offset."""
    grid = e2m1_grid(device=block.device)
    block_amax = block.abs().amax(dim=-1)
    base = round_to_e4m3(block_amax / (E2M1_MAX * tensor_scale))
    base = torch.clamp(base, min=E4M3_MIN_SUBNORMAL)
    best_scale = base
    best_effective = base * tensor_scale
    best_err = _block_scale_error(block, best_effective, grid, metric)
    best_offset = torch.zeros_like(best_err)
    for offset in range(-offset_down, offset_up + 1):
        if offset == 0:
            continue
        candidate = _e4m3_neighbor_scales(base, offset)
        effective = candidate * tensor_scale
        err = _block_scale_error(block, effective, grid, metric)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, candidate, best_scale)
        best_effective = torch.where(better, effective, best_effective)
        best_offset = torch.where(better, torch.full_like(best_offset, float(offset)), best_offset)
    return best_scale, best_effective, best_offset


def _sweep_mxfp4_offsets(
    block: torch.Tensor,
    metric: Optional[torch.Tensor],
    offset_down: int,
    offset_up: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sweep MXFP4 E8M0 offsets around the ceil scale."""
    grid = e2m1_grid(device=block.device)
    block_amax = block.abs().amax(dim=-1)
    base = e8m0_block_scale(block_amax)
    base_exp = torch.round(torch.log2(base))
    best_scale = base
    best_err = _block_scale_error(block, base, grid, metric)
    best_offset = torch.zeros_like(best_err)
    for offset in range(-offset_down, offset_up + 1):
        if offset == 0:
            continue
        exponent = torch.clamp(base_exp + offset, min=-127.0, max=127.0)
        candidate = torch.exp2(exponent)
        err = _block_scale_error(block, candidate, grid, metric)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, candidate, best_scale)
        best_offset = torch.where(better, torch.full_like(best_offset, float(offset)), best_offset)
    return best_scale, best_offset


def _sweep_mxfp4_full_grid(
    block: torch.Tensor,
    metric: Optional[torch.Tensor],
) -> torch.Tensor:
    """Evaluate every representable E8M0 scale for MXFP4 ablations."""
    grid = e2m1_grid(device=block.device)
    block_amax = block.abs().amax(dim=-1)
    best_scale = e8m0_block_scale(block_amax)
    best_err = _block_scale_error(block, best_scale, grid, metric)
    for exponent in range(-127, 128):
        candidate = torch.full_like(best_scale, 2.0**exponent)
        err = _block_scale_error(block, candidate, grid, metric)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, candidate, best_scale)
    return best_scale


def _sweep_one_block_scales(
    block: torch.Tensor,
    fmt: str,
    tensor_scale: Optional[torch.Tensor],
    metric: Optional[torch.Tensor],
    candidate_strategy: str = "local",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sweep the scale candidates for one physical FP4 block.

    The metric may be diagonal or full. The latter is used for the
    conditional Schur-complement objective. Candidate strategies expose
    the reviewer-critical distinction between local, full-grid, and
    adaptively expanded scale searches.
    """
    if fmt == "nvfp4":
        offset_down, offset_up = _nvfp4_local_window(metric)
        if candidate_strategy == "full":
            offset_down = offset_up = 125
        if candidate_strategy != "adaptive":
            best_scale, best_effective, _ = _sweep_nvfp4_offsets(
                block, tensor_scale, metric, offset_down, offset_up
            )
            return best_scale, best_effective

        while True:
            best_scale, best_effective, best_offset = _sweep_nvfp4_offsets(
                block, tensor_scale, metric, offset_down, offset_up
            )
            hit_lower = bool((best_offset == -offset_down).any()) and offset_down < 125
            hit_upper = bool((best_offset == offset_up).any()) and offset_up < 125
            if not (hit_lower or hit_upper):
                return best_scale, best_effective
            if hit_lower:
                offset_down = min(125, max(offset_down + 1, offset_down * 2))
            if hit_upper:
                offset_up = min(125, max(offset_up + 1, offset_up * 2))

    if candidate_strategy == "full":
        best_scale = _sweep_mxfp4_full_grid(block, metric)
        return best_scale, best_scale
    offset_down, offset_up = 2, 1
    if candidate_strategy != "adaptive":
        best_scale, _ = _sweep_mxfp4_offsets(block, metric, offset_down, offset_up)
        return best_scale, best_scale

    while True:
        best_scale, best_offset = _sweep_mxfp4_offsets(block, metric, offset_down, offset_up)
        hit_lower = bool((best_offset == -offset_down).any()) and offset_down < 254
        hit_upper = bool((best_offset == offset_up).any()) and offset_up < 254
        if not (hit_lower or hit_upper):
            return best_scale, best_scale
        if hit_lower:
            offset_down = min(254, max(offset_down + 1, offset_down * 2))
        if hit_upper:
            offset_up = min(254, max(offset_up + 1, offset_up * 2))


def _conditional_block_metric(
    hessian_inverse_chol: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Compute ``((H^{-1})_BB)^{-1}`` from GPTQ's inverse-Hessian factor."""
    cols = hessian_inverse_chol[:, start:stop]
    block_inverse = cols.T.matmul(cols)
    eye = torch.eye(block_inverse.shape[0], device=block_inverse.device, dtype=block_inverse.dtype)
    jitter = torch.finfo(block_inverse.dtype).eps * block_inverse.diag().abs().mean().clamp_min(
        1.0
    )
    return torch.linalg.inv(block_inverse + jitter * eye)


def _static_block_scales(
    matrix_W: torch.Tensor,
    fmt: str,
    block_size: int,
    tensor_scale: Optional[torch.Tensor],
    objective: str,
    hessian_diag: Optional[torch.Tensor],
    candidate_strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preselect all block scales before GPTQ compensation begins."""
    out_features, columns = matrix_W.shape
    num_blocks = columns // block_size
    block_scales = torch.zeros(out_features, num_blocks, device=matrix_W.device)
    effective_scales = torch.zeros_like(block_scales)
    for block_idx, col in enumerate(range(0, columns, block_size)):
        metric = None
        if objective == "diag_wmse":
            metric = hessian_diag[col : col + block_size]
        scale, effective = _select_block_scale(
            matrix_W[:, col : col + block_size],
            fmt,
            tensor_scale,
            metric,
            sweep=True,
            candidate_strategy=candidate_strategy,
        )
        block_scales[:, block_idx] = scale
        effective_scales[:, block_idx] = effective
    return block_scales, effective_scales


def _run_fp4_direct(
    matrix_W: torch.Tensor,
    fmt: str,
    block_size: int,
    scale_search: bool = False,
    scale_timing: str = "auto",
    scale_candidate_strategy: str = "local",
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Direct FP4 quantization with non-local candidate strategies."""
    out_features, columns = matrix_W.shape
    num_blocks = columns // block_size
    block_scales = torch.zeros(out_features, num_blocks, device=matrix_W.device)
    codes = torch.zeros(out_features, columns, dtype=torch.int8, device=matrix_W.device)
    dequantized = torch.zeros_like(matrix_W)
    tensor_scale = None
    if fmt == "nvfp4":
        global_amax = matrix_W.abs().amax()
        tensor_scale = global_amax / (E4M3_MAX * E2M1_MAX)
        if tensor_scale == 0:
            tensor_scale = torch.ones_like(tensor_scale)
    grid = e2m1_grid(device=matrix_W.device)
    timing = _resolve_scale_timing(scale_search, scale_timing, use_hessian=False)
    for block_idx, col in enumerate(range(0, columns, block_size)):
        block = matrix_W[:, col : col + block_size]
        scale, effective_scale = _select_block_scale(
            block,
            fmt,
            tensor_scale,
            metric=None,
            sweep=timing != "none",
            candidate_strategy=scale_candidate_strategy,
        )
        block_scales[:, block_idx] = scale
        block_codes = quantize_to_grid(block / effective_scale.unsqueeze(-1), grid)
        codes[:, col : col + block_size] = block_codes
        dequantized[:, col : col + block_size] = dequantize_from_grid(
            block_codes, grid
        ) * effective_scale.unsqueeze(-1)
    return codes, block_scales, tensor_scale, dequantized


def _run_fp4_gptq(  # pylint: disable=too-many-locals, too-many-statements
    hessian: torch.Tensor,
    matrix_W: torch.Tensor,
    fmt: str,
    block_size: int,
    blocksize: int = 128,
    percdamp: float = 0.01,
    scale_search: bool = False,
    scale_timing: str = "auto",
    scale_objective: str = "auto",
    scale_candidate_strategy: str = "local",
) -> tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """GPTQ-style error-compensated quantization onto FP grids.

    Follows the column-sequential error-compensation structure of GPTQ
    (see ``onecomp.quantizer.gptq``), replacing the integer
    round-to-nearest step with rounding onto the target floating-point
    grid (E2M1 for nvfp4/mxfp4, E4M3 for fp8). Block scales are computed
    from the error-compensated weights at each scale-block boundary.

    Args:
        hessian (torch.Tensor): Hessian matrix (X^T X), shape
            (in_features, in_features).
        matrix_W (torch.Tensor): FP32 weight matrix of shape
            (out_features, in_features). Modified in place.
        fmt (str): Target format ("nvfp4", "mxfp4", or "fp8").
        block_size (int): Scale-block size (-1 for per-channel fp8).
        blocksize (int): Number of columns processed together in the
            outer error-compensation loop.
        percdamp (float): Dampening fraction for the Hessian inverse.
        scale_search (bool): If True, block scales are selected by a
            local sweep minimizing the Hessian-diagonal weighted
            reconstruction error (WMSE) of the block, instead of the
            AbsMax / ceil heuristics. The sweep uses the
            error-compensated weights at each block boundary, so it
            composes with the GPTQ compensation.
        scale_timing (str): ``"static"`` selects swept scales before
            compensation, ``"in_loop"`` selects them at block boundaries,
            ``"auto"`` preserves the old ``scale_search`` behavior.
        scale_objective (str): ``"mse"``, ``"diag_wmse"``, or
            ``"conditional"``. Conditional uses the block Schur
            complement ``((H^{-1})_BB)^{-1}``.
        scale_candidate_strategy (str): ``"local"``, ``"full"``, or
            ``"adaptive"`` candidate set for FP4 block-scale sweeps.

    Returns:
        tuple: (codes, block_scales, tensor_scale, dequantized)
            - codes: E2M1 grid indices (torch.int8) or None for fp8.
            - block_scales: Block or per-channel scales (FP32).
            - tensor_scale: Per-tensor scale (nvfp4) or None.
            - dequantized: Dequantized FP32 weight matrix.
    """
    device = matrix_W.device
    out_features, columns = matrix_W.shape

    hessian = hessian.clone().float()
    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    matrix_W[:, dead] = 0
    timing = _resolve_scale_timing(scale_search, scale_timing, use_hessian=True)
    objective = _resolve_scale_objective(scale_objective, timing, use_hessian=True)
    # Per-column importance (diagonal approximation of the layer-wise
    # reconstruction objective) for the WMSE scale sweep.
    hessian_diag = torch.diag(hessian).clone() if objective == "diag_wmse" else None

    hessian_inverse = _compute_inverse_hessian(hessian, percdamp)

    grid = e2m1_grid(device=device)

    # Align the outer loop block to the scale-block boundaries
    if fmt in ("nvfp4", "mxfp4"):
        if block_size <= blocksize:
            blocksize = block_size * max(1, blocksize // block_size)
        else:
            blocksize = block_size
        num_blocks = columns // block_size
        block_scales = torch.zeros(out_features, num_blocks, device=device)
        codes = torch.zeros(out_features, columns, dtype=torch.int8, device=device)
    else:  # fp8: per-channel scales fixed from the original weights
        row_amax = matrix_W.abs().amax(dim=-1, keepdim=True)
        block_scales = torch.where(row_amax == 0, torch.ones_like(row_amax), row_amax) / E4M3_MAX
        codes = None

    tensor_scale = None
    if fmt == "nvfp4":
        global_amax = matrix_W.abs().amax()
        tensor_scale = global_amax / (E4M3_MAX * E2M1_MAX)
        if tensor_scale == 0:
            tensor_scale = torch.ones_like(tensor_scale)

    static_effective_scales = None
    if fmt in ("nvfp4", "mxfp4") and timing == "static":
        block_scales, static_effective_scales = _static_block_scales(
            matrix_W,
            fmt,
            block_size,
            tensor_scale,
            objective,
            hessian_diag,
            scale_candidate_strategy,
        )

    dequantized = torch.zeros_like(matrix_W)
    effective_scale = None

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = matrix_W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = hessian_inverse[i1:i2, i1:i2]

        for i in range(count):
            col = i1 + i
            w = W1[:, i]
            d = Hinv1[i, i]

            if fmt in ("nvfp4", "mxfp4") and col % block_size == 0:
                block_idx = col // block_size
                if timing == "static":
                    effective_scale = static_effective_scales[:, block_idx]
                else:
                    metric = None
                    if timing == "in_loop":
                        if objective == "diag_wmse":
                            metric = hessian_diag[col : col + block_size]
                        elif objective == "conditional":
                            metric = _conditional_block_metric(
                                hessian_inverse, col, col + block_size
                            )
                    scale, effective_scale = _select_block_scale(
                        W1[:, i : i + block_size],
                        fmt,
                        tensor_scale,
                        metric,
                        sweep=timing == "in_loop",
                        candidate_strategy=scale_candidate_strategy,
                    )
                    block_scales[:, block_idx] = scale

            if fmt in ("nvfp4", "mxfp4"):
                col_codes = quantize_to_grid(w / effective_scale, grid)
                q = dequantize_from_grid(col_codes, grid) * effective_scale
                codes[:, col] = col_codes
            else:  # fp8
                q = round_to_e4m3(w / block_scales.squeeze(-1)) * block_scales.squeeze(-1)

            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        dequantized[:, i1:i2] = Q1
        matrix_W[:, i2:] -= Err1.matmul(hessian_inverse[i1:i2, i2:])

    del hessian, hessian_inverse
    gc.collect()
    torch.cuda.empty_cache()

    return codes, block_scales, tensor_scale, dequantized
