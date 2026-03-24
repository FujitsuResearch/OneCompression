"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa, Keiji Kimura

"""

from dataclasses import dataclass
import re
from typing import Any, Optional

import gc

import torch
from torch import nn
from transformers import Conv1D

from onecomp.quantizer._quantizer import Quantizer, QuantizationResult
from onecomp.utils.quant_config import get_quant_param


@dataclass
class GPTQResult(QuantizationResult):
    """GPTQ quantization result class.

    Inherits from QuantizationResult and adds GPTQ-specific parameters.

    Attributes:
        dequantized_weight: Dequantized weights (FP16, CPU) - inherited from parent class.

        [Quantization configuration parameters]
        wbits: Quantization bit width.
        groupsize: Group size (-1 means no grouping).
        actorder: Whether columns were reordered by activation order.
        sym: Whether symmetric quantization was used.

        [Weight reconstruction data]
        qweight: Quantized weights (INT type, CPU).
        scales: Scale coefficients (FP16, CPU).
        qzeros: Zero points (FP16, CPU).
        perm: Column permutation order (used when actorder=True).

    Note:
        - g_idx (group index) is not stored since it can be computed from groupsize and perm.
          Computation: g_idx[perm[i]] = i // groupsize (when actorder=True)
                       g_idx[i] = i // groupsize (when actorder=False)
        - invperm (inverse permutation) is not stored since it can be computed from perm.
          Computation: invperm = torch.argsort(perm)
    """

    # =========================================
    # Quantization configuration parameters
    # =========================================
    wbits: int = None
    groupsize: int = None
    actorder: bool = None
    sym: bool = None

    # =========================================
    # Weight reconstruction data
    # =========================================
    qweight: Optional[torch.Tensor] = None  # Quantized weights (INT type)
    scales: Optional[torch.Tensor] = None  # Scale coefficients
    qzeros: Optional[torch.Tensor] = None  # Zero points
    perm: Optional[torch.Tensor] = None  # Column permutation order (actorder=True)


@dataclass
class GPTQ(Quantizer):
    """GPTQ (Accurate Post-Training Quantization for Generative Pre-trained Transformers) quantizer.

    Performs layer-wise weight quantization using second-order (Hessian) information
    to minimize the output reconstruction error. Supports grouped quantization,
    activation-order column reordering, and MSE-based grid search for optimal
    scale/zero-point parameters.

    GPTQ requires calibration data and Hessian matrix computation.

    Attributes:
        flag_calibration (bool): Whether calibration data is needed (True for GPTQ).
        flag_hessian (bool): Whether Hessian matrix is needed (True for GPTQ).
        blocksize (int): Number of columns quantized together in each block.
            Larger values may improve quality but increase memory usage. Default is 128.
        percdamp (float): Percentage of the Hessian diagonal average added for
            numerical stability (dampening). Default is 0.01.
        wbits (int): Quantization bit width (1-8). Default is 4.
        groupsize (int): Number of columns sharing the same scale/zero-point.
            -1 means per-channel (no grouping). Must be -1 or in 1..blocksize. Default is -1.
        actorder (bool): If True, reorder columns by decreasing activation magnitude
            before quantization (desc_act). Default is False.
        mse (bool): If True, use MSE-based grid search to find optimal
            scale and zero-point parameters. Default is False.
        sym (bool): If True, use symmetric quantization (zero-point fixed at midpoint).
            If False, use asymmetric quantization (zero-point computed from data).
            Default is True.
        q_grid (int): Number of grid points for MSE-based scale search
            (used when mse=True). Default is 600.
        q_norm (float): Norm exponent for MSE grid search error metric
            (used when mse=True). Default is 2.4.

    Example:
        >>> from onecomp.quantizer.gptq import GPTQ
        >>> quantizer = GPTQ(wbits=4, groupsize=128)
        >>> quantizer = GPTQ(wbits=4, groupsize=128, sym=False, actorder=True)
    """

    flag_calibration: bool = True
    flag_hessian: bool = True

    # Parameters for the GPTQ quantizer
    blocksize: int = 128
    percdamp: float = 0.01
    wbits: int = 4
    groupsize: int = -1
    actorder: bool = False
    mse: bool = False
    # perccorr: float = 0.5
    sym: bool = True
    q_grid: int = 600
    q_norm: float = 2.4
    mlp_wbits: Optional[int] = None
    mlp_groupsize: Optional[int] = None
    module_wbits: Optional[dict[str, int]] = None

    @staticmethod
    def resolve_bits(
        layer_name: Optional[str],
        default_bits: int,
        mlp_bits: Optional[int] = None,
        module_bits: Optional[dict[str, int]] = None,
    ) -> int:
        """Resolve bit-width from overrides (GPTQ semantics: module > mlp > default).

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

    @staticmethod
    def resolve_groupsize(
        layer_name: Optional[str],
        default_groupsize: int,
        mlp_groupsize: Optional[int] = None,
    ) -> int:
        """Resolve group_size (mlp override > default)."""
        if mlp_groupsize is not None and layer_name is not None and "mlp" in layer_name:
            return mlp_groupsize
        return default_groupsize

    def validate_params(self):
        """Validate GPTQ parameters once at quantizer initialization."""
        bad = []

        if not (isinstance(self.blocksize, int) and self.blocksize >= 1):
            bad.append(
                f"Invalid GPTQ parameter 'blocksize': {self.blocksize!r} (expected int >= 1)"
            )

        if not (isinstance(self.percdamp, (int, float)) and self.percdamp >= 3.95e-4):
            bad.append(
                f"Invalid GPTQ parameter 'percdamp': {self.percdamp!r} "
                f"(expected numeric >= 3.95e-4)"
            )

        if not (isinstance(self.wbits, int) and 1 <= self.wbits <= 64):
            bad.append(f"Invalid GPTQ parameter 'wbits': {self.wbits!r} (expected int in 1..64)")

        if not (
            isinstance(self.groupsize, int)
            and (self.groupsize == -1 or (1 <= self.groupsize <= self.blocksize))
        ):
            bad.append(
                "Invalid GPTQ parameter 'groupsize': "
                f"{self.groupsize!r} (expected int -1 or 1..blocksize)"
            )

        if not (isinstance(self.q_grid, int) and 1 <= self.q_grid <= 1000):
            bad.append(
                f"Invalid GPTQ parameter 'q_grid': {self.q_grid!r} (expected int in 1..1000)"
            )

        if not (isinstance(self.q_norm, (int, float)) and self.q_norm >= 1e-5):
            bad.append(
                f"Invalid GPTQ parameter 'q_norm': {self.q_norm!r} (expected numeric >= 1e-5)"
            )

        if self.mlp_wbits is not None:
            if not (isinstance(self.mlp_wbits, int) and 1 <= self.mlp_wbits <= 64):
                bad.append(
                    f"Invalid GPTQ parameter 'mlp_wbits': {self.mlp_wbits!r} (expected int in 1..64)"
                )

        if self.mlp_groupsize is not None:
            if not (
                isinstance(self.mlp_groupsize, int)
                and (self.mlp_groupsize == -1 or (1 <= self.mlp_groupsize <= self.blocksize))
            ):
                bad.append(
                    f"Invalid GPTQ parameter 'mlp_groupsize': {self.mlp_groupsize!r} "
                    f"(expected int -1 or 1..blocksize)"
                )

        if self.module_wbits is not None:
            if not isinstance(self.module_wbits, dict):
                bad.append(
                    f"Invalid GPTQ parameter 'module_wbits': must be a dict[str, int], got {type(self.module_wbits).__name__!r}"
                )
            else:
                for layer_name, bits in self.module_wbits.items():
                    if not isinstance(layer_name, str):
                        bad.append(
                            "Invalid GPTQ parameter 'module_wbits': keys must be layer name strings."
                        )
                    elif not (isinstance(bits, int) and 1 <= bits <= 64):
                        bad.append(
                            f"Invalid GPTQ parameter 'module_wbits[{layer_name!r}]': {bits!r} (expected int in 1..64)"
                        )

        if bad:
            raise ValueError("; ".join(bad))

    def quantize_layer(self, module, input, hessian=None):
        """Quantize the layer

        Args:
            module (torch.nn.Module): The layer module
            input (tuple or torch.Tensor): The input to the layer (input activations)
            hessian (torch.Tensor): The Hessian matrix

        Returns:
            GPTQResult: GPTQ quantization result object.
        """

        layer_name = self.module_to_name.get(module)
        resolved_wbits = GPTQ.resolve_bits(
            layer_name,
            self.wbits,
            self.mlp_wbits,
            self.module_wbits,
        )
        resolved_groupsize = GPTQ.resolve_groupsize(
            layer_name,
            self.groupsize,
            self.mlp_groupsize,
        )

        # Quantize the layer
        result_dict = run_gptq(
            hessian,
            module,
            blocksize=self.blocksize,
            percdamp=self.percdamp,
            wbits=resolved_wbits,
            groupsize=resolved_groupsize,
            actorder=self.actorder,
            mse=self.mse,
            # perccorr=self.perccorr,
            sym=self.sym,
            q_grid=self.q_grid,
            q_norm=self.q_norm,
        )

        return GPTQResult(
            dequantized_weight=result_dict["dequantized_weight"],
            wbits=resolved_wbits,
            groupsize=resolved_groupsize,
            actorder=self.actorder,
            sym=self.sym,
            qweight=result_dict["qweight"],
            scales=result_dict["scales"],
            qzeros=result_dict["qzeros"],
            perm=result_dict["perm"],
        )

    def get_quant_config(self) -> dict:
        """Return quantization_config dict for save_quantized_model(HF/vLLM compatible keys).

        Structure: all keys at top-level (quant_method, bits, group_size, actorder, sym,
        checkpoint_format, optional mlp_wbits / module_wbits).

        When ``module_wbits`` is non-empty (mixed-bit model), ``quant_method`` is set to
        ``"mixed_gptq"`` so vLLM can dispatch per-module kernels via the mixed_gptq plugin.
        The ``quantization_bits`` list (indexed by transformer layer) is injected by
        ``finalize_quant_config_for_save`` after the quantized names are known.

        checkpoint_format is always ``"gptq"`` (v1).  OneComp GPTQLinear stores zero-points
        with the -1 offset convention (v1) unconditionally, so ``"gptq_v2"`` would cause an
        off-by-one mismatch when loaded by vLLM.
        """
        # mixed_gptq when any per-module or per-group bit overrides exist
        quant_method = (
            "mixed_gptq"
            if (self.module_wbits or self.mlp_wbits or self.mlp_groupsize is not None)
            else "gptq"
        )
        result: dict[str, Any] = {
            "quant_method": quant_method,
            "bits": self.wbits,
            "groupsize": self.groupsize,
            "actorder": self.actorder,
            "group_size": self.groupsize,
            "desc_act": self.actorder,
            "sym": self.sym,
            "checkpoint_format": "gptq",
        }
        if self.mlp_wbits is not None:
            result["mlp_wbits"] = self.mlp_wbits
        if self.mlp_groupsize is not None:
            result["mlp_groupsize"] = self.mlp_groupsize
        if self.module_wbits:
            result["module_wbits"] = dict(self.module_wbits)
        return result

    @staticmethod
    def _build_quantization_bits(
        quantized_names: list[str],
        quant_config: dict[str, Any],
        num_layers: int,
    ) -> list[dict[str, Any]]:
        _LAYER_RE = re.compile(r"\.layers\.(\d+)\.(.*)")
        default_wbits = quant_config.get("bits", quant_config.get("wbits", 4))
        mlp_wbits = get_quant_param(quant_config, "mlp_wbits")
        module_wbits: dict[str, int] = get_quant_param(quant_config, "module_wbits") or {}
        default_gs = get_quant_param(quant_config, "group_size", "groupsize", default=-1)
        mlp_gs = get_quant_param(quant_config, "mlp_groupsize")

        layer_modules: dict[int, dict[str, Any]] = {}
        for name in quantized_names:
            m = _LAYER_RE.search(name)
            if m is None:
                continue
            layer_idx = int(m.group(1))
            suffix = m.group(2)

            bits = GPTQ.resolve_bits(name, default_wbits, mlp_wbits, module_wbits)
            gs = GPTQ.resolve_groupsize(name, default_gs, mlp_gs)

            layer_modules.setdefault(layer_idx, {})[suffix] = {
                "bits": bits,
                "method": "gptq",
                "params": {"group_size": gs},
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
                "num_hidden_layers is required for GPTQ quantization_bits "
                "(Runner passes model.config.num_hidden_layers)"
            )
        quant_config["quantization_bits"] = GPTQ._build_quantization_bits(
            quantized_layer_names, quant_config, num_hidden_layers
        )
        return quant_config

    def create_inference_layer(self, result, linear_module, **kwargs):
        """Build GPTQLinear from GPTQResult."""
        from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

        pack_weights = kwargs.get("pack_weights", True)
        return GPTQLinear.from_quantization_result(
            result=result,
            bias=(
                linear_module.bias
                if hasattr(linear_module, "bias") and linear_module.bias is not None
                else None
            ),
            device=linear_module.weight.device,
            pack_weights=pack_weights,
            use_gemlite=kwargs.get("use_gemlite"),
        )


def run_gptq(  # pylint: disable=too-many-positional-arguments
    hessian: torch.Tensor,
    layer: torch.nn.Module,
    blocksize: int = 128,
    percdamp: float = 0.01,
    wbits: int = 16,
    groupsize: int = -1,
    actorder: bool = False,
    mse: bool = False,
    # perccorr=0.5
    sym: bool = True,
    q_grid: int = 600,
    q_norm: float = 2.4,
) -> dict[str, torch.Tensor]:
    """GPTQ quantization process.

    Ported from QEP-dev: src/method/gptq/gptq_impl.py (2024/09/23)

    """

    # Initial code modified from the original

    quantizer = GPTQExcecutor()
    quantizer.configure(
        wbits,
        perchannel=True,
        sym=sym,
        mse=mse,
        norm=q_norm,
        grid=q_grid,
    )

    matrix_W = layer.weight.data.clone()
    if isinstance(layer, nn.Conv2d):
        matrix_W = matrix_W.flatten(1)
    if isinstance(layer, Conv1D):
        matrix_W = matrix_W.t()
    matrix_W = matrix_W.float()

    if not quantizer.ready():
        quantizer.find_params(matrix_W, weight=True)

    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    matrix_W[:, dead] = 0

    perm = None
    if actorder:
        perm = torch.argsort(torch.diag(hessian), descending=True)
        matrix_W = matrix_W[:, perm]
        hessian = hessian[perm][:, perm]
        invperm = torch.argsort(perm)

    Q = torch.zeros_like(matrix_W)
    Q_int = torch.zeros_like(matrix_W, dtype=torch.int32)

    damp = percdamp * torch.mean(torch.diag(hessian))
    diag = torch.arange(hessian.shape[0], device=hessian.device)
    hessian[diag, diag] += damp
    hessian = torch.linalg.cholesky(hessian)
    hessian = torch.cholesky_inverse(hessian)
    hessian = torch.linalg.cholesky(hessian, upper=True)
    Hinv = hessian

    # Accumulate per-group scale/zero for grouped quantization
    if groupsize != -1:
        num_groups = (hessian.shape[0] + groupsize - 1) // groupsize
        all_scales = torch.zeros(
            matrix_W.shape[0], num_groups, dtype=matrix_W.dtype, device=matrix_W.device
        )
        all_zeros = torch.zeros(
            matrix_W.shape[0], num_groups, dtype=matrix_W.dtype, device=matrix_W.device
        )

    # total_blocks = (hessian.shape[0] + blocksize - 1) // blocksize

    for block_idx, i1 in enumerate(range(0, hessian.shape[0], blocksize)):
        i2 = min(i1 + blocksize, hessian.shape[0])
        count = i2 - i1

        # Per-block progress display
        # if block_idx % 10 == 0 or block_idx == total_blocks - 1:
        #    print(f"[GPTQ Block {block_idx}/{total_blocks-1}] Processing columns {i1}-{i2-1}")

        W1 = matrix_W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Q1_int = torch.zeros_like(W1, dtype=torch.int32)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if groupsize != -1:
                if (i1 + i) % groupsize == 0:
                    quantizer.find_params(
                        matrix_W[:, (i1 + i) : (i1 + i + groupsize)], weight=True
                    )
                    # Accumulate group scale/zero
                    group_idx = (i1 + i) // groupsize
                    all_scales[:, group_idx] = quantizer.scale.squeeze(-1)
                    all_zeros[:, group_idx] = quantizer.zero.squeeze(-1)

            q_int = quantize(w.unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq)

            if q_int is not None:
                q = dequantize(q_int, quantizer.scale, quantizer.zero, quantizer.maxq).flatten()
                q_int = q_int.flatten()
            else:
                w_expanded = w.unsqueeze(1)  # (out_features, 1)
                q = quantize_trits(w_expanded, quantizer.scale, quantizer.zero).flatten()

            Q1[:, i] = q
            if q_int is not None:
                Q1_int[:, i] = q_int

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        Q_int[:, i1:i2] = Q1_int

        matrix_W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    if actorder:
        Q = Q[:, invperm]
        Q_int = Q_int[:, invperm]

    if isinstance(layer, Conv1D):
        Q = Q.t()
        Q_int = Q_int.t()

    # layer.weight.data = Q.reshape(layer.weight.shape).to(layer.weight.data.dtype) # original code
    dequantized_weight = Q.reshape(layer.weight.shape).to(layer.weight.data.dtype).cpu()
    quantized_weight = Q_int.reshape(layer.weight.shape).cpu()

    if groupsize != -1:
        scale = all_scales.to(dtype=torch.float16, device="cpu").T
        zero = all_zeros.to(dtype=torch.int32, device="cpu").T
    else:
        scale = quantizer.scale.to(dtype=torch.float16, device="cpu")
        zero = quantizer.zero.to(dtype=torch.int32, device="cpu")
    perm = perm.cpu() if perm is not None else None

    del hessian, Hinv, matrix_W, Q, Q_int
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "dequantized_weight": dequantized_weight,
        "qweight": quantized_weight,
        "scales": scale,
        "qzeros": zero,
        "perm": perm,
    }


### Code below is ported from QEP-dev: src/method/gptq/quant.py (2024/09/23)


def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: torch.Tensor,
) -> torch.Tensor | None:
    """Quantize floating point values to integers.

    Args:
        x: Input tensor (floating point).
        scale: Scale coefficients.
        zero: Zero points.
        maxq: Maximum quantization level.

    Returns:
        Quantized integer tensor (range 0 to maxq), or None if maxq < 0.
    """
    if maxq < 0:  # trits=True
        return None
    maxq_val = maxq.item() if isinstance(maxq, torch.Tensor) else maxq
    return torch.clamp(torch.round(x / scale) + zero, 0, maxq_val).int()


def dequantize(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: torch.Tensor,
) -> torch.Tensor:
    """Dequantize integer values back to floating point.

    Args:
        quantized: Quantized integer tensor.
        scale: Scale coefficients.
        zero: Zero points.
        maxq: Maximum quantization level.

    Returns:
        Dequantized floating point tensor.
    """
    return scale * (quantized.float() - zero)


def quantize_trits(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero


class GPTQExcecutor(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxq = None
        self.scale = None
        self.zero = None
        self.perchannel = None
        self.sym = None
        self.mse = None
        self.norm = None
        self.grid = None
        self.maxshrink = None

    def configure(  # pylint: disable=too-many-positional-arguments
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        trits=False,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q_int = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                if q_int is not None:
                    q = dequantize(q_int, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                    q -= x
                    q.abs_()
                    q.pow_(self.norm)
                    err = torch.sum(q, 1)
                else:
                    # Skip MSE optimization when trits=True
                    continue
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        """Quantize and dequantize input tensor using current scale/zero parameters.

        If parameters are not yet computed (ready() is False), returns the input unchanged.

        Args:
            x (torch.Tensor): Input tensor to quantize.

        Returns:
            torch.Tensor: Dequantized tensor, or the original input if not ready.
        """
        if self.ready():
            q_int = quantize(x, self.scale, self.zero, self.maxq)
            if q_int is not None:
                return dequantize(q_int, self.scale, self.zero, self.maxq)
            else:
                return quantize_trits(x, self.scale, self.zero)
        return x

    def enabled(self):
        """Check whether quantization is active (maxq > 0).

        Returns:
            bool: True if quantization bit width has been configured and is positive.
        """
        return self.maxq is not None and self.maxq > 0

    def ready(self):
        """Check whether scale parameters have been computed.

        Returns:
            bool: True if scale is set and all values are non-zero.
        """
        return self.scale is not None and torch.all(self.scale != 0)
