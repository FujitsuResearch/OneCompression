"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa, Keiji Kimura

"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import transformers
import gc

from onecomp.quantizer._quantizer import Quantizer, QuantizationResult


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
    scales: Optional[torch.Tensor] = None   # Scale coefficients
    qzeros: Optional[torch.Tensor] = None   # Zero points
    perm: Optional[torch.Tensor] = None     # Column permutation order (actorder=True)


@dataclass
class GPTQ(Quantizer):
    """GPTQ quantizer class"""

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
    sym: bool = False
    q_grid: int = 600
    q_norm: float = 2.4

    def validate_params(self):
        """Validate GPTQ parameters once at quantizer initialization."""
        bad = []

        if not (isinstance(self.blocksize, int) and self.blocksize >= 1):
            bad.append(
                f"Invalid GPTQ parameter 'blocksize': {self.blocksize!r} (expected int >= 1)"
            )

        if not (isinstance(self.percdamp, (int, float)) and self.percdamp >= 3.95e-4):
            bad.append(
                f"Invalid GPTQ parameter 'percdamp': {self.percdamp!r} (expected numeric >= 3.95e-4)"
            )

        if not (isinstance(self.wbits, int) and 1 <= self.wbits <= 64):
            bad.append(
                f"Invalid GPTQ parameter 'wbits': {self.wbits!r} (expected int in 1..64)"
            )

        if not (
            isinstance(self.groupsize, int)
            and (
                self.groupsize == -1
                or (1 <= self.groupsize <= self.blocksize)
            )
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

        # Quantize the layer
        result_dict = run_gptq(
            hessian,
            module,
            blocksize=self.blocksize,
            percdamp=self.percdamp,
            wbits=self.wbits,
            groupsize=self.groupsize,
            actorder=self.actorder,
            mse=self.mse,
            # perccorr=self.perccorr,
            sym=self.sym,
            q_grid=self.q_grid,
            q_norm=self.q_norm,
        )

        return GPTQResult(
            dequantized_weight=result_dict["dequantized_weight"],
            wbits=self.wbits,
            groupsize=self.groupsize,
            actorder=self.actorder,
            sym=self.sym,
            qweight=result_dict["qweight"],
            scales=result_dict["scales"],
            qzeros=result_dict["qzeros"],
            perm=result_dict["perm"],
        )

    def get_quant_config(self) -> dict:
        """Return quantization_config dict for save_quantized_model(HF/vLLM compatible keys)."""
        return {
            "quant_method": "gptq",
            "bits": self.wbits,
            "group_size": self.groupsize,
            "desc_act": self.actorder,
            "sym": self.sym,
        }

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

def run_gptq( # pylint: disable=too-many-positional-arguments
    H: torch.Tensor,  # Hessian matrix
    layer: torch.nn.Module,
    blocksize: int = 128,
    percdamp: float = 0.01,
    wbits: int = 16,
    groupsize: int = -1,
    actorder: bool = False,
    mse: bool = False,
    # perccorr=0.5
    sym: bool = False,
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

    W = layer.weight.data.clone()
    if isinstance(layer, nn.Conv2d):
        W = W.flatten(1)
    if isinstance(layer, transformers.Conv1D):
        W = W.t()
    W = W.float()
    # H = H.clone()

    if not quantizer.ready():
        quantizer.find_params(W, weight=True)

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    perm = None
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

    Q = torch.zeros_like(W)
    Q_int = torch.zeros_like(W, dtype=torch.int32)

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[0], device=H.device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    # Accumulate per-group scale/zero for grouped quantization
    if groupsize != -1:
        num_groups = (H.shape[0] + groupsize - 1) // groupsize
        all_scales = torch.zeros(W.shape[0], num_groups, dtype=W.dtype, device=W.device)
        all_zeros = torch.zeros(W.shape[0], num_groups, dtype=W.dtype, device=W.device)

    # total_blocks = (H.shape[0] + blocksize - 1) // blocksize

    for block_idx, i1 in enumerate(range(0, H.shape[0], blocksize)):
        i2 = min(i1 + blocksize, H.shape[0])
        count = i2 - i1

        # Per-block progress display
        # if block_idx % 10 == 0 or block_idx == total_blocks - 1:
        #    print(f"[GPTQ Block {block_idx}/{total_blocks-1}] Processing columns {i1}-{i2-1}")

        W1 = W[:, i1:i2].clone()
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
                        W[:, (i1 + i) : (i1 + i + groupsize)], weight=True
                    )
                    # Accumulate group scale/zero
                    group_idx = (i1 + i) // groupsize
                    all_scales[:, group_idx] = quantizer.scale.squeeze(-1)
                    all_zeros[:, group_idx] = quantizer.zero.squeeze(-1)

            q_int = quantize(
                w.unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
            )
            
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

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    if actorder:
        Q = Q[:, invperm]
        Q_int = Q_int[:, invperm]

    if isinstance(layer, transformers.Conv1D):
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

    del H, Hinv, W, Q, Q_int
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

    def __init__(self, shape=1):
        super(GPTQExcecutor, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

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
                zero1 = (
                    torch.round(-xmin1 / scale1) if not self.sym else self.zero
                )
                q_int = quantize(
                    x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                )
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
        if self.ready():
            q_int = quantize(x, self.scale, self.zero, self.maxq)
            if q_int is not None:
                return dequantize(q_int, self.scale, self.zero, self.maxq)
            else:
                return quantize_trits(x, self.scale, self.zero)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)