"""
MDBF (Multi-scale Double Binary Factorization) Layer implementation

Constructs an efficient inference layer from MDBF parameters.
Based on the DBF implementation, achieves bit-packing and memory efficiency.

Structure:
- MDBFLinear: MDBF inference layer for a single pass
- MultipathMDBFLinear: MDBF inference layer for P passes
- Packing/Unpacking: Compresses sign matrices to 1-bit

Weight representation:
    W ≈ Σ_{p=1}^{P} W^{(p)}
    W^{(p)} = F^{(p)} @ G^{(p)}
    where F = S_A * (A_amp @ Q_U_amp^T)
          G = S_B * (Q_V_amp @ B_amp^T)

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import gc
import json
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from .initialize import MDBFParams

logger = getLogger(__name__)


# =============================================================================
# Bit-packing/Unpacking
# =============================================================================


def pack_binary(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Convert ±1 to {0,1} and pack into uint8 with 8:1 ratio. Pad the end with +1.

    Args:
        x: Tensor with ±1 values (any shape)

    Returns:
        (packed, original_shape): Packed uint8 tensor and original shape
    """
    original_shape = x.shape
    flat = (x.flatten() >= 0).to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = F.pad(flat, (0, pad), value=1)

    out = torch.zeros((flat.numel() // 8,), device=flat.device, dtype=torch.uint8)
    for i in range(8):
        out += (flat[i::8] << (7 - i))
    return out, original_shape


def unpack_binary(packed: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Expand uint8 to {−1,+1} int8 and reshape.

    Args:
        packed: Packed uint8 tensor
        original_shape: Original shape

    Returns:
        ±1 int8 tensor
    """
    numel = 1
    for dim in original_shape:
        numel *= dim

    out = torch.zeros((packed.shape[0], 8), device=packed.device, dtype=torch.int8)
    for i in range(8):
        out[:, i] = (packed >> (7 - i)) & 1
    return (out.flatten()[:numel].reshape(original_shape) * 2 - 1)


# =============================================================================
# Packed MDBF Parameters
# =============================================================================


@dataclass
class PackedMDBFParams:
    """Packed MDBF parameters (for memory-efficient storage)"""
    A_sign_packed: torch.Tensor
    B_sign_packed: torch.Tensor
    A_sign_shape: Tuple[int, ...]
    B_sign_shape: Tuple[int, ...]
    A_amp: torch.Tensor
    B_amp: torch.Tensor
    Q_U_amp: torch.Tensor
    Q_V_amp: torch.Tensor


def pack_MDBF_params(params: MDBFParams) -> PackedMDBFParams:
    """Convert MDBFParams to packed format"""
    A_sign_packed, A_sign_shape = pack_binary(params.A_sign)
    B_sign_packed, B_sign_shape = pack_binary(params.B_sign)

    return PackedMDBFParams(
        A_sign_packed=A_sign_packed,
        B_sign_packed=B_sign_packed,
        A_sign_shape=A_sign_shape,
        B_sign_shape=B_sign_shape,
        A_amp=params.A_amp,
        B_amp=params.B_amp,
        Q_U_amp=params.Q_U_amp,
        Q_V_amp=params.Q_V_amp,
    )


def unpack_MDBF_params(packed: PackedMDBFParams) -> MDBFParams:
    """Restore MDBFParams from packed format"""
    A_sign = unpack_binary(packed.A_sign_packed, packed.A_sign_shape)
    B_sign = unpack_binary(packed.B_sign_packed, packed.B_sign_shape)

    return MDBFParams(
        A_sign=A_sign.float(),
        B_sign=B_sign.float(),
        A_amp=packed.A_amp,
        B_amp=packed.B_amp,
        Q_U_amp=packed.Q_U_amp,
        Q_V_amp=packed.Q_V_amp,
    )


# =============================================================================
# Scaling Layer
# =============================================================================


class ScalingLayer(nn.Module):
    """Element-wise scaling layer"""

    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.register_buffer("w", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.w.to(x.dtype)


# =============================================================================
# Packed Binary Matrix Layer
# =============================================================================


class PackedBinaryLinear(nn.Module):
    """
    Packed binary matrix × input linear layer

    If preunpack=True, unpack and store at initialization (fast = high memory).
    """

    def __init__(self, binary_matrix: torch.Tensor, preunpack: bool = True):
        super().__init__()

        if binary_matrix.ndim != 2:
            raise ValueError("PackedBinaryLinear: expected 2D ±1 tensor")

        self.shape = tuple(binary_matrix.shape)
        self._numel = binary_matrix.numel()
        self._preunpack = preunpack

        packed, _ = pack_binary(binary_matrix)
        self.register_buffer("packed", packed)
        self.register_buffer("_shape_tensor", torch.tensor(self.shape, dtype=torch.int64))

        if preunpack:
            unpacked = unpack_binary(self.packed, self.shape)
            self.register_buffer("bit_mat", unpacked, persistent=False)
        else:
            self.register_buffer("bit_mat", None, persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        """Reconstruct bit_mat during loading"""
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                       missing_keys, unexpected_keys, error_msgs)

        if hasattr(self, '_shape_tensor') and self._shape_tensor is not None:
            self.shape = tuple(self._shape_tensor.tolist())
            self._numel = 1
            for dim in self.shape:
                self._numel *= dim

        if hasattr(self, 'packed') and self.packed is not None:
            if not hasattr(self, 'bit_mat') or self.bit_mat is None:
                if getattr(self, '_preunpack', True):
                    unpacked = unpack_binary(self.packed, self.shape)
                    self.register_buffer("bit_mat", unpacked, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bit_mat is None:
            bit_mat = unpack_binary(self.packed, self.shape)
        else:
            bit_mat = self.bit_mat

        weight = bit_mat.to(x.dtype).t()
        return x.matmul(weight)


# =============================================================================
# MDBFLinear Layer (1-pass)
# =============================================================================


class MDBFLinear(nn.Module):
    """
    1-pass MDBF inference layer (always on-the-fly unpack).

    W^{(p)} = F @ G
    where F = S_A * (A_amp @ Q_U_amp^T)
          G = S_B * (Q_V_amp @ B_amp^T)

    Inference: y = x @ W^T = x @ G^T @ F^T
    """

    def __init__(self, params: MDBFParams):
        super().__init__()

        n, r = params.A_sign.shape
        r2, m = params.B_sign.shape
        assert r == r2, f"Rank mismatch: A_sign has rank {r}, B_sign has rank {r2}"

        self.n = n
        self.m = m
        self.r = r
        self.l = params.A_amp.shape[1]

        # Packed sign matrices
        A_sign_packed, _ = pack_binary(params.A_sign)
        B_sign_packed, _ = pack_binary(params.B_sign)

        self.register_buffer("A_sign_packed", A_sign_packed)
        self.register_buffer("B_sign_packed", B_sign_packed)
        self.register_buffer("_A_sign_shape", torch.tensor([n, r], dtype=torch.int64))
        self.register_buffer("_B_sign_shape", torch.tensor([r, m], dtype=torch.int64))

        # Scales (FP16)
        self.register_buffer("A_amp", params.A_amp.half())
        self.register_buffer("B_amp", params.B_amp.half())
        self.register_buffer("Q_U_amp", params.Q_U_amp.half())
        self.register_buffer("Q_V_amp", params.Q_V_amp.half())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        """Reconstruct dimensions from shape buffers during loading"""
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                       missing_keys, unexpected_keys, error_msgs)

        if hasattr(self, '_A_sign_shape') and self._A_sign_shape is not None:
            A_shape = tuple(self._A_sign_shape.tolist())
            B_shape = tuple(self._B_sign_shape.tolist())

            self.n, self.r = A_shape
            _, self.m = B_shape
            self.l = self.A_amp.shape[1] if hasattr(self, 'A_amp') else 1

    def _get_factor_matrices(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute factor matrices F, G (always on-the-fly unpack)"""
        A_sign = unpack_binary(self.A_sign_packed, (self.n, self.r)).to(dtype)
        B_sign = unpack_binary(self.B_sign_packed, (self.r, self.m)).to(dtype)

        amp_A = self.A_amp.to(dtype) @ self.Q_U_amp.to(dtype).T
        F = A_sign * amp_A

        amp_B = self.Q_V_amp.to(dtype) @ self.B_amp.to(dtype).T
        G = B_sign * amp_B

        return F, G

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F, G = self._get_factor_matrices(x.dtype)
        y = x @ G.T
        y = y @ F.T
        return y

    def get_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Reconstruct weight matrix"""
        F, G = self._get_factor_matrices(dtype)
        return F @ G


# =============================================================================
# MultipathMDBFLinear Layer (P-pass)
# =============================================================================


class MultipathMDBFLinear(nn.Module):
    """P-pass MDBF inference layer: W ≈ Σ_{p=1}^{P} W^{(p)}"""

    def __init__(
        self,
        params_list: List[MDBFParams],
        bias: Optional[torch.Tensor] = None,
        device=None,
    ):
        super().__init__()

        if len(params_list) == 0:
            raise ValueError("params_list must not be empty")

        self.P = len(params_list)
        self.n = params_list[0].A_sign.shape[0]
        self.m = params_list[0].B_sign.shape[1]

        self.paths = nn.ModuleList([
            MDBFLinear(params)
            for params in params_list
        ])

        if bias is not None:
            self.register_buffer("bias", bias.clone().to(torch.float16))
        else:
            self.bias = None

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.paths[0](x)
        for i in range(1, self.P):
            y = y + self.paths[i](x)

        if self.bias is not None:
            y = y + self.bias.to(x.dtype)

        return y

    def get_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Reconstruct weight matrix"""
        W = self.paths[0].get_weight(dtype)
        for i in range(1, self.P):
            W = W + self.paths[i].get_weight(dtype)
        return W

    @classmethod
    def from_quantization_result(
        cls,
        result,
        bias=None,
        device=None,
    ) -> "MultipathMDBFLinear":
        """Build MultipathMDBFLinear from MDBFResult.

        Args:
            result: MDBFResult from quantizer.
            bias: Optional bias tensor (from original Linear).
            device: Device to place the layer on.

        Returns:
            MultipathMDBFLinear instance.
        """
        params_list = result.get_MDBF_params_list()
        return cls(params_list=params_list, bias=bias, device=device)

    @classmethod
    def from_saved_state(
        cls,
        layer_state_dict: dict,
        in_features: int,
        out_features: int,
        empty: bool = False,
        target_bits: float = None,
    ) -> "MultipathMDBFLinear":
        """Build MultipathMDBFLinear from saved state_dict tensors.

        P (number of paths) and l (multi-scale rank) are inferred from the
        state_dict (path index keys and A_amp.shape[1] respectively), so
        callers do not need to pass them.

        Args:
            layer_state_dict: Sub-state_dict for this layer.
                Keys follow the pattern:
                  paths.{p}.A_sign_packed, paths.{p}.B_sign_packed,
                  paths.{p}._A_sign_shape, paths.{p}._B_sign_shape,
                  paths.{p}.A_amp, paths.{p}.B_amp,
                  paths.{p}.Q_U_amp, paths.{p}.Q_V_amp
                  bias (optional)
            in_features: Input feature size (m).
            out_features: Output feature size (n).
            empty: If True, create zero-initialized tensors (for
                "replace then load_state_dict" flow).
            target_bits: Nominal bit-width (from config).

        Returns:
            MultipathMDBFLinear instance.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.target_bits = target_bits
        self.n = out_features
        self.m = in_features

        def _t(k):
            t = layer_state_dict[k]
            return torch.zeros_like(t) if empty else t

        # Detect P from state_dict keys
        path_indices = set()
        for key in layer_state_dict:
            if key.startswith("paths."):
                parts = key.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    path_indices.add(int(parts[1]))
        if not path_indices:
            raise ValueError(
                "MultipathMDBFLinear.from_saved_state: no `paths.{p}.*` keys "
                "found in layer_state_dict."
            )
        self.P = max(path_indices) + 1

        paths = nn.ModuleList()
        for p_idx in range(self.P):
            path_prefix = f"paths.{p_idx}."

            path_layer = MDBFLinear.__new__(MDBFLinear)
            nn.Module.__init__(path_layer)

            # Recover rank r from Q_U_amp's shape ((r, l)).
            # (DBF analog: mid_dim = layer_state_dict["scaling2"].numel().)
            r = layer_state_dict[f"{path_prefix}Q_U_amp"].shape[0]
            path_layer.n = out_features
            path_layer.r = r
            path_layer.m = in_features
            shape_A = (out_features, r)
            shape_B = (r, in_features)

            path_layer.register_buffer("A_sign_packed", _t(f"{path_prefix}A_sign_packed"))
            path_layer.register_buffer("B_sign_packed", _t(f"{path_prefix}B_sign_packed"))
            path_layer.register_buffer("_A_sign_shape",
                                        torch.tensor(shape_A, dtype=torch.int64))
            path_layer.register_buffer("_B_sign_shape",
                                        torch.tensor(shape_B, dtype=torch.int64))
            path_layer.register_buffer("A_amp", _t(f"{path_prefix}A_amp"))
            path_layer.register_buffer("B_amp", _t(f"{path_prefix}B_amp"))
            path_layer.register_buffer("Q_U_amp", _t(f"{path_prefix}Q_U_amp"))
            path_layer.register_buffer("Q_V_amp", _t(f"{path_prefix}Q_V_amp"))

            path_layer.l = path_layer.A_amp.shape[1]

            paths.append(path_layer)

        self.paths = paths

        bias = layer_state_dict.get("bias")
        if bias is not None:
            self.register_buffer("bias", torch.zeros_like(bias) if empty else bias)
        else:
            self.bias = None

        return self


# =============================================================================
# Layer replacement functions (QEP-DEV compatible, not used in OneComp framework)
# =============================================================================


def create_mdbf_layer_from_linear(
    module: nn.Module,
    preunpack: bool = True
) -> Optional[MultipathMDBFLinear]:
    """Create a MultipathMDBFLinear from a quantized MDBF Linear layer"""
    if not hasattr(module, 'MDBF_params'):
        return None
    if not getattr(module, 'is_quantized', False):
        return None

    params_list = module.MDBF_params
    if not isinstance(params_list, list) or len(params_list) == 0:
        return None

    bias = module.bias.clone() if hasattr(module, 'bias') and module.bias is not None else None

    return MultipathMDBFLinear(
        params_list=params_list,
        bias=bias,
    )


def replace_linear_with_mdbf(
    module: nn.Module,
    name: str,
    parent_module: nn.Module,
    preunpack: bool = True
) -> bool:
    """Replace a Linear layer with a MultipathMDBFLinear layer"""
    mdbf_layer = create_mdbf_layer_from_linear(module, preunpack=preunpack)

    if mdbf_layer is None:
        return False

    device = module.weight.device
    mdbf_layer = mdbf_layer.to(device)
    setattr(parent_module, name, mdbf_layer)

    del module.weight
    if hasattr(module, 'MDBF_params'):
        del module.MDBF_params

    return True


def replace_all_MDBF_layers(
    model: nn.Module,
    preunpack: bool = True
) -> int:
    """Replace all MDBF quantized layers in the model with MultipathMDBFLinear"""
    replaced_count = 0

    for parent_name, parent_module in model.named_modules():
        for name, module in list(parent_module.named_children()):
            if isinstance(module, (nn.Linear, transformers.Conv1D)):
                if replace_linear_with_mdbf(module, name, parent_module, preunpack):
                    replaced_count += 1
                    logger.debug(f"[MDBF] Replaced {parent_name}.{name} with MultipathMDBFLinear")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"[MDBF] Total replaced layers: {replaced_count}")
    return replaced_count


# =============================================================================
# Checkpoint management (QEP-DEV compatible, not used in OneComp framework)
# =============================================================================


def save_MDBF_weights(
    model: nn.Module,
    save_path: Path,
    packed: bool = True
) -> Dict[str, int]:
    """Save MDBF weights"""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    weights = {}
    stats = {"layers": 0, "params": 0}

    for name, module in model.named_modules():
        if not (hasattr(module, 'MDBF_params') and getattr(module, 'is_quantized', False)):
            continue

        params_list = module.MDBF_params

        for p_idx, params in enumerate(params_list):
            prefix = f"{name}.path{p_idx}"

            if packed:
                A_packed, A_shape = pack_binary(params.A_sign)
                B_packed, B_shape = pack_binary(params.B_sign)

                weights[f"{prefix}.A_sign_packed"] = A_packed.cpu()
                weights[f"{prefix}.B_sign_packed"] = B_packed.cpu()
                weights[f"{prefix}.A_sign_shape"] = torch.tensor(A_shape, dtype=torch.int64)
                weights[f"{prefix}.B_sign_shape"] = torch.tensor(B_shape, dtype=torch.int64)
            else:
                weights[f"{prefix}.A_sign"] = params.A_sign.cpu()
                weights[f"{prefix}.B_sign"] = params.B_sign.cpu()

            weights[f"{prefix}.A_amp"] = params.A_amp.cpu()
            weights[f"{prefix}.B_amp"] = params.B_amp.cpu()
            weights[f"{prefix}.Q_U_amp"] = params.Q_U_amp.cpu()
            weights[f"{prefix}.Q_V_amp"] = params.Q_V_amp.cpu()

            stats["params"] += 6

        if hasattr(module, 'bias') and module.bias is not None:
            weights[f"{name}.bias"] = module.bias.cpu()

        stats["layers"] += 1

    torch.save(weights, save_path / "MDBF_weights.pt")

    metadata = {
        "packed": packed,
        "layers": stats["layers"],
        "params": stats["params"],
    }
    with open(save_path / "MDBF_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"[MDBF] Saved {stats['layers']} layers to {save_path}")
    return stats


def load_MDBF_weights(load_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load MDBF weights"""
    load_path = Path(load_path)
    weights = torch.load(load_path / "MDBF_weights.pt", map_location="cpu")

    with open(load_path / "MDBF_metadata.json", "r") as f:
        metadata = json.load(f)

    return weights, metadata


# =============================================================================
# Verification functions
# =============================================================================


def verify_binary_values(params: MDBFParams) -> Tuple[bool, str]:
    """Verify that S_A and S_B are valid binary {-1, +1} matrices"""
    A_unique = torch.unique(params.A_sign)
    A_valid = len(A_unique) <= 2 and all(v in [-1.0, 1.0] for v in A_unique.tolist())

    B_unique = torch.unique(params.B_sign)
    B_valid = len(B_unique) <= 2 and all(v in [-1.0, 1.0] for v in B_unique.tolist())

    if A_valid and B_valid:
        return True, "S_A and S_B are valid binary {-1, +1} matrices"

    msg_parts = []
    if not A_valid:
        msg_parts.append(f"S_A has invalid values: {A_unique.tolist()}")
    if not B_valid:
        msg_parts.append(f"S_B has invalid values: {B_unique.tolist()}")
    return False, "; ".join(msg_parts)


def verify_all_params(params_list: List[MDBFParams]) -> Tuple[bool, List[str]]:
    """Verify all parameters in all paths"""
    all_valid = True
    messages = []

    for i, params in enumerate(params_list):
        valid, msg = verify_binary_values(params)
        messages.append(f"Path {i}: {msg}")
        if not valid:
            all_valid = False

    return all_valid, messages
