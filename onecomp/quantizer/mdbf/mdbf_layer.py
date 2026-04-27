"""
MDBF (Multi-scale Double Binary Factorization) Layer実装

MSVIDパラメータから効率的な推論用レイヤーを構築。
DBF実装を参考に、ビットパッキングとメモリ効率を実現。

構造:
- MSVIDLinear: 1パス分のMSVID推論層
- MultipathMSVIDLinear: Pパス対応のMSVID推論層
- パッキング/アンパッキング: 符号行列を1-bitに圧縮

重み表現:
    W ≈ Σ_{p=1}^{P} W^{(p)}
    W^{(p)} = F^{(p)} @ G^{(p)}
    where F = S_A * (A_amp @ Q_U_amp^T)
          G = S_B * (Q_V_amp @ B_amp^T)
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

from .initialize import MSVIDParams

logger = getLogger(__name__)


# =============================================================================
# ビットパッキング/アンパッキング
# =============================================================================


def pack_binary(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    ±1 を {0,1} に変換して uint8 に 8:1 パック。末尾は +1 でパディング。

    Args:
        x: ±1 のテンソル（任意の形状）

    Returns:
        (packed, original_shape): パックされたuint8テンソルと元の形状
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
    uint8 を {−1,+1} の int8 に展開してreshape。

    Args:
        packed: パックされたuint8テンソル
        original_shape: 元の形状

    Returns:
        ±1 の int8 テンソル
    """
    numel = 1
    for dim in original_shape:
        numel *= dim

    out = torch.zeros((packed.shape[0], 8), device=packed.device, dtype=torch.int8)
    for i in range(8):
        out[:, i] = (packed >> (7 - i)) & 1
    return (out.flatten()[:numel].reshape(original_shape) * 2 - 1)


# =============================================================================
# パック済みMSVIDパラメータ
# =============================================================================


@dataclass
class PackedMSVIDParams:
    """パック済みMSVIDパラメータ（メモリ効率的な保存用）"""
    A_sign_packed: torch.Tensor
    B_sign_packed: torch.Tensor
    A_sign_shape: Tuple[int, ...]
    B_sign_shape: Tuple[int, ...]
    A_amp: torch.Tensor
    B_amp: torch.Tensor
    Q_U_amp: torch.Tensor
    Q_V_amp: torch.Tensor


def pack_msvid_params(params: MSVIDParams) -> PackedMSVIDParams:
    """MSVIDParamsをパック済み形式に変換"""
    A_sign_packed, A_sign_shape = pack_binary(params.A_sign)
    B_sign_packed, B_sign_shape = pack_binary(params.B_sign)

    return PackedMSVIDParams(
        A_sign_packed=A_sign_packed,
        B_sign_packed=B_sign_packed,
        A_sign_shape=A_sign_shape,
        B_sign_shape=B_sign_shape,
        A_amp=params.A_amp,
        B_amp=params.B_amp,
        Q_U_amp=params.Q_U_amp,
        Q_V_amp=params.Q_V_amp,
    )


def unpack_msvid_params(packed: PackedMSVIDParams) -> MSVIDParams:
    """パック済み形式からMSVIDParamsを復元"""
    A_sign = unpack_binary(packed.A_sign_packed, packed.A_sign_shape)
    B_sign = unpack_binary(packed.B_sign_packed, packed.B_sign_shape)

    return MSVIDParams(
        A_sign=A_sign.float(),
        B_sign=B_sign.float(),
        A_amp=packed.A_amp,
        B_amp=packed.B_amp,
        Q_U_amp=packed.Q_U_amp,
        Q_V_amp=packed.Q_V_amp,
    )


# =============================================================================
# スケーリング層
# =============================================================================


class ScalingLayer(nn.Module):
    """要素ごとのスケーリング層"""

    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.register_buffer("w", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.w.to(x.dtype)


# =============================================================================
# パック済みバイナリ行列層
# =============================================================================


class PackedBinaryLinear(nn.Module):
    """
    パック済みバイナリ行列 × 入力の線形層

    preunpack=True なら初期化時に展開して保持（高速＝大メモリ）。
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
        """ロード時にbit_matを再構築"""
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
# MSVIDLinear層（1パス用）
# =============================================================================


class MSVIDLinear(nn.Module):
    """
    1パス分のMSVID推論層

    W^{(p)} = F @ G
    where F = S_A * (A_amp @ Q_U_amp^T)
          G = S_B * (Q_V_amp @ B_amp^T)

    推論: y = x @ W^T = x @ G^T @ F^T
    """

    def __init__(self, params: MSVIDParams, preunpack: bool = True):
        super().__init__()

        n, r = params.A_sign.shape
        r2, m = params.B_sign.shape
        assert r == r2, f"Rank mismatch: A_sign has rank {r}, B_sign has rank {r2}"

        self.n = n
        self.m = m
        self.r = r
        self.l = params.A_amp.shape[1]
        self._preunpack = preunpack

        # 符号行列（パック）
        A_sign_packed, _ = pack_binary(params.A_sign)
        B_sign_packed, _ = pack_binary(params.B_sign)

        self.register_buffer("A_sign_packed", A_sign_packed)
        self.register_buffer("B_sign_packed", B_sign_packed)
        self.register_buffer("_A_sign_shape", torch.tensor([n, r], dtype=torch.int64))
        self.register_buffer("_B_sign_shape", torch.tensor([r, m], dtype=torch.int64))

        # スケール（FP16）
        self.register_buffer("A_amp", params.A_amp.half())
        self.register_buffer("B_amp", params.B_amp.half())
        self.register_buffer("Q_U_amp", params.Q_U_amp.half())
        self.register_buffer("Q_V_amp", params.Q_V_amp.half())

        # 展開済み符号行列
        if preunpack:
            self.register_buffer("A_sign", unpack_binary(A_sign_packed, (n, r)), persistent=False)
            self.register_buffer("B_sign", unpack_binary(B_sign_packed, (r, m)), persistent=False)
        else:
            self.register_buffer("A_sign", None, persistent=False)
            self.register_buffer("B_sign", None, persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        """ロード時に展開済み符号行列を再構築"""
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                       missing_keys, unexpected_keys, error_msgs)

        if hasattr(self, '_A_sign_shape') and self._A_sign_shape is not None:
            A_shape = tuple(self._A_sign_shape.tolist())
            B_shape = tuple(self._B_sign_shape.tolist())

            self.n, self.r = A_shape
            _, self.m = B_shape
            self.l = self.A_amp.shape[1] if hasattr(self, 'A_amp') else 1

            if getattr(self, '_preunpack', True):
                if not hasattr(self, 'A_sign') or self.A_sign is None:
                    self.register_buffer("A_sign", unpack_binary(self.A_sign_packed, A_shape), persistent=False)
                if not hasattr(self, 'B_sign') or self.B_sign is None:
                    self.register_buffer("B_sign", unpack_binary(self.B_sign_packed, B_shape), persistent=False)

    def _get_factor_matrices(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """因子行列 F, G を計算"""
        if self.A_sign is None:
            A_sign = unpack_binary(self.A_sign_packed, (self.n, self.r)).to(dtype)
        else:
            A_sign = self.A_sign.to(dtype)

        if self.B_sign is None:
            B_sign = unpack_binary(self.B_sign_packed, (self.r, self.m)).to(dtype)
        else:
            B_sign = self.B_sign.to(dtype)

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
        """重み行列を再構成"""
        F, G = self._get_factor_matrices(dtype)
        return F @ G


# =============================================================================
# MultipathMSVIDLinear層（Pパス対応）
# =============================================================================


class MultipathMSVIDLinear(nn.Module):
    """Pパス対応のMSVID推論層: W ≈ Σ_{p=1}^{P} W^{(p)}"""

    def __init__(
        self,
        params_list: List[MSVIDParams],
        preunpack: bool = True,
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
            MSVIDLinear(params, preunpack=preunpack)
            for params in params_list
        ])

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("bias", None)

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
        """重み行列を再構成"""
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
    ) -> "MultipathMSVIDLinear":
        """Build MultipathMSVIDLinear from MDBFResult.

        Args:
            result: MDBFResult from quantizer.
            bias: Optional bias tensor (from original Linear).
            device: Device to place the layer on.

        Returns:
            MultipathMSVIDLinear instance.
        """
        params_list = [
            MSVIDParams(
                A_sign=result.mdbf_A_sign[p],
                B_sign=result.mdbf_B_sign[p],
                A_amp=result.mdbf_A_amp[p],
                B_amp=result.mdbf_B_amp[p],
                Q_U_amp=result.mdbf_Q_U_amp[p],
                Q_V_amp=result.mdbf_Q_V_amp[p],
            )
            for p in range(result.P)
        ]
        return cls(params_list=params_list, bias=bias, device=device)


# =============================================================================
# レイヤー置換関数（QEP-DEV 互換、OneComp フレームワークでは不使用）
# =============================================================================


def create_mdbf_layer_from_linear(
    module: nn.Module,
    preunpack: bool = True
) -> Optional[MultipathMSVIDLinear]:
    """MSVID量子化済みLinear層からMultipathMSVIDLinearを作成"""
    if not hasattr(module, 'msvid_params'):
        return None
    if not getattr(module, 'is_quantized', False):
        return None

    params_list = module.msvid_params
    if not isinstance(params_list, list) or len(params_list) == 0:
        return None

    bias = module.bias.clone() if hasattr(module, 'bias') and module.bias is not None else None

    return MultipathMSVIDLinear(
        params_list=params_list,
        preunpack=preunpack,
        bias=bias,
    )


def replace_linear_with_mdbf(
    module: nn.Module,
    name: str,
    parent_module: nn.Module,
    preunpack: bool = True
) -> bool:
    """Linear層をMultipathMSVIDLinearに置換"""
    mdbf_layer = create_mdbf_layer_from_linear(module, preunpack=preunpack)

    if mdbf_layer is None:
        return False

    device = module.weight.device
    mdbf_layer = mdbf_layer.to(device)
    setattr(parent_module, name, mdbf_layer)

    del module.weight
    if hasattr(module, 'msvid_params'):
        del module.msvid_params

    return True


def replace_all_msvid_layers(
    model: nn.Module,
    preunpack: bool = True
) -> int:
    """モデル内のすべてのMSVID量子化層をMultipathMSVIDLinearに置換"""
    replaced_count = 0

    for parent_name, parent_module in model.named_modules():
        for name, module in list(parent_module.named_children()):
            if isinstance(module, (nn.Linear, transformers.Conv1D)):
                if replace_linear_with_mdbf(module, name, parent_module, preunpack):
                    replaced_count += 1
                    logger.debug(f"[MDBF] Replaced {parent_name}.{name} with MultipathMSVIDLinear")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"[MDBF] Total replaced layers: {replaced_count}")
    return replaced_count


# =============================================================================
# チェックポイント管理（QEP-DEV 互換、OneComp フレームワークでは不使用）
# =============================================================================


def save_msvid_weights(
    model: nn.Module,
    save_path: Path,
    packed: bool = True
) -> Dict[str, int]:
    """MSVID重みを保存"""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    weights = {}
    stats = {"layers": 0, "params": 0}

    for name, module in model.named_modules():
        if not (hasattr(module, 'msvid_params') and getattr(module, 'is_quantized', False)):
            continue

        params_list = module.msvid_params

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

    torch.save(weights, save_path / "msvid_weights.pt")

    metadata = {
        "packed": packed,
        "layers": stats["layers"],
        "params": stats["params"],
    }
    with open(save_path / "msvid_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"[MDBF] Saved {stats['layers']} layers to {save_path}")
    return stats


def load_msvid_weights(load_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """MSVID重みを読み込み"""
    load_path = Path(load_path)
    weights = torch.load(load_path / "msvid_weights.pt", map_location="cpu")

    with open(load_path / "msvid_metadata.json", "r") as f:
        metadata = json.load(f)

    return weights, metadata


# =============================================================================
# 検証関数
# =============================================================================


def verify_binary_values(params: MSVIDParams) -> Tuple[bool, str]:
    """S_A, S_Bが適切に±1の二値になっているか検証"""
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


def verify_all_params(params_list: List[MSVIDParams]) -> Tuple[bool, List[str]]:
    """全パスのパラメータを検証"""
    all_valid = True
    messages = []

    for i, params in enumerate(params_list):
        valid, msg = verify_binary_values(params)
        messages.append(f"Path {i}: {msg}")
        if not valid:
            all_valid = False

    return all_valid, messages
