"""Copyright 2025-2026 Fujitsu Ltd."""

import os
from typing import Any, List

import torch
from torch.nn import Parameter

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm_plugins.utils.module import (
    _lookup_module_config,
    _parse_layer_and_module,
    _validate_quant_config_within_shard,
)
from vllm_plugins.dbf.modules.naive import unpack_sign_bits

logger = init_logger(__name__)


def _parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "")
    if value == "":
        return default
    return value.strip().upper() in {"1", "TRUE", "YES", "ON"}


_USE_NAIVE = _parse_bool_env("ONECOMP_DBF_NAIVE_LINEAR", default=False)


def _try_import_gemlite():
    try:
        from vllm_plugins.dbf.modules.gemlite_linear import (
            get_gemlite_linear,
            GROUP_SIZE,
        )
    except Exception as exc:  # pragma: no cover - best effort import
        logger.warning("DBF gemlite unavailable; falling back to naive. (%s)", exc)
        return None
    return get_gemlite_linear, GROUP_SIZE


def _shard_id_to_index(shard_id: Any) -> int:
    if shard_id is None:
        return 0
    if isinstance(shard_id, int):
        return shard_id
    if isinstance(shard_id, str):
        mapping = {"q": 0, "k": 1, "v": 2}
        if shard_id not in mapping:
            raise ValueError(f"Unknown shard id: {shard_id}")
        return mapping[shard_id]
    raise TypeError(f"Unsupported shard id type: {type(shard_id)}")


def _compute_mid_features(w_bit: float, in_features: int, out_features: int) -> int:
    mid_features1 = int(w_bit * (in_features * out_features) / (in_features + out_features))
    return min(min(in_features, out_features), mid_features1)


class DBFLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: "DbfConfig"):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # DBF implementation in this plugin assumes TP size == 1.
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size != 1:
            raise ValueError("DBF quantization currently supports only tensor_parallel_size=1.")

        del params_dtype  # not used (DBF weights are stored as fp16/uint8)
        del output_size  # derived from output_partition_sizes

        part_count = len(output_partition_sizes)
        in_features = input_size

        mod_cfg = getattr(layer, "_dbf_mod_cfg", None)
        if mod_cfg is None:
            logger.warning("DBF module config is missing on layer; defaulting bits=1.0.")
            w_bit = 1.0
        else:
            w_bit = mod_cfg.get("bits", 0)

        mid_sizes = [
            _compute_mid_features(w_bit, in_features, out_features)
            for out_features in output_partition_sizes
        ]
        bp1_sizes = [(mid * in_features + 7) // 8 for mid in mid_sizes]
        bp3_sizes = [
            (out_features * mid + 7) // 8
            for out_features, mid in zip(output_partition_sizes, mid_sizes)
        ]

        def _offsets(sizes: List[int]) -> List[int]:
            offsets = []
            current = 0
            for size in sizes:
                offsets.append(current)
                current += size
            return offsets

        scaling2_offsets = _offsets(mid_sizes)
        scaling4_offsets = _offsets(output_partition_sizes)
        bp1_offsets = _offsets(bp1_sizes)
        bp3_offsets = _offsets(bp3_sizes)

        layer._dbf_meta = {
            "part_count": part_count,
            "in_features": in_features,
            "out_sizes": output_partition_sizes,
            "mid_sizes": mid_sizes,
            "scaling2_offsets": scaling2_offsets,
            "scaling4_offsets": scaling4_offsets,
            "bp1_offsets": bp1_offsets,
            "bp3_offsets": bp3_offsets,
            "bp1_sizes": bp1_sizes,
            "bp3_sizes": bp3_sizes,
        }

        def _make_loader(kind: str, offsets: List[int], sizes: List[int]):
            def _loader(param: Parameter, loaded_weight: torch.Tensor, shard_id: Any = None):
                part_idx = _shard_id_to_index(shard_id)
                if part_count > 1 and shard_id is None:
                    raise ValueError(f"DBF expects shard_id for fused module parameter {kind}.")
                if kind == "scaling0":
                    if param.data.ndim == 2:
                        param.data[part_idx].copy_(loaded_weight)
                    else:
                        param.data.copy_(loaded_weight)
                    return
                offset = offsets[part_idx]
                size = sizes[part_idx]
                param.data.narrow(0, offset, size).copy_(loaded_weight)

            return _loader

        # scaling0: store per-part scaling0 if fused
        if part_count > 1:
            scaling0 = Parameter(
                torch.empty((part_count, in_features), dtype=torch.float16),
                requires_grad=False,
            )
        else:
            scaling0 = Parameter(
                torch.empty((in_features,), dtype=torch.float16),
                requires_grad=False,
            )
        set_weight_attrs(
            scaling0,
            extra_weight_attrs | {"weight_loader": _make_loader("scaling0", [], [])},
        )
        layer.register_parameter("scaling0", scaling0)

        scaling2 = Parameter(
            torch.empty((sum(mid_sizes),), dtype=torch.float16),
            requires_grad=False,
        )
        set_weight_attrs(
            scaling2,
            extra_weight_attrs
            | {"weight_loader": _make_loader("scaling2", scaling2_offsets, mid_sizes)},
        )
        layer.register_parameter("scaling2", scaling2)

        scaling4 = Parameter(
            torch.empty((sum(output_partition_sizes),), dtype=torch.float16),
            requires_grad=False,
        )
        set_weight_attrs(
            scaling4,
            extra_weight_attrs
            | {
                "weight_loader": _make_loader("scaling4", scaling4_offsets, output_partition_sizes)
            },
        )
        layer.register_parameter("scaling4", scaling4)

        bp1 = Parameter(
            torch.empty((sum(bp1_sizes),), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            bp1,
            extra_weight_attrs | {"weight_loader": _make_loader("bp1", bp1_offsets, bp1_sizes)},
        )
        layer.register_parameter("bp1", bp1)

        bp3 = Parameter(
            torch.empty((sum(bp3_sizes),), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            bp3,
            extra_weight_attrs | {"weight_loader": _make_loader("bp3", bp3_offsets, bp3_sizes)},
        )
        layer.register_parameter("bp3", bp3)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_dbf_initialized", False):
            return
        layer._dbf_initialized = True
        layer._dbf_use_gemlite = False

        if _USE_NAIVE:
            return

        gemlite = _try_import_gemlite()
        if gemlite is None:
            return
        get_gemlite_linear, group_size = gemlite

        meta = getattr(layer, "_dbf_meta", None)
        if meta is None:
            return

        parts = []
        for idx in range(meta["part_count"]):
            mid = meta["mid_sizes"][idx]
            out_size = meta["out_sizes"][idx]
            bp1_offset = meta["bp1_offsets"][idx]
            bp3_offset = meta["bp3_offsets"][idx]
            bp1_size = meta["bp1_sizes"][idx]
            bp3_size = meta["bp3_sizes"][idx]

            bp1_packed = layer.bp1.narrow(0, bp1_offset, bp1_size)
            bp3_packed = layer.bp3.narrow(0, bp3_offset, bp3_size)

            bp1_int8 = unpack_sign_bits(bp1_packed, (mid, meta["in_features"])).to(torch.int8)
            bp3_int8 = unpack_sign_bits(bp3_packed, (out_size, mid)).to(torch.int8)

            binary1 = get_gemlite_linear(bp1_int8)
            binary3 = get_gemlite_linear(bp3_int8)
            if binary1 is None or binary3 is None:
                logger.warning("DBF gemlite creation failed; falling back to naive.")
                return
            parts.append((binary1, binary3))

        layer._dbf_gemlite_parts = parts
        layer._dbf_use_gemlite = True

    def _apply_naive(
        self,
        x: torch.Tensor,
        scaling0: torch.Tensor,
        scaling2: torch.Tensor,
        scaling4: torch.Tensor,
        bp1: torch.Tensor,
        bp3: torch.Tensor,
        in_features: int,
        mid_features: int,
        out_features: int,
    ) -> torch.Tensor:
        x = x * scaling0.to(x.dtype)
        w1 = unpack_sign_bits(bp1, (mid_features, in_features)).to(x.dtype)
        x = x.matmul(w1.t())
        x = x * scaling2.to(x.dtype)
        w3 = unpack_sign_bits(bp3, (out_features, mid_features)).to(x.dtype)
        x = x.matmul(w3.t())
        x = x * scaling4.to(x.dtype)
        return x

    def _apply_gemlite(
        self,
        x: torch.Tensor,
        scaling0: torch.Tensor,
        scaling2: torch.Tensor,
        scaling4: torch.Tensor,
        binary1: torch.nn.Module,
        binary3: torch.nn.Module,
        group_size: int,
        mid_features: int,
        out_features: int,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x * scaling0.to(x.dtype)
        if x.dtype != torch.bfloat16:
            x = x.bfloat16()
        if x.shape[-1] % group_size != 0:
            pad = group_size - (x.shape[-1] % group_size)
            x = torch.nn.functional.pad(x, (0, pad), mode="constant", value=0)
        x = binary1(x)
        x = x[..., :mid_features]
        x = x * scaling2.to(x.dtype)
        if x.dtype != torch.bfloat16:
            x = x.bfloat16()
        if x.shape[-1] % group_size != 0:
            pad = group_size - (x.shape[-1] % group_size)
            x = torch.nn.functional.pad(x, (0, pad), mode="constant", value=0)
        x = binary3(x)
        x = x[..., :out_features]
        x = x * scaling4.to(x.dtype)
        if x.dtype != input_dtype:
            x = x.to(dtype=input_dtype)
        return x

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        meta = getattr(layer, "_dbf_meta", None)
        if meta is None:
            raise RuntimeError("DBF meta not initialized on layer.")

        part_count = meta["part_count"]
        outputs = []

        use_gemlite = getattr(layer, "_dbf_use_gemlite", False)
        gemlite = _try_import_gemlite() if use_gemlite else None
        group_size = gemlite[1] if gemlite is not None else 128

        for idx in range(part_count):
            if layer.scaling0.ndim == 2:
                scaling0 = layer.scaling0[idx]
            else:
                scaling0 = layer.scaling0

            scaling2 = layer.scaling2.narrow(
                0, meta["scaling2_offsets"][idx], meta["mid_sizes"][idx]
            )
            scaling4 = layer.scaling4.narrow(
                0, meta["scaling4_offsets"][idx], meta["out_sizes"][idx]
            )
            bp1 = layer.bp1.narrow(0, meta["bp1_offsets"][idx], meta["bp1_sizes"][idx])
            bp3 = layer.bp3.narrow(0, meta["bp3_offsets"][idx], meta["bp3_sizes"][idx])

            if use_gemlite and hasattr(layer, "_dbf_gemlite_parts"):
                binary1, binary3 = layer._dbf_gemlite_parts[idx]
                out = self._apply_gemlite(
                    x,
                    scaling0,
                    scaling2,
                    scaling4,
                    binary1,
                    binary3,
                    group_size,
                    meta["mid_sizes"][idx],
                    meta["out_sizes"][idx],
                )
            else:
                out = self._apply_naive(
                    x,
                    scaling0,
                    scaling2,
                    scaling4,
                    bp1,
                    bp3,
                    meta["in_features"],
                    meta["mid_sizes"][idx],
                    meta["out_sizes"][idx],
                )
            outputs.append(out)

        out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if bias is not None:
            out = out + bias
        return out


@register_quantization_config("dbf")
class DbfConfig(QuantizationConfig):
    def __init__(self, quantization_bits: List[dict[str, Any]]):
        super().__init__()
        self.quantization_bits = quantization_bits
        self._method = DBFLinearMethod(self)

    @classmethod
    def get_name(cls) -> str:
        return "dbf"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DbfConfig":
        quantization_bits = config.get("quantization_bits", [])
        return cls(quantization_bits=quantization_bits)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> LinearMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None

        layer_idx, module_suffix = _parse_layer_and_module(prefix)
        if layer_idx is None:
            return UnquantizedLinearMethod()

        mod_cfg = _lookup_module_config(self.quantization_bits, layer_idx, module_suffix)
        if mod_cfg is None:
            logger.debug("No module config found for %s, using UnquantizedLinearMethod.", prefix)
            return UnquantizedLinearMethod()

        if not _validate_quant_config_within_shard(
            self.quantization_bits, layer_idx, module_suffix
        ):
            raise ValueError(
                f"Detected some but not all shards of {prefix} "
                "are quantized. All shards of fused layers "
                "to have the same precision."
            )

        bits = mod_cfg.get("bits", 0)
        method = mod_cfg.get("method", "dbf")

        if bits == 0:
            logger.info(
                "DBF config for %s has bits=0, using UnquantizedLinearMethod.",
                prefix,
            )
            return UnquantizedLinearMethod()

        if method != "dbf":
            logger.warning(
                "DBF config for %s has unsupported method=%s, falling back to UnquantizedLinearMethod.",
                prefix,
                method,
            )
            return UnquantizedLinearMethod()

        # create_weights() has no prefix argument, so carry per-module config on layer.
        layer._dbf_prefix = prefix
        layer._dbf_mod_cfg = mod_cfg

        return self._method


def register_vllm_plugin():
    # Register DBF quantization with vLLM's plugin system.
    register_quantization_config(DbfConfig)
