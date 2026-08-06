"""Read an OneComp quantized checkpoint and recover per-layer GPTQ tensors.

This module parses a directory produced by ``Runner.save_quantized_model`` (HF
``config.json`` with a ``quantization_config`` plus AutoGPTQ-style safetensors)
and yields fully unpacked GPTQ layers: integer codes ``(out, in)``, per-group
scales and (restored) zero points. The unpacking reuses OneComp's own
``gptq_layer`` helpers so the recovered values match ``GPTQLinear.forward``.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from glob import glob
from logging import getLogger
from typing import Dict, Iterator

import torch

from onecomp.quantizer.gptq.gptq_layer import unpack_int_weights, unpack_zeros

logger = getLogger(__name__)


@dataclass
class GPTQLayer:
    """One unpacked GPTQ linear layer."""

    name: str  # HF module name, e.g. "model.layers.0.self_attn.q_proj"
    q_int: torch.Tensor  # (out_features, in_features) int32 codes in [0, 2^wbits-1]
    scales: torch.Tensor  # (num_groups, out_features) float
    zeros: torch.Tensor  # (num_groups, out_features) restored integer zero points
    wbits: int
    sym: bool
    groupsize: int
    actorder: bool
    in_features: int
    out_features: int
    g_idx: torch.Tensor  # (in_features,) group index per input column

    @property
    def weight_key(self) -> str:
        return f"{self.name}.weight"


def _get(cfg: dict, *keys, default=None):
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def load_quant_config(save_directory: str) -> dict:
    """Load the ``quantization_config`` block from a saved model's config.json."""
    with open(os.path.join(save_directory, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    quant_config = config.get("quantization_config")
    if quant_config is None:
        raise ValueError(f"{save_directory}/config.json has no 'quantization_config'.")
    return quant_config


@dataclass
class QuantMeta:
    """High-level descriptor of a saved OneComp checkpoint (for export routing)."""

    quant_method: str  # gptq / mixed_gptq / dbf / onebit / autobit / ...
    rotated: bool  # rotation pre-processing was applied (online Hadamard on down_proj)
    fp32_had: bool  # online Hadamard computed in fp32
    is_gptq_family: bool  # weights use the AutoGPTQ qweight/qzeros/scales layout
    actorder: bool = False  # desc_act / act-order: input columns permuted per layer

    @property
    def supports_direct(self) -> bool:
        """True when layers can be packed losslessly into GGUF blocks."""
        return self.is_gptq_family and not self.rotated


# quant_method values whose tensors use the AutoGPTQ qweight/qzeros/scales layout
# (so iter_gptq_layers can read them): GPTQ, QEP (same codes), JointQ, RTN, mixed.
_GPTQ_FAMILY = {"gptq", "mixed_gptq", "jointq", "rtn"}


def configured_bit_widths(quant_config: dict) -> set:
    """All weight bit-widths in a checkpoint (default + per-layer ``quantization_bits``)."""
    bits = {int(_get(quant_config, "bits", "wbits", default=4))}
    qbits = quant_config.get("quantization_bits")
    if isinstance(qbits, list):
        for layer_map in qbits:
            if not isinstance(layer_map, dict):
                continue
            for info in layer_map.values():
                if isinstance(info, dict) and "bits" in info:
                    bits.add(int(info["bits"]))
    return bits


def needs_mixed_export(quant_config: dict) -> bool:
    """True when any layer cannot use the lossless direct GGUF block packer.

    Triggers include act-order, bit-widths with no legacy GGUF mapping (2/3/5/6),
    and 8-bit asymmetric quantization (no Q8_1 legacy type).
    """
    if bool(_get(quant_config, "actorder", "desc_act", default=False)):
        return True
    sym = bool(_get(quant_config, "sym", default=True))
    for b in configured_bit_widths(quant_config):
        if b in (2, 3, 5, 6):
            return True
        if b == 8 and not sym:
            return True
    return False


def read_quant_meta(save_directory: str) -> QuantMeta:
    """Read the quantization method / rotation flags from a saved checkpoint."""
    cfg = load_quant_config(save_directory)
    method = str(_get(cfg, "quant_method", default="gptq")).lower()
    # The loader strips a "mixed_" prefix for dbf; normalize the family check.
    base = (
        method[len("mixed_") :]
        if method.startswith("mixed_") and method != "mixed_gptq"
        else method
    )
    is_gptq = base in _GPTQ_FAMILY or method == "mixed_gptq"
    return QuantMeta(
        quant_method=method,
        rotated=bool(_get(cfg, "rotated", default=False)),
        fp32_had=bool(_get(cfg, "fp32_had", default=False)),
        is_gptq_family=is_gptq,
        actorder=bool(_get(cfg, "actorder", "desc_act", default=False)),
    )


def _load_state_dict(save_directory: str) -> Dict[str, torch.Tensor]:
    """Load all tensors from one or more safetensors shards into a flat dict."""
    from safetensors.torch import load_file

    shards = sorted(glob(os.path.join(save_directory, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {save_directory}")
    state: Dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(shard, device="cpu"))
    return state


def _per_layer_overrides(quant_config: dict) -> Dict[str, Dict[str, int]]:
    """Build a {module_name: {bits, group_size}} map from ``quantization_bits``.

    ``quantization_bits`` is a list indexed by transformer block; each entry maps
    a submodule suffix (e.g. ``self_attn.q_proj``) to ``{bits, method, params}``.
    """
    overrides: Dict[str, Dict[str, int]] = {}
    qbits = quant_config.get("quantization_bits")
    if not isinstance(qbits, list):
        return overrides
    for layer_idx, layer_map in enumerate(qbits):
        if not isinstance(layer_map, dict):
            continue
        for suffix, info in layer_map.items():
            if not isinstance(info, dict):
                continue
            name = f"model.layers.{layer_idx}.{suffix}"
            entry: Dict[str, int] = {}
            if "bits" in info:
                entry["bits"] = int(info["bits"])
            params = info.get("params") or {}
            if "group_size" in params:
                entry["group_size"] = int(params["group_size"])
            if entry:
                overrides[name] = entry
    return overrides


def iter_gptq_layers(save_directory: str) -> Iterator[GPTQLayer]:
    """Yield every GPTQ-quantized linear in a saved OneComp model, fully unpacked.

    Only ``gptq`` / ``mixed_gptq`` checkpoints expose ``qweight`` tensors; other
    methods (dbf/onebit) are skipped here and must use the dequantize path.
    """
    quant_config = load_quant_config(save_directory)
    state = _load_state_dict(save_directory)

    default_bits = int(_get(quant_config, "bits", "wbits", default=4))
    default_gs = int(_get(quant_config, "group_size", "groupsize", default=128))
    sym = bool(_get(quant_config, "sym", default=True))
    actorder = bool(_get(quant_config, "actorder", "desc_act", default=False))
    checkpoint_format = str(_get(quant_config, "checkpoint_format", default="gptq"))
    overrides = _per_layer_overrides(quant_config)

    qweight_keys = sorted(k for k in state if k.endswith(".qweight"))
    for qk in qweight_keys:
        name = qk[: -len(".qweight")]
        qweight = state[qk]
        scales = state[f"{name}.scales"]
        qzeros = state[f"{name}.qzeros"]

        ov = overrides.get(name, {})
        wbits = int(ov.get("bits", default_bits))
        groupsize = int(ov.get("group_size", default_gs))

        out_features = qweight.shape[1]
        in_features = qweight.shape[0] * 32 // wbits

        q_int = unpack_int_weights(qweight, wbits, (out_features, in_features))
        zeros_unpacked = unpack_zeros(qzeros, wbits, out_features)
        if checkpoint_format != "gptq_v2":
            mask = (1 << wbits) - 1
            zeros_unpacked = (zeros_unpacked + 1) & mask

        perm = state.get(f"{name}.perm")
        g_idx_t = state.get(f"{name}.g_idx")
        if g_idx_t is not None:
            g_idx = g_idx_t.to(torch.long)
        elif groupsize == -1:
            g_idx = torch.zeros(in_features, dtype=torch.long)
        elif actorder and perm is not None:
            g_idx = (torch.argsort(perm) // groupsize).to(torch.long)
        else:
            g_idx = torch.arange(in_features, dtype=torch.long) // groupsize

        yield GPTQLayer(
            name=name,
            q_int=q_int.to(torch.int32),
            scales=scales.to(torch.float32),
            zeros=zeros_unpacked.to(torch.float32),
            wbits=wbits,
            sym=sym,
            groupsize=groupsize,
            actorder=actorder,
            in_features=in_features,
            out_features=out_features,
            g_idx=g_idx,
        )


def dequantize_layer(layer: GPTQLayer) -> torch.Tensor:
    """Reconstruct the dense fp32 weight (out, in) for one GPTQ layer.

    Mirrors ``GPTQLinear.forward``: weight = scale * (q - zero), expanded per
    group via g_idx. Used for the dequantize->fp16 fallback path and for
    numerical round-trip verification.
    """
    g_idx = layer.g_idx
    scale_exp = layer.scales[g_idx, :].T  # (out, in)
    zero_exp = layer.zeros[g_idx, :].T  # (out, in)
    return scale_exp * (layer.q_int.float() - zero_exp)
