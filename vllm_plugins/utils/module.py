"""Copyright 2025-2026 Fujitsu Ltd."""

import re

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

# Prefixes belonging to vision/audio encoders -- never quantized.
_NON_TEXT_PREFIXES = ("vision_tower", "vision_model", "multi_modal_projector", "audio")

# Map from vLLM's fused module name to the constituent config keys.
# When vLLM fuses q/k/v into qkv_proj, the prefix becomes "...qkv_proj".
# We look up the first constituent's config as representative.
_FUSED_TO_CONSTITUENTS = {
    "qkv_proj": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "gate_up_proj": ["mlp.gate_proj", "mlp.up_proj"],
}


def _parse_layer_and_module(prefix: str) -> tuple[int | None, str | None]:
    if any(p in prefix for p in _NON_TEXT_PREFIXES):
        return None, None
    m = _LAYER_RE.search(prefix)
    if m is None:
        return None, None
    layer_idx = int(m.group(1))
    after = prefix[m.end() :]
    return layer_idx, after


def _resolve_fused_bits(layer_cfg: dict, module_suffix: str) -> dict | None:
    for fused_name, constituents in _FUSED_TO_CONSTITUENTS.items():
        if fused_name in module_suffix:
            for name in constituents:
                if name in layer_cfg:
                    return layer_cfg[name]
            return None
    return None


def _lookup_module_config(
    quantization_bits: list[dict], layer_idx: int, module_suffix: str
) -> dict | None:
    if layer_idx >= len(quantization_bits):
        return None
    layer_cfg = quantization_bits[layer_idx]
    for name, cfg in layer_cfg.items():
        if module_suffix.startswith(name):
            return cfg
    fused = _resolve_fused_bits(layer_cfg, module_suffix)
    if fused is not None:
        return fused
    if "_all" in layer_cfg:
        return layer_cfg["_all"]
    return None


# Check whether all quantization configs within the same shard are identical
def _validate_quant_config_within_shard(
    quantization_bits: list[dict], layer_idx: int, module_suffix: str
) -> bool:
    if layer_idx >= len(quantization_bits):
        return False
    layer_cfg = quantization_bits[layer_idx]

    for fused_name, constituents in _FUSED_TO_CONSTITUENTS.items():
        # If fused_name is found in module_suffix, verify that all configs in the shard are identical.
        # Each config has 'bits' and 'method' fields; both must match across sub-modules.

        # If not a fused module, skip the within-shard check.
        if fused_name not in module_suffix:
            continue

        configs = []
        for name in constituents:
            # If at least one sub-module in the shard has a quantization config,
            # all sub-modules in the shard must also have one.
            if name not in layer_cfg:
                return False

            cfg = layer_cfg[name]
            if cfg is None:
                return False

            configs.append(cfg)

        # If configs is empty all sub-modules are unquantized, which is fine.
        # Verify that all quantization configs within the shard are identical.
        for cfg in configs:
            if cfg != configs[0]:
                return False
        return True

    # Not a fused module: no within-shard check needed.
    return True
