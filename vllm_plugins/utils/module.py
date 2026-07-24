"""Copyright 2025-2026 Fujitsu Ltd."""

import re

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

# Prefixes belonging to vision/audio encoders -- never quantized.
_NON_TEXT_PREFIXES = ("vision_tower", "vision_model", "multi_modal_projector", "audio")

# Matches a per-expert MoE projection name. "experts" sits at different
# depths per architecture (e.g. "mlp.experts.N.down_proj" vs top-level
# "experts.N.down_proj"), hence the optional dotted prefix.
_MOE_EXPERT_RE = re.compile(r"^(?:[\w.]+\.)?experts\.\d+\.(gate_proj|up_proj|down_proj)$")

# Map from vLLM's fused module leaf name to its constituent leaf names.
# Constituents are substituted into module_suffix at the fused name's own
# position (not looked up under a hardcoded parent path), since fusion
# doesn't change the parent path, only the leaf
# (e.g. "self_attn.q_proj" -> "self_attn.qkv_proj").
_FUSED_TO_CONSTITUENTS = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
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
        if fused_name not in module_suffix:
            continue
        for constituent in constituents:
            candidate = module_suffix.replace(fused_name, constituent)
            if candidate in layer_cfg:
                return layer_cfg[candidate]
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


def _lookup_moe_config(
    quantization_bits: list[dict], layer_idx: int, num_experts: int
) -> dict | None:
    """Aggregate one uniform GPTQ config across all experts in a FusedMoE layer.

    Returns:
        None if the layer's experts are not quantized at all.

    Raises:
        ValueError: If only some experts are quantized, or experts disagree
            on bits/method/group_size -- not representable by one MoE kernel.
    """
    if layer_idx >= len(quantization_bits):
        return None
    layer_cfg = quantization_bits[layer_idx]

    expert_cfgs = {name: cfg for name, cfg in layer_cfg.items() if _MOE_EXPERT_RE.match(name)}
    if not expert_cfgs:
        return None

    # *3 : gate_proj, up_proj, and down_proj per expert (see _MOE_EXPERT_RE)
    expected = num_experts * 3
    if len(expert_cfgs) != expected:
        raise ValueError(
            f"Layer {layer_idx}: only {len(expert_cfgs)}/{expected} expert "
            "projections have a quantization config. vLLM's FusedMoE "
            "kernel requires every expert in a layer to share the same "
            "quantization scheme; a partially-quantized MoE layer cannot "
            "be served."
        )

    variants = {
        (cfg["bits"], cfg["method"], cfg.get("params", {}).get("group_size"))
        for cfg in expert_cfgs.values()
    }
    if len(variants) != 1:
        raise ValueError(
            f"Layer {layer_idx}: experts have inconsistent GPTQ configs "
            f"{sorted(variants)}; vLLM's FusedMoE kernel requires one "
            "scheme per layer."
        )

    bits, method, group_size = next(iter(variants))
    return {"bits": bits, "method": method, "group_size": group_size}


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
        for constituent in constituents:
            # If at least one sub-module in the shard has a quantization config,
            # all sub-modules in the shard must also have one.
            candidate = module_suffix.replace(fused_name, constituent)
            if candidate not in layer_cfg:
                return False

            cfg = layer_cfg[candidate]
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
