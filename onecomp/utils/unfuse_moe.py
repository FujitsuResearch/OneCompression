"""
Unfuse / fuse 3D fused MoE expert parameters for quantization and vLLM export.

Copyright 2025-2026 Fujitsu Ltd.

"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

_MOE_EXPERT_KEY_RE = re.compile(r"\.mlp\.experts\.")


class _ExpertMLP(nn.Module):
    """Single MoE expert with gate/up/down projections."""

    __slots__ = ("act_fn",)

    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear, act_fn):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn


class _UnfusedExperts(nn.Module):
    """Drop-in replacement for fused 3D expert modules."""

    def __init__(
        self,
        num_experts: int,
        experts: list,
        act_fn=None,
        combine_fn=None,
    ):
        super().__init__()
        self._num_experts = num_experts
        for i, expert in enumerate(experts):
            self.add_module(str(i), expert)
        # act_fn: Gemma4 etc. — forward does act_fn(gate) * up (unary activation).
        # combine_fn: GPT-OSS — (gate, up) -> hidden; not expressible as act_fn(gate) * up.
        self.act_fn = act_fn
        self.combine_fn = combine_fn

    def __len__(self):
        return self._num_experts

    def __getitem__(self, idx):
        return getattr(self, str(int(idx)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor | None = None,
        top_k_weights: torch.Tensor | None = None,
        router_indices: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # MoE models disagree on router arg names (Gemma4 etc.: top_k_*;
        # GPT-OSS: router_indices / routing_weights). Accept both so
        # _UnfusedExperts stays a drop-in replacement after unfuse.
        if top_k_index is None:
            top_k_index = router_indices
        if top_k_weights is None:
            top_k_weights = routing_weights

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index,
                num_classes=self._num_experts,
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)),
                0,
            ).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self._num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            expert = self[expert_idx]
            gate = expert.gate_proj(current_state)
            up = expert.up_proj(current_state)
            if self.combine_fn is not None:
                current_hidden_states = self.combine_fn(gate, up)
            else:
                current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = expert.down_proj(current_hidden_states)

            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )

        return final_hidden_states


class _GenericFusedExperts(nn.Module):
    """Minimal fused MoE container for non-GPT-OSS architectures."""

    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor, act_fn):
        super().__init__()
        self.gate_up_proj = nn.Parameter(gate_up_proj)
        self.down_proj = nn.Parameter(down_proj)
        self.act_fn = act_fn


def _is_fused_experts(module: nn.Module) -> bool:
    """Return True if module holds fused 3D expert parameters."""
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    return (
        isinstance(gate_up, nn.Parameter)
        and isinstance(down, nn.Parameter)
        and gate_up.ndim == 3
        and down.ndim == 3
    )


def _is_unfused_experts(module: nn.Module) -> bool:
    return isinstance(module, _UnfusedExperts)


@lru_cache(maxsize=1)
def _gpt_oss_experts_type():
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

        return GptOssExperts
    except ImportError:
        return None


def _is_gpt_oss_experts(module: nn.Module) -> bool:
    """Return True if module is transformers' GptOssExperts."""
    cls = _gpt_oss_experts_type()
    return cls is not None and isinstance(module, cls)


def _copy_into_gpt_oss_experts(
    fused: nn.Module,
    gate_up_3d: torch.Tensor,
    gate_up_bias_2d: torch.Tensor,
    down_3d: torch.Tensor,
    down_bias_2d: torch.Tensor,
) -> None:
    """Copy fused tensors into existing GptOssExperts Parameters in-place."""
    with torch.no_grad():
        fused.gate_up_proj.copy_(gate_up_3d.to(dtype=fused.gate_up_proj.dtype))
        fused.gate_up_proj_bias.copy_(gate_up_bias_2d.to(dtype=fused.gate_up_proj_bias.dtype))
        fused.down_proj.copy_(down_3d.to(dtype=fused.down_proj.dtype))
        fused.down_proj_bias.copy_(down_bias_2d.to(dtype=fused.down_proj_bias.dtype))


def _ensure_unique_fused_moe_parameters(model: nn.Module) -> None:
    """Give each fused MoE layer independent parameter storage after fuse."""
    param_locations: dict[int, list[tuple[nn.Module, str]]] = {}
    for module in model.modules():
        if not (_is_gpt_oss_experts(module) or _is_fused_experts(module)):
            continue
        for pname in ("gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias"):
            param = getattr(module, pname, None)
            if not isinstance(param, nn.Parameter):
                continue
            param_locations.setdefault(id(param), []).append((module, pname))

    for locations in param_locations.values():
        if len(locations) == 1:
            module, pname = locations[0]
            param = getattr(module, pname)
            with torch.no_grad():
                param.data = param.detach().clone()
            continue
        source = getattr(locations[0][0], locations[0][1]).detach().clone()
        for module, pname in locations:
            module._parameters[pname] = nn.Parameter(
                source.clone(),
                requires_grad=False,
            )


def _purge_orphan_parameters(model: nn.Module) -> None:
    """Remove ``name$`` ghost parameters left by Parameter replacement."""
    for module in model.modules():
        for key in list(module._parameters.keys()):
            if key.endswith("$"):
                del module._parameters[key]


def _infer_fused_moe_dtype(model: nn.Module) -> torch.dtype | None:
    """Infer the activation dtype used by non-MoE layers in *model*."""
    for name, param in model.named_parameters():
        if ".mlp.experts." in name:
            continue
        if param.dtype in (torch.float16, torch.bfloat16):
            return param.dtype
    return None


def _cast_fused_moe_parameters(model: nn.Module, dtype: torch.dtype | None) -> None:
    """Cast fused MoE expert tensors to *dtype* (e.g. after dequant fuse or load)."""
    if dtype is None:
        return
    for module in model.modules():
        if not (_is_gpt_oss_experts(module) or _is_fused_experts(module)):
            continue
        for pname in ("gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias"):
            param = getattr(module, pname, None)
            if isinstance(param, nn.Parameter) and param.dtype != dtype:
                with torch.no_grad():
                    param.data = param.detach().to(dtype=dtype)


def _checkpoint_uses_fused_moe(state_dict: dict[str, torch.Tensor]) -> bool:
    """Return True when the checkpoint stores fused 3D expert tensors."""
    if any(k.endswith(".mlp.experts.gate_up_proj") for k in state_dict):
        return True
    if "gate_up_proj$" in state_dict or "down_proj$" in state_dict:
        return True
    if any(_MOE_EXPERT_KEY_RE.search(k) and ".qweight" in k for k in state_dict):
        return False
    if any(k.endswith(".mlp.experts.gate_up_proj_bias") for k in state_dict):
        return True
    return False


def _expand_deduped_moe_keys(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Expand legacy ``gate_up_proj$`` / ``down_proj$`` keys to per-layer names."""
    shared_suffixes = {
        "gate_up_proj$": "gate_up_proj",
        "down_proj$": "down_proj",
    }
    for shared_key, suffix in shared_suffixes.items():
        if shared_key not in state_dict:
            continue
        tensor = state_dict.pop(shared_key)
        for name, module in model.named_modules():
            if not name:
                continue
            if not (_is_gpt_oss_experts(module) or _is_fused_experts(module)):
                continue
            state_dict[f"{name}.{suffix}"] = tensor
    return state_dict


def validate_fused_moe_checkpoint_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
) -> None:
    """Raise if a checkpoint state_dict is missing fused MoE keys or has ``$`` keys."""
    bad_keys = [k for k in state_dict if k.endswith("$")]
    if bad_keys:
        raise RuntimeError(
            "Fused MoE checkpoint has orphaned/shared tensor keys: "
            f"{bad_keys[:5]}{'...' if len(bad_keys) > 5 else ''}"
        )

    missing: list[str] = []
    gate_ptrs: dict[int, list[str]] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        if not (_is_gpt_oss_experts(module) or _is_fused_experts(module)):
            continue
        for suffix in ("gate_up_proj", "down_proj"):
            key = f"{name}.{suffix}"
            if key not in state_dict:
                missing.append(key)
            else:
                ptr = state_dict[key].untyped_storage().data_ptr()
                gate_ptrs.setdefault(ptr, []).append(key)
    if missing:
        raise RuntimeError(
            "Fused MoE checkpoint missing expert weight keys: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    shared = [names for names in gate_ptrs.values() if len(names) > 1]
    if shared:
        raise RuntimeError(
            "Fused MoE layers share expert weight storage across layers: "
            f"{shared[0][:3]}{'...' if len(shared[0]) > 3 else ''}"
        )


def validate_fused_moe_state_dict(model: nn.Module) -> None:
    """Raise if fused MoE weights are missing or deduped into ``$`` keys."""
    _purge_orphan_parameters(model)
    validate_fused_moe_checkpoint_state_dict(model.state_dict(), model)


def verify_saved_moe_checkpoint(save_directory: str | Path) -> int:
    """Verify safetensors on disk have per-layer fused MoE weights (no ``$`` keys).

    Returns the number of per-layer ``gate_up_proj`` tensors found.
    """
    from safetensors import safe_open

    save_path = Path(save_directory)
    shard_paths = sorted(save_path.glob("model*.safetensors"))
    if not shard_paths:
        raise RuntimeError(f"No safetensors shards under {save_directory}")

    keys: list[str] = []
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt") as f:
            keys.extend(f.keys())

    layer_gate = [k for k in keys if k.endswith(".mlp.experts.gate_up_proj") and ".layers." in k]
    bad_keys = [k for k in keys if k.endswith("$")]
    if bad_keys:
        raise RuntimeError(
            "Fused MoE checkpoint has deduped keys: "
            f"{bad_keys[:5]}{'...' if len(bad_keys) > 5 else ''}"
        )
    if not layer_gate:
        raise RuntimeError("Fused MoE checkpoint missing per-layer gate_up_proj weights")
    return len(layer_gate)


def verify_saved_moe_quant_checkpoint(save_directory: str | Path) -> int:
    """Verify safetensors on disk have per-expert GPTQ MoE weights.

    Used for the quant-experts save mode where experts are kept as per-expert
    GPTQLinear tensors (``...mlp.experts.{i}.{gate,up,down}_proj.qweight`` etc.)
    instead of being dequantized/fused.  Returns the number of per-expert
    ``qweight`` tensors found.
    """
    from safetensors import safe_open

    save_path = Path(save_directory)
    shard_paths = sorted(save_path.glob("model*.safetensors"))
    if not shard_paths:
        raise RuntimeError(f"No safetensors shards under {save_directory}")

    keys: list[str] = []
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt") as f:
            keys.extend(f.keys())

    expert_qweight = [k for k in keys if _MOE_EXPERT_KEY_RE.search(k) and k.endswith(".qweight")]
    bad_keys = [k for k in keys if k.endswith("$")]
    if bad_keys:
        raise RuntimeError(
            "Quant-experts MoE checkpoint has deduped keys: "
            f"{bad_keys[:5]}{'...' if len(bad_keys) > 5 else ''}"
        )
    fused_dense = [k for k in keys if k.endswith(".mlp.experts.gate_up_proj") and ".layers." in k]
    if fused_dense:
        raise RuntimeError(
            "Quant-experts MoE checkpoint unexpectedly contains dense fused "
            f"gate_up_proj tensors: {fused_dense[:3]}"
        )
    if not expert_qweight:
        raise RuntimeError("Quant-experts MoE checkpoint missing per-expert GPTQ qweight tensors")
    return len(expert_qweight)


def _gpt_oss_combine_gate_up(alpha: float, limit: float):
    def combine(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        return (up + 1) * glu

    return combine


def _dequantized_weight_bias(linear: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return dense (weight, bias) from nn.Linear or GPTQLinear."""
    weight = getattr(linear, "weight", None)
    if weight is not None:
        return weight, getattr(linear, "bias", None)

    qweight = getattr(linear, "qweight", None)
    if qweight is None:
        raise TypeError(f"Cannot fuse expert layer without weight: {type(linear)}")

    from onecomp.quantizer.gptq.gptq_layer import GPTQLinear, unpack_int_weights, unpack_zeros

    if not isinstance(linear, GPTQLinear):
        raise TypeError(f"Cannot fuse expert layer without weight: {type(linear)}")

    wbits = linear.wbits
    if linear._weight_is_packed:
        weight_int = unpack_int_weights(
            linear.qweight, wbits, (linear.out_features, linear.in_features)
        )
    else:
        weight_int = linear.qweight

    _v1 = getattr(linear, "checkpoint_format", "gptq") != "gptq_v2"
    wbits_mask = (1 << wbits) - 1
    if linear._weight_is_packed:
        zeros = unpack_zeros(linear.qzeros, wbits, linear.out_features)
        if _v1:
            zeros = (zeros + 1) & wbits_mask
    else:
        zeros = ((linear.qzeros + 1) & wbits_mask) if _v1 else linear.qzeros

    scale_expanded = linear.scales[linear.g_idx, :].T
    zero_expanded = zeros[linear.g_idx, :].T
    weight = scale_expanded * (weight_int.float() - zero_expanded)
    return weight, linear.bias


# GPT-OSS reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py
def _unfuse_gpt_oss_one(module: nn.Module) -> _UnfusedExperts:
    """Convert GPT-OSS fused experts to per-expert nn.Linear layers."""
    gate_up_3d = module.gate_up_proj.data  # [E, hidden, 2*inter]
    gate_up_bias_2d = module.gate_up_proj_bias.data  # [E, 2*inter]
    down_3d = module.down_proj.data  # [E, inter, hidden]
    down_bias_2d = module.down_proj_bias.data  # [E, hidden]
    num_experts = gate_up_3d.shape[0]
    hidden = gate_up_3d.shape[1]
    inter = gate_up_3d.shape[2] // 2
    dtype = gate_up_3d.dtype
    combine_fn = _gpt_oss_combine_gate_up(module.alpha, module.limit)

    experts = []
    for i in range(num_experts):
        gate_proj = nn.Linear(hidden, inter, bias=True, dtype=dtype)
        gate_proj.weight = nn.Parameter(gate_up_3d[i, :, ::2].T.contiguous())
        gate_proj.bias = nn.Parameter(gate_up_bias_2d[i, ::2].contiguous())

        up_proj = nn.Linear(hidden, inter, bias=True, dtype=dtype)
        up_proj.weight = nn.Parameter(gate_up_3d[i, :, 1::2].T.contiguous())
        up_proj.bias = nn.Parameter(gate_up_bias_2d[i, 1::2].contiguous())

        down_proj = nn.Linear(inter, hidden, bias=True, dtype=dtype)
        down_proj.weight = nn.Parameter(down_3d[i].T.contiguous())
        down_proj.bias = nn.Parameter(down_bias_2d[i].contiguous())

        experts.append(_ExpertMLP(gate_proj, up_proj, down_proj, act_fn=None))

    result = _UnfusedExperts(num_experts, experts, combine_fn=combine_fn)

    del (
        module.gate_up_proj,
        module.gate_up_proj_bias,
        module.down_proj,
        module.down_proj_bias,
    )

    return result


def _unfuse_one(module: nn.Module) -> _UnfusedExperts:
    """Convert a single fused-experts module to per-expert nn.Linear."""
    gate_up_3d = module.gate_up_proj.data  # [E, 2*inter, hidden]
    down_3d = module.down_proj.data  # [E, hidden, inter]
    num_experts = gate_up_3d.shape[0]
    inter = gate_up_3d.shape[1] // 2
    hidden = gate_up_3d.shape[2]
    act_fn = module.act_fn
    dtype = gate_up_3d.dtype

    experts = []
    for i in range(num_experts):
        gate_proj = nn.Linear(hidden, inter, bias=False, dtype=dtype)
        gate_proj.weight = nn.Parameter(gate_up_3d[i, :inter].contiguous())

        up_proj = nn.Linear(hidden, inter, bias=False, dtype=dtype)
        up_proj.weight = nn.Parameter(gate_up_3d[i, inter:].contiguous())

        down_proj = nn.Linear(inter, hidden, bias=False, dtype=dtype)
        down_proj.weight = nn.Parameter(down_3d[i].contiguous())

        experts.append(_ExpertMLP(gate_proj, up_proj, down_proj, act_fn))

    result = _UnfusedExperts(num_experts, experts, act_fn)

    del module.gate_up_proj, module.down_proj

    return result


def _fuse_gpt_oss_one(
    unfused: _UnfusedExperts,
    override: dict[str, torch.Tensor | float] | None = None,
) -> nn.Module:
    """Convert per-expert nn.Linear layers back to GPT-OSS fused experts."""
    gpt_oss_cls = _gpt_oss_experts_type()
    if gpt_oss_cls is None:
        raise ImportError("transformers GptOssExperts is required to fuse GPT-OSS MoE experts")

    if override is not None:
        num_experts = override["gate_up_proj"].shape[0]
        hidden = override["gate_up_proj"].shape[1]
        inter = override["gate_up_proj"].shape[2] // 2
    else:
        num_experts = len(unfused)
        expert0 = unfused[0]
        hidden = expert0.gate_proj.in_features
        inter = expert0.gate_proj.out_features

    config = SimpleNamespace(
        intermediate_size=inter,
        num_local_experts=num_experts,
        hidden_size=hidden,
    )
    fused = gpt_oss_cls(config)

    if override is not None:
        _copy_into_gpt_oss_experts(
            fused,
            override["gate_up_proj"],
            override["gate_up_proj_bias"],
            override["down_proj"],
            override["down_proj_bias"],
        )
        if "alpha" in override:
            fused.alpha = override["alpha"]
        if "limit" in override:
            fused.limit = override["limit"]
        return fused

    gate_w0, gate_b0 = _dequantized_weight_bias(unfused[0].gate_proj)
    down_w0, down_b0 = _dequantized_weight_bias(unfused[0].down_proj)
    bias_dtype = (
        gate_b0.dtype
        if gate_b0 is not None
        else (down_b0.dtype if down_b0 is not None else gate_w0.dtype)
    )
    gate_up_3d = torch.empty(num_experts, hidden, 2 * inter, dtype=gate_w0.dtype)
    gate_up_bias_2d = torch.zeros(num_experts, 2 * inter, dtype=bias_dtype)
    down_3d = torch.empty(num_experts, inter, hidden, dtype=down_w0.dtype)
    down_bias_2d = torch.zeros(num_experts, hidden, dtype=bias_dtype)

    for i in range(num_experts):
        expert = unfused[i]
        gate_w, gate_b = _dequantized_weight_bias(expert.gate_proj)
        up_w, up_b = _dequantized_weight_bias(expert.up_proj)
        down_w, down_b = _dequantized_weight_bias(expert.down_proj)

        gate_up_3d[i, :, ::2] = gate_w.T
        gate_up_3d[i, :, 1::2] = up_w.T
        if gate_b is not None:
            gate_up_bias_2d[i, ::2] = gate_b
        if up_b is not None:
            gate_up_bias_2d[i, 1::2] = up_b
        down_3d[i] = down_w.T
        if down_b is not None:
            down_bias_2d[i] = down_b

    _copy_into_gpt_oss_experts(
        fused,
        gate_up_3d.contiguous(),
        gate_up_bias_2d.contiguous(),
        down_3d.contiguous(),
        down_bias_2d.contiguous(),
    )
    return fused


def _fuse_one(
    unfused: _UnfusedExperts,
    override: dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    """Convert per-expert nn.Linear layers back to generic fused experts."""
    if override is not None:
        gate_up_3d = override["gate_up_proj"]
        down_3d = override["down_proj"]
    else:
        num_experts = len(unfused)
        expert0 = unfused[0]
        inter = expert0.gate_proj.out_features
        hidden = expert0.gate_proj.in_features
        gate_up_3d = torch.empty(
            num_experts,
            2 * inter,
            hidden,
            dtype=expert0.gate_proj.weight.dtype,
        )
        down_3d = torch.empty(
            num_experts,
            hidden,
            inter,
            dtype=expert0.down_proj.weight.dtype,
        )
        for i in range(num_experts):
            expert = unfused[i]
            gate_w, _ = _dequantized_weight_bias(expert.gate_proj)
            up_w, _ = _dequantized_weight_bias(expert.up_proj)
            down_w, _ = _dequantized_weight_bias(expert.down_proj)
            gate_up_3d[i, :inter] = gate_w
            gate_up_3d[i, inter:] = up_w
            down_3d[i] = down_w

    return _GenericFusedExperts(
        gate_up_3d.contiguous(),
        down_3d.contiguous(),
        unfused.act_fn,
    )


def strip_moe_experts_from_quant_config(quant_config: dict) -> None:
    """Remove per-expert MoE entries from a vLLM quantization config."""
    for key in ("modules_in_block_to_quantize", "quantized_layer_names"):
        names = quant_config.get(key)
        if not names:
            continue
        quant_config[key] = [name for name in names if not _MOE_EXPERT_KEY_RE.search(name)]

    quantization_bits = quant_config.get("quantization_bits")
    if not quantization_bits:
        return
    for layer_cfg in quantization_bits:
        if not isinstance(layer_cfg, dict):
            continue
        for suffix in list(layer_cfg):
            if suffix.startswith("mlp.experts."):
                del layer_cfg[suffix]


def unfuse_moe_experts(model: nn.Module, logger: logging.Logger) -> bool:
    """Replace fused 3D expert modules with per-expert nn.Linear layers.

    Args:
        model: The model to modify in place.

    Returns:
        True if at least one module was unfused, False otherwise.
    """
    replacements: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if _is_fused_experts(module):
            replacements.append((name, module))

    if not replacements:
        return False

    for name, fused_module in replacements:
        if _is_gpt_oss_experts(fused_module):
            unfused = _unfuse_gpt_oss_one(fused_module)
        else:
            unfused = _unfuse_one(fused_module)
        *parent_path, attr = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        setattr(parent, attr, unfused)
        num = len(unfused)
        logger.info(
            "Unfused %s: %d experts -> %d nn.Linear layers",
            name,
            num,
            num * 3,
        )

    return True


def fuse_moe_experts(
    model: nn.Module,
    logger: logging.Logger,
    fused_weight_overrides: dict[str, dict[str, torch.Tensor | float]] | None = None,
) -> bool:
    """Replace per-expert nn.Linear MoE modules with fused 3D expert tensors.

    vLLM's gpt-oss loader expects fused ``gate_up_proj`` / ``down_proj`` keys.
    When experts are GPTQLinear layers, their dequantized dense weights are
    fused (vLLM MoE uses UnquantizedFusedMoEMethod, so INT4 tensors are not
    kept in the checkpoint).  Pass ``fused_weight_overrides`` only to force
    pre-quantization BF16 tensors instead of dequantized values.

    Args:
        model: The model to modify in place.
        fused_weight_overrides: Optional pre-unfuse fused tensors keyed by module path.

    Returns:
        True if at least one module was fused, False otherwise.
    """
    replacements: list[tuple[str, _UnfusedExperts]] = []
    for name, module in model.named_modules():
        if _is_unfused_experts(module):
            replacements.append((name, module))

    if not replacements:
        return False

    overrides = fused_weight_overrides or {}
    for name, unfused in replacements:
        override = overrides.get(name)

        if unfused.combine_fn is not None:
            fused = _fuse_gpt_oss_one(unfused, override)
        else:
            fused = _fuse_one(unfused, override)

        *parent_path, attr = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        setattr(parent, attr, fused)
        logger.info(
            "Fused %s: %d experts -> fused 3D tensors",
            name,
            len(unfused),
        )

    _ensure_unique_fused_moe_parameters(model)
    _purge_orphan_parameters(model)
    _cast_fused_moe_parameters(model, _infer_fused_moe_dtype(model))
    validate_fused_moe_state_dict(model)
    return True
