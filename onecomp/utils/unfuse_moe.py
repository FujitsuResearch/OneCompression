"""
Unfuse 3D fused MoE expert parameters into per-expert nn.Linear modules.

Copyright 2025-2026 Fujitsu Ltd.

"""

import logging

import torch
from torch import nn


class _ExpertMLP(nn.Module):
    """Single MoE expert with gate/up/down projections."""

    __slots__ = ("act_fn",)

    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear,
                 down_proj: nn.Linear, act_fn):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn


class _UnfusedExperts(nn.Module):
    """Drop-in replacement for fused 3D expert modules.

    """

    def __init__(self, num_experts: int, experts: list, act_fn):
        super().__init__()
        self._num_experts = num_experts
        for i, expert in enumerate(experts):
            self.add_module(str(i), expert)
        self.act_fn = act_fn

    def __len__(self):
        return self._num_experts

    def __getitem__(self, idx):
        return getattr(self, str(int(idx)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self._num_experts,
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0,
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
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = expert.down_proj(current_hidden_states)

            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )

        return final_hidden_states


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


def _unfuse_one(module: nn.Module) -> _UnfusedExperts:
    """Convert a single fused-experts module to per-expert nn.Linear."""
    gate_up_3d = module.gate_up_proj.data   # [E, 2*inter, hidden]
    down_3d = module.down_proj.data         # [E, hidden, inter]
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
        unfused = _unfuse_one(fused_module)
        *parent_path, attr = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        setattr(parent, attr, unfused)
        num = len(unfused)
        logger.info(
            "Unfused %s: %d experts -> %d nn.Linear layers",
            name, num, num * 3,
        )

    return True
