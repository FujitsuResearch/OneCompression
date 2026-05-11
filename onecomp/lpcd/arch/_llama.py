"""
Copyright 2025-2026 Fujitsu Ltd.

"""
import torch
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv
)

from ._llama_cf import (
    closed_form_solver_o_proj,
    closed_form_solver_down_proj
)

from .._lpcd_config import LPCDConfig
from .._metric import ClosedFormSolverArgument, LpcdMetric, LpcdMetricGroup
from typing import Callable



class LlamaQueryKey(LpcdMetric):
    """LPCD metric for query/key projection of Llama"""

    def named_targets(self) -> list[tuple[str, nn.Module]]:
        return [
            ('self_attn.q_proj', self.block.self_attn.q_proj),
            ('self_attn.k_proj', self.block.self_attn.k_proj),
        ]

    def closed_form_solvers(self) -> list[Callable[[ClosedFormSolverArgument], None] | None]:
        return [None, None]

    def forward(
        self, 
        block_inps: torch.Tensor, 
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: dict
    ) -> torch.Tensor:

        self_attn: LlamaAttention = self.block.self_attn

        hidden_states = self.block.input_layernorm(block_inps)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self_attn.head_dim)

        query_states = self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self_attn.scaling
        
        # apply logical attention mask
        if attention_mask is not None:
            addition_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            bool_mask = (addition_mask == 0)
            attn_weights = attn_weights * bool_mask

        return attn_weights


# v_proj / o_proj
class LlamaValueOut(LpcdMetric):
    """LPCD metric for value/o_proj of Llama"""

    def named_targets(self):
        return [
            ("self_attn.v_proj", self.block.self_attn.v_proj),
            ("self_attn.o_proj", self.block.self_attn.o_proj),
        ]
    
    def closed_form_solvers(self) -> list[Callable[[ClosedFormSolverArgument], None] | None]:
        return [None, closed_form_solver_o_proj]

    def forward(self, block_inps: torch.Tensor, **kwargs: dict) -> torch.Tensor:
        hidden_states = self.block.input_layernorm(block_inps)
        hidden_states, _ = self.block.self_attn(
            hidden_states=hidden_states, **kwargs,
        )
        return block_inps + hidden_states


# Only o_proj only (residual)
class LlamaOut(LlamaValueOut):

    def named_targets(self) -> list[tuple[str, nn.Module]]:
        return [("self_attn.o_proj", self.block.self_attn.o_proj)]

    def closed_form_solvers(self) -> list[Callable[[ClosedFormSolverArgument], None] | None]:
        return [closed_form_solver_o_proj]


# up_proj / down_proj
class LlamaUpDown(LpcdMetric):
    """LPCD metric for up/down projection of Llama"""

    def named_targets(self) -> list[tuple[str, nn.Module]]:
        return [
            ("mlp.up_proj", self.block.mlp.up_proj),
            ("mlp.down_proj", self.block.mlp.down_proj),
        ]

    def closed_form_solvers(self) -> list[Callable[[ClosedFormSolverArgument], None] | None]:
        return [None, closed_form_solver_down_proj]

    def forward(self, block_inps: torch.Tensor, **kwargs: dict) -> torch.Tensor:
        return self.block(block_inps, **kwargs)


# Only down_proj (residual)
class LlamaDown(LlamaUpDown):

    def named_targets(self) -> list[tuple[str, nn.Module]]:
        return [("mlp.down_proj", self.block.mlp.down_proj)]

    def closed_form_solvers(self) -> list[Callable[[ClosedFormSolverArgument], None] | None]:
        return [closed_form_solver_down_proj]


def make_llama_lpcd_metrics(
    lpcd_config: LPCDConfig, 
    block_q: LlamaDecoderLayer,
    block_f: LlamaDecoderLayer,
) -> LpcdMetricGroup:
    """ Make LPCD metrics for Llama models.

    Args:
        lpcd_config (LPCDConfig): LPCD configuration
        block_q (LlamaDecoderLayer): Quantizing Transformer block
        block_f (LlamaDecoderLayer): Full-precision Transformer block

    Returns:
        LpcdMetricGroup: LPCD metrics for the given blocks
    """
    metrics = []
    if lpcd_config.enable_qk:
        metrics.append((LlamaQueryKey(block_q), LlamaQueryKey(block_f)))
    
    if lpcd_config.enable_vo:
        metrics.append((LlamaValueOut(block_q), LlamaValueOut(block_f)))
    elif lpcd_config.enable_residual:
        metrics.append((LlamaOut(block_q), LlamaOut(block_f)))
    
    if lpcd_config.enable_ud:
        metrics.append((LlamaUpDown(block_q), LlamaUpDown(block_f)))
    elif lpcd_config.enable_residual:
        metrics.append((LlamaDown(block_q), LlamaDown(block_f)))

    return LpcdMetricGroup(metrics)
