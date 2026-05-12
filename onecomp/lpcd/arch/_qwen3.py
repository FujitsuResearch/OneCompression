"""
Copyright 2025-2026 Fujitsu Ltd.

"""
import torch
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Attention,
    apply_rotary_pos_emb,
    repeat_kv
)


from .._lpcd_config import LPCDConfig
from .._metric import LpcdMetric, LpcdMetricGroup
from ._llama import (
    LlamaQueryKey,
    LlamaValueOut,
    LlamaOut,
    LlamaUpDown,
    LlamaDown,
)

# q_proj / k_proj
class Qwen3QueryKey(LlamaQueryKey):

    def forward(
        self, 
        block_inps: torch.Tensor, 
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: dict
    ) -> torch.Tensor:
        self_attn: Qwen3Attention = self.block.self_attn

        hidden_states = self.block.input_layernorm(block_inps)
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self_attn.head_dim)

        query_states = self_attn.q_norm(self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self_attn.k_norm(self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

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


# v_proj / o_proj (same as Llama)
class Qwen3ValueOut(LlamaValueOut):
    pass

# Only o_proj (same as Llama)
class Qwen3Out(LlamaOut):
    pass


# up_proj / down_proj (same as Llama)
class Qwen3UpDown(LlamaUpDown):
    pass

# Only down_proj (same as Llama)
class Qwen3Down(LlamaDown):
    pass


def make_qwen3_lpcd_metrics(
    lpcd_config: LPCDConfig, 
    block_q: Qwen3DecoderLayer,
    block_f: Qwen3DecoderLayer,
) -> LpcdMetricGroup:
    """ Make LPCD metrics for Qwen3 models.

    Args:
        lpcd_config (LPCDConfig): LPCD configuration
        block_q (Qwen3DecoderLayer): Quantized Transformer block
        block_f (Qwen3DecoderLayer): Full-precision Transformer block

    Returns:
        LpcdMetricGroup: LPCD metrics for the given block
    """
    metrics = []
    if lpcd_config.enable_qk:
        metrics.append((Qwen3QueryKey(block_q), Qwen3QueryKey(block_f)))

    if lpcd_config.enable_vo:
        metrics.append((Qwen3ValueOut(block_q), Qwen3ValueOut(block_f)))
    elif lpcd_config.enable_residual:
        metrics.append((Qwen3Out(block_q), Qwen3Out(block_f)))

    if lpcd_config.enable_ud:
        metrics.append((Qwen3UpDown(block_q), Qwen3UpDown(block_f)))
    elif lpcd_config.enable_residual:
        metrics.append((Qwen3Down(block_q), Qwen3Down(block_f)))

    return LpcdMetricGroup(metrics)
