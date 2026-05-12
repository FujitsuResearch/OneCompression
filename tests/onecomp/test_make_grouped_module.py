"""Unit tests for make_grouped_module.

"""

import torch
import torch.nn as nn

from onecomp.qep._quantize_with_qep_arch import make_grouped_module


class _Attention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, **kwargs):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.o_proj(attn)


class _MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _SequentialBlock(nn.Module):
    """Llama-style sequential block: attn and MLP receive different inputs."""

    def __init__(self, hidden: int = 64, intermediate: int = 128):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.self_attn = _Attention(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.mlp = _MLP(hidden, intermediate)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _ParallelBlock(nn.Module):
    """Parallel-residual block: attn and MLP share the same normed input."""

    def __init__(self, hidden: int = 64, intermediate: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.self_attn = _Attention(hidden)
        self.mlp = _MLP(hidden, intermediate)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        normed = self.norm(hidden_states)
        attn_out = self.self_attn(normed)
        mlp_out = self.mlp(normed)
        return residual + attn_out + mlp_out


class _ZeroAttention(nn.Module):
    """Attention that outputs all zeros (simulates skip-like behaviour)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        for p in self.parameters():
            nn.init.zeros_(p)

    def forward(self, x, **kwargs):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.o_proj(attn)


class _BugTriggerBlock(nn.Module):
    """Block that triggers the value-equality bug."""

    def __init__(self, hidden: int = 64, intermediate: int = 128):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden)
        self.self_attn = _ZeroAttention(hidden)
        self.post_attention_layernorm = nn.RMSNorm(hidden)
        self.mlp = _MLP(hidden, intermediate)
        self.post_attention_layernorm.load_state_dict(
            self.input_layernorm.state_dict()
        )

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states  # residual + 0 = residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _MoEBlock(nn.Module):
    """Simplified MoE block modelled after Gemma-4."""

    def __init__(self, hidden: int = 64, intermediate: int = 128, n_experts: int = 2):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden)
        self.router = nn.Linear(hidden, n_experts, bias=False)
        self.expert_norm = nn.LayerNorm(hidden)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden_states, **kwargs):
        flat = self.pre_norm(hidden_states)
        _routing = self.router(flat)
        expert_in = self.expert_norm(flat)
        gate = nn.functional.silu(self.gate_proj(expert_in))
        up = self.up_proj(expert_in)
        return hidden_states + self.down_proj(gate * up)


def _get_names(block, groups):
    """Convert groups of modules to sets of short names for readability."""
    mod_to_name = {}
    for name, mod in block.named_modules():
        if isinstance(mod, nn.Linear):
            mod_to_name[mod] = name
    return [set(mod_to_name[m] for m in g) for g in groups]


def _run(block, hidden: int = 64, seq_len: int = 8, device: str = "cpu"):
    inps = torch.randn(2, seq_len, hidden, device=device)
    kwargs: dict[str, torch.Tensor] = {}
    return make_grouped_module(block.to(device), inps, kwargs, torch.device(device))


class TestSequentialBlock:
    """Standard sequential transformer block (Llama-like)."""

    def test_qkv_grouped(self):
        block = _SequentialBlock()
        groups = _get_names(block, _run(block))
        qkv = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
        assert qkv in groups, f"q/k/v should be in one group, got {groups}"

    def test_gate_up_grouped(self):
        block = _SequentialBlock()
        groups = _get_names(block, _run(block))
        gate_up = {"mlp.gate_proj", "mlp.up_proj"}
        assert gate_up in groups, f"gate/up should be in one group, got {groups}"

    def test_attn_and_mlp_separated(self):
        block = _SequentialBlock()
        groups = _get_names(block, _run(block))
        for g in groups:
            has_attn = any("self_attn" in n for n in g)
            has_mlp = any("mlp" in n for n in g)
            assert not (has_attn and has_mlp), (
                f"Attention and MLP should never share a group, got {g}"
            )

    def test_o_proj_alone(self):
        block = _SequentialBlock()
        groups = _get_names(block, _run(block))
        assert {"self_attn.o_proj"} in groups, f"o_proj should be alone, got {groups}"

    def test_down_proj_alone(self):
        block = _SequentialBlock()
        groups = _get_names(block, _run(block))
        assert {"mlp.down_proj"} in groups, f"down_proj should be alone, got {groups}"

    def test_total_group_count(self):
        block = _SequentialBlock()
        groups = _run(block)
        assert len(groups) == 4, (
            f"Expected 4 groups (qkv, o, gate_up, down), got {len(groups)}"
        )


class TestParallelBlock:
    """Parallel-residual block: attn and MLP genuinely share the same input."""

    def test_qkv_gate_up_share_group(self):
        block = _ParallelBlock()
        groups = _get_names(block, _run(block))
        shared = {
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "mlp.gate_proj", "mlp.up_proj",
        }
        assert shared in groups, (
            f"In a parallel block, q/k/v/gate/up should share a group, got {groups}"
        )


class TestBugTriggerBlock:
    """Regression test for the Gemma-4 mis-grouping bug."""

    def test_attn_and_mlp_stay_separate(self):
        block = _BugTriggerBlock()
        groups = _get_names(block, _run(block))
        for g in groups:
            has_attn = any("self_attn" in n for n in g)
            has_mlp = any("mlp" in n for n in g)
            assert not (has_attn and has_mlp), (
                f"Bug regression: attn and MLP must not share a group "
                f"even when norm outputs are value-equal. Got {g}"
            )

    def test_qkv_still_grouped(self):
        block = _BugTriggerBlock()
        groups = _get_names(block, _run(block))
        qkv = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
        assert qkv in groups, f"q/k/v should still be grouped, got {groups}"

    def test_gate_up_still_grouped(self):
        block = _BugTriggerBlock()
        groups = _get_names(block, _run(block))
        gate_up = {"mlp.gate_proj", "mlp.up_proj"}
        assert gate_up in groups, f"gate/up should still be grouped, got {groups}"


class TestMoEBlock:
    """MoE block where the router has different in_features than expert layers."""

    def test_router_separate_from_experts(self):
        block = _MoEBlock()
        groups = _get_names(block, _run(block))
        for g in groups:
            has_router = "router" in g
            has_proj = any(n in g for n in ("gate_proj", "up_proj"))
            assert not (has_router and has_proj), (
                f"Router should not share a group with expert projections "
                f"(different in_features), got {g}"
            )

    def test_gate_up_grouped(self):
        block = _MoEBlock()
        groups = _get_names(block, _run(block))
        gate_up = {"gate_proj", "up_proj"}
        assert any(g >= gate_up for g in groups), (
            f"gate/up should be grouped, got {groups}"
        )
