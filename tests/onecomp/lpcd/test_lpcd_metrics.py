"""Unit tests for LPCD metric dispatch (``make_lpcd_metrics``).

Builds tiny Llama / Qwen3 decoder blocks on CPU (no weights downloaded)
and checks that ``make_lpcd_metrics`` returns the correct metric classes
for each combination of ``enable_*`` flags.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    pytest tests/onecomp/lpcd/test_lpcd_metrics.py -v
"""

import pytest
import torch
from torch import nn

from onecomp import LPCDConfig
from onecomp.lpcd._metric import (
    LpcdMetric,
    LpcdMetricGroup,
    make_lpcd_metrics,
)
from onecomp.lpcd.arch._llama import (
    LlamaDown,
    LlamaOut,
    LlamaQueryKey,
    LlamaUpDown,
    LlamaValueOut,
)
from onecomp.lpcd.arch._qwen3 import (
    Qwen3Down,
    Qwen3Out,
    Qwen3QueryKey,
    Qwen3UpDown,
    Qwen3ValueOut,
)


def _make_llama_block():
    """Build a tiny ``LlamaDecoderLayer`` on CPU (no weights downloaded)."""
    from transformers.models.llama.modeling_llama import (
        LlamaConfig,
        LlamaDecoderLayer,
    )

    config = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        vocab_size=128,
    )
    return LlamaDecoderLayer(config, layer_idx=0)


def _make_qwen3_block():
    """Build a tiny ``Qwen3DecoderLayer`` on CPU (no weights downloaded)."""
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Config,
        Qwen3DecoderLayer,
    )

    config = Qwen3Config(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        vocab_size=128,
        head_dim=8,
    )
    return Qwen3DecoderLayer(config, layer_idx=0)


def _metric_classes(group: LpcdMetricGroup) -> list[type]:
    """Return the class of each (q, f) metric pair's q-side."""
    return [type(metric_q) for metric_q, _ in group.metrics]


def _metric_target_names(group: LpcdMetricGroup) -> list[list[str]]:
    """Return the list of target names for each metric in the group."""
    return [
        [name for name, _ in metric_q.named_targets()]
        for metric_q, _ in group.metrics
    ]


class TestMakeLpcdMetricsLlama:
    """Dispatch tests for Llama blocks."""

    @pytest.fixture(scope="class")
    def blocks(self):
        return _make_llama_block(), _make_llama_block()

    def test_default_residual_only(self, blocks):
        """Default config → o_proj (residual) + down_proj (residual)."""
        block_q, block_f = blocks
        group = make_lpcd_metrics(LPCDConfig(), block_q, block_f)

        classes = _metric_classes(group)
        assert classes == [LlamaOut, LlamaDown]

        names = _metric_target_names(group)
        assert names == [["self_attn.o_proj"], ["mlp.down_proj"]]

    def test_all_enabled(self, blocks):
        """enable_qk/vo/ud → QK + VO + UD."""
        block_q, block_f = blocks
        cfg = LPCDConfig(
            enable_qk=True,
            enable_vo=True,
            enable_ud=True,
            enable_residual=False,
        )
        group = make_lpcd_metrics(cfg, block_q, block_f)

        classes = _metric_classes(group)
        assert classes == [LlamaQueryKey, LlamaValueOut, LlamaUpDown]

        names = _metric_target_names(group)
        assert names == [
            ["self_attn.q_proj", "self_attn.k_proj"],
            ["self_attn.v_proj", "self_attn.o_proj"],
            ["mlp.up_proj", "mlp.down_proj"],
        ]

    def test_vo_overrides_residual_out(self, blocks):
        """enable_vo=True suppresses the residual-only LlamaOut."""
        block_q, block_f = blocks
        cfg = LPCDConfig(enable_vo=True, enable_residual=True)
        group = make_lpcd_metrics(cfg, block_q, block_f)

        classes = _metric_classes(group)
        assert LlamaValueOut in classes
        assert LlamaOut not in classes

    def test_ud_overrides_residual_down(self, blocks):
        """enable_ud=True suppresses the residual-only LlamaDown."""
        block_q, block_f = blocks
        cfg = LPCDConfig(enable_ud=True, enable_residual=True)
        group = make_lpcd_metrics(cfg, block_q, block_f)

        classes = _metric_classes(group)
        assert LlamaUpDown in classes
        assert LlamaDown not in classes

    def test_no_metrics_when_all_disabled(self, blocks):
        """All flags off → empty metric group."""
        block_q, block_f = blocks
        cfg = LPCDConfig(
            enable_qk=False, enable_vo=False, enable_ud=False, enable_residual=False
        )
        group = make_lpcd_metrics(cfg, block_q, block_f)
        assert group.metrics == []

    def test_module_to_metric_mapping(self, blocks):
        """LpcdMetricGroup.module_to_metric should point to the q-side metric."""
        block_q, block_f = blocks
        cfg = LPCDConfig(enable_qk=True, enable_residual=False)
        group = make_lpcd_metrics(cfg, block_q, block_f)

        q_proj = block_q.self_attn.q_proj
        k_proj = block_q.self_attn.k_proj
        assert q_proj in group.module_to_metric
        assert k_proj in group.module_to_metric
        assert isinstance(group.module_to_metric[q_proj], LlamaQueryKey)


class TestMakeLpcdMetricsQwen3:
    """Dispatch tests for Qwen3 blocks."""

    @pytest.fixture(scope="class")
    def blocks(self):
        return _make_qwen3_block(), _make_qwen3_block()

    def test_default_residual_only(self, blocks):
        block_q, block_f = blocks
        group = make_lpcd_metrics(LPCDConfig(), block_q, block_f)

        classes = _metric_classes(group)
        assert classes == [Qwen3Out, Qwen3Down]

    def test_all_enabled(self, blocks):
        block_q, block_f = blocks
        cfg = LPCDConfig(
            enable_qk=True,
            enable_vo=True,
            enable_ud=True,
            enable_residual=False,
        )
        group = make_lpcd_metrics(cfg, block_q, block_f)

        classes = _metric_classes(group)
        assert classes == [Qwen3QueryKey, Qwen3ValueOut, Qwen3UpDown]


class TestMakeLpcdMetricsUnsupported:
    """Unsupported architectures should raise ``NotImplementedError``."""

    def test_plain_module_raises(self):
        block = nn.Linear(4, 4)
        with pytest.raises(NotImplementedError):
            make_lpcd_metrics(LPCDConfig(), block, block)


class TestLpcdMetricReadiness:
    """Verify ``is_refineable`` / ``mark_as_ready`` state machine."""

    def test_is_refineable_flow(self):
        block_q, block_f = _make_llama_block(), _make_llama_block()
        cfg = LPCDConfig(enable_qk=True, enable_residual=False)
        group = make_lpcd_metrics(cfg, block_q, block_f)

        assert group.get_refineable_metrics() == []

        group.mark_as_ready(block_q.self_attn.q_proj)
        assert group.get_refineable_metrics() == []

        group.mark_as_ready(block_q.self_attn.k_proj)
        refineable = group.get_refineable_metrics()
        assert len(refineable) == 1
        metric_q, _ = refineable[0]
        assert isinstance(metric_q, LlamaQueryKey)

        metric_q.is_refined = True
        assert group.get_refineable_metrics() == []

    def test_mark_as_ready_ignores_unknown_module(self):
        block_q, block_f = _make_llama_block(), _make_llama_block()
        cfg = LPCDConfig(enable_qk=True, enable_residual=False)
        group = make_lpcd_metrics(cfg, block_q, block_f)

        unrelated = nn.Linear(4, 4)
        group.mark_as_ready(unrelated)
