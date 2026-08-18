"""
Tests for MoE expert unfusing in ``QuantizedModelLoader.load_quantized_model``.

Copyright 2025-2026 Fujitsu Ltd.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from safetensors.torch import save_file

from onecomp.quantized_model_loader import QuantizedModelLoader


class TestUnfuseMoeBeforeLoad:
    def test_unfuse_moe_experts_runs_before_state_dict_is_loaded(self, tmp_path):
        # Checkpoint keys such as "model.layers.0.mlp.experts.0.down_proj.weight"
        # only resolve if the empty model's fused gate_up_proj/down_proj
        # parameters have already been unfused into per-expert nn.Linear
        # modules, so unfuse_moe_experts must run before the state_dict is
        # materialized against the model.  The unfuse decision is made after the
        # checkpoint is read from disk (so fused-MoE checkpoints can skip it),
        # but still before model.load_state_dict materializes the tensors.
        fake_model = MagicMock(name="empty_model")
        call_order = []

        with (
            patch.object(
                QuantizedModelLoader,
                "_load_config_and_quant_config",
                return_value=(
                    {"model_type": "llama"},
                    {"quant_method": "gptq", "modules_in_block_to_quantize": []},
                ),
            ),
            patch("onecomp.quantized_model_loader.needs_bfloat16", return_value=False),
            patch.object(
                QuantizedModelLoader,
                "_build_empty_model_from_config",
                side_effect=lambda *a, **k: (call_order.append("build"), fake_model)[1],
            ),
            patch(
                "onecomp.quantized_model_loader.unfuse_moe_experts",
                side_effect=lambda *a, **k: call_order.append("unfuse") or False,
            ) as mock_unfuse,
            patch.object(
                QuantizedModelLoader,
                "_load_state_dict_from_dir",
                side_effect=lambda *a, **k: call_order.append("load_state_dict") or {},
            ),
            patch.object(
                QuantizedModelLoader, "_remap_state_dict_keys", side_effect=lambda sd, m: sd
            ),
            patch.object(
                QuantizedModelLoader, "_replace_quantized_layers", side_effect=lambda m, sd, qc: sd
            ),
            patch.object(QuantizedModelLoader, "_retie_lm_head_if_needed"),
            patch.object(QuantizedModelLoader, "_cast_fp16_to_target_dtype", return_value=[]),
            patch.object(QuantizedModelLoader, "_assert_quantized_modules_loaded"),
            patch.object(QuantizedModelLoader, "_load_generation_config"),
            patch.object(QuantizedModelLoader, "_apply_lora_adapters_from_sidecar"),
            patch("onecomp.quantized_model_loader.AutoTokenizer.from_pretrained"),
        ):
            QuantizedModelLoader.load_quantized_model(str(tmp_path), device_map="")

        mock_unfuse.assert_called_once()
        assert mock_unfuse.call_args[0][0] is fake_model
        assert call_order == ["build", "load_state_dict", "unfuse"]


class _FakeMoEExpertsBlock(nn.Module):
    """Minimal stand-in for a fused 3D MoE expert block."""

    def __init__(self, num_experts: int, hidden: int, inter: int):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.zeros(num_experts, 2 * inter, hidden, dtype=torch.float16)
        )
        self.down_proj = nn.Parameter(torch.zeros(num_experts, hidden, inter, dtype=torch.float16))
        self.act_fn = nn.SiLU()


class _FakeMoEModel(nn.Module):
    """Minimal CausalLM stand-in with one fused-MoE decoder layer."""

    def __init__(self, num_experts: int, hidden: int, inter: int):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.experts = _FakeMoEExpertsBlock(num_experts, hidden, inter)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(hidden, 8, bias=False, dtype=torch.float16)


class TestUnfuseMoeLoadRoundTrip:
    def test_unfuse_moe_load_state_dict_fills_actual_per_expert_weights(self, tmp_path):
        """Runs the real unfuse + load_state_dict and checks each expert's
        weight matches its saved checkpoint tensor."""
        num_experts, hidden, inter = 2, 4, 8
        save_dir = tmp_path / "moe_saved"
        save_dir.mkdir()

        expert_weights = {}
        state_dict = {"lm_head.weight": torch.zeros(8, hidden, dtype=torch.float16)}
        for i in range(num_experts):
            gate = torch.randn(inter, hidden, dtype=torch.float16) + i
            up = torch.randn(inter, hidden, dtype=torch.float16) + i
            down = torch.randn(hidden, inter, dtype=torch.float16) + i
            expert_weights[i] = (gate, up, down)
            prefix = f"model.layers.0.mlp.experts.{i}"
            state_dict[f"{prefix}.gate_proj.weight"] = gate
            state_dict[f"{prefix}.up_proj.weight"] = up
            state_dict[f"{prefix}.down_proj.weight"] = down

        save_file(state_dict, str(save_dir / "model.safetensors"))
        (save_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "quantization_config": {
                        "quant_method": "gptq",
                        "modules_in_block_to_quantize": [],
                    },
                }
            ),
            encoding="utf-8",
        )

        with (
            patch.object(
                QuantizedModelLoader,
                "_build_empty_model_from_config",
                return_value=_FakeMoEModel(num_experts, hidden, inter),
            ),
            patch(
                "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained",
                return_value=object(),
            ),
        ):
            model, _ = QuantizedModelLoader.load_quantized_model(str(save_dir), device_map="")

        experts = model.model.layers[0].mlp.experts
        assert len(experts) == num_experts
        for i in range(num_experts):
            gate, up, down = expert_weights[i]
            assert torch.equal(experts[i].gate_proj.weight, gate)
            assert torch.equal(experts[i].up_proj.weight, up)
            assert torch.equal(experts[i].down_proj.weight, down)
