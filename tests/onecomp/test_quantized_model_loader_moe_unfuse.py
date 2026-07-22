"""
Tests for MoE expert unfusing in ``QuantizedModelLoader.load_quantized_model``.

Copyright 2025-2026 Fujitsu Ltd.
"""

from unittest.mock import MagicMock, patch

from onecomp.quantized_model_loader import QuantizedModelLoader


class TestUnfuseMoeBeforeLoad:
    def test_unfuse_moe_experts_runs_before_state_dict_is_loaded(self, tmp_path):
        # Checkpoint keys such as "model.layers.0.mlp.experts.0.down_proj.weight"
        # only resolve if the empty model's fused gate_up_proj/down_proj
        # parameters have already been unfused into per-expert nn.Linear
        # modules, so unfuse_moe_experts must run before the state_dict is
        # materialized against the model.
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
        assert call_order == ["build", "unfuse", "load_state_dict"]
