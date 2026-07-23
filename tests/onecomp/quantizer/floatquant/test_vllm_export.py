"""Tests for the vLLM-native checkpoint exports (FP8 / compressed-tensors).

Copyright 2025-2026 Fujitsu Ltd.
"""

import json
import os

import pytest
import torch
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

from onecomp.quantizer.floatquant import (
    FloatQuant,
    diagnose_nvfp4_fused_export_gap,
    save_vllm_fp8_model,
    save_vllm_native_model,
)
from onecomp.quantizer.floatquant.formats import (
    E4M3_MAX,
    fp8_dequantize,
    mxfp4_dequantize,
    nvfp4_dequantize,
    uint8_to_e8m0_scales,
    unpack_fp4_codes,
)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(3)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def exported(tiny_model, tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("fp8_vllm_ckpt")
    save_vllm_fp8_model(tiny_model, str(save_dir))
    return tiny_model, str(save_dir)


class TestSaveVllmFp8Model:
    """Exported checkpoint follows the vLLM-native FP8 layout."""

    def test_config_records_native_fp8(self, exported):
        _, save_dir = exported
        with open(os.path.join(save_dir, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        quant_config = config["quantization_config"]
        assert quant_config["quant_method"] == "fp8"
        assert quant_config["activation_scheme"] == "dynamic"
        assert "lm_head" in quant_config["ignored_layers"]
        # Native checkpoints must NOT carry the fake-quant marker.
        assert quant_config.get("checkpoint_format") != "fake_quant"

    def test_linear_weights_are_fp8_with_per_tensor_scale(self, exported):
        model, save_dir = exported
        state = load_file(os.path.join(save_dir, "model.safetensors"))

        target = "model.layers.0.self_attn.q_proj"
        assert state[f"{target}.weight"].dtype == torch.float8_e4m3fn
        scale = state[f"{target}.weight_scale"]
        assert scale.dtype == torch.float32
        assert scale.numel() == 1

        original = dict(model.named_parameters())[f"{target}.weight"].float()
        expected_scale = original.abs().amax() / E4M3_MAX
        assert torch.isclose(scale.reshape(()), expected_scale)

    def test_every_block_linear_quantized(self, exported):
        model, save_dir = exported
        state = load_file(os.path.join(save_dir, "model.safetensors"))
        num_layers = model.config.num_hidden_layers
        projections = ("q_proj", "k_proj", "v_proj", "o_proj")
        mlps = ("gate_proj", "up_proj", "down_proj")
        for i in range(num_layers):
            for proj in projections:
                key = f"model.layers.{i}.self_attn.{proj}.weight"
                assert state[key].dtype == torch.float8_e4m3fn
                assert f"model.layers.{i}.self_attn.{proj}.weight_scale" in state
            for proj in mlps:
                key = f"model.layers.{i}.mlp.{proj}.weight"
                assert state[key].dtype == torch.float8_e4m3fn

    def test_non_target_tensors_unchanged(self, exported):
        model, save_dir = exported
        state = load_file(os.path.join(save_dir, "model.safetensors"))
        embed = state["model.embed_tokens.weight"]
        assert embed.dtype == model.model.embed_tokens.weight.dtype
        assert torch.equal(embed, model.model.embed_tokens.weight.data)

    def test_dequantized_weight_close_to_original(self, exported):
        model, save_dir = exported
        state = load_file(os.path.join(save_dir, "model.safetensors"))
        target = "model.layers.0.mlp.down_proj"
        original = dict(model.named_parameters())[f"{target}.weight"].float()
        scale = state[f"{target}.weight_scale"].reshape(())
        dequantized = state[f"{target}.weight"].float() * scale
        # E4M3 rounding plus per-tensor scaling: coarse but bounded error.
        assert (dequantized - original).abs().max() <= 0.1 * original.abs().max()
        assert torch.isfinite(dequantized).all()

    def test_model_without_blocks_raises(self, tmp_path):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with pytest.raises(ValueError):
            save_vllm_fp8_model(model, str(tmp_path / "bad"))


def _quantize_block_linears(model, fmt):
    """Quantize the transformer-block Linears of ``model`` with FloatQuant."""
    quantizer = FloatQuant(fmt=fmt)
    results = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Module) and name.startswith("model.layers."):
            if isinstance(module, torch.nn.Linear):
                results[name] = quantizer.quantize_layer(module)
    return results


class _DiagnosticSelfAttention(torch.nn.Module):
    """Minimal q/k/v module using vLLM-fused leaf names."""

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(16, 4, bias=False)
        self.k_proj = torch.nn.Linear(16, 4, bias=False)
        self.v_proj = torch.nn.Linear(16, 4, bias=False)


class _DiagnosticLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _DiagnosticSelfAttention()


class _DiagnosticBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_DiagnosticLayer()])


class _DiagnosticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DiagnosticBackbone()


def _diagnostic_nvfp4_results(fmt="nvfp4"):
    """Build tiny fused q/k/v results with intentionally different scales."""
    model = _DiagnosticModel()
    base = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).reshape(4, 16)
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight.copy_(base * 0.01)
        model.model.layers[0].self_attn.k_proj.weight.copy_(base)
        model.model.layers[0].self_attn.v_proj.weight.copy_(base * 512.0)

    quantizer = FloatQuant(fmt=fmt)
    results = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            results[name] = quantizer.quantize_layer(module)
    return model, results


@pytest.fixture(scope="module")
def native_model():
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


class TestSaveVllmNativeModel:
    """Exported checkpoints follow the vLLM compressed-tensors layouts."""

    @pytest.mark.parametrize(
        "fmt,expected_format",
        [
            ("nvfp4", "nvfp4-pack-quantized"),
            ("mxfp4", "mxfp4-pack-quantized"),
            ("fp8", "float-quantized"),
        ],
    )
    def test_config_records_compressed_tensors(self, native_model, tmp_path, fmt, expected_format):
        results = _quantize_block_linears(native_model, fmt)
        save_dir = tmp_path / f"ct_{fmt}"
        save_vllm_native_model(native_model, results, str(save_dir))

        with open(save_dir / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        qc = config["quantization_config"]
        assert qc["quant_method"] == "compressed-tensors"
        assert qc["format"] == expected_format
        assert "lm_head" in qc["ignore"]
        group = qc["config_groups"]["group_0"]
        assert group["targets"] == ["Linear"]
        weights = group["weights"]
        if fmt == "fp8":
            assert weights["num_bits"] == 8
            assert weights["strategy"] == "channel"
            assert group["input_activations"]["dynamic"] is True
            assert group["input_activations"]["strategy"] == "token"
        else:
            assert weights["num_bits"] == 4
            assert weights["type"] == "float"
            assert group["input_activations"] is None
            if fmt == "nvfp4":
                assert weights["strategy"] == "tensor_group"
                assert weights["group_size"] == 16
            else:
                assert weights["strategy"] == "group"
                assert weights["group_size"] == 32

    def test_nvfp4_tensors_reconstruct_dequantized_weight(self, native_model, tmp_path):
        results = _quantize_block_linears(native_model, "nvfp4")
        save_dir = tmp_path / "ct_nvfp4_roundtrip"
        save_vllm_native_model(native_model, results, str(save_dir))
        state = load_file(str(save_dir / "model.safetensors"))

        # o_proj / down_proj are never fused, so their global scale is
        # untouched and reconstruction must be bit-exact.
        target = "model.layers.0.self_attn.o_proj"
        packed = state[f"{target}.weight_packed"]
        assert packed.dtype == torch.uint8
        scales = state[f"{target}.weight_scale"]
        assert scales.dtype == torch.float8_e4m3fn
        global_scale = state[f"{target}.weight_global_scale"]
        assert global_scale.dtype == torch.float32 and global_scale.numel() == 1

        codes = unpack_fp4_codes(packed)
        tensor_scale = (1.0 / global_scale).reshape(())
        reconstructed = nvfp4_dequantize(codes, scales.float(), tensor_scale, 16)
        expected = results[target].compute_dequantized_weight().float()
        assert torch.allclose(reconstructed, expected, atol=1e-3, rtol=1e-3)

    def test_nvfp4_fused_layers_share_global_scale(self, native_model, tmp_path):
        results = _quantize_block_linears(native_model, "nvfp4")
        save_dir = tmp_path / "ct_nvfp4_fused"
        save_vllm_native_model(native_model, results, str(save_dir))
        state = load_file(str(save_dir / "model.safetensors"))

        for i in range(2):
            attn = f"model.layers.{i}.self_attn"
            qkv = [
                state[f"{attn}.{p}.weight_global_scale"] for p in ("q_proj", "k_proj", "v_proj")
            ]
            assert torch.equal(qkv[0], qkv[1]) and torch.equal(qkv[1], qkv[2])
            mlp = f"model.layers.{i}.mlp"
            gate_up = [state[f"{mlp}.{p}.weight_global_scale"] for p in ("gate_proj", "up_proj")]
            assert torch.equal(gate_up[0], gate_up[1])

    def test_nvfp4_fused_export_gap_diagnostic(self):
        model, results = _diagnostic_nvfp4_results()
        report = diagnose_nvfp4_fused_export_gap(results, model=model, top_k=1)

        assert report["has_reference"] is True
        assert report["num_fused_groups"] == 1
        assert report["num_fused_layers"] == 3
        assert report["num_requantized_layers"] == 2
        assert report["gap_to_pre_export"]["squared_error"] > 0.0
        assert report["pre_export"]["squared_error"] >= 0.0
        assert report["post_export"]["squared_error"] >= 0.0
        assert report["delta_export"]["squared_error"] == pytest.approx(
            report["post_export"]["squared_error"] - report["pre_export"]["squared_error"]
        )
        assert len(report["worst_layers"]) == 1
        assert len(report["worst_groups"]) == 1
        json.dumps(report)

    def test_nvfp4_fused_export_gap_diagnostic_without_model(self):
        _, results = _diagnostic_nvfp4_results()
        report = diagnose_nvfp4_fused_export_gap(results, top_k=2)

        assert report["has_reference"] is False
        assert report["pre_export"] is None
        assert report["post_export"] is None
        assert report["delta_export"] is None
        assert report["gap_to_pre_export"]["squared_error"] > 0.0
        assert len(report["worst_layers"]) == 2
        assert "pre_export" not in report["worst_layers"][0]

    def test_nvfp4_fused_export_gap_diagnostic_rejects_non_nvfp4(self):
        _, results = _diagnostic_nvfp4_results(fmt="fp8")
        with pytest.raises(ValueError, match="only supports nvfp4"):
            diagnose_nvfp4_fused_export_gap(results)

    def test_mxfp4_tensors_reconstruct_dequantized_weight(self, native_model, tmp_path):
        results = _quantize_block_linears(native_model, "mxfp4")
        save_dir = tmp_path / "ct_mxfp4_roundtrip"
        save_vllm_native_model(native_model, results, str(save_dir))
        state = load_file(str(save_dir / "model.safetensors"))

        target = "model.layers.1.mlp.down_proj"
        packed = state[f"{target}.weight_packed"]
        scales = state[f"{target}.weight_scale"]
        assert packed.dtype == torch.uint8
        assert scales.dtype == torch.uint8

        codes = unpack_fp4_codes(packed)
        reconstructed = mxfp4_dequantize(codes, uint8_to_e8m0_scales(scales), 32)
        expected = results[target].compute_dequantized_weight()
        assert torch.equal(reconstructed.to(expected.dtype), expected)

    def test_fp8_tensors_reconstruct_dequantized_weight(self, native_model, tmp_path):
        results = _quantize_block_linears(native_model, "fp8")
        save_dir = tmp_path / "ct_fp8_roundtrip"
        save_vllm_native_model(native_model, results, str(save_dir))
        state = load_file(str(save_dir / "model.safetensors"))

        target = "model.layers.0.mlp.gate_proj"
        weight = state[f"{target}.weight"]
        scales = state[f"{target}.weight_scale"]
        assert weight.dtype == torch.float8_e4m3fn
        assert scales.dtype == torch.float32
        out_features = results[target].dequantized_weight.shape[0]
        assert scales.shape == (out_features, 1)

        reconstructed = fp8_dequantize(weight.float(), scales)
        expected = results[target].compute_dequantized_weight().float()
        assert torch.allclose(reconstructed, expected, atol=1e-3, rtol=1e-3)

    def test_mxfp4_checkpoint_uses_bfloat16(self, native_model, tmp_path):
        """mxfp4 checkpoints store bfloat16 (vLLM MXFP4 kernel requirement)."""
        results = _quantize_block_linears(native_model, "mxfp4")
        save_dir = tmp_path / "ct_mxfp4_dtype"
        save_vllm_native_model(native_model, results, str(save_dir))

        with open(save_dir / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        assert config["torch_dtype"] == "bfloat16"
        state = load_file(str(save_dir / "model.safetensors"))
        assert state["model.embed_tokens.weight"].dtype == torch.bfloat16
        bias_keys = [k for k in state if k.endswith(".bias")]
        for key in bias_keys:
            assert state[key].dtype == torch.bfloat16

    def test_unquantized_linears_listed_in_ignore(self, native_model, tmp_path):
        results = _quantize_block_linears(native_model, "mxfp4")
        removed = "model.layers.1.mlp.down_proj"
        results.pop(removed)
        save_dir = tmp_path / "ct_ignore"
        save_vllm_native_model(native_model, results, str(save_dir))

        with open(save_dir / "config.json", encoding="utf-8") as f:
            qc = json.load(f)["quantization_config"]
        assert removed in qc["ignore"]
        state = load_file(str(save_dir / "model.safetensors"))
        assert f"{removed}.weight" in state
        assert f"{removed}.weight_packed" not in state

    def test_empty_results_raises(self, native_model, tmp_path):
        with pytest.raises(ValueError):
            save_vllm_native_model(native_model, {}, str(tmp_path / "bad"))

    def test_mixed_formats_raise(self, native_model, tmp_path):
        results = _quantize_block_linears(native_model, "nvfp4")
        other = _quantize_block_linears(native_model, "mxfp4")
        key = next(iter(other))
        results[key] = other[key]
        with pytest.raises(ValueError):
            save_vllm_native_model(native_model, results, str(tmp_path / "bad"))


class TestW4A4Export:
    """NVFP4 checkpoints with quantized activations (W4A4)."""

    def _collect(self, native_model, results, **kwargs):
        from onecomp.quantizer.floatquant import collect_input_global_scales

        class _Tok:
            def __call__(self, text, **kwargs):
                class _Enc:
                    input_ids = torch.arange(8).reshape(1, 8)

                return _Enc()

        return collect_input_global_scales(
            native_model, _Tok(), list(results), ["calibration text"], device="cpu", **kwargs
        )

    def _collect_tiny_activation_scales(self, **kwargs):
        from onecomp.quantizer.floatquant import collect_input_global_scales

        class _Tok:
            def __call__(self, text, **kwargs):
                class _Enc:
                    input_ids = torch.arange(8).reshape(1, 8)

                return _Enc()

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

            def forward(self, input_ids):
                del input_ids
                x = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4)
                return self.proj(x)

        return collect_input_global_scales(
            _Model(), _Tok(), ["proj"], ["calibration text"], device="cpu", **kwargs
        )

    def test_w4a4_checkpoint_layout(self, native_model, tmp_path):
        results = _quantize_block_linears(native_model, "nvfp4")
        scales = self._collect(native_model, results)
        save_dir = tmp_path / "ct_w4a4"
        save_vllm_native_model(native_model, results, str(save_dir), input_global_scales=scales)

        state = load_file(str(save_dir / "model.safetensors"))
        target = "model.layers.0.self_attn.q_proj"
        assert state[f"{target}.input_global_scale"].dtype == torch.float32
        assert state[f"{target}.input_global_scale"].numel() == 1
        assert state[f"{target}.input_global_scale"].item() > 0

        with open(save_dir / "config.json", encoding="utf-8") as f:
            qc = json.load(f)["quantization_config"]
        acts = qc["config_groups"]["group_0"]["input_activations"]
        assert acts["num_bits"] == 4
        assert acts["type"] == "float"
        assert acts["strategy"] == "tensor_group"
        assert acts["group_size"] == 16
        assert acts["dynamic"] == "local"

    def test_fused_shards_share_input_scale(self, native_model):
        """q/k/v receive the same input, so their scales must be equal."""
        results = _quantize_block_linears(native_model, "nvfp4")
        scales = self._collect(native_model, results)
        base = "model.layers.0.self_attn"
        q = scales[f"{base}.q_proj"]
        for leaf in ("k_proj", "v_proj"):
            assert torch.isclose(scales[f"{base}.{leaf}"], q)

    def test_activation_percentile_scale_search_controls_resolution(self):
        """Lower activation percentiles increase the divisor scale."""
        absmax = self._collect_tiny_activation_scales(percentile=100.0)
        pct = self._collect_tiny_activation_scales(percentile=90.0)
        target = "proj"
        assert pct[target] >= absmax[target]

        doubled = self._collect_tiny_activation_scales(scale_multiplier=2.0)
        assert torch.isclose(doubled[target], absmax[target] * 2.0)

    def test_activation_scale_search_validates_parameters(self):
        with pytest.raises(ValueError, match="percentile"):
            self._collect_tiny_activation_scales(percentile=0.0)
        with pytest.raises(ValueError, match="scale_multiplier"):
            self._collect_tiny_activation_scales(scale_multiplier=0.0)

    def test_w4a4_rejected_for_mxfp4(self, native_model, tmp_path):
        results = _quantize_block_linears(native_model, "mxfp4")
        fake_scales = {name: torch.tensor(1.0) for name in results}
        with pytest.raises(ValueError):
            save_vllm_native_model(
                native_model,
                results,
                str(tmp_path / "bad_w4a4"),
                input_global_scales=fake_scales,
            )

    def test_missing_scale_entries_raise(self, native_model, tmp_path):
        results = _quantize_block_linears(native_model, "nvfp4")
        scales = self._collect(native_model, results)
        scales.pop(next(iter(scales)))
        with pytest.raises(ValueError):
            save_vllm_native_model(
                native_model,
                results,
                str(tmp_path / "bad_missing"),
                input_global_scales=scales,
            )


class TestMixedExport:
    """Mixed NVFP4 / FP8 checkpoints (compressed-tensors mixed-precision)."""

    @staticmethod
    def _mixed_results(native_model, fp8_fraction=0.5):
        from onecomp.quantizer.floatquant import select_mixed_formats

        nvfp4 = _quantize_block_linears(native_model, "nvfp4")
        fp8 = _quantize_block_linears(native_model, "fp8")
        return select_mixed_formats(native_model, nvfp4, fp8, fp8_fraction=fp8_fraction)

    def test_budget_controls_fp8_share(self, native_model):
        all_nvfp4 = self._mixed_results(native_model, fp8_fraction=0.0)
        assert all(result.fmt == "nvfp4" for result in all_nvfp4.values())

        all_fp8 = self._mixed_results(native_model, fp8_fraction=1.0)
        assert any(result.fmt == "fp8" for result in all_fp8.values())

    def test_fused_groups_share_format(self, native_model):
        mixed = self._mixed_results(native_model, fp8_fraction=0.5)
        for layer_idx in range(native_model.config.num_hidden_layers):
            base = f"model.layers.{layer_idx}.self_attn"
            fmts = {mixed[f"{base}.{leaf}"].fmt for leaf in ("q_proj", "k_proj", "v_proj")}
            assert len(fmts) == 1

    def test_mixed_checkpoint_layout(self, native_model, tmp_path):
        from onecomp.quantizer.floatquant import save_vllm_mixed_model

        mixed = self._mixed_results(native_model, fp8_fraction=0.5)
        fp8_names = sorted(n for n, r in mixed.items() if r.fmt == "fp8")
        nvfp4_names = sorted(n for n, r in mixed.items() if r.fmt == "nvfp4")
        assert fp8_names and nvfp4_names, "test needs both formats present"

        save_dir = tmp_path / "ct_mixed"
        save_vllm_mixed_model(native_model, mixed, str(save_dir))

        state = load_file(str(save_dir / "model.safetensors"))
        assert state[f"{nvfp4_names[0]}.weight_packed"].dtype == torch.uint8
        assert state[f"{nvfp4_names[0]}.weight_global_scale"].dtype == torch.float32
        assert state[f"{fp8_names[0]}.weight"].dtype == torch.float8_e4m3fn

        with open(save_dir / "config.json", encoding="utf-8") as f:
            qc = json.load(f)["quantization_config"]
        assert qc["format"] == "mixed-precision"
        groups = qc["config_groups"]
        formats = {g["format"] for g in groups.values()}
        assert formats == {"nvfp4-pack-quantized", "float-quantized"}
        for group in groups.values():
            if group["format"] == "nvfp4-pack-quantized":
                assert group["targets"] == nvfp4_names
            else:
                assert group["targets"] == fp8_names

    def test_fused_group_mixing_formats_raises(self, native_model, tmp_path):
        from onecomp.quantizer.floatquant import save_vllm_mixed_model

        nvfp4 = _quantize_block_linears(native_model, "nvfp4")
        fp8 = _quantize_block_linears(native_model, "fp8")
        mixed = dict(nvfp4)
        mixed["model.layers.0.self_attn.q_proj"] = fp8["model.layers.0.self_attn.q_proj"]
        with pytest.raises(ValueError):
            save_vllm_mixed_model(native_model, mixed, str(tmp_path / "bad_mixed"))

    def test_exact_assignment_beats_ratio_greedy_counterexample(self):
        """Exact DP must handle the standard 0-1 knapsack counterexample."""
        from onecomp.quantizer.floatquant.vllm_export import (
            _select_upgrade_units_exact,
            _select_upgrade_units_greedy,
        )

        candidates = [
            (6.0, 60.0, 10.0, ["A"]),
            (5.0, 100.0, 20.0, ["B"]),
            (4.0, 120.0, 30.0, ["C"]),
        ]
        assert _select_upgrade_units_greedy(candidates, 50.0) == {0, 1}
        assert _select_upgrade_units_exact(candidates, 50.0) == {1, 2}
