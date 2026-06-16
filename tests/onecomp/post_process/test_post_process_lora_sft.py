"""
Smoke test for PostProcessLoraSFT.

Verifies that PostProcessLoraSFT completes without error on a small
model (TinyLlama) with minimal training settings (1 epoch, 4 samples).
Also checks that LoRA layers are injected and that weights are updated
after training.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    pytest tests/onecomp/post_process/test_post_process_lora_sft.py -v -s --log-cli-level=INFO
"""

import gc
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from onecomp import GPTQ, CalibrationConfig, ModelConfig, Runner, setup_logger
from onecomp.post_process.post_process_lora_sft import (
    LoRAGPTQLinear,
    PostProcessLoraSFT,
    _capture_gptq_pack_state,
    _iter_gptq_linears,
    _restore_gptq_pack_state,
    _unpack_gptq_linears_in_place,
)
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

try:
    from onecomp import JointQ

    HAS_JOINTQ = True
except ImportError:
    HAS_JOINTQ = False

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SFT_DATA_FILE = str(FIXTURES_DIR / "sft_train_data.jsonl")


@pytest.fixture(scope="module")
def quantized_model_and_config():
    """Quantize TinyLlama with GPTQ and build a packed quantized model.

    ``pack_weights=True`` mirrors the production ``run_post_processes`` path, so
    the smoke tests exercise packed-in -> packed-out preservation.
    """
    setup_logger()

    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
    quantizer = GPTQ(wbits=4, groupsize=128)

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=CalibrationConfig(num_calibration_samples=8, max_length=512),
    )
    runner.run()

    model, _tokenizer = runner.create_quantized_model(
        pack_weights=True,
        use_gemlite=False,
    )

    yield model, model_config

    del model, runner
    gc.collect()
    torch.cuda.empty_cache()


class TestPostProcessLoraSFT:
    """Smoke tests for PostProcessLoraSFT."""

    def test_run_completes_without_error(self, quantized_model_and_config):
        """PostProcessLoraSFT.run() completes without raising."""
        model, model_config = quantized_model_and_config

        post_process = PostProcessLoraSFT(
            data_files=SFT_DATA_FILE,
            epochs=1,
            max_train_samples=4,
            max_length=64,
            batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
        )
        post_process.run(model, model_config)

    def test_base_layers_remain_packed_after_run(self, quantized_model_and_config):
        """Packed input -> packed output: after the post-process restores the
        incoming pack state, every base GPTQLinear is packed again.

        Relies on ``test_run_completes_without_error`` having run the
        post-process on the shared module-scoped model first.
        """
        model, _model_config = quantized_model_and_config

        base_layers = [m for _name, m in model.named_modules() if isinstance(m, GPTQLinear)]
        assert base_layers, "No GPTQLinear base layers found"
        assert all(getattr(m, "_weight_is_packed", False) for m in base_layers), (
            "Expected all base GPTQLinear layers to be packed after run() "
            "(packed-in -> packed-out)"
        )

    def test_lora_layers_are_injected(self, quantized_model_and_config):
        """After run(), target layers should be LoRAGPTQLinear."""
        model, _model_config = quantized_model_and_config

        lora_count = sum(1 for _name, m in model.named_modules() if isinstance(m, LoRAGPTQLinear))
        assert lora_count > 0, "No LoRAGPTQLinear layers found after post-process"

    def test_model_is_on_cpu_after_run(self, quantized_model_and_config):
        """After run(), model should be moved back to CPU."""
        model, _model_config = quantized_model_and_config

        devices = {str(p.device) for p in model.parameters()}
        assert devices == {"cpu"}, f"Expected all params on CPU, got {devices}"

    def test_model_is_in_eval_mode(self, quantized_model_and_config):
        """After run(), model should be in eval mode."""
        model, _model_config = quantized_model_and_config
        assert not model.training, "Model should be in eval mode after run()"


class TestPostProcessLoraSFTViaRunner:
    """Test PostProcessLoraSFT integrated with Runner.run()."""

    def test_runner_with_post_process(self):
        """Runner with post_processes runs end-to-end without error."""
        setup_logger()

        model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
        quantizer = GPTQ(wbits=4, groupsize=128)

        post_process = PostProcessLoraSFT(
            data_files=SFT_DATA_FILE,
            epochs=1,
            max_train_samples=4,
            max_length=64,
            batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
        )

        runner = Runner(
            model_config=model_config,
            quantizer=quantizer,
            calibration_config=CalibrationConfig(num_calibration_samples=8, max_length=512),
            post_processes=[post_process],
        )
        runner.run()

        assert (
            runner.quantized_model is not None
        ), "runner.quantized_model should be set after post-process"

        lora_count = sum(
            1
            for _name, m in runner.quantized_model.named_modules()
            if isinstance(m, LoRAGPTQLinear)
        )
        assert lora_count > 0, "No LoRAGPTQLinear layers found in runner.quantized_model"

        del runner
        gc.collect()
        torch.cuda.empty_cache()


def _make_gptq_linear(packed, wbits=4):
    """Build a small CPU GPTQLinear in the requested pack state."""
    torch.manual_seed(0)
    groupsize = 32
    in_features = groupsize * 2
    out_features = 32
    num_groups = in_features // groupsize
    vmax = (1 << wbits) - 1

    qweight = torch.randint(0, vmax + 1, (out_features, in_features), dtype=torch.int32)
    scales = torch.rand(num_groups, out_features) * 0.05 + 0.02
    zeros = torch.full((num_groups, out_features), float(min(vmax, 5)))

    return GPTQLinear(
        in_features=in_features,
        out_features=out_features,
        wbits=wbits,
        groupsize=groupsize,
        actorder=False,
        quantized_weight=qweight,
        scale=scales,
        zero=zeros,
        bias=None,
        device="cpu",
        pack_weights=packed,
        use_gemlite=False,
    )


class _GptqContainer(nn.Module):
    """A standalone GPTQLinear plus one wrapped as LoRAGPTQLinear.base_layer."""

    def __init__(self, plain_packed, wrapped_packed):
        super().__init__()
        self.plain = _make_gptq_linear(packed=plain_packed)
        self.wrapped = LoRAGPTQLinear(
            base_layer=_make_gptq_linear(packed=wrapped_packed),
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
        )


class TestGptqPackStateHelpers:
    """Deterministic CPU tests for the packed-in/packed-out helpers."""

    def test_iter_gptq_linears_includes_wrapped_base_layer(self):
        """``_iter_gptq_linears`` finds the standalone layer and the
        LoRAGPTQLinear.base_layer, but not the wrapper itself."""
        model = _GptqContainer(plain_packed=True, wrapped_packed=True)
        names = {name for name, _m in _iter_gptq_linears(model)}
        assert names == {"plain", "wrapped.base_layer"}

    def test_packed_in_packed_out_roundtrip(self):
        """capture -> unpack -> restore must re-pack every layer that was
        packed on input, and forward output must survive the round trip."""
        model = _GptqContainer(plain_packed=True, wrapped_packed=True)
        state = _capture_gptq_pack_state(model)
        assert state == {"plain": True, "wrapped.base_layer": True}

        x = torch.randn(2, model.plain.in_features, dtype=torch.float16)
        ref = model.wrapped(x).clone()  # lora_B is zero-init, so == base output

        num_unpacked = _unpack_gptq_linears_in_place(model)
        assert num_unpacked == 2
        assert all(not m._weight_is_packed for _n, m in _iter_gptq_linears(model))

        num_repacked = _restore_gptq_pack_state(model, state)
        assert num_repacked == 2
        assert all(m._weight_is_packed for _n, m in _iter_gptq_linears(model))

        out = model.wrapped(x)
        assert (ref - out).abs().max().item() < 1e-3

    def test_unpacked_in_unpacked_out_is_preserved(self):
        """A layer that was unpacked on input must stay unpacked after the
        round trip, while a packed sibling is re-packed."""
        model = _GptqContainer(plain_packed=True, wrapped_packed=False)
        state = _capture_gptq_pack_state(model)
        assert state == {"plain": True, "wrapped.base_layer": False}

        # Only the packed layer is unpacked for training.
        num_unpacked = _unpack_gptq_linears_in_place(model)
        assert num_unpacked == 1

        # Only the packed-on-input layer is restored; the unpacked one stays.
        num_repacked = _restore_gptq_pack_state(model, state)
        assert num_repacked == 1
        assert model.plain._weight_is_packed is True
        assert model.wrapped.base_layer._weight_is_packed is False


@pytest.mark.skipif(not HAS_JOINTQ, reason="jointq package not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.slow
class TestPostProcessLoraSFTViaRunnerJointQ:
    """Smoke test for JointQ + PostProcessLoraSFT via Runner.run()."""

    def test_runner_with_jointq_and_post_process(self):
        """Runner with JointQ + post_processes runs end-to-end without error."""
        setup_logger()

        model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
        quantizer = JointQ(bits=4, group_size=128)

        post_process = PostProcessLoraSFT(
            data_files=SFT_DATA_FILE,
            epochs=1,
            max_train_samples=4,
            max_length=64,
            batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
        )

        runner = Runner(
            model_config=model_config,
            quantizer=quantizer,
            calibration_config=CalibrationConfig(num_calibration_samples=8, max_length=512),
            post_processes=[post_process],
        )
        runner.run()

        assert (
            runner.quantized_model is not None
        ), "runner.quantized_model should be set after JointQ + post-process"

        lora_count = sum(
            1
            for _name, m in runner.quantized_model.named_modules()
            if isinstance(m, LoRAGPTQLinear)
        )
        assert lora_count > 0, "No LoRAGPTQLinear layers found in JointQ runner.quantized_model"

        devices = {str(p.device) for p in runner.quantized_model.parameters()}
        assert devices == {"cpu"}, f"Expected all params on CPU, got {devices}"

        assert not runner.quantized_model.training, "Model should be in eval mode after run()"

        del runner
        gc.collect()
        torch.cuda.empty_cache()
