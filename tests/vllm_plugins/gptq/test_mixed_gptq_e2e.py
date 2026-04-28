"""End-to-end test: mixed group-size AutoBit quantization → vLLM inference.

Validates:
  1. AutoBit with mixed group-size candidates produces a valid saved model
  2. The saved config.json contains per-module group_size in quantization_bits
  3. vLLM loads the model and generates output without errors

Quantization runs once per module (shared via fixture) with qep=False and
minimal calibration samples to keep runtime short.

Requirements: CUDA GPU.  vLLM tests additionally require vLLM.

Copyright 2025-2026 Fujitsu Ltd.

"""

import gc
import json
import os
import tempfile

import pytest
import torch

try:
    from vllm import LLM, SamplingParams

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

from onecomp import CalibrationConfig, ModelConfig, Runner
from onecomp.quantizer.autobit._autobit import AutoBitQuantizer
from onecomp.quantizer.gptq import GPTQ
from onecomp.utils import estimate_wbits_from_vram

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

SMALL_MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

MIXED_GS_CANDIDATES = [
    GPTQ(wbits=2, groupsize=128),
    GPTQ(wbits=4, groupsize=128),
    GPTQ(wbits=4, groupsize=32),
]


@pytest.fixture(scope="module")
def quantized_model_dir(tmp_path_factory):
    """Quantize once with mixed group-size AutoBit and save to a temp directory.

    Shared across all tests in this module.
    Uses qep=False and num_calibration_samples=8 for speed.
    """
    target_bit = estimate_wbits_from_vram(SMALL_MODEL_ID, total_vram_gb=0.8).target_bitwidth

    quantizer = AutoBitQuantizer(
        assignment_strategy="activation_aware",
        target_bit=target_bit,
        quantizers=MIXED_GS_CANDIDATES,
        enable_fused_groups=True,
    )
    runner = Runner(
        model_config=ModelConfig(model_id=SMALL_MODEL_ID, device="cuda:0"),
        quantizer=quantizer,
        calibration_config=CalibrationConfig(num_calibration_samples=8, max_length=512),
        qep=False,
    )
    runner.run()

    save_dir = str(tmp_path_factory.mktemp("mixed_gs_model"))
    runner.save_quantized_model(save_dir)

    del runner
    gc.collect()
    torch.cuda.empty_cache()

    return save_dir


# ---------------------------------------------------------------------------
# Config verification (no vLLM needed)
# ---------------------------------------------------------------------------


class TestMixedGroupSizeQuantizeSave:
    """Verify the saved model contains per-module group_size in its config."""

    def test_config_json_exists(self, quantized_model_dir):
        config_path = os.path.join(quantized_model_dir, "config.json")
        assert os.path.exists(config_path)

    def test_quant_method_is_mixed_gptq(self, quantized_model_dir):
        with open(os.path.join(quantized_model_dir, "config.json")) as f:
            qcfg = json.load(f).get("quantization_config", {})
        assert qcfg.get("quant_method") == "mixed_gptq"

    def test_quantization_bits_not_empty(self, quantized_model_dir):
        with open(os.path.join(quantized_model_dir, "config.json")) as f:
            qcfg = json.load(f).get("quantization_config", {})
        qbits = qcfg.get("quantization_bits", [])
        assert len(qbits) > 0, "quantization_bits is empty"

    def test_multiple_distinct_group_sizes(self, quantized_model_dir):
        with open(os.path.join(quantized_model_dir, "config.json")) as f:
            qcfg = json.load(f).get("quantization_config", {})
        qbits = qcfg.get("quantization_bits", [])

        found_group_sizes = set()
        for layer_cfg in qbits:
            for mod_cfg in layer_cfg.values():
                params = mod_cfg.get("params", {})
                gs = mod_cfg.get("group_size") or params.get("group_size")
                if gs is not None:
                    found_group_sizes.add(gs)

        assert len(found_group_sizes) > 1, (
            f"Expected multiple distinct group_sizes, found {found_group_sizes}. "
            "All modules may have been assigned the same quantizer."
        )


# ---------------------------------------------------------------------------
# vLLM inference (requires vLLM)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestMixedGroupSizeVllmInference:
    """Load the quantized model with vLLM and verify generation works."""

    def test_generate_produces_non_empty_output(self, quantized_model_dir):
        # gpu_memory_utilization is lowered from the vLLM default (0.92) to
        # accommodate DGX Spark's 128 GB Unified Memory and the test job's
        # SLURM cgroup limit (--mem=115G in run_test_vllm.sh):
        #   - 0.92 (~112 GiB) trips vLLM's own startup OOM check (only ~106
        #     GiB of UMA is free after AutoBit quantization runs in the same
        #     process).
        #   - 0.85 (~103 GiB) clears vLLM's check but the resulting Python
        #     residual (~16 GiB for vllm/transformers/torch imports + pytest
        #     state) plus 103 GiB allocation overflows the 115 GiB cgroup
        #     and the kernel OOM-kills the process.
        #   - 0.78 (~95 GiB) leaves ~4 GiB cgroup headroom and is the
        #     largest value we can use without cgroup OOM.
        llm = LLM(
            model=quantized_model_dir,
            max_model_len=512,
            dtype="float16",
            enforce_eager=True,
            gpu_memory_utilization=0.78,
        )

        outputs = llm.generate(
            ["The capital of France is"],
            SamplingParams(max_tokens=16, temperature=0.0),
        )

        assert len(outputs) == 1
        text = outputs[0].outputs[0].text
        assert len(text) > 0, "vLLM generated empty output"

        del llm
        gc.collect()
        torch.cuda.empty_cache()
