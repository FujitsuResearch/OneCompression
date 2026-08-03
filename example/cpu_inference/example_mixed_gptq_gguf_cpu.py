"""Example: mixed-bit GPTQ -> mixed-precision GGUF -> llama.cpp CPU inference.

Quantizes a model with per-module bit-widths (4-bit attention, 8-bit MLP gate,
3-bit MLP up, 2-bit MLP down) in a single GPTQ run, then exports it to ONE GGUF
where each tensor carries its own quantization type:

    4-bit sym -> Q4_0   8-bit sym -> Q8_0   3-bit -> Q3_K   2-bit -> Q2_K

The 4/8-bit layers keep the exact GPTQ codes (lossless); the 2/3-bit layers are
re-quantized to K-quants via llama-quantize (requires the llama-quantize binary).

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import json
import os

from transformers import AutoConfig

from llamacpp_plugins.gptq import export_mixed_gptq_gguf, plan_mixed_export
from onecomp import GPTQ, CalibrationConfig, ModelConfig, Runner, setup_logger
from onecomp.cpu import LlamaCppModel

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUT_DIR = "./results/mixed-gptq"
OUT_GGUF = "./results/mixed-gptq.gguf"


def main():
    setup_logger()

    num_layers = AutoConfig.from_pretrained(MODEL_ID).num_hidden_layers
    module_wbits = {}
    for i in range(num_layers):
        module_wbits[f"model.layers.{i}.mlp.gate_proj"] = 8
        module_wbits[f"model.layers.{i}.mlp.up_proj"] = 3
        module_wbits[f"model.layers.{i}.mlp.down_proj"] = 2

    quantizer = GPTQ(wbits=4, groupsize=128, sym=True, module_wbits=module_wbits)
    runner = Runner(
        model_config=ModelConfig(model_id=MODEL_ID, device="cpu"),
        quantizer=quantizer,
        calibration_config=CalibrationConfig(num_calibration_samples=32, max_length=512),
        qep=False,
    )
    runner.run()
    runner.save_quantized_model(OUT_DIR)

    # Preview the per-module GGUF routing.
    for p in plan_mixed_export(OUT_DIR)[:6]:
        print(f"{p.name:45s} bits={p.bits} -> {p.route}/{p.ggml_type}")

    summary = export_mixed_gptq_gguf(OUT_DIR, OUT_GGUF)
    print(json.dumps(summary["plan"], indent=2))

    model = LlamaCppModel(OUT_GGUF, n_ctx=512)
    print(model.generate("The capital of Japan is", max_tokens=32, temperature=0.0))


if __name__ == "__main__":
    main()
