"""GPTQ quantization -> direct GGUF export -> CPU inference with llama.cpp.

This example shows the OneComp CPU inference path:
  1. Quantize a model with GPTQ (4-bit symmetric, group size 128).
  2. Export it to GGUF without re-quantization (preserving the GPTQ codes).
  3. Run CPU text generation via llama-cpp-python.

Only the GGUF export and inference are CPU-bound; quantization runs on the GPU
when one is available and falls back to CPU (ModelConfig default
``device="auto"``; both are supported).

Run:
    python example/cpu_inference/example_gptq_gguf_cpu.py

Requires: pip install 'onecomp[llamacpp]'  (gguf + llama-cpp-python)

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import os

from onecomp import GPTQ, CalibrationConfig, ModelConfig, Runner
from onecomp.cpu import LlamaCppModel, convert_gptq_to_gguf
from onecomp.log import setup_logger

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SAVE_DIR = "./TinyLlama-1.1B-gptq-4bit"
GGUF_PATH = "./TinyLlama-1.1B-gptq-4bit.gguf"


def main():
    setup_logger()

    # 1. Quantize (4-bit symmetric, group size 128, no actorder => lossless to Q4_0).
    runner = Runner(
        model_config=ModelConfig(model_id=MODEL_ID),
        quantizer=GPTQ(wbits=4, groupsize=128, sym=True, actorder=False),
        calibration_config=CalibrationConfig(num_calibration_samples=64, max_length=512),
        qep=True,
    )
    runner.run()
    runner.save_quantized_model(SAVE_DIR)

    # 2. Direct GGUF export (no re-quantization; QEP-corrected codes preserved).
    summary = convert_gptq_to_gguf(quantized_dir=SAVE_DIR, out_gguf=GGUF_PATH)
    print("Export summary:", summary)

    # 3. CPU inference.
    model = LlamaCppModel(GGUF_PATH, n_ctx=1024)
    print(model.generate("Fujitsu is", max_tokens=64, temperature=0.0))


if __name__ == "__main__":
    main()
