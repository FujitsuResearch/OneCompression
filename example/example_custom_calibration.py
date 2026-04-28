"""
Example: Custom calibration data vs default C4 calibration

Demonstrates how to use CalibrationConfig with a custom calibration dataset
and compares the results against the default C4 calibration.

Steps:
  1. Quantize with default C4 calibration (GPTQ 3-bit)
  2. Quantize with custom calibration using CalibrationConfig
  3. Compare inference outputs

The custom calibration data is a collection of Python code snippets
stored in example/data/python_calibration.txt.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    python example/example_custom_calibration.py
"""

import os

import torch
from onecomp import (
    CalibrationConfig,
    GPTQ,
    ModelConfig,
    Runner,
    setup_logger,
)

setup_logger()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

PROMPTS = [
    "def fibonacci(n):\n",
    "def binary_search(arr, target):\n",
    "# Reverse a linked list\nclass Node:\n",
    "The capital of France is",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "python_calibration.txt")

# ── 1. Quantize with default C4 calibration ──────────────────────

print("=" * 60)
print("Quantizing with default C4 calibration (GPTQ 3-bit)")
print("=" * 60)

model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
gptq_default = GPTQ(wbits=3, groupsize=128)

runner_default = Runner(
    model_config=model_config,
    quantizer=gptq_default,
    calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=128),
)
runner_default.run()

# ── 2. Quantize with custom calibration (CalibrationConfig) ──────

print("\n" + "=" * 60)
print("Quantizing with custom calibration (Python code snippets, GPTQ 3-bit)")
print("=" * 60)

gptq_custom = GPTQ(wbits=3, groupsize=128)

calib_config = CalibrationConfig(
    calibration_dataset=CUSTOM_DATA_PATH,
    max_length=512,
    num_calibration_samples=128,
    strategy="concat_chunk",
)

runner_custom = Runner(
    model_config=model_config,
    quantizer=gptq_custom,
    calibration_config=calib_config,
)
runner_custom.run()

# ── 3. Compare inference outputs ─────────────────────────────────

print("\n" + "=" * 60)
print("Comparing inference outputs")
print("=" * 60)

tokenizer = model_config.load_tokenizer()

models = {}
for label, runner in [("C4 (default)", runner_default), ("Custom (Python)", runner_custom)]:
    model, _ = runner.create_quantized_model()
    model.to("cuda:0")
    models[label] = model

for prompt in PROMPTS:
    print(f"\n{'─' * 60}")
    print(f"Prompt: {prompt!r}")
    print(f"{'─' * 60}")

    inputs = tokenizer(prompt, return_tensors="pt")

    for label, model in models.items():
        with torch.no_grad():
            output_ids = model.generate(
                **{k: v.to("cuda:0") for k, v in inputs.items()},
                max_new_tokens=64,
                do_sample=False,
            )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\n  [{label}]")
        print(f"  {generated}")

for model in models.values():
    del model
torch.cuda.empty_cache()
