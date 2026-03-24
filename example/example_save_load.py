"""

Example: GPTQ quantization → save → load → verify

Demonstrates the save/load pipeline:
  1. Quantize with GPTQ (4-bit) + QEP
  2. Save the quantized model via runner.save_quantized_model()
  3. Load the saved model via load_quantized_model()
  4. Run a simple generation to verify the loaded model works

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import torch
from onecomp import ModelConfig, Runner, GPTQ, load_quantized_model, setup_logger

setup_logger()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SAVE_DIR = "./tinyllama_gptq4"

# ── 1. Quantize with GPTQ ────────────────────────────────────
model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")

gptq = GPTQ(
    wbits=4,
    groupsize=128,
)

runner = Runner(model_config=model_config, quantizer=gptq, qep=True)
runner.run()

# ── 2. Save ───────────────────────────────────────────────────
runner.save_quantized_model(SAVE_DIR)
print(f"\nQuantized model saved to: {SAVE_DIR}")

# ── 3. Load ───────────────────────────────────────────────────
model, tokenizer = load_quantized_model(SAVE_DIR)
print(f"Loaded model type : {type(model).__name__}")
print(f"Loaded model device: {next(model.parameters()).device}")

# ── 4. Verify: simple text generation ─────────────────────────
prompt = "Fujitsu is"
inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\nPrompt   : {prompt}")
print(f"Generated: {generated}")
