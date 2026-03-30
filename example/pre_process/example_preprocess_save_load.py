"""

Example: Rotation preprocessing + GPTQ quantization → save → load → verify

Demonstrates the full save/load pipeline for preprocessed+quantized models
with perplexity measurement at each stage:
  1. Measure original model (fp16) perplexity as baseline
  2. Apply SpinQuant-style rotation and measure perplexity
  3. Quantize the rotated model with GPTQ and measure perplexity
  4. Save the quantized model via runner.save_quantized_model()
  5. Load via load_quantized_model() and measure perplexity
     (rotated: true in quantization_config triggers automatic
      Hadamard hook registration)
  6. Run a simple generation to verify the loaded model works

Copyright 2025-2026 Fujitsu Ltd.

"""

import torch
from onecomp import (
    ModelConfig,
    Runner,
    GPTQ,
    prepare_rotated_model,
    load_quantized_model,
    setup_logger,
)
from onecomp.utils.perplexity import calculate_perplexity

setup_logger()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
ROTATED_DIR = "./rotated_model_saveload"
SAVE_DIR = "./quantized_model_saveload"

WBITS = 4
GROUPSIZE = 128
NUM_CALIBRATION_SAMPLES = 128
SEED = 0

# ── 1. Original model perplexity (baseline) ───────────────────
print("=" * 60)
print("Step 1: Original model (fp16) perplexity")
print("=" * 60)
model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
original_ppl = calculate_perplexity(model_config=model_config)
print(f"Original model PPL: {original_ppl:.2f}")

# ── 2. Rotation preprocessing + perplexity ────────────────────
print("\n" + "=" * 60)
print("Step 2: Rotation preprocessing")
print("=" * 60)
rotated_config = prepare_rotated_model(
    model_config=model_config,
    save_directory=ROTATED_DIR,
    seed=SEED,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    wbits=WBITS,
    groupsize=GROUPSIZE,
)
rotated_ppl = calculate_perplexity(model_config=rotated_config)
print(f"Rotated model PPL : {rotated_ppl:.2f}")

# ── 3. Quantize with GPTQ + perplexity ───────────────────────
print("\n" + "=" * 60)
print("Step 3: GPTQ quantization")
print("=" * 60)
gptq = GPTQ(wbits=WBITS, groupsize=GROUPSIZE)

runner = Runner(
    model_config=rotated_config,
    quantizer=gptq,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    calibration_seed=SEED,
)
runner.run()

_, _, quantized_ppl = runner.calculate_perplexity()
print(f"Quantized model PPL: {quantized_ppl:.2f}")

# ── 4. Save ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Save quantized model")
print("=" * 60)
runner.save_quantized_model(SAVE_DIR)
print(f"Quantized model saved to: {SAVE_DIR}")

# ── 5. Load + perplexity ──────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: Load and measure perplexity")
print("=" * 60)
model, tokenizer = load_quantized_model(SAVE_DIR)
print(f"Loaded model type : {type(model).__name__}")
print(f"Loaded model device: {next(model.parameters()).device}")

loaded_ppl = calculate_perplexity(model=model, tokenizer=tokenizer)
print(f"Loaded model PPL  : {loaded_ppl:.2f}")

# ── 6. Verify: simple text generation ─────────────────────────
print("\n" + "=" * 60)
print("Step 6: Text generation")
print("=" * 60)
prompt = "Fujitsu is"
inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Prompt   : {prompt}")
print(f"Generated: {generated}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PPL Summary")
print("=" * 60)
print(f"  Original (fp16)     : {original_ppl:.2f}")
print(f"  After rotation      : {rotated_ppl:.2f}")
print(f"  After quantization  : {quantized_ppl:.2f}")
print(f"  After save → load   : {loaded_ppl:.2f}")
