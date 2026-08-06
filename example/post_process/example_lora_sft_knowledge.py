"""
Example: Knowledge injection via LoRA SFT on a GPTQ-quantized model

Demonstrates how to teach a quantized model new knowledge using
LoRA SFT post-processing. The model learns about "OneCompression"
(a topic it has never seen) and can answer questions about it
after training.

Flow:
    1. Quantize TinyLlama with GPTQ 4-bit (groupsize=128)
    2. Build quantized model via create_quantized_model
    3. Generate text BEFORE LoRA SFT (model does not know OneCompression)
    4. Run LoRA SFT with OneCompression knowledge data
    5. Save the LoRA-applied quantized model in HF-compatible format
    6. Load the saved model and auto-apply the LoRA adapter
    7. Generate text AFTER LoRA SFT (model can describe OneCompression)
    8. Compare results side by side

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    python example/post_process/example_lora_sft_knowledge.py
"""

from pathlib import Path

import torch

from onecomp import (
    GPTQ,
    CalibrationConfig,
    ModelConfig,
    PostProcessLoraSFT,
    Runner,
    load_quantized_model,
    setup_logger,
)

setup_logger()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
KNOWLEDGE_DATA = str(Path(__file__).parent / "onecomp_knowledge.jsonl")
PROMPT = "Q: What is OneCompression?\nA:"
SAVE_DIR = "./tinyllama_gptq4_lora_knowledge"


def generate_text(model, tokenizer, prompt, device, max_new_tokens=128):
    """Generate text from a prompt using the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.2,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


# ================================================================
# Step 1: Quantize the model with GPTQ 4-bit
# ================================================================
print("=" * 70)
print("Step 1: Quantizing TinyLlama with GPTQ 4-bit (groupsize=128)")
print("=" * 70)

model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
gptq = GPTQ(wbits=4, groupsize=128)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=128),
)
runner.run()

# ================================================================
# Step 2: Build quantized model
# ================================================================
print("\n" + "=" * 70)
print("Step 2: Building quantized model via create_quantized_model")
print("=" * 70)

model, tokenizer = runner.create_quantized_model(
    pack_weights=False,
    use_gemlite=False,
)

# ================================================================
# Step 3: Generate BEFORE LoRA SFT
# ================================================================
print("\n" + "=" * 70)
print("Step 3: Generating text BEFORE LoRA SFT")
print("=" * 70)

model.to("cuda:0")
before_text = generate_text(model, tokenizer, PROMPT, "cuda:0")
model.to("cpu")
torch.cuda.empty_cache()

print(f"\nPrompt: {PROMPT}")
print(f"Response:\n{before_text}")

# ================================================================
# Step 4: Run LoRA SFT with OneCompression knowledge
# ================================================================
print("\n" + "=" * 70)
print("Step 4: Running LoRA SFT with OneCompression knowledge data")
print("=" * 70)

post_process = PostProcessLoraSFT(
    data_files=KNOWLEDGE_DATA,
    max_length=256,
    epochs=50,
    batch_size=2,
    gradient_accumulation_steps=1,
    lr=3e-4,
    lora_r=16,
    lora_alpha=32,
    logging_steps=5,
)
runner.post_processes = [post_process]
runner.run_post_processes()

# ================================================================
# Step 5: Save the LoRA-applied quantized model (HF safetensors + adapter sidecar)
# ================================================================
print("\n" + "=" * 70)
print(f"Step 5: Saving LoRA-applied model to {SAVE_DIR}")
print("=" * 70)
runner.save_quantized_model(SAVE_DIR)
print(f"Model saved to: {SAVE_DIR}")
print(
    "  - model.safetensors, config.json                  : base GPTQ model (HF-compatible)\n"
    "  - lora_adapter/adapter_model.safetensors           : PEFT-format LoRA adapter\n"
    "  - lora_adapter/adapter_config.json                 : PEFT-format adapter config"
)
# Release references and clear CUDA cache before reload to reduce OOM risk
del runner
del model
torch.cuda.empty_cache()

# ================================================================
# Step 6: Load the saved model with LoRA adapter
# ================================================================
print("\n" + "=" * 70)
print(f"Step 6: Loading model from {SAVE_DIR}")
print("=" * 70)

loaded_model, loaded_tokenizer = load_quantized_model(SAVE_DIR)
print(f"Loaded model type : {type(loaded_model).__name__}")
print(f"Loaded model device: {next(loaded_model.parameters()).device}")

# ================================================================
# Step 7: Generate AFTER LoRA SFT
# ================================================================
print("\n" + "=" * 70)
print("Step 7: Generating text AFTER LoRA SFT")
print("=" * 70)

loaded_model.to("cuda:0")
after_text = generate_text(loaded_model, loaded_tokenizer, PROMPT, "cuda:0")
loaded_model.to("cpu")
torch.cuda.empty_cache()

print(f"\nPrompt: {PROMPT}")
print(f"Response:\n{after_text}")

# ================================================================
# Step 8: Compare results
# ================================================================
print("\n" + "=" * 70)
print("Step 8: Comparison: Before vs After LoRA SFT")
print("=" * 70)
print(f"\nPrompt: {PROMPT}")
print(f"\n--- BEFORE LoRA SFT ---")
print(before_text)
print(f"\n--- AFTER LoRA SFT ---")
print(after_text)
print("=" * 70)
