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
    5. Generate text AFTER LoRA SFT (model can describe OneCompression)
    6. Compare results side by side

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    python example/post_process/example_lora_sft_knowledge.py
"""

from pathlib import Path

import torch

from onecomp import GPTQ, ModelConfig, Runner, PostProcessLoraSFT, setup_logger

setup_logger()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
KNOWLEDGE_DATA = str(Path(__file__).parent / "onecomp_knowledge.jsonl")
PROMPT = "Q: What is OneCompression?\nA:"


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

runner = Runner(model_config=model_config, quantizer=gptq)
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
post_process.run(model, model_config)

# ================================================================
# Step 5: Generate AFTER LoRA SFT
# ================================================================
print("\n" + "=" * 70)
print("Step 5: Generating text AFTER LoRA SFT")
print("=" * 70)

model.to("cuda:0")
after_text = generate_text(model, tokenizer, PROMPT, "cuda:0")
model.to("cpu")
torch.cuda.empty_cache()

print(f"\nPrompt: {PROMPT}")
print(f"Response:\n{after_text}")

# ================================================================
# Step 6: Compare results
# ================================================================
print("\n" + "=" * 70)
print("Comparison: Before vs After LoRA SFT")
print("=" * 70)
print(f"\nPrompt: {PROMPT}")
print(f"\n--- BEFORE LoRA SFT ---")
print(before_text)
print(f"\n--- AFTER LoRA SFT ---")
print(after_text)
print("=" * 70)
