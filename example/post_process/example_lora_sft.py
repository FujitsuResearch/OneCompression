"""
Example: GPTQ quantization + LoRA SFT + save/load

End-to-end demonstration of the LoRA SFT post-process workflow:
    1. Quantize TinyLlama with GPTQ 4-bit (groupsize=128)
    2. Apply LoRA SFT post-process (WikiText-2)
    3. Evaluate PPL (original vs quantized+LoRA)
    4. Save the LoRA-applied model via save_quantized_model_pt()
    5. Load the saved model via load_quantized_model_pt()
    6. Generate text with the loaded model to verify it works

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    python example/post_process/example_lora_sft.py
"""

import torch

from onecomp import (
    GPTQ,
    ModelConfig,
    PostProcessLoraSFT,
    Runner,
    load_quantized_model_pt,
    setup_logger,
)

setup_logger()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SAVE_DIR = "./tinyllama_gptq4_lora"
PROMPT = "Fujitsu is"


def generate_text(model, tokenizer, prompt, device, max_new_tokens=64):
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
# Step 1: Quantize + LoRA SFT via Runner
# ================================================================
print("=" * 70)
print("Step 1: Quantize TinyLlama (GPTQ 4-bit) + LoRA SFT (WikiText-2)")
print("=" * 70)

model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
gptq = GPTQ(wbits=4, groupsize=128)

post_process = PostProcessLoraSFT(
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    train_split="train",
    text_column="text",
    max_train_samples=256,
    max_length=512,
    epochs=2,
    batch_size=2,
    gradient_accumulation_steps=4,
    lr=1e-4,
    lora_r=16,
    lora_alpha=32,
    logging_steps=5,
)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    post_processes=[post_process],
)
runner.run()

# ================================================================
# Step 2: Evaluate PPL (original vs quantized+LoRA)
# ================================================================
print("\n" + "=" * 70)
print("Step 2: Evaluate PPL")
print("=" * 70)

original_ppl, _, quantized_ppl = runner.calculate_perplexity(
    original_model=True,
    quantized_model=True,
)

print(f"  Original model PPL:              {original_ppl:.4f}")
print(f"  Quantized + LoRA SFT model PPL:  {quantized_ppl:.4f}")

# ================================================================
# Step 3: Save the LoRA-applied model (PyTorch .pt format)
# ================================================================
print("\n" + "=" * 70)
print(f"Step 3: Saving LoRA-applied model to {SAVE_DIR}")
print("=" * 70)

runner.save_quantized_model_pt(SAVE_DIR)
print(f"Model saved to: {SAVE_DIR}")

del runner
torch.cuda.empty_cache()

# ================================================================
# Step 4: Load the saved model
# ================================================================
print("\n" + "=" * 70)
print(f"Step 4: Loading model from {SAVE_DIR}")
print("=" * 70)

loaded_model, loaded_tokenizer = load_quantized_model_pt(SAVE_DIR)
print(f"Loaded model type : {type(loaded_model).__name__}")
print(f"Loaded model device: {next(loaded_model.parameters()).device}")

# ================================================================
# Step 5: Generate text with the loaded model
# ================================================================
print("\n" + "=" * 70)
print("Step 5: Generate text with loaded model")
print("=" * 70)

loaded_text = generate_text(
    loaded_model,
    loaded_tokenizer,
    PROMPT,
    device=next(loaded_model.parameters()).device,
)

print(f"\nPrompt   : {PROMPT}")
print(f"Generated: {loaded_text}")
print("=" * 70)
