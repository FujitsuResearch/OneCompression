"""
Example: Knowledge injection via LoRA SFT on a JointQ-quantized model

Demonstrates how to teach a quantized model new knowledge using
LoRA SFT post-processing. The model learns about "OneCompression"
(a topic it has never seen) and can answer questions about it
after training.

Flow:
    1. Quantize TinyLlama with JointQ 4-bit (groupsize=128)
    2. Build quantized model via create_quantized_model
    3. Generate text BEFORE LoRA SFT (model does not know OneCompression)
    4. Run LoRA SFT with OneCompression knowledge data
    4-1. Save via save_quantized_model() - writes HF-compatible safetensors
         plus a PEFT-format LoRA adapter sidecar
         (``adapter_model.safetensors`` + ``adapter_config.json``)
    4-2. Load via load_quantized_model() - the sidecar is auto-detected and
         matching GPTQLinear layers are re-wrapped with LoRAGPTQLinear
    5. Generate text AFTER LoRA SFT (model can describe OneCompression)
    6. Compare results side by side

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    python example/post_process/example_lora_sft_knowledge.py
"""

from pathlib import Path

import torch
import torch.nn as nn

from onecomp import CalibrationConfig, JointQ, ModelConfig, Runner, PostProcessLoraSFT, setup_logger, load_quantized_model

setup_logger()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
KNOWLEDGE_DATA = str(Path(__file__).parent / "onecomp_knowledge.jsonl")
PROMPT = "Q: What is OneCompression?\nA:"
SAVE_DIR = "./tinyllama_jointq4_lora_knowledge"

def dump_model_summary(model, max_modules=1000):
    print("\n" + "=" * 70)
    print("Model summary before save")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1) Linear系モジュールの一覧
    # ------------------------------------------------------------
    print("\n[Linear-like modules]")
    linear_count = 0

    for name, module in model.named_modules():
        cls_name = module.__class__.__name__

        # 通常の Linear、またはクラス名に Linear を含むものを拾う
        if isinstance(module, nn.Linear) or "Linear" in cls_name:
            linear_count += 1

            in_features = getattr(module, "in_features", None)
            out_features = getattr(module, "out_features", None)

            print(f"- {name}")
            print(f"    class        : {cls_name}")
            print(f"    in_features  : {in_features}")
            print(f"    out_features : {out_features}")

            # 代表的な重みテンソルを表示
            for attr in ["weight", "qweight", "scales", "lora_A", "lora_B"]:
                if hasattr(module, attr):
                    obj = getattr(module, attr)
                    if isinstance(obj, torch.Tensor):
                        print(
                            f"    {attr:<12}: shape={tuple(obj.shape)}, "
                            f"dtype={obj.dtype}, device={obj.device}"
                        )
                    elif hasattr(obj, "weight") and isinstance(obj.weight, torch.Tensor):
                        print(
                            f"    {attr}.weight : shape={tuple(obj.weight.shape)}, "
                            f"dtype={obj.weight.dtype}, device={obj.weight.device}"
                        )

            print()

            if linear_count >= max_modules:
                print(f"... truncated after {max_modules} modules")
                break

    print(f"Detected linear-like modules: {linear_count}")

    # ------------------------------------------------------------
    # 2) trainable parameter 一覧
    # ------------------------------------------------------------
    print("\n[Trainable parameters]")
    trainable = 0
    total = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            print(
                f"- {name}: shape={tuple(param.shape)}, "
                f"dtype={param.dtype}, device={param.device}"
            )

    print(f"\nTrainable params: {trainable:,}")
    print(f"Total params    : {total:,}")
    if total > 0:
        print(f"Trainable ratio : {100.0 * trainable / total:.6f}%")

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
# Step 1: Quantize the model with JointQ 4-bit
# ================================================================
print("=" * 70)
print("Step 1: Quantizing TinyLlama with JointQ 4-bit (groupsize=128)")
print("=" * 70)

model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
jointq = JointQ(bits=4, group_size=128)

runner = Runner(
    model_config=model_config,
    quantizer=jointq,
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
model = runner.quantized_model

# ================================================================
# Step 4.1: Save the LoRA-applied quantized model (HF safetensors + adapter sidecar)
# ================================================================
print("\n" + "=" * 70)
print("Step 4.1-a: Inspecting model before save")
print("=" * 70)
dump_model_summary(model)

print("\n" + "=" * 70)
print(f"Step 4.1-b: Saving LoRA-applied model to {SAVE_DIR}")
print("=" * 70)
runner.save_quantized_model(SAVE_DIR)
print(f"Model saved to: {SAVE_DIR}")
print(
    "  - model.safetensors, config.json                  : base JointQ model (HF-compatible)\n"
    "  - lora_adapter/adapter_model.safetensors           : PEFT-format LoRA adapter\n"
    "  - lora_adapter/adapter_config.json                 : PEFT-format adapter config"
)

# ================================================================
# Step 4.2: Load the saved model
# ================================================================
print("\n" + "=" * 70)
print(f"Step 4.2: Loading model from {SAVE_DIR}")
print("=" * 70)

loaded_model, loaded_tokenizer = load_quantized_model(SAVE_DIR)
print(f"Loaded model type : {type(loaded_model).__name__}")
print(f"Loaded model device: {next(loaded_model.parameters()).device}")

# ================================================================
# Step 5: Generate AFTER LoRA SFT
# ================================================================
print("\n" + "=" * 70)
print("Step 5: Generating text AFTER LoRA SFT")
print("=" * 70)

loaded_model.to("cuda:0")
after_text = generate_text(loaded_model, loaded_tokenizer, PROMPT, "cuda:0")
loaded_model.to("cpu")
torch.cuda.empty_cache()

print(f"\nPrompt: {PROMPT}")
print(f"Response:\n{after_text}")

print("\n" + "=" * 70)
print("Step 5(ex): Generating text Before Save model")
print("=" * 70)

# test before save model
model.to("cuda:0")
bs_text = generate_text(model, tokenizer, PROMPT, "cuda:0")
model.to("cpu")
torch.cuda.empty_cache()

print(f"\nPrompt: {PROMPT}")
print(f"Response:\n{bs_text}")

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



