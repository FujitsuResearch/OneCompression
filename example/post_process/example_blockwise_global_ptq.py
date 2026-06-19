"""
Example: quantization + BlockWisePTQ + GlobalPTQ -> save -> load inference

End-to-end demonstration of chaining two structure-preserving post-processes
in a single Runner:
    1. Quantize TinyLlama with GPTQ 4-bit (groupsize=128)
    2. Apply BlockWisePTQ (post-process 1) to refine the quantized blocks
    3. Apply GlobalPTQ (post-process 2) to globally optimise the quantization
       parameters via KL-divergence distillation from the FP teacher
    4. Save via save_quantized_model() - both BlockWisePTQ and GlobalPTQ keep
       the quantized layer structure, so the result is HF-compatible
       safetensors with no adapter sidecar
    5. Load via load_quantized_model()
    6. Generate text with the loaded model to verify it works

Runner executes the post_processes list in order, so the chain is
    Quantize -> BlockWisePTQ -> GlobalPTQ
and GlobalPTQ optimises the parameters of the BlockWisePTQ-refined model.

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    python example/post_process/example_blockwise_global_ptq.py
"""

import torch

from onecomp import (
    GPTQ,
    BlockWisePTQ,
    CalibrationConfig,
    GlobalPTQ,
    ModelConfig,
    Runner,
    load_quantized_model,
    setup_logger,
)

setup_logger()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SAVE_DIR = "./tinyllama-gptq4-blockwise-globalptq"
PROMPT = "Fujitsu is"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


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
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ================================================================
# Step 1: Quantize + BlockWisePTQ + GlobalPTQ via Runner
# ================================================================
print("=" * 70)
print("Step 1: Quantize TinyLlama (GPTQ 4-bit) + BlockWisePTQ + GlobalPTQ")
print("=" * 70)

model_config = ModelConfig(model_id=MODEL_ID, device=DEVICE)
quantizer = GPTQ(wbits=4, groupsize=128)

# Post-process 1: refine the quantized transformer blocks.
blockwise_ptq = BlockWisePTQ(
    lr=1e-4,
    epochs=10,
    cbq_enable=True,
    gptq_lr=1e-3,
    calibration_config=CalibrationConfig(
        num_calibration_samples=128,
        max_length=2048,
    ),
)

# Post-process 2: globally optimise the quantization parameters.
global_ptq = GlobalPTQ(
    epochs=3,
    gptq_lr=1e-5,
    dbf_lr=5e-4,
    calibration_config=CalibrationConfig(
        num_calibration_samples=32,
        max_length=512,
    ),
    eval_interval=1,
    use_gradient_checkpointing=True,
)

runner = Runner(
    model_config=model_config,
    quantizer=quantizer,
    calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=128),
    post_processes=[blockwise_ptq, global_ptq],
)
# NOTE: The calibration settings above are kept compact so the demo runs
# fast and may be insufficient for real quantization.  For higher quality,
# prefer the CalibrationConfig() defaults
# (max_length=2048, num_calibration_samples=512).
# Runner.run() quantizes, then executes the post_processes list in order:
# BlockWisePTQ refines the quantized blocks, then GlobalPTQ optimises the
# quantization parameters of the refined model. Both keep the quantized layer
# structure, so the result is stored in runner.quantized_model as plain
# GPTQLinear-backed layers (no adapter sidecar).
runner.run()

# ================================================================
# Step 2: Evaluate PPL (original vs quantized + BlockWisePTQ + GlobalPTQ)
# ================================================================
print("\n" + "=" * 70)
print("Step 2: Evaluate PPL")
print("=" * 70)

original_ppl, _, quantized_ppl = runner.calculate_perplexity(
    original_model=True,
    quantized_model=True,
)
print(f"  Original model PPL:                        {original_ppl:.4f}")
print(f"  Quantized + BlockWisePTQ + GlobalPTQ PPL:  {quantized_ppl:.4f}")

# ================================================================
# Step 3: Save (HF-compatible safetensors, structure-preserving)
# ================================================================
print("\n" + "=" * 70)
print(f"Step 3: Saving model to {SAVE_DIR}")
print("=" * 70)

runner.save_quantized_model(SAVE_DIR)
print(f"Model saved (safetensors) to: {SAVE_DIR}")

del runner
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ================================================================
# Step 4: Load the saved model
# ================================================================
print("\n" + "=" * 70)
print(f"Step 4: Loading model from {SAVE_DIR}")
print("=" * 70)

loaded_model, loaded_tokenizer = load_quantized_model(SAVE_DIR)
print(f"Loaded model type  : {type(loaded_model).__name__}")
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
