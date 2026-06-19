"""
Example: staged quantize -> save/load -> BlockWisePTQ -> save/load -> GlobalPTQ -> inference

Unlike example_blockwise_global_ptq.py, which runs the whole chain inside a
single Runner (post_processes=[BlockWisePTQ, GlobalPTQ]), this example performs
each stage separately and persists the model to disk between stages:

    1. Runner.run() with a quantizer only -> quantize TinyLlama (GPTQ 4-bit)
       -> save_quantized_model() -> load_quantized_model()
    2. Attach the loaded model to a Runner (quantizer=None) and refine it with
       BlockWisePTQ via run_post_processes() -> save -> load
    3. Attach the reloaded model to a Runner and optimise it with GlobalPTQ via
       run_post_processes()
    4. Save the final model, reload it, and generate text to verify it works

Why stage it this way:
    - Each save/load boundary behaves like a fresh process that only has the
      on-disk checkpoint, so the example doubles as a check that the
      structure-preserving safetensors round-trips between post-processes.
    - Post-process metadata accumulates in config.json across the cycles, so the
      final checkpoint records both BlockWisePTQ and GlobalPTQ in order.

Both BlockWisePTQ and GlobalPTQ keep the quantized layer structure, so every
checkpoint is HF-compatible safetensors with no adapter sidecar.

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    python example/post_process/example_blockwise_global_ptq_staged.py
"""

import json
from pathlib import Path

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
QUANT_DIR = "./tinyllama-gptq4-staged-quant"
BLOCKWISE_DIR = "./tinyllama-gptq4-staged-blockwise"
GLOBALPTQ_DIR = "./tinyllama-gptq4-staged-globalptq"
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


def free(*objs):
    """Drop references and clear the CUDA cache between stages."""
    del objs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# NOTE: The calibration settings throughout this example are kept compact so the
# demo runs fast and may be insufficient for real quantization.  For higher
# quality, prefer the CalibrationConfig() defaults
# (max_length=2048, num_calibration_samples=512).
model_config = ModelConfig(model_id=MODEL_ID, device=DEVICE)

# ================================================================
# Stage 1: Quantize via Runner.run() -> save -> load
# ================================================================
print("=" * 70)
print("Stage 1: Quantize TinyLlama (GPTQ 4-bit), save, reload")
print("=" * 70)

quant_runner = Runner(
    model_config=model_config,
    quantizer=GPTQ(wbits=4, groupsize=128),
    calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=128),
)
# Runner.run() quantizes only -- no post_processes are configured on this
# Runner, so the chain is driven explicitly stage by stage below.
quant_runner.run()
quant_runner.save_quantized_model(QUANT_DIR)
print(f"Quantized checkpoint saved to: {QUANT_DIR}")

# Drop the quantization Runner so the next stage starts from the checkpoint
# alone, just like a fresh process would.
free(quant_runner)

# ================================================================
# Stage 2: Load -> BlockWisePTQ -> save -> load
# ================================================================
print("\n" + "=" * 70)
print("Stage 2: Reload, refine with BlockWisePTQ, save, reload")
print("=" * 70)

# NOTE: device_map=None keeps the model on CPU. Post-process entry points
# validate and normalize CPU placement before moving only the working pieces to
# the target device.
blockwise_model, _ = load_quantized_model(QUANT_DIR, device_map=None)
print(f"Reloaded quantized model device: {next(blockwise_model.parameters()).device}")

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

# NOTE: quantizer=None is valid here because the loaded quantized model is
# assigned before calling run_post_processes(). Do not call runner.run() on this
# Runner; run() still starts from quantization and therefore requires a
# quantizer.
blockwise_runner = Runner(
    model_config=model_config,
    quantizer=None,
    post_processes=[blockwise_ptq],
)
blockwise_runner.quantized_model = blockwise_model
blockwise_runner.run_post_processes()

blockwise_runner.save_quantized_model(BLOCKWISE_DIR)
print(f"BlockWisePTQ checkpoint saved to: {BLOCKWISE_DIR}")

free(blockwise_runner, blockwise_model)

# ================================================================
# Stage 3: Load -> GlobalPTQ -> save
# ================================================================
print("\n" + "=" * 70)
print("Stage 3: Reload, optimise with GlobalPTQ, save")
print("=" * 70)

globalptq_model, _ = load_quantized_model(BLOCKWISE_DIR, device_map=None)
print(f"Reloaded BlockWisePTQ model device: {next(globalptq_model.parameters()).device}")

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

globalptq_runner = Runner(
    model_config=model_config,
    quantizer=None,
    post_processes=[global_ptq],
)
globalptq_runner.quantized_model = globalptq_model
globalptq_runner.run_post_processes()

globalptq_runner.save_quantized_model(GLOBALPTQ_DIR)
print(f"GlobalPTQ checkpoint saved to: {GLOBALPTQ_DIR}")

# The post-process history accumulates across the staged save/load cycles, so
# the final checkpoint records BlockWisePTQ then GlobalPTQ in order.
config_path = Path(GLOBALPTQ_DIR) / "config.json"
saved_config = json.loads(config_path.read_text(encoding="utf-8"))
history = saved_config["quantization_config"].get("onecomp_post_processes", [])
print("Recorded post-processes (in order):")
for entry in history:
    print(f"  - {entry['name']} ({entry['class']})")

free(globalptq_runner, globalptq_model)

# ================================================================
# Stage 4: Load the final model and run inference
# ================================================================
print("\n" + "=" * 70)
print("Stage 4: Reload final model and generate text")
print("=" * 70)

# device_map="auto" (the default) places the model on GPU for inference.
loaded_model, loaded_tokenizer = load_quantized_model(GLOBALPTQ_DIR)
print(f"Loaded model type  : {type(loaded_model).__name__}")
print(f"Loaded model device: {next(loaded_model.parameters()).device}")

loaded_text = generate_text(
    loaded_model,
    loaded_tokenizer,
    PROMPT,
    device=next(loaded_model.parameters()).device,
)
print(f"\nPrompt   : {PROMPT}")
print(f"Generated: {loaded_text}")
print("=" * 70)
