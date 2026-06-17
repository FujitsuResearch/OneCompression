"""
Example: load a saved quantized checkpoint, post-process it, and re-save it.

Demonstrates the load -> post-process -> re-save workflow:
    1. Quantize TinyLlama with GPTQ and save the initial checkpoint
    2. Reload that checkpoint with load_quantized_model(..., device_map=None)
       so the model stays on CPU for the post-process entry point
    3. Attach the loaded model to a Runner with quantizer=None
    4. Run BlockWisePTQ via runner.run_post_processes()
    5. Re-save the refined checkpoint with accumulated post-process metadata

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    python example/post_process/example_reload_post_process_resave.py
"""

import json
from pathlib import Path

import torch

from onecomp import (
    BlockWisePTQ,
    CalibrationConfig,
    GPTQ,
    ModelConfig,
    Runner,
    load_quantized_model,
    setup_logger,
)


def main():
    setup_logger()

    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    initial_save_dir = "./tinyllama-gptq4-initial"
    resave_dir = "./tinyllama-gptq4-blockwise-resaved"

    model_config = ModelConfig(model_id=model_id, device=device)

    # ================================================================
    # Step 1: Quantize and save an initial checkpoint
    # ================================================================
    quantize_runner = Runner(
        model_config=model_config,
        quantizer=GPTQ(wbits=4, groupsize=128),
        calibration_config=CalibrationConfig(
            max_length=512,
            num_calibration_samples=128,
        ),
    )
    quantize_runner.run()
    quantize_runner.save_quantized_model(initial_save_dir)
    print(f"Initial quantized checkpoint saved to: {initial_save_dir}")

    # NOTE: Intentionally drop the quantization Runner here. The reload flow
    # below should behave like a fresh process that only has the saved
    # checkpoint, not an in-memory Runner from the quantization step.
    del quantize_runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ================================================================
    # Step 2: Reload on CPU for post-processing
    # ================================================================
    # NOTE: device_map=None keeps the model on CPU. Post-process entry points
    # validate and normalize CPU placement before moving only the working
    # pieces to the target device.
    loaded_model, _ = load_quantized_model(
        initial_save_dir,
        device_map=None,
    )
    print(f"Reloaded model device: {next(loaded_model.parameters()).device}")

    # ================================================================
    # Step 3: Post-process the loaded model and re-save
    # ================================================================
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

    # NOTE: quantizer=None is valid for this reload flow because the loaded
    # quantized model is assigned before calling run_post_processes(). Do not
    # call runner.run() on this Runner; run() still starts from quantization and
    # therefore requires a quantizer.
    post_runner = Runner(
        model_config=model_config,
        quantizer=None,
        post_processes=[blockwise_ptq],
    )
    post_runner.quantized_model = loaded_model

    # NOTE: Main path. Let Runner execute the post-process list so this reuses
    # the same orchestration as Runner.run(), but starts from the loaded model
    # instead of quantizing again.
    post_runner.run_post_processes()

    # NOTE: Alternative direct path. This is useful for quick experiments
    # without Runner orchestration. Keep it commented out for the standard
    # sample, and keep post_runner.quantized_model set before saving.
    # blockwise_ptq.run(loaded_model, model_config)
    # post_runner.quantized_model = loaded_model

    post_runner.save_quantized_model(resave_dir)
    print(f"Post-processed checkpoint re-saved to: {resave_dir}")

    # The history is persisted to config.json and accumulates across repeated
    # load -> post-process -> re-save cycles.
    config_path = Path(resave_dir) / "config.json"
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    history = saved_config["quantization_config"].get("onecomp_post_processes", [])
    print("Recorded post-processes:")
    for entry in history:
        print(f"  - {entry['name']} ({entry['class']})")


if __name__ == "__main__":
    main()
