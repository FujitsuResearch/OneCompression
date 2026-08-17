"""Example: MDBF quantization followed by Global PTQ.

This example quantizes TinyLlama with MDBF, optimizes both amplitude and
binary sign parameters with Global PTQ, evaluates perplexity, and saves the
optimized model.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii

Usage:
    python example/example_global_ptq_mdbf.py
"""

import torch
from onecomp_globalptq import GlobalPTQ

from onecomp import MDBF, CalibrationConfig, ModelConfig, Runner, setup_logger


def main():
    setup_logger()

    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_config = ModelConfig(model_id=model_id, device=device)
    quantizer = MDBF(target_bits=1.0)

    global_ptq = GlobalPTQ(
        epochs=3,
        dbf_lr=5e-4,
        optimize_binary=True,
        mdbf_ste_k=2.0,
        num_calibration_samples=32,
        max_length=512,
        eval_interval=1,
        use_gradient_checkpointing=True,
    )

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=CalibrationConfig(
            max_length=512,
            num_calibration_samples=128,
        ),
        post_processes=[global_ptq],
        qep=False,
    )
    runner.run()

    original_ppl, _, quantized_ppl = runner.calculate_perplexity(
        original_model=True,
        quantized_model=True,
    )
    print(f"\nOriginal PPL:                  {original_ppl:.4f}")
    print(f"Quantized + Global PTQ PPL:    {quantized_ppl:.4f}")

    save_dir = "./tinyllama-mdbf-globalptq"
    runner.save_quantized_model(save_dir)
    print(f"\nModel saved to {save_dir}")


if __name__ == "__main__":
    main()
