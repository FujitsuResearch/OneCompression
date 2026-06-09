"""
Example: DBF quantization + Global PTQ

End-to-end demonstration of Global PTQ with DBF backend:
    1. Quantize TinyLlama with DBF
    2. Apply GlobalPTQ to optimise DBF scalings via KL distillation
    3. Evaluate PPL (original vs quantized+GlobalPTQ)
    4. Save the optimised model

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii

Usage:
    python example/post_process/example_global_ptq_dbf.py
"""

import torch

from onecomp import (
    CalibrationConfig,
    GlobalPTQ,
    ModelConfig,
    Runner,
    setup_logger,
)
from onecomp.quantizer.dbf import DBF


def main():
    setup_logger()

    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_config = ModelConfig(model_id=model_id, device=device)
    quantizer = DBF()

    global_ptq = GlobalPTQ(
        epochs=3,
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
        calibration_config=CalibrationConfig(
            max_length=512,
            num_calibration_samples=128,
        ),
        post_processes=[global_ptq],
    )
    runner.run()

    original_ppl, _, quantized_ppl = runner.calculate_perplexity(
        original_model=True,
        quantized_model=True,
    )
    print(f"\nOriginal PPL:                  {original_ppl:.4f}")
    print(f"Quantized + Global PTQ PPL:    {quantized_ppl:.4f}")

    runner.save_quantized_model_pt("./tinyllama-dbf-globalptq")
    print("\nModel saved to ./tinyllama-dbf-globalptq")


if __name__ == "__main__":
    main()
