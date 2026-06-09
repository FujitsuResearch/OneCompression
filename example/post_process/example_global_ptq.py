"""
Example: GPTQ quantization + Global PTQ

End-to-end demonstration of the Global PTQ post-process workflow:
    1. Quantize TinyLlama with GPTQ 4-bit (groupsize=128)
    2. Apply GlobalPTQ to optimise scales/zeros via KL distillation
    3. Evaluate PPL (original vs quantized+GlobalPTQ)
    4. Save the optimised model

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii

Usage:
    python example/post_process/example_global_ptq.py
"""

import torch

from onecomp import (
    GPTQ,
    CalibrationConfig,
    GlobalPTQ,
    ModelConfig,
    Runner,
    setup_logger,
)


def main():
    setup_logger()

    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_config = ModelConfig(model_id=model_id, device=device)
    quantizer = GPTQ(wbits=4, groupsize=128)

    global_ptq = GlobalPTQ(
        epochs=3,
        gptq_lr=1e-5,
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

    runner.save_quantized_model_pt("./tinyllama-gptq-globalptq")
    print("\nModel saved to ./tinyllama-gptq-globalptq")


if __name__ == "__main__":
    main()
