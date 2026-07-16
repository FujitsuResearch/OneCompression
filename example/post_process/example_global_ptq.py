"""
Example: quantization + Global PTQ (packed buffers)

End-to-end demonstration of the Global PTQ post-process workflow:
    1. Quantize TinyLlama with the selected quantizer
    2. Apply GlobalPTQ to optimise quantization parameters via KL distillation
    3. Evaluate PPL (original vs quantized+GlobalPTQ)
    4. Save the optimised model in both safetensors and PyTorch .pt formats

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii

Usage:
    python example/post_process/example_global_ptq.py
"""

import torch

from onecomp import DBF  # noqa: F401
from onecomp import RTN  # noqa: F401
from onecomp import JointQ  # noqa: F401
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

    # GlobalPTQ currently supports these quantized module families:
    #   GPTQLinear:         GPTQ, RTN, JointQ
    #   DoubleBinaryLinear: DBF
    quantizer = GPTQ(wbits=4, groupsize=128)
    # quantizer = RTN(wbits=4, groupsize=128)
    # quantizer = JointQ(bits=4, group_size=128)
    # quantizer = DBF(target_bits=1.5)
    quantizer_name = type(quantizer).__name__.lower()

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
        calibration_config=CalibrationConfig(
            max_length=512,
            num_calibration_samples=128,
        ),
        post_processes=[global_ptq],
    )
    # NOTE: The calibration settings above are kept compact so the demo runs
    # fast and may be insufficient for real quantization.  For higher quality,
    # prefer the CalibrationConfig() defaults
    # (max_length=2048, num_calibration_samples=512).
    # For qep=False runs with large calibration data, also pass ``batch_size``
    # as a CalibrationConfig argument, e.g.
    #   CalibrationConfig(
    #       max_length=2048,
    #       num_calibration_samples=512,
    #       batch_size=128,
    #   )
    # so that Runner.quantize_with_calibration_chunked runs instead of a
    # single all-at-once forward pass.
    runner.run()

    original_ppl, _, quantized_ppl = runner.calculate_perplexity(
        original_model=True,
        quantized_model=True,
    )
    print(f"\nOriginal PPL:                  {original_ppl:.4f}")
    print(f"Quantized + Global PTQ PPL:    {quantized_ppl:.4f}")

    # GlobalPTQ only tweaks the quantized layers' buffers (no custom modules),
    # so the optimised model can be saved either way:
    #   - save_quantized_model:    HF-compatible safetensors
    #   - save_quantized_model_pt: whole-object PyTorch .pt (torch.save)
    save_dir = f"./tinyllama-{quantizer_name}-globalptq"
    save_dir_pt = f"{save_dir}-pt"

    runner.save_quantized_model(save_dir)
    print(f"\nModel saved (safetensors) to {save_dir}")

    runner.save_quantized_model_pt(save_dir_pt)
    print(f"Model saved (.pt) to {save_dir_pt}")


if __name__ == "__main__":
    main()
