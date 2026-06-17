"""
Example: GPTQ quantization + GlobalPTQDistributed (multi-GPU)

Same workflow as example_global_ptq.py but using GlobalPTQDistributed
with DeepSpeed ZeRO-2, designed for torchrun with multiple GPUs.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii

Usage:
    # 4-GPU with DeepSpeed ZeRO-2
    torchrun --nproc_per_node=4 example/post_process/example_global_ptq_distributed.py

    # 2-GPU without DeepSpeed (Trainer-only mode)
    torchrun --nproc_per_node=2 \
        example/post_process/example_global_ptq_distributed.py --no-deepspeed
"""

import argparse
import os

import torch.distributed as dist

from onecomp import (
    GPTQ,
    CalibrationConfig,
    GlobalPTQDistributed,
    ModelConfig,
    Runner,
    setup_logger,
)

DS_CONFIG = os.path.join(os.path.dirname(__file__), "ds_zero2.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-deepspeed", action="store_true", help="Disable DeepSpeed (Trainer-only mode)"
    )
    args = parser.parse_args()

    setup_logger()
    rank = int(os.environ.get("RANK", 0))

    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"

    model_config = ModelConfig(model_id=model_id, device=device)
    quantizer = GPTQ(wbits=4, groupsize=128)

    ds_cfg = None if args.no_deepspeed else DS_CONFIG

    global_ptq = GlobalPTQDistributed(
        epochs=3,
        gptq_lr=1e-5,
        calibration_config=CalibrationConfig(
            num_calibration_samples=32,
            max_length=512,
        ),
        eval_interval=1,
        use_gradient_checkpointing=True,
        deepspeed_config=ds_cfg,
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

    if rank == 0:
        original_ppl, _, quantized_ppl = runner.calculate_perplexity(
            original_model=True,
            quantized_model=True,
        )
        print(f"\nOriginal PPL:                          {original_ppl:.4f}")
        print(f"Quantized + GlobalPTQDistributed PPL:   {quantized_ppl:.4f}")

        # GlobalPTQDistributed only tweaks the quantized layers' buffers (no
        # custom modules), so the optimised model can be saved either way:
        #   - save_quantized_model:    HF-compatible safetensors
        #   - save_quantized_model_pt: whole-object PyTorch .pt (torch.save)
        runner.save_quantized_model("./tinyllama-gptq-globalptq-dist")
        print("\nModel saved (safetensors) to ./tinyllama-gptq-globalptq-dist")

        runner.save_quantized_model_pt("./tinyllama-gptq-globalptq-dist-pt")
        print("Model saved (.pt) to ./tinyllama-gptq-globalptq-dist-pt")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
