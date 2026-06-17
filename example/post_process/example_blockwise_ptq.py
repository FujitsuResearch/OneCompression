"""
Example: quantization + Block-wise PTQ (packed buffers)

Demonstrates the BlockWisePTQ post-process workflow:
    1. Quantize TinyLlama with the selected quantizer
    2. Apply BlockWisePTQ through Runner.run()
    3. Evaluate PPL (original vs quantized+BlockWisePTQ)
    4. Save the packed post-processed checkpoint with save_quantized_model()

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    python example/post_process/example_blockwise_ptq.py
"""

import torch

from onecomp import (
    DBF,  # noqa: F401
    GPTQ,
    JointQ,  # noqa: F401
    Onebit,  # noqa: F401
    RTN,  # noqa: F401
    BlockWisePTQ,
    CalibrationConfig,
    ModelConfig,
    Runner,
    setup_logger,
)

setup_logger()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ================================================================
# Step 1: Quantize + BlockWisePTQ via Runner
# ================================================================
print("=" * 70)
print("Step 1: Quantize TinyLlama + BlockWisePTQ")
print("=" * 70)

model_config = ModelConfig(model_id=MODEL_ID, device=DEVICE)

# BlockWisePTQ currently supports these quantized module families:
#   GPTQLinear:         GPTQ, RTN, JointQ
#   DoubleBinaryLinear: DBF
#   OneBitLinear:       Onebit
quantizer = GPTQ(wbits=4, groupsize=128)
# quantizer = RTN(wbits=4, groupsize=128)
# quantizer = JointQ(bits=4, group_size=128)
# quantizer = DBF(target_bits=1.5)
# quantizer = Onebit()
quantizer_name = type(quantizer).__name__.lower()

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

runner = Runner(
    model_config=model_config,
    quantizer=quantizer,
    calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=128),
    post_processes=[blockwise_ptq],
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
#
# NOTE: This is the normal Runner-managed path. Runner.run() quantizes, builds
# the post-process input with packed buffers by default, applies BlockWisePTQ,
# and stores the result in runner.quantized_model. Use
# example_blockwise_ptq_unpacked.py only when you explicitly need unpacked
# GPTQLinear buffers for research/debug workflows.
runner.run()

# ================================================================
# Step 2: Evaluate PPL
# ================================================================
print("\n" + "=" * 70)
print("Step 2: Evaluate PPL after BlockWisePTQ")
print("=" * 70)

original_ppl, _, blockwise_ppl = runner.calculate_perplexity(
    original_model=True,
    quantized_model=True,
)

print(f"\n  Original model PPL:           {original_ppl:.4f}")
print(f"  Quantized + BlockWisePTQ PPL: {blockwise_ppl:.4f}")
print("=" * 70)

# ================================================================
# Step 3: Save packed post-processed checkpoint
# ================================================================
SAVE_DIR = f"./tinyllama-{quantizer_name}-blockwise-packed"
print("\n" + "=" * 70)
print(f"Step 3: Save packed checkpoint to {SAVE_DIR}")
print("=" * 70)

runner.save_quantized_model(SAVE_DIR)
print(f"Packed BlockWisePTQ model saved to: {SAVE_DIR}")
