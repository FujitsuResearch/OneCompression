"""
Example: GPTQ quantization + Block-wise PTQ

Demonstrates the BlockWisePTQ post-process workflow:
    1. Quantize TinyLlama with GPTQ 4-bit (groupsize=128)
    2. Evaluate baseline PPL (original vs GPTQ-only)
    3. Apply BlockWisePTQ directly (Phase 1 greedy + Phase 2 CBQ)
    4. Evaluate improved PPL and compare

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    python example/post_process/example_blockwise_ptq.py
"""

from onecomp import (
    GPTQ,
    BlockWisePTQ,
    CalibrationConfig,
    ModelConfig,
    Runner,
    setup_logger,
)

setup_logger()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# ================================================================
# Step 1: Quantize with GPTQ
# ================================================================
print("=" * 70)
print("Step 1: Quantize TinyLlama (GPTQ 4-bit)")
print("=" * 70)

model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
gptq = GPTQ(wbits=4, groupsize=128)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=128),
)
# NOTE: The calibration settings above are kept compact so the demo runs
# fast and may be insufficient for real quantisation.  For higher quality,
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

# ================================================================
# Step 2: Evaluate baseline PPL
# ================================================================
print("\n" + "=" * 70)
print("Step 2: Evaluate baseline PPL (original vs GPTQ-only)")
print("=" * 70)

original_ppl, _, baseline_ppl = runner.calculate_perplexity(
    original_model=True,
    quantized_model=True,
)

print(f"  Original model PPL:  {original_ppl:.4f}")
print(f"  GPTQ baseline PPL:  {baseline_ppl:.4f}")

# ================================================================
# Step 3: Apply BlockWisePTQ directly
# ================================================================
print("\n" + "=" * 70)
print("Step 3: Apply BlockWisePTQ (Phase 1 greedy + Phase 2 CBQ)")
print("=" * 70)

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

model, _ = runner.create_quantized_model(pack_weights=False, use_gemlite=False)
blockwise_ptq.run(model, model_config)
runner.quantized_model = model

# ================================================================
# Step 4: Evaluate improved PPL
# ================================================================
print("\n" + "=" * 70)
print("Step 4: Evaluate PPL after BlockWisePTQ")
print("=" * 70)

_, _, blockwise_ppl = runner.calculate_perplexity(
    quantized_model=True,
)

print(f"\n  Original model PPL:           {original_ppl:.4f}")
print(f"  GPTQ baseline PPL:            {baseline_ppl:.4f}")
print(f"  GPTQ + BlockWisePTQ PPL:      {blockwise_ppl:.4f}")
print(f"  PPL improvement:              {baseline_ppl - blockwise_ppl:.4f}")
print("=" * 70)
