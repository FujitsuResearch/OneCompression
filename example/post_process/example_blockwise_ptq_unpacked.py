"""
Example: quantization + Block-wise PTQ (unpacked buffers)

Demonstrates the explicit unpacked BlockWisePTQ workflow:
    1. Quantize TinyLlama with the selected quantizer
    2. Evaluate baseline PPL (original vs quantized-only)
    3. Build an unpacked quantized model with pack_weights=False
    4. Apply BlockWisePTQ directly and evaluate the result

This is a research/debug counterpart to example_blockwise_ptq.py.  The normal
Runner-managed path builds packed buffers by default and is the recommended
path for reusable saved checkpoints and vLLM-serving workflows.

Current cases that must use unpacked GPTQLinear buffers:
    - JointQ(bits=1, ...), because GPTQLinear bit packing does not support
      1-bit JointQ output.
    - GPTQ/RTN bit widths in {1, 5, 6, 7}, because the GPTQLinear packing
      helpers currently support only {2, 3, 4, 8}.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    python example/post_process/example_blockwise_ptq_unpacked.py
"""

import torch

from onecomp import DBF  # noqa: F401
from onecomp import RTN  # noqa: F401
from onecomp import JointQ  # noqa: F401
from onecomp import Onebit  # noqa: F401
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
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ================================================================
# Step 1: Quantize
# ================================================================
print("=" * 70)
print("Step 1: Quantize TinyLlama")
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
#
# Mandatory unpacked GPTQLinear-backed settings today:
# quantizer = GPTQ(wbits=1, groupsize=128)
# quantizer = RTN(wbits=1, groupsize=128)
# quantizer = JointQ(bits=1, group_size=128)

runner = Runner(
    model_config=model_config,
    quantizer=quantizer,
    calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=128),
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

# ================================================================
# Step 2: Evaluate baseline PPL
# ================================================================
print("\n" + "=" * 70)
print("Step 2: Evaluate baseline PPL (original vs quantized-only)")
print("=" * 70)

original_ppl, _, baseline_ppl = runner.calculate_perplexity(
    original_model=True,
    quantized_model=True,
)

print(f"  Original model PPL:  {original_ppl:.4f}")
print(f"  Quantized baseline PPL:  {baseline_ppl:.4f}")

# ================================================================
# Step 3: Apply BlockWisePTQ directly on an unpacked quantized model
# ================================================================
print("\n" + "=" * 70)
print("Step 3: Apply BlockWisePTQ on unpacked buffers")
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

# NOTE: Explicitly request unpacked qweight/qzeros buffers for
# GPTQLinear-backed quantizers. This is not the normal Runner-managed
# post-process path; use the packed example for standard saved checkpoints.
model, _ = runner.create_quantized_model(pack_weights=False, use_gemlite=False)
blockwise_ptq.run(model, model_config)
runner.quantized_model = model

# ================================================================
# Step 4: Evaluate improved PPL
# ================================================================
print("\n" + "=" * 70)
print("Step 4: Evaluate PPL after unpacked BlockWisePTQ")
print("=" * 70)

_, _, blockwise_ppl = runner.calculate_perplexity(
    quantized_model=True,
)

print(f"\n  Original model PPL:           {original_ppl:.4f}")
print(f"  Quantized baseline PPL:       {baseline_ppl:.4f}")
print(f"  Quantized + BlockWisePTQ PPL: {blockwise_ppl:.4f}")
print(f"  PPL improvement:              {baseline_ppl - blockwise_ppl:.4f}")
print("=" * 70)
