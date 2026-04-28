"""
Example: TinyLlama-1.1B-intermediate-step-1431k-3T Rotation preprocessing + RTN quantization

Step 1: Apply SpinQuant-style rotation to the model and save.
Step 2: Load the rotated model and quantize with RTN.

Copyright 2025-2026 Fujitsu Ltd.
"""

from onecomp import ModelConfig, Runner, RTN, prepare_rotated_model, setup_logger

setup_logger()

# ============================================================
# Step 1: Rotation preprocessing
# ============================================================

model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    device="cuda:0",
)

rotated_config = prepare_rotated_model(
    model_config=model_config,
    save_directory="./rotated_model_llama_rtn",
    seed=0,
    wbits=3,
    groupsize=-1,
    sym=False,
)

# ============================================================
# Step 2: Quantization with the rotated model
# wbits/groupsize/sym must match the values used in Step 1
# ============================================================

rtn = RTN(wbits=3, groupsize=-1, sym=False)
runner = Runner(model_config=rotated_config, quantizer=rtn)
runner.run()

# Calculate perplexity
original_ppl, dequantized_ppl, quantized_ppl = runner.calculate_perplexity(
    original_model=True, dequantized_model=True, quantized_model=False
)
print(f"Original model perplexity: {original_ppl}")
print(f"Dequantized model perplexity: {dequantized_ppl}")
