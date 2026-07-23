"""

Example: Quantization using FloatQuant (NVFP4 microscaling format)

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from onecomp import CalibrationConfig, FloatQuant, ModelConfig, Runner, setup_logger

# Set up logger (output logs to stdout)
setup_logger()

# Prepare the model
model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", device="cuda:0"
)

# Configure the quantization method.
# fmt selects the microscaling format:
#   - "nvfp4": FP4 (E2M1) elements, block-16 FP8 E4M3 scales + per-tensor FP32 scale
#   - "mxfp4": FP4 (E2M1) elements, block-32 E8M0 power-of-two scales (OCP MX)
#   - "fp8":   FP8 E4M3 elements with per-channel scales
# use_hessian=True enables GPTQ-style error-compensated rounding
# (requires calibration data); use_hessian=False quantizes directly (RTN-style).
floatquant = FloatQuant(fmt="nvfp4", use_hessian=True)

# Configure the runner
runner = Runner(
    model_config=model_config,
    quantizer=floatquant,
    calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=128),
    qep=False,
)
# NOTE: The calibration settings above are kept compact so the demo runs
# fast and may be insufficient for real quantisation.  For higher quality,
# prefer the CalibrationConfig() defaults
# (max_length=2048, num_calibration_samples=512).

# Run quantization
runner.run()

# Calculate perplexity.
# FloatQuant is a fake-quantizer, so the dequantized model reflects the
# NVFP4-quantized weights.
original_ppl, dequantized_ppl, quantized_ppl = runner.calculate_perplexity(
    original_model=True, dequantized_model=True, quantized_model=False
)

# Display perplexity
print(f"Original model perplexity: {original_ppl}")
print(f"Dequantized (NVFP4) model perplexity: {dequantized_ppl}")
print(f"Quantized model perplexity: {quantized_ppl}")
