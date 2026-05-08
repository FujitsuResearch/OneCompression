"""

Example: Quantization using JointQ(4bit, groupsize=128)

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from onecomp import CalibrationConfig, JointQ, ModelConfig, Runner, setup_logger

# Set up logger (output logs to stdout)
setup_logger()

# Prepare the model
model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", device="cuda:0"
)

# Configure the quantization method
jointq = JointQ(bits=4, group_size=128)

# Configure the runner
runner = Runner(
    model_config=model_config,
    quantizer=jointq,
    calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=128),
    qep=False,
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

# Run quantization
runner.run()

# Calculate perplexity
# Set True for models to evaluate, False returns None
original_ppl, dequantized_ppl, quantized_ppl = runner.calculate_perplexity(
    original_model=True, dequantized_model=True, quantized_model=False
)

# Display perplexity
print(f"Original model perplexity: {original_ppl}")
print(f"Dequantized model perplexity: {dequantized_ppl}")
print(f"Quantized model perplexity: {quantized_ppl}")
