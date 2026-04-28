"""

Example: Quantization using AutoBitQuantizer

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

from onecomp import setup_logger
from onecomp import CalibrationConfig, ModelConfig, Runner, AutoBitQuantizer, GPTQ
from onecomp.utils import estimate_wbits_from_vram

setup_logger()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

result = estimate_wbits_from_vram(MODEL_ID, total_vram_gb=0.8)
target_bit = result.target_bitwidth

quantizer = AutoBitQuantizer(
    assignment_strategy="activation_aware",
    target_bit=target_bit,
    quantizers=[GPTQ(wbits=b, groupsize=128) for b in (2, 3, 4, 8)],
    save_path="./results/vram_0.8",
)

runner = Runner(
    model_config=ModelConfig(model_id=MODEL_ID, device="cuda:0"),
    quantizer=quantizer,
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
runner.run()

# Calculate perplexity
# Set True for models to evaluate, False returns None
original_ppl, dequantized_ppl, quantized_ppl = runner.calculate_perplexity(
    original_model=True, dequantized_model=False, quantized_model=True
)

# Display perplexity
print(f"Original model perplexity: {original_ppl}")
print(f"Dequantized model perplexity: {dequantized_ppl}")
print(f"Quantized model perplexity: {quantized_ppl}")
