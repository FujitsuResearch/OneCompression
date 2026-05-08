"""

Example: Quantization using GPTQ(3bit) + QEP + LPCD

LPCD (Layer-Projected Coordinate Descent) refines quantized weights by
jointly optimising sub-module groups (QK / VO / MLP / residual) with
closed-form and gradient-based solvers on top of the base quantiser.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from onecomp import (
    CalibrationConfig,
    GPTQ,
    LPCDConfig,
    ModelConfig,
    Runner,
    setup_logger,
)

setup_logger()

model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", device="cuda:0"
)

gptq = GPTQ(wbits=3, groupsize=128)

# Default: enable_residual=True (refine o_proj / down_proj via
# closed-form solvers).  Set enable_qk / enable_vo / enable_ud=True
# for stronger refinement at the cost of longer runtime.
lpcd_config = LPCDConfig(
    enable_residual=True,
    perccorr=0.5,
    percdamp=0.01,
    use_closed_form=True,
    device="cuda:0",
)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=128),
    qep=True,
    lpcd=True,
    lpcd_config=lpcd_config,
)
# NOTE: The calibration settings above are kept compact so the demo runs
# fast and may be insufficient for real quantisation.  For higher quality,
# prefer the CalibrationConfig() defaults
# (max_length=2048, num_calibration_samples=512).

runner.run()

original_ppl, dequantized_ppl, quantized_ppl = runner.calculate_perplexity(
    original_model=True, dequantized_model=False, quantized_model=True
)

print(f"Original model perplexity:    {original_ppl}")
print(f"Dequantized model perplexity: {dequantized_ppl}")
print(f"Quantized model perplexity:   {quantized_ppl}")
