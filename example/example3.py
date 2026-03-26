"""

Example: Quantization using AutoBitQuantizer

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

from onecomp import setup_logger
from onecomp import ModelConfig, Runner, AutoBitQuantizer, GPTQ
from onecomp.utils import estimate_wbits_from_vram

setup_logger()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

result = estimate_wbits_from_vram(MODEL_ID, total_vram_gb=0.8)
target_bit = result.target_bitwidth

quantizer = AutoBitQuantizer(
    assignment_strategy="activation_aware",
    target_bit=target_bit,
    quantizers=[GPTQ(wbits=b) for b in (2, 3, 4, 5)],
    save_path="./results/vram_0.8",
)

runner = Runner(
    model_config=ModelConfig(model_id=MODEL_ID, device="cuda:0"),
    quantizer=quantizer,
    qep=False,
)
runner.run()

# Calculate perplexity
original_ppl, quantized_ppl = runner.calculate_perplexity()

# Display perplexity
print(f"Original model perplexity: {original_ppl}")
print(f"Quantized model perplexity: {quantized_ppl}")
