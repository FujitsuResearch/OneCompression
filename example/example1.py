"""

Example: Quantization using GPTQ(3bit) and QEP

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from onecomp import ModelConfig, Runner, GPTQ, setup_logger

# Set up logger (output logs to stdout)
setup_logger()

# Prepare the model
model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    device="cuda:0"
)

# Configure the quantization method
gptq = GPTQ(wbits=3)

# Configure the runner
runner = Runner(model_config=model_config, quantizer=gptq, qep=True)

# Run quantization
runner.run()

# Calculate perplexity
original_ppl, quantized_ppl = runner.calculate_perplexity()

# Display perplexity
print(f"Original model perplexity: {original_ppl}")
print(f"Quantized model perplexity: {quantized_ppl}")
