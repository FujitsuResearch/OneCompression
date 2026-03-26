"""

Example: One-liner quantization using auto_run

Performs the following steps automatically:
  1. Load the model and tokenizer from Hugging Face Hub
  2. Bit-allocation based on activation-aware error estimation with ILP
  3. Quantize with GPTQ + QEP
  4. Evaluate perplexity (wikitext-2) and zero-shot accuracy
  5. Save the quantized model to disk

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from onecomp import Runner

Runner.auto_run(model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

# You can specify the total VRAM
Runner.auto_run(model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", total_vram_gb=1)
