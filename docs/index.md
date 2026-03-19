# Fujitsu One Compression

**Open-source Python library for post-training quantization of Large Language Models**

---

Fujitsu One Compression (OneComp) is an open-source Python library for post-training quantization of Large Language Models (LLMs).
It implements state-of-the-art quantization algorithms including GPTQ, DBF, RTN, and the novel
**Quantization Error Propagation (QEP)** method proposed in our NeurIPS 2025 paper.

## Key Features

- **Multiple quantization algorithms** -- GPTQ, DBF (Double Binary Factorization), RTN (Round-To-Nearest), JointQ, QuIP, and more
- **Quantization Error Propagation (QEP)** -- A novel method that adjusts weights before quantization to compensate for error propagation across layers
- **Simple, unified API** -- Configure model, quantizer, and runner in a few lines of code
- **Save/Load pipeline** -- Save quantized models in a format compatible with Hugging Face Transformers and vLLM
- **Evaluation tools** -- Built-in perplexity and zero-shot accuracy benchmarks

## Quick Example

```python
from onecomp import ModelConfig, Runner, GPTQ, setup_logger

setup_logger()

model_config = ModelConfig(
    model_id="meta-llama/Llama-2-7b-hf",
    device="cuda:0",
)
quantizer = GPTQ(wbits=4, groupsize=128)

runner = Runner(model_config=model_config, quantizer=quantizer, qep=True)
runner.run()

original_ppl, quantized_ppl = runner.calculate_perplexity()
print(f"Original: {original_ppl:.2f}, Quantized: {quantized_ppl:.2f}")
```

## Getting Started

<div class="grid cards" markdown>

-   **Installation**

    Set up OneComp with pip or uv.

    [:octicons-arrow-right-24: Installation guide](getting-started/installation.md)

-   **Quick Start**

    Quantize your first LLM in minutes.

    [:octicons-arrow-right-24: Quick start guide](getting-started/quickstart.md)

-   **User Guide**

    Learn the full workflow: configure, quantize, evaluate, and save.

    [:octicons-arrow-right-24: Basic usage](user-guide/basic-usage.md)

-   **Algorithms**

    Understand the quantization algorithms and QEP.

    [:octicons-arrow-right-24: Algorithm overview](algorithms/overview.md)

</div>

## Citation

If you use OneComp in your research, please cite our paper:

```bibtex
@inproceedings{
arai2025quantization,
title={Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization},
author={Yamato Arai and Yuma Ichikawa},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=a3l3K9khbL}
}
```

## License

Fujitsu One Compression is released under the terms of the [LICENSE](https://github.com/FujitsuResearch/OneCompression/blob/main/LICENSE) file included in the repository.

Copyright 2025-2026 Fujitsu Ltd.
