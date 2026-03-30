# Fujitsu One Compression

**Open-source Python library for post-training quantization of Large Language Models**

---

Fujitsu One Compression (OneComp) is an open-source Python library for post-training quantization of Large Language Models (LLMs).
It implements state-of-the-art quantization algorithms including GPTQ, DBF, RTN, and the novel
**Quantization Error Propagation (QEP)** method proposed in our NeurIPS 2025 paper.

## Key Features

- **Quantization Error Propagation (QEP)** -- A post-training quantization method that corrects quantization errors by propagating them to subsequent layers, improving the accuracy of quantized LLMs. See [Arai & Ichikawa, NeurIPS 2025](https://openreview.net/forum?id=a3l3K9khbL) for details.
- **vLLM Plugin Integration** -- Serve OneComp-quantized models with [vLLM](https://docs.vllm.ai/) via built-in plugins for DBF and Mixed-GPTQ quantization methods.
- **AutoBit** -- Mixed-precision quantization with ILP-based bitwidth assignment. Automatically estimates the target bitwidth from available VRAM and assigns per-layer bitwidths to minimize quantization error under the memory budget.
- **JointQ** -- Joint quantization method that optimizes weight assignments and scale parameters simultaneously for improved quantization accuracy. Supports group-wise quantization (e.g., 4-bit, groupsize=128).
- **LoRA SFT Post-Process** -- Fine-tune quantized models with LoRA adapters for accuracy recovery or domain-specific knowledge injection. Supports SFT loss, teacher distillation, and intermediate block alignment.
- **Rotation Preprocessing** -- SpinQuant/OstQuant-based rotation preprocessing that reduces quantization error by learning optimal rotation matrices before quantization. Rotation/scaling matrices are absorbed into model weights, with online Hadamard hooks automatically registered at load time. Supports Llama and Qwen3 architectures.

## Quick Example

Quantize any Hugging Face model in a single line -- with QEP, GPTQ 4-bit quantization,
evaluation (perplexity + accuracy), and model saving all handled automatically:

=== "Python"

    ```python
    from onecomp import Runner

    Runner.auto_run(model_id="meta-llama/Llama-2-7b-hf")
    ```

=== "CLI"

    ```bash
    onecomp meta-llama/Llama-2-7b-hf
    ```

For full control over each step, see the [step-by-step workflow](user-guide/basic-usage.md#detailed-workflow).

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
