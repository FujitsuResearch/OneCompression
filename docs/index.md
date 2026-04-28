# Fujitsu One Compression

**Open-source Python library for post-training quantization of Large Language Models**

<p align="center">
  <img src="assets/onecomp.gif" alt="OneComp" />
</p>

---

Fujitsu One Compression (OneComp) is an open-source Python library for post-training quantization of Large Language Models (LLMs).
It implements state-of-the-art quantization algorithms including GPTQ, DBF, RTN, and the
research methods **Quantization Error Propagation (QEP)** and
**Layer-Projected Coordinate Descent (LPCD)**.

## Just one line.

```bash
onecomp <generative AI>
```

**That's all you need.** OneComp detects your GPU VRAM, picks the best bit-width per layer, quantizes with error propagation, evaluates, and saves — fully automatic.

=== "CLI"

    ```bash
    onecomp meta-llama/Llama-2-7b-hf
    ```

=== "Python"

    ```python
    from onecomp import Runner

    Runner.auto_run(model_id="meta-llama/Llama-2-7b-hf")
    ```

For full control over each step, see the [step-by-step workflow](user-guide/basic-usage.md#detailed-workflow).

## Key Features

- **Quantization Error Propagation (QEP)** -- A post-training quantization method that corrects quantization errors by propagating them to subsequent layers, improving the accuracy of quantized LLMs. See [Arai & Ichikawa, NeurIPS 2025](https://openreview.net/forum?id=a3l3K9khbL) for details.
- **Layer-Projected Coordinate Descent (LPCD)** -- A unified Post Training Quantization (PTQ) framework that extends layer-wise quantization to arbitrary submodules by optimising relaxed objectives and projecting the solutions with layer-wise quantizers. See [Ichikawa et al., 2025](https://arxiv.org/abs/2512.01546) for details.
- **vLLM Plugin Integration** -- Serve OneComp-quantized models with [vLLM](https://docs.vllm.ai/) via built-in plugins for DBF and Mixed-GPTQ quantization methods. Pair with [Open WebUI](https://github.com/open-webui/open-webui) for a ChatGPT-like chat experience on your local machine. See the [setup guide](user-guide/vllm-inference.md#3-chat-with-open-webui-optional).
- **AutoBit** -- Mixed-precision quantization with ILP-based bitwidth assignment. Automatically estimates the target bitwidth from available VRAM and assigns per-layer bitwidths to minimize quantization error under the memory budget.
- **JointQ** -- Joint quantization method that optimizes weight assignments and scale parameters simultaneously for improved quantization accuracy. Supports group-wise quantization (e.g., 4-bit, groupsize=128).
- **Block-wise PTQ** -- Post-quantization block-wise distillation that minimises intermediate-representation MSE against an FP16 teacher model at Transformer-block granularity. Includes greedy per-block optimisation (Phase 1) and cross-block sliding-window optimisation (Phase 2 CBQ). Supports GPTQ, DBF, and OneBit quantizers.
- **LoRA SFT Post-Process** -- Fine-tune quantized models with LoRA adapters for accuracy recovery or domain-specific knowledge injection. Supports SFT loss, teacher distillation, and intermediate block alignment.
- **Rotation Preprocessing** -- SpinQuant/OstQuant-based rotation preprocessing that reduces quantization error by learning optimal rotation matrices before quantization. Rotation/scaling matrices are absorbed into model weights, with online Hadamard hooks automatically registered at load time. Supports Llama and Qwen3 architectures.

## Supported Models

OneComp has been verified with the following model architectures.
Other Hugging Face-compatible models may work but are currently untested.

| # | Architecture | Verified Models | Status |
|---|-------------|-----------------|--------|
| 1 | Llama | TinyLlama, Llama-2, Llama-3 | :white_check_mark: Verified |
| 2 | Qwen3 | Qwen3-0.6B ~ 32B | :white_check_mark: Verified |
| 3 | Gemma | Gemma 2, Gemma 3, Gemma 4 | :white_check_mark: Verified |

!!! note
    Support for additional architectures is planned. Contributions and test reports are welcome.

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

    Understand the quantization algorithms, QEP, and LPCD.

    [:octicons-arrow-right-24: Algorithm overview](algorithms/overview.md)

</div>

## Citation

If you use OneComp in your research, please cite our paper:

OneComp technical report (coming soon on ArXiv):

```bibtex
@misc{ichikawa2026onecomponelinerevolutiongenerative,
      title={OneComp: One-Line Revolution for Generative AI Model Compression}, 
      author={Yuma Ichikawa and Keiji Kimura and Akihiro Yoshida and Yudai Fujimoto and Hiroki Tokura and Yamato Arai and Yoshiyuki Ishii and Yusei Kawakami and Genki Shikada and Achille Jacquemond and Yoshihiko Fujisawa and Katsuki Fujisawa and Takumi Honda and Akira Sakai},
      year={2026},
      eprint={2603.28845},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.28845}, 
}
```

QEP (Quantization Error Propagation):

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

LPCD (Layer-Projected Coordinate Descent):

```bibtex
@article{ichikawa2025lpcd,
  title={LPCD: Unified Framework from Layer-Wise to Submodule Quantization},
  author={Yuma Ichikawa and Yudai Fujimoto and Akira Sakai},
  journal={arXiv preprint arXiv:2512.01546},
  year={2025},
  url={https://arxiv.org/abs/2512.01546}
}
```

## License

Fujitsu One Compression is released under the terms of the [LICENSE](https://github.com/FujitsuResearch/OneCompression/blob/main/LICENSE) file included in the repository.

Copyright 2025-2026 Fujitsu Ltd.
