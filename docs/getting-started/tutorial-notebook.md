# Tutorial Notebook

OneComp ships an interactive Jupyter notebook under [`notebook/`](https://github.com/FujitsuResearch/OneCompression/tree/main/notebook).
It walks through the same workflow as the CLI and Python API, with visualizations and a chat demo at the end.

If you prefer learning by running cells instead of reading scripts, start here.

## Notebooks

| Notebook | What you will do |
|---|---|
| [`01_tutorial.ipynb`](https://github.com/FujitsuResearch/OneCompression/blob/main/notebook/01_tutorial.ipynb) | Visualize 4-bit RTN on a single layer, run `Runner.auto_run`, compare FP16 vs quantized output with vLLM |

### Sections in `01_tutorial.ipynb`

| Section | Topic |
|---|---|
| 0. Setup | Install OneComp and verify the version |
| 1. Select the model | Load TinyLlama and generate an FP16 baseline response |
| 2. 4-bit RTN | Inspect one weight matrix and plot the quantization error |
| 3. One-line quantization | Quantize with `Runner.auto_run` and compare model size / perplexity |
| 4. Run with vLLM | Load the saved model in-process with `vllm.LLM` |
| 5. Try a chat | Multi-turn chat with the quantized model |

The notebook uses **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`** because it is small, ungated, and fits a Colab T4 GPU.

## Run locally

The `notebook/` directory is a standalone uv project that depends on the repository root.

```bash
cd notebook
uv sync
uv run jupyter lab
```

Then open `01_tutorial.ipynb`.

!!! tip
    You can also open the notebook from any Jupyter environment if OneComp is already installed.
    The setup cell prints the installed version so you can confirm dependencies before continuing.

## Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FujitsuResearch/OneCompression/blob/main/notebook/01_tutorial.ipynb)

### 1. Choose a runtime

In Colab, open **Runtime → Change runtime type** and select:

- **Python 3.12** (OneComp requires Python 3.12+)
- **GPU** (T4 is sufficient for this tutorial)

### 2. Install OneComp

In section **0. Setup**, uncomment and run the install lines in the first code cell:

```python
!pip install -q "onecomp[cu130,vllm]" matplotlib
```

Colab images ship with a CUDA-enabled PyTorch build, so you normally do not need to install PyTorch separately.

### 3. Run the notebook

Execute the cells from top to bottom. The notebook frees GPU memory between major sections
(FP16 model → quantization → vLLM) so each step fits on a single T4.

### Colab notes for the vLLM section

- Prefer **`vllm.LLM` in-process inference** (as shown in the notebook) over `vllm serve`.
  Running an HTTP server on Colab is fragile because of port and process management.
- The OneComp vLLM plugin registers automatically when you install `onecomp[vllm]`; no extra vLLM configuration is required.
- For a browser chat UI with `vllm serve` and Open WebUI, run on your own machine instead.
  See the [vLLM Inference guide](../user-guide/vllm-inference.md).

## Next steps

- [Quick Start](quickstart.md) — one-line quantization from Python or the CLI
- [Basic Usage](../user-guide/basic-usage.md) — full step-by-step workflow
- [Examples](../user-guide/examples.md) — copy-paste patterns for each quantizer
