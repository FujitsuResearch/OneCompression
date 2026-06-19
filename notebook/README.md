# notebook

Interactive tutorial notebooks for OneComp.

For Google Colab setup and a walkthrough of each section, see the
[Tutorial Notebook guide](https://FujitsuResearch.github.io/OneCompression/getting-started/tutorial-notebook/)
in the documentation.

## Notebooks

| Notebook | Description |
|---|---|
| [01_tutorial.ipynb](01_tutorial.ipynb) | Hands-on tour: 4-bit RTN visualization, `Runner.auto_run`, and vLLM chat inference |

## Local setup

This directory has its own [uv](https://docs.astral.sh/uv/) project that installs OneComp
from the repository root in editable mode.

```bash
cd notebook
uv sync
uv run jupyter lab
```

Open `01_tutorial.ipynb` and run the cells from top to bottom.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FujitsuResearch/OneCompression/blob/main/notebook/01_tutorial.ipynb)

1. Open the notebook in Colab with the badge above (or upload `01_tutorial.ipynb`).
2. Select a **Python 3.12** runtime with a GPU (T4 is enough).
3. In section **0. Setup**, uncomment and run the `pip install` lines in the first code cell.
4. Run the remaining cells in order.

For the vLLM section (section 4), install the vLLM extra:

```python
!pip install -q "onecomp[cu130,vllm]" matplotlib
```

The tutorial uses **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`** so it runs comfortably on a Colab T4
without a Hugging Face login.

## Requirements

- Python 3.12+ (OneComp requirement)
- `onecomp >= 1.2.0`

## Related

- [Tutorial Notebook guide (docs)](../docs/getting-started/tutorial-notebook.md)
- [Quick Start](../docs/getting-started/quickstart.md)
- [Example scripts](../example/)
