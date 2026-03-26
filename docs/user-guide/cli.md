# CLI Reference

OneComp provides the `onecomp` command for quantizing models directly from the terminal.

## Installation

The `onecomp` command is installed automatically with the package:

```bash
pip install git+https://github.com/FujitsuResearch/OneCompression.git
```

Verify the installation:

```bash
onecomp --version
```

You can also use `python -m onecomp` as an alternative:

```bash
python -m onecomp --version
```

## Usage

```
onecomp [-h] [--wbits WBITS] [--total-vram-gb GB] [--groupsize GROUPSIZE]
        [--device DEVICE] [--no-qep] [--no-eval] [--eval-original]
        [--save-dir SAVE_DIR] [--version]
        model_id
```

### Positional Arguments

| Argument   | Description                              |
|------------|------------------------------------------|
| `model_id` | Hugging Face model ID or local path      |

### Options

| Option                    | Default      | Description                                              |
|---------------------------|--------------|----------------------------------------------------------|
| `--wbits WBITS`           | `None` (auto)| Target bitwidth. When omitted, estimated from VRAM       |
| `--total-vram-gb GB`      | `None` (auto)| VRAM budget in GB for bitwidth estimation. When omitted, detected from GPU |
| `--groupsize GROUPSIZE`   | `128`        | GPTQ group size (`-1` to disable grouping)               |
| `--device DEVICE`         | `cuda:0`     | Device to place the model on                             |
| `--no-qep`                |              | Disable QEP (enabled by default)                         |
| `--no-eval`               |              | Skip perplexity and accuracy evaluation                  |
| `--eval-original`         |              | Also evaluate the original (unquantized) model           |
| `--save-dir SAVE_DIR`     | `auto`       | Save directory (`auto` = derived from model name, `none` to skip) |
| `--version`               |              | Show version and exit                                    |

## Examples

### Basic usage (AutoBit with VRAM auto-estimation)

Quantize with defaults (AutoBit mixed-precision + QEP, evaluate, auto-save):

```bash
onecomp TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

### Specify VRAM budget

```bash
onecomp meta-llama/Llama-2-7b-hf --total-vram-gb 8
```

### Fixed bitwidth (skip VRAM estimation)

```bash
onecomp meta-llama/Llama-2-7b-hf --wbits 4
```

### 3-bit quantization

```bash
onecomp meta-llama/Llama-2-7b-hf --wbits 3
```

### Custom group size

```bash
onecomp meta-llama/Llama-2-7b-hf --wbits 4 --groupsize 64
```

### Without QEP

```bash
onecomp meta-llama/Llama-2-7b-hf --no-qep
```

### Skip evaluation (quantize and save only)

```bash
onecomp meta-llama/Llama-2-7b-hf --no-eval
```

### Custom save directory

```bash
onecomp meta-llama/Llama-2-7b-hf --save-dir ./my_quantized_model
```

### Skip saving

```bash
onecomp meta-llama/Llama-2-7b-hf --save-dir none
```

### Evaluate original model too

```bash
onecomp meta-llama/Llama-2-7b-hf --eval-original
```

### Use a specific GPU

```bash
onecomp meta-llama/Llama-2-7b-hf --device cuda:1
```

## Default Behavior

When run with no options, the `onecomp` command:

1. Loads the model and tokenizer from Hugging Face Hub
2. Estimates the target bitwidth from available VRAM
3. Quantizes with AutoBit (ILP-based mixed-precision) + QEP
4. Evaluates perplexity (wikitext-2) and zero-shot accuracy
5. Saves the quantized model to `<model_name>-autobit-<X>bit/`

## Equivalent Python API

The CLI is a thin wrapper around `Runner.auto_run`. Every CLI invocation maps directly
to the Python API:

```bash
onecomp meta-llama/Llama-2-7b-hf --wbits 3 --no-qep --save-dir ./output
```

is equivalent to:

```python
from onecomp import Runner

Runner.auto_run(
    model_id="meta-llama/Llama-2-7b-hf",
    wbits=3,
    qep=False,
    save_dir="./output",
)
```
