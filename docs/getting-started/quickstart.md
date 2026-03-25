# Quick Start

This guide walks you through quantizing your first LLM with Fujitsu One Compression (OneComp).

## The Fastest Way: `auto_run`

`Runner.auto_run` handles everything -- model loading, GPTQ quantization with QEP,
evaluation (perplexity + zero-shot accuracy), and saving the quantized model:

=== "Python"

    ```python
    from onecomp import Runner

    Runner.auto_run(model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    ```

=== "CLI"

    ```bash
    onecomp TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
    ```

That's it. The quantized model is saved to
`TinyLlama-1.1B-intermediate-step-1431k-3T-gptq-4bit/` by default.

### `auto_run` Parameters

| Parameter   | Default    | Description                                              |
|-------------|------------|----------------------------------------------------------|
| `model_id`  | (required) | Hugging Face model ID or local path                      |
| `wbits`     | `4`        | Quantization bit width                                   |
| `groupsize` | `128`      | GPTQ group size (`-1` to disable)                        |
| `device`    | `"cuda:0"` | Device for computation                                   |
| `qep`       | `True`     | Enable QEP (Quantization Error Propagation)              |
| `evaluate`  | `True`     | Calculate perplexity and zero-shot accuracy              |
| `eval_original_model` | `False` | Also evaluate the original (unquantized) model        |
| `save_dir`  | `"auto"`   | Save directory (`"auto"` = derived from model name, `None` to skip) |

### Examples

```python
from onecomp import Runner

# 3-bit quantization, no QEP, skip saving
Runner.auto_run(
    model_id="meta-llama/Llama-2-7b-hf",
    wbits=3,
    qep=False,
    save_dir=None,
)

# Custom save directory, skip evaluation
Runner.auto_run(
    model_id="meta-llama/Llama-2-7b-hf",
    save_dir="./my_quantized_model",
    evaluate=False,
)
```

---

## Step-by-step Workflow

For full control over each component, use the manual configuration approach.

The workflow involves three components:

1. **ModelConfig** -- specifies which model to quantize
2. **Quantizer** (e.g., GPTQ) -- defines the quantization method and parameters
3. **Runner** -- orchestrates the quantization pipeline

```python
from onecomp import ModelConfig, Runner, GPTQ, setup_logger

setup_logger()

model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    device="cuda:0",
)
gptq = GPTQ(wbits=4, groupsize=128)

runner = Runner(model_config=model_config, quantizer=gptq)
runner.run()
```

## Evaluating the Quantized Model

After quantization, measure the impact on model quality:

```python
# Perplexity (lower is better)
# Returns a 3-tuple: (original, dequantized, quantized)
# By default, only quantized is computed (others are None)
_, _, quantized_ppl = runner.calculate_perplexity()
print(f"Quantized: {quantized_ppl:.2f}")

# To also evaluate the original model, pass original_model=True
original_ppl, _, quantized_ppl = runner.calculate_perplexity(original_model=True)
print(f"Original:  {original_ppl:.2f}")
print(f"Quantized: {quantized_ppl:.2f}")

# Zero-shot accuracy (same 3-tuple pattern)
_, _, quantized_acc = runner.calculate_accuracy()
```

!!! note
    - Evaluating the original or dequantized model requires loading the full model on GPU.
    - Quantized-model evaluation is currently supported only for **GPTQ** and **DBF** quantizers. Support for other methods is planned.

## Using QEP (Quantization Error Propagation)

QEP compensates for error propagation across layers, improving quantization quality
-- especially at lower bit-widths:

```python
runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    qep=True,
)
runner.run()
```

For fine-grained control over QEP, use `QEPConfig`:

```python
from onecomp import QEPConfig

qep_config = QEPConfig(
    percdamp=0.01,
    perccorr=0.5,
)
runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    qep=True,
    qep_config=qep_config,
)
runner.run()
```

## Saving and Loading

### Save a dequantized model (FP16 weights)

```python
runner.save_dequantized_model("./output/dequantized_model")
```

### Save a quantized model (packed integer weights)

```python
runner.save_quantized_model("./output/quantized_model")
```

### Load a saved quantized model

```python
from onecomp import load_quantized_model

model, tokenizer = load_quantized_model("./output/quantized_model")
```

## Next Steps

- [CLI Reference](../user-guide/cli.md) -- full CLI options and usage
- [Configuration](../user-guide/configuration.md) -- detailed explanation of `ModelConfig`, `QEPConfig`, and `Runner` parameters
- [Examples](../user-guide/examples.md) -- more usage patterns including multi-GPU and chunked calibration
- [Algorithms](../algorithms/overview.md) -- learn about the quantization algorithms available in OneComp
