# Basic Usage

This guide covers the core workflow of Fujitsu One Compression (OneComp): configure a model, select a quantizer, run quantization, and evaluate results.

## Quick Path: `Runner.auto_run()`

For the common case of GPTQ + QEP quantization, `auto_run` handles everything in one call:

```python
from onecomp import Runner

runner = Runner.auto_run(model_id="meta-llama/Llama-2-7b-hf")
```

This automatically:

1. Loads the model and tokenizer
2. Quantizes with GPTQ (4-bit, groupsize=128) + QEP
3. Evaluates perplexity and zero-shot accuracy
4. Saves the quantized model to disk

The returned `runner` instance gives access to quantization results for further analysis.
See the [Quick Start](../getting-started/quickstart.md) for `auto_run` parameters, or use
the [CLI](cli.md) for command-line usage.

---

## Detailed Workflow

For full control over each component, use the manual configuration approach.

```
ModelConfig ──┐
              ├──► Runner.run() ──► Evaluate / Save
Quantizer ────┘
```

Every quantization session follows the same pattern:

1. Create a `ModelConfig` to specify which model to quantize
2. Create a `Quantizer` (e.g., `GPTQ`, `RTN`, `DBF`) with desired parameters
3. Pass both to a `Runner` and call `runner.run()`
4. Evaluate or save the result

## Step 1: Configure the Model

```python
from onecomp import ModelConfig

model_config = ModelConfig(
    model_id="meta-llama/Llama-2-7b-hf",
    device="cuda:0",
)
```

| Parameter  | Description                         | Default      |
|------------|-------------------------------------|--------------|
| `model_id` | Hugging Face Hub model ID          | —            |
| `path`     | Local path to a saved model         | —            |
| `dtype`    | Data type (`"float16"`, `"float32"`)| `"float16"`  |
| `device`   | Device (`"cpu"`, `"cuda"`, `"auto"`)| `"auto"`     |

You must provide either `model_id` or `path`.

## Step 2: Choose a Quantizer

```python
from onecomp import GPTQ

gptq = GPTQ(wbits=4, groupsize=128)
```

Available quantizers and their typical parameters:

| Quantizer | Key Parameters               | Calibration Required |
|-----------|------------------------------|----------------------|
| `GPTQ`    | `wbits`, `groupsize`, `sym`  | Yes                  |
| `RTN`     | `wbits`, `groupsize`, `sym`  | No                   |
| `DBF`     | `target_bits`, `iters`       | Yes                  |
| `JointQ`  | `bits`, `group_size`         | Yes                  |

All quantizers share common parameters:

| Parameter              | Description                                      | Default        |
|------------------------|--------------------------------------------------|----------------|
| `num_layers`           | Max layers to quantize (None = all)              | `None`         |
| `calc_quant_error`     | Calculate quantization error per layer           | `False`        |
| `exclude_layer_names`  | Layer names to skip (exact match)                | `["lm_head"]`  |
| `include_layer_keywords` | Only quantize layers matching keywords         | `None`         |

## Step 3: Run Quantization

```python
from onecomp import Runner, setup_logger

setup_logger()  # Optional: enable logging output

runner = Runner(model_config=model_config, quantizer=gptq)
runner.run()
```

## Step 4: Evaluate

### Perplexity

`calculate_perplexity()` returns a 3-tuple `(original, dequantized, quantized)`.
By default, only the quantized model is evaluated:

```python
_, _, quantized_ppl = runner.calculate_perplexity()
print(f"Quantized: {quantized_ppl:.2f}")

# To also evaluate the original model:
original_ppl, _, quantized_ppl = runner.calculate_perplexity(original_model=True)
print(f"Original:  {original_ppl:.2f}")
print(f"Quantized: {quantized_ppl:.2f}")
```

!!! note
    - Evaluating the original or dequantized model requires loading the full model on GPU.
    - Quantized-model evaluation is currently supported only for **GPTQ** and **DBF** quantizers. Support for other methods is planned.

### Zero-shot Accuracy

```python
_, _, quantized_acc = runner.calculate_accuracy()
```

### Quantization Statistics

```python
runner.print_quantization_results()
runner.save_quantization_statistics("stats.json")
```

## Step 5: Save the Model

```python
# Save dequantized weights (FP16, compatible with any HF pipeline)
runner.save_dequantized_model("./output/dequantized")

# Save quantized model (packed integer weights, compatible with vLLM)
runner.save_quantized_model("./output/quantized")
```

## Enabling QEP

QEP adjusts weights before quantization to compensate for error propagation across layers.
Simply set `qep=True` on the Runner:

```python
runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    qep=True,
)
runner.run()
```

See [QEP Algorithm](../algorithms/qep.md) for the theory behind QEP.
