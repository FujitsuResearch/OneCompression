# Basic Usage

This guide covers the core workflow of Fujitsu One Compression (OneComp): configure a model, select a quantizer, run quantization, and evaluate results.

## Quick Path: `Runner.auto_run()`

`auto_run` handles everything in one call -- VRAM-based bitwidth estimation,
AutoBit mixed-precision quantization with QEP, evaluation, and saving:

```python
from onecomp import Runner

runner = Runner.auto_run(model_id="meta-llama/Llama-2-7b-hf")
```

This automatically:

1. Loads the model and tokenizer
2. Estimates the target bitwidth from available VRAM
3. Quantizes with AutoBit (ILP-based mixed-precision) + QEP
4. Evaluates perplexity and zero-shot accuracy
5. Saves the quantized model to disk

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

| Quantizer          | Key Parameters                          | Calibration Required |
|--------------------|------------------------------------------|----------------------|
| `AutoBitQuantizer` | `quantizers`, `target_bit`, `assignment_strategy`, `enable_fused_groups` | Yes                  |
| `GPTQ`             | `wbits`, `groupsize`, `sym`              | Yes                  |
| `RTN`              | `wbits`, `groupsize`, `sym`              | No                   |
| `DBF`              | `target_bits`, `iters`                   | Yes                  |
| `JointQ`           | `bits`, `group_size`                     | Yes                  |

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
    - Quantized-model evaluation (`quantized_model=True`) is supported only for quantizers
      that implement `create_quantized_model()` (**GPTQ**, **DBF**, **AutoBitQuantizer**).
      For other quantizers, evaluation automatically falls back to the dequantized (FP16) model.

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

!!! note "Quantizer feature support"
    `save_quantized_model()`, `create_quantized_model()`, and quantized-model PPL/ACC evaluation
    require the quantizer to implement `get_quant_config()` and `create_inference_layer()`.
    Currently only **GPTQ**, **DBF**, and **AutoBitQuantizer** support these features.

    | Quantizer          | Save | Quantized PPL/ACC | Fallback                  |
    |--------------------|:----:|:-----------------:|---------------------------|
    | `GPTQ`             | Yes  | Yes               | —                         |
    | `DBF`              | Yes  | Yes               | —                         |
    | `AutoBitQuantizer` | Yes  | Yes               | —                         |
    | `RTN`              | —    | —                 | Dequantized (FP16) model  |
    | `JointQ`           | —    | —                 | Dequantized (FP16) model  |
    | `QUIP`             | —    | —                 | Dequantized (FP16) model  |
    | `CQ`               | —    | —                 | Dequantized (FP16) model  |
    | `ARB`              | —    | —                 | Dequantized (FP16) model  |
    | `QBB`              | —    | —                 | Dequantized (FP16) model  |
    | `Onebit`           | —    | —                 | Dequantized (FP16) model  |

    For unsupported quantizers:

    - **PPL/ACC evaluation**: automatically falls back to the dequantized (FP16) model. No error is raised.
    - **Saving**: use `save_dequantized_model()` (FP16) or `save_quantization_results()` instead.

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

## Enabling LPCD

LPCD refines quantization at the submodule level after moving beyond a purely
layer-wise objective. In OneComp, it is enabled through `Runner` together with
an `LPCDConfig`:

```python
from onecomp import LPCDConfig

lpcd_config = LPCDConfig(
    enable_residual=True,  # default and fastest mode
    perccorr=0.5,
    percdamp=0.01,
    device="cuda:0",
)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    qep=True,
    lpcd=True,
    lpcd_config=lpcd_config,
)
runner.run()
```

!!! tip
    `LPCDConfig()` defaults to residual-only refinement (`enable_residual=True`),
    which is a good starting point. Enable `enable_qk`, `enable_vo`, and
    `enable_ud` for stronger but slower refinement.

See [LPCD](../algorithms/lpcd.md) for the algorithm details.
