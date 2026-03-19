# Quick Start

This guide walks you through quantizing your first LLM with Fujitsu One Compression (OneComp).

## Basic Quantization with GPTQ

The simplest workflow involves three components:

1. **ModelConfig** -- specifies which model to quantize
2. **Quantizer** (e.g., GPTQ) -- defines the quantization method and parameters
3. **Runner** -- orchestrates the quantization pipeline

```python
from onecomp import ModelConfig, Runner, GPTQ, setup_logger

# Enable logging to see progress
setup_logger()

# 1. Configure the model
model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    device="cuda:0",
)

# 2. Choose a quantizer
gptq = GPTQ(wbits=4, groupsize=128)

# 3. Create the runner and quantize
runner = Runner(model_config=model_config, quantizer=gptq)
runner.run()
```

## Evaluating the Quantized Model

After quantization, measure the impact on model quality:

```python
# Perplexity (lower is better)
original_ppl, quantized_ppl = runner.calculate_perplexity()
print(f"Original: {original_ppl:.2f}")
print(f"Quantized: {quantized_ppl:.2f}")

# Zero-shot accuracy
original_acc, quantized_acc = runner.calculate_accuracy()
```

## Using QEP (Quantization Error Propagation)

QEP compensates for error propagation across layers, improving quantization quality
-- especially at lower bit-widths:

```python
runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    qep=True,  # Enable QEP
)
runner.run()
```

For fine-grained control over QEP, use `QEPConfig`:

```python
from onecomp import QEPConfig

qep_config = QEPConfig(
    percdamp=0.01,     # Hessian damping
    perccorr=0.5,      # Correction strength
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

- [Configuration](../user-guide/configuration.md) -- detailed explanation of `ModelConfig`, `QEPConfig`, and `Runner` parameters
- [Examples](../user-guide/examples.md) -- more usage patterns including multi-GPU and chunked calibration
- [Algorithms](../algorithms/overview.md) -- learn about the quantization algorithms available in OneComp
