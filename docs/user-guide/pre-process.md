# Pre-Process (Rotation Preprocessing)

Rotation preprocessing applies SpinQuant/OstQuant-style rotation matrices to model weights
before quantization, reducing quantization error. This is particularly effective for
low-bit quantization (e.g., 3-bit).

## Overview

The rotation preprocessing pipeline:

1. **Trains** rotation/scaling matrices using calibration data with an RTN quantization proxy
2. **Absorbs** the learned matrices into model weights (fuses LayerNorms, rotates projections)
3. **Registers** online Hadamard hooks on `down_proj` layers for inference correctness
4. **Saves** the rotated model as a standard Hugging Face model directory

The saved model can then be quantized with any quantizer (GPTQ, RTN, etc.) using the
standard `Runner` pipeline.

## Quick Start

```python
from onecomp import ModelConfig, Runner, GPTQ, prepare_rotated_model, setup_logger

setup_logger()

# Step 1: Rotation preprocessing
model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf", device="cuda:0")

rotated_config = prepare_rotated_model(
    model_config=model_config,
    save_directory="./rotated_model",
    wbits=4,
    groupsize=128,
)

# Step 2: Quantize (wbits/groupsize/sym must match Step 1)
gptq = GPTQ(wbits=4, groupsize=128)
runner = Runner(model_config=rotated_config, quantizer=gptq)
runner.run()
```

## Supported Architectures

| Architecture | Status |
|-------------|--------|
| Llama       | Supported |
| Qwen3       | Supported |

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rotation` | Apply rotation matrices (R1, R2) | `True` |
| `scaling` | Apply scaling diagonals (S_*) | `False` |
| `enable_training` | Train rotation matrices (vs. random init) | `True` |
| `wbits` | RTN proxy bit-width (must match quantizer) | `4` |
| `groupsize` | RTN proxy group size (must match quantizer) | `-1` |
| `sym` | RTN proxy symmetric quantization | `False` |
| `fp32_had` | Use FP32 for online Hadamard transform | `False` |
| `seed` | Seed for rotation init and calibration data | `0` |

!!! warning "Parameter matching"
    The `wbits`, `groupsize`, and `sym` parameters **must match** the quantizer
    used in Step 2. Mismatched values will degrade quantization quality because
    the rotation matrices were optimized for different quantization settings.

## Save and Load

Rotation-preprocessed quantized models support the standard save/load API:

```python
# Save
runner.save_quantized_model("./quantized_model")

# Load (Hadamard hooks are auto-registered)
from onecomp import load_quantized_model
model, tokenizer = load_quantized_model("./quantized_model")
```

The saved `config.json` includes `"rotated": true` and `"fp32_had": false`,
which `load_quantized_model()` uses to automatically register the required
Hadamard hooks on `down_proj` layers.

## Limitations

- **vLLM inference is not supported.** vLLM kernels do not apply the online Hadamard
  transform required by rotation-preprocessed models.
- Only Llama and Qwen3 architectures are currently supported.

## API Reference

See [Pre-Process API](../api/pre_process.md) for full parameter documentation.
