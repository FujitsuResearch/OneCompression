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
| `rotation_mode` | Rotation init mode: `"random_hadamard"`, `"hadamard"`, `"random"`, `"identity"` | `"random_hadamard"` |
| `scaling_mode` | Scaling init mode: `"identity"`, `"random_ones"`, `"random"` | `"identity"` |
| `enable_training` | Train rotation matrices (vs. random init) | `True` |
| `wbits` | RTN proxy bit-width (must match quantizer) | `4` |
| `groupsize` | RTN proxy group size (must match quantizer) | `-1` |
| `sym` | RTN proxy symmetric quantization | `False` |
| `mse` | Enable MSE grid search for RTN proxy clipping | `False` |
| `norm` | Lp norm exponent for MSE search | `2.4` |
| `grid` | Number of candidate shrink levels for MSE search | `100` |
| `fp32_had` | Use FP32 for online Hadamard transform | `False` |
| `num_calibration_samples` | Number of calibration samples | `512` |
| `calibration_strategy` | Calibration strategy: `"concat_chunk"`, `"concat_chunk_align"`, `"drop_head"`, `"drop_rand"` | `"drop_rand"` |
| `seed` | Seed for rotation init and calibration data | `0` |

!!! note "Input validation"
    `prepare_rotated_model` validates all parameters on entry. Invalid values for
    `rotation_mode`, `scaling_mode`, `calibration_strategy`, or out-of-range numeric
    parameters (e.g. `wbits < 1`, `grid < 1`) raise `ValueError`.

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

After `runner.save_quantized_model()`, the saved `config.json` includes
`"rotated": true` and the `"fp32_had"` value used during preprocessing.
`load_quantized_model()` uses these flags to automatically register the
required Hadamard hooks on `down_proj` layers.

## Examples

!!! tip
    Complete working examples are available in the repository:

    - [`example/pre_process/example_llama_preprocess_rtn.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/pre_process/example_llama_preprocess_rtn.py) -- Rotation preprocessing + RTN quantization (TinyLlama)
    - [`example/pre_process/example_preprocess_save_load.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/pre_process/example_preprocess_save_load.py) -- Rotation preprocessing + GPTQ with save/load and perplexity comparison

## Limitations

- **vLLM inference is not supported.** vLLM kernels do not apply the online Hadamard
  transform required by rotation-preprocessed models.
- Only Llama and Qwen3 architectures are currently supported.

## API Reference

See [Pre-Process API](../api/pre_process.md) for full parameter documentation.
