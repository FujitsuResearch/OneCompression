# Configuration

This page describes all configurable components in Fujitsu One Compression (OneComp).

## ModelConfig

`ModelConfig` wraps model loading and tokenizer initialization.

```python
from onecomp import ModelConfig

model_config = ModelConfig(
    model_id="meta-llama/Llama-2-7b-hf",
    dtype="float16",
    device="cuda:0",
)
```

| Parameter  | Type  | Description                             | Default      |
|------------|-------|-----------------------------------------|--------------|
| `model_id` | `str` | Hugging Face Hub model ID              | `None`       |
| `path`     | `str` | Local path to model directory           | `None`       |
| `dtype`    | `str` | Model precision (`"float16"`, `"float32"`)| `"float16"` |
| `device`   | `str` | Device placement (`"cpu"`, `"cuda"`, `"auto"`)| `"auto"` |

!!! note
    Provide exactly one of `model_id` or `path`. A `ValueError` is raised if neither is specified.

## Runner

`Runner` is the main entry point for quantization. It manages the full pipeline: loading the model, preparing calibration data, executing quantization, and providing evaluation utilities.

```python
from onecomp import Runner

runner = Runner(
    model_config=model_config,
    quantizer=quantizer,
    max_length=512,
    num_calibration_samples=128,
    qep=False,
)
```

### Core Parameters

| Parameter                   | Type              | Description                                      | Default          |
|-----------------------------|-------------------|--------------------------------------------------|------------------|
| `model_config`              | `ModelConfig`     | Model and tokenizer configuration                | —                |
| `quantizer`                 | `Quantizer`       | Quantization method                              | `None`           |
| `quantizers`                | `list[Quantizer]` | Multiple quantizers (for benchmarking)           | `None`           |
| `qep`                       | `bool`            | Enable QEP                                       | `False`          |
| `qep_config`                | `QEPConfig`       | QEP configuration                                | `None`           |

### Calibration Parameters

| Parameter                   | Type   | Description                                      | Default          |
|-----------------------------|--------|--------------------------------------------------|------------------|
| `calibration_dataset`       | `Dataset` | Custom calibration dataset                    | `None`           |
| `max_length`                | `int`  | Maximum input sequence length                    | `512`            |
| `num_calibration_samples`   | `int`  | Number of calibration samples                    | `128`            |
| `calibration_strategy`      | `str`  | Strategy for preparing calibration inputs        | `"drop_rand"`    |
| `calibration_seed`          | `int`  | Random seed for calibration                      | `0`              |
| `calibration_batch_size`    | `int`  | Batch size for chunked calibration               | `None`           |
| `num_layers_per_group`      | `int`  | Layers processed simultaneously in chunked mode  | `7`              |

### Advanced Parameters

| Parameter     | Type        | Description                                      | Default  |
|---------------|-------------|--------------------------------------------------|----------|
| `multi_gpu`   | `bool`      | Enable multi-GPU layer-wise parallel quantization| `False`  |
| `gpu_ids`     | `list[int]` | Specific GPU IDs to use                          | `None`   |

### Calibration Strategies

| Strategy             | Description                                                          |
|----------------------|----------------------------------------------------------------------|
| `"drop_rand"`        | Tokenize each document independently; take a random window of `max_length` tokens. |
| `"drop_head"`        | Same, but always take the first `max_length` tokens.                 |
| `"concat_chunk"`     | Concatenate all texts, tokenize, and split into fixed-length chunks. |
| `"concat_chunk_align"` | Same as `concat_chunk`, but adjusts samples so chunk count equals `num_calibration_samples`. |

### Valid Parameter Combinations

| `quantizers` | `qep`  | `multi_gpu` | `calibration_batch_size` |
|:------------:|:------:|:-----------:|:------------------------:|
| Specified    | False  | False       | Specified                |
| None         | True   | False       | None                     |
| None         | False  | True        | None                     |
| None         | False  | False       | Specified                |
| None         | False  | False       | None                     |

## QEPConfig

`QEPConfig` controls Quantization Error Propagation behavior.

```python
from onecomp import QEPConfig

qep_config = QEPConfig(
    general=False,
    percdamp=0.01,
    perccorr=0.5,
    device="cuda:0",
    exclude_layer_keywords=["mlp.down_proj"],
)
```

| Parameter                 | Type        | Description                                      | Default              |
|---------------------------|-------------|--------------------------------------------------|----------------------|
| `general`                 | `bool`      | Use generic (architecture-independent) QEP       | `False`              |
| `percdamp`                | `float`     | Damping percentage for Hessian regularization     | `0.01`               |
| `perccorr`                | `float`     | Correction percentage for error propagation       | `0.5`                |
| `device`                  | `str`       | GPU device for QEP computations                   | `"cuda:0"`           |
| `exclude_layer_keywords`  | `list[str]` | Layer keywords excluded from error propagation    | `["mlp.down_proj"]`  |

!!! tip
    The default `general=False` uses the architecture-aware implementation, which is faster because it exploits shared activations (e.g., QKV layers sharing the same input in Llama-like models).

## Quantizer Common Parameters

All quantizers inherit from the `Quantizer` base class and share these parameters:

| Parameter                | Type          | Description                                       | Default        |
|--------------------------|---------------|---------------------------------------------------|----------------|
| `name`                   | `str`         | Quantizer name (defaults to class name)           | `None`         |
| `num_layers`             | `int`         | Maximum layers to quantize                        | `None`         |
| `calc_quant_error`       | `bool`        | Calculate quantization error per layer            | `False`        |
| `include_layer_names`    | `list[str]`   | Layers to quantize (exact match)                  | `None`         |
| `exclude_layer_names`    | `list[str]`   | Layers to skip (exact match)                      | `["lm_head"]`  |
| `include_layer_keywords` | `list[str]`   | Quantize layers containing any keyword            | `None`         |
| `exclude_layer_keywords` | `list[str]`   | Skip layers containing any keyword                | `None`         |

### Layer Selection Priority

1. Filter by layer type (`target_layer_types`)
2. If `include_layer_names` is set, only include exact matches
3. If `include_layer_keywords` is set, only include layers containing any keyword
4. Exclude `exclude_layer_names` (exact match)
5. Exclude `exclude_layer_keywords` (keyword match)
6. Limit by `num_layers`
