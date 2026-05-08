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
from onecomp import CalibrationConfig, Runner

calib_config = CalibrationConfig(
    max_length=2048,
    num_calibration_samples=512,
)

runner = Runner(
    model_config=model_config,
    quantizer=quantizer,
    calibration_config=calib_config,
    qep=False,
    lpcd=False,
)
```

### Core Parameters

| Parameter                   | Type                | Description                                      | Default          |
|-----------------------------|---------------------|--------------------------------------------------|------------------|
| `model_config`              | `ModelConfig`       | Model and tokenizer configuration                | —                |
| `quantizer`                 | `Quantizer`         | Quantization method                              | `None`           |
| `quantizers`                | `list[Quantizer]`   | Multiple quantizers (for benchmarking)           | `None`           |
| `calibration_config`        | `CalibrationConfig` | Calibration data configuration                   | `None` (auto)    |
| `qep`                       | `bool`              | Enable QEP                                       | `False`          |
| `qep_config`                | `QEPConfig`         | QEP configuration                                | `None`           |
| `lpcd`                      | `bool`              | Enable LPCD                                      | `False`          |
| `lpcd_config`               | `LPCDConfig`        | LPCD configuration                               | `None`           |

### Advanced Parameters

| Parameter     | Type        | Description                                      | Default  |
|---------------|-------------|--------------------------------------------------|----------|
| `multi_gpu`   | `bool`      | Enable multi-GPU layer-wise parallel quantization| `False`  |
| `gpu_ids`     | `list[int]` | Specific GPU IDs to use                          | `None`   |

!!! note
    When `calibration_config` is `None`, a `CalibrationConfig()` with default values is created automatically.

## CalibrationConfig

`CalibrationConfig` groups all calibration-related parameters into a single dataclass.

```python
from onecomp import CalibrationConfig

calib_config = CalibrationConfig(
    calibration_dataset="wikitext2",
    max_length=2048,
    num_calibration_samples=256,
    strategy="concat_rand",
)
```

| Parameter                   | Type   | Description                                      | Default          |
|-----------------------------|--------|--------------------------------------------------|------------------|
| `calibration_dataset`       | `str`  | Dataset name (`"c4"`, `"wikitext2"`), local file path, or HuggingFace Hub ID | `"c4"`           |
| `max_length`                | `int`  | Maximum token length per calibration chunk        | `2048`           |
| `num_calibration_samples`   | `int`  | Target number of calibration samples              | `512`            |
| `strategy`                  | `str`  | Chunking strategy (see table below)               | `"drop_rand"`    |
| `seed`                      | `int`  | Random seed for stochastic strategies             | `0`              |
| `batch_size`                | `int`  | Batch size for chunked calibration forward passes | `None`           |
| `num_layers_per_group`      | `int`  | Layers processed simultaneously in chunked mode   | `7`              |
| `text_key`                  | `str`  | Column name when loading custom or Hub datasets   | `"text"`         |
| `use_quality_filter`        | `bool` | Apply C4 quality filtering                        | `False`          |
| `max_documents`             | `int`  | Cap on documents loaded from custom/Hub sources   | `10000`          |

### Calibration Strategies

| Strategy             | Description                                                          |
|----------------------|----------------------------------------------------------------------|
| `"drop_rand"`        | Tokenize each document independently; take a random window of `max_length` tokens. |
| `"drop_head"`        | Same, but always take the first `max_length` tokens.                 |
| `"concat_chunk"`     | Concatenate all texts, tokenize, and split into fixed-length chunks. |
| `"concat_chunk_align"` | Same as `concat_chunk`, but adjusts samples so chunk count equals `num_calibration_samples`. |
| `"concat_rand"`      | Concatenate all texts, tokenize, then randomly sample windows. Standard GPTQ/AWQ approach. |

### Supported Calibration Datasets

| Value               | Source                                                               |
|---------------------|----------------------------------------------------------------------|
| `"c4"`              | AllenAI C4 dataset (default)                                         |
| `"wikitext2"`       | WikiText-2 dataset (Salesforce)                                      |
| Local file path     | `.txt`, `.json`, `.jsonl`, `.csv`, `.tsv`, `.parquet`, `.arrow`, or HuggingFace Dataset directory |
| HuggingFace Hub ID  | Any public dataset (e.g. `"username/dataset"`)                       |

### Valid Parameter Combinations

| `quantizers` | `qep`  | `multi_gpu` | `calibration_config.batch_size` |
|:------------:|:------:|:-----------:|:-------------------------------:|
| Specified    | False  | False       | Specified                       |
| None         | True   | False       | None                            |
| None         | False  | True        | None                            |
| None         | False  | False       | Specified                       |
| None         | False  | False       | None                            |

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

## LPCDConfig

`LPCDConfig` controls Layer-Projected Coordinate Descent (LPCD) refinement.

```python
from onecomp import LPCDConfig

lpcd_config = LPCDConfig(
    enable_residual=True,
    percdamp=0.01,
    perccorr=0.5,
    use_closed_form=True,
    device="cuda:0",
)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `enable_qk` | `bool` | Jointly refine `q_proj` / `k_proj` | `False` |
| `enable_vo` | `bool` | Jointly refine `v_proj` / `o_proj` | `False` |
| `enable_ud` | `bool` | Jointly refine `up_proj` / `down_proj` | `False` |
| `enable_residual` | `bool` | Refine residual-path modules (`o_proj`, `down_proj`) | `True` |
| `alt_steps` | `int` | Alternating coordinate-descent steps | `1` |
| `percdamp` | `float` | Damping percentage for Hessian regularization | `0.01` |
| `perccorr` | `float` | Correction percentage for relaxed weights | `0.5` |
| `use_closed_form` | `bool` | Use closed-form solvers where available | `True` |
| `gd_steps` | `int` | Gradient-descent steps per sub-problem | `20` |
| `gd_batch_size` | `int` | Effective batch size for gradient accumulation | `16` |
| `gd_base_lr` | `float` | Base learning rate for gradient solver | `1e-4` |
| `device` | `str` | Device for LPCD computation | `"cuda:0"` |

!!! tip
    `LPCDConfig()` defaults to residual-only refinement, which is the fastest
    way to get started. Enable `enable_qk`, `enable_vo`, and `enable_ud` for
    broader submodule refinement.

!!! note
    When combining LPCD with QEP, use the architecture-aware QEP path
    (`QEPConfig(general=False)`). The current LPCD implementation does not
    support `QEPConfig(general=True)`.

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
