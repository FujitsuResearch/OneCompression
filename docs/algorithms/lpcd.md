# LPCD (Layer-Projected Coordinate Descent)

LPCD is a unified framework that extends post-training quantization beyond
individual linear layers to larger Transformer submodules.

!!! abstract "Reference"
    Yuma Ichikawa, Yudai Fujimoto, and Akira Sakai,
    "LPCD: Unified Framework from Layer-Wise to Submodule Quantization," 2025.
    [arXiv:2512.01546](https://arxiv.org/abs/2512.01546)

## Motivation

Standard layer-wise PTQ methods such as GPTQ optimize one linear layer at a time.
QEP improves this by compensating for error propagation across layers, but the
optimization target is still fundamentally layer-wise.

LPCD lifts the optimization target from a single layer to a **submodule**. This
lets OneComp refine interactions inside attention and MLP blocks while keeping
compatibility with existing layer-wise quantizers.

## How LPCD Works

For each Transformer block, LPCD:

1. Builds a baseline quantized block using the chosen quantizer, optionally with QEP
2. Selects refineable submodules such as Q/K, V/O, up/down, or residual paths
3. Optimizes a relaxed objective on the selected submodule group
4. Projects the refined solution back through the underlying layer-wise quantizer

In OneComp, some residual-path refinements can use closed-form solvers, while
larger submodule groups are refined with an iterative gradient-based solver.

## Supported Targets

`LPCDConfig` enables the following submodule groups:

| Flag | Target modules | Typical purpose |
|------|----------------|-----------------|
| `enable_qk` | `q_proj`, `k_proj` | Refine attention score computation |
| `enable_vo` | `v_proj`, `o_proj` | Refine value/output projection path |
| `enable_ud` | `up_proj`, `down_proj` | Refine MLP transformation |
| `enable_residual` | `o_proj`, `down_proj` | Fast residual-only refinement |

By default, `LPCDConfig()` enables only `enable_residual=True`, which is the
fastest and most practical starting point.

## Usage

### Basic LPCD with GPTQ + QEP

```python
from onecomp import (
    CalibrationConfig,
    GPTQ,
    LPCDConfig,
    ModelConfig,
    Runner,
    setup_logger,
)

setup_logger()

model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    device="cuda:0",
)
gptq = GPTQ(wbits=3, groupsize=128)

lpcd_config = LPCDConfig(
    enable_residual=True,
    perccorr=0.5,
    percdamp=0.01,
    use_closed_form=True,
    device="cuda:0",
)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=128),
    qep=True,
    lpcd=True,
    lpcd_config=lpcd_config,
)
runner.run()
```

### Stronger LPCD Refinement

Enable more submodule groups when you want higher-quality refinement and can
accept longer runtime:

```python
lpcd_config = LPCDConfig(
    enable_qk=True,
    enable_vo=True,
    enable_ud=True,
    enable_residual=True,
    alt_steps=1,
    gd_steps=20,
    gd_base_lr=1e-4,
)
```

## Relationship to QEP

LPCD and QEP are complementary:

- **QEP** compensates for error propagation across sequential layers
- **LPCD** refines the objective inside a submodule after moving beyond a purely layer-wise view

You can use LPCD without QEP, but the common setup in OneComp is `GPTQ + QEP + LPCD`.

!!! note
    When combining LPCD with QEP, use the architecture-aware QEP path
    (`QEPConfig(general=False)`, which is also the default). The current LPCD
    implementation does not support `QEPConfig(general=True)`.

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `enable_qk` | `bool` | Jointly refine `q_proj` / `k_proj` | `False` |
| `enable_vo` | `bool` | Jointly refine `v_proj` / `o_proj` | `False` |
| `enable_ud` | `bool` | Jointly refine `up_proj` / `down_proj` | `False` |
| `enable_residual` | `bool` | Refine residual-path modules (`o_proj`, `down_proj`) | `True` |
| `alt_steps` | `int` | Number of alternating coordinate-descent steps | `1` |
| `perccorr` | `float` | Correction strength for relaxed weights | `0.5` |
| `percdamp` | `float` | Hessian damping ratio | `0.01` |
| `use_closed_form` | `bool` | Use closed-form solvers where available | `True` |
| `gd_steps` | `int` | Gradient-descent steps per sub-problem | `20` |
| `gd_batch_size` | `int` | Effective batch size for gradient accumulation | `16` |
| `gd_base_lr` | `float` | Base learning rate for gradient solver | `1e-4` |
| `device` | `str` | Device for LPCD optimization | `"cuda:0"` |

## Current Support

- Supported architectures: **Llama** and **Qwen3**
- LPCD runs through `Runner(..., lpcd=True, lpcd_config=...)`
- LPCD is a refinement framework, not a standalone quantizer
- The current examples and tests focus on **GPTQ-based** workflows

See also the [examples guide](../user-guide/examples.md) and
[API reference for `LPCDConfig`](../api/lpcd_config.md).
