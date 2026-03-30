# Pre-Process (Rotation Preprocessing)

Rotation preprocessing reduces quantization error by learning optimal rotation matrices
(SpinQuant/OstQuant) and absorbing them into model weights before quantization.

## `prepare_rotated_model`

::: onecomp.pre_process.prepare_rotated_model.prepare_rotated_model
    options:
      show_source: false

## `RotatedModelConfig`

`ModelConfig` subclass for loading rotation-preprocessed models.
Automatically registers Hadamard `forward_pre_hook` on `down_proj` layers.

::: onecomp.rotated_model_config.RotatedModelConfig
    options:
      show_source: false

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Rotation Preprocessing                             │
│                                                             │
│  ModelConfig ──► prepare_rotated_model() ──► RotatedModelConfig
│                  (train rotation matrices,                  │
│                   absorb into weights,                      │
│                   save rotated model)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Step 2: Quantization                                       │
│                                                             │
│  RotatedModelConfig ──► Runner(quantizer=GPTQ/RTN/...) ──► run()
│  (auto-registers            ──► save_quantized_model()      │
│   Hadamard hooks)                                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Step 3: Load                                               │
│                                                             │
│  load_quantized_model()                                     │
│  (auto-detects "rotated: true" in config.json,              │
│   registers Hadamard hooks automatically)                   │
└─────────────────────────────────────────────────────────────┘
```

!!! note
    The `wbits`, `groupsize`, and `sym` parameters passed to `prepare_rotated_model()`
    control the RTN proxy used during rotation training. These values **must match**
    the quantizer parameters used in Step 2.
