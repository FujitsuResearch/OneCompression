# Qwen3-8B GPTQ Benchmark

GPTQ benchmark for [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) using OneComp v1.1.1.

All combinations of `bits × group_size` are run in a single pass, sharing calibration data accumulation across quantizers for efficiency.

Four configurations are benchmarked (the 2×2 grid of `actorder × mse`):

1. **GPTQ (default)** — `actorder=false`, `mse=false`
2. **GPTQ (actorder)** — `actorder=true`, `mse=false`
3. **GPTQ (mse)** — `actorder=false`, `mse=true`
4. **GPTQ (mse+actorder)** — `actorder=true`, `mse=true` (strongest GPTQ setting)

## Benchmark Configuration

### Common Parameters

| Parameter | Values |
|---|---|
| bits | 4, 3 |
| group_size | 128, per-channel |
| symmetric | true |
| num_calibration_samples | 1024 |
| calibration_strategy | drop_rand |
| max_length | 2048 |
| dtype | bfloat16 |

This produces **4 quantizers** (2 bits × 2 group sizes) per configuration.

### Configuration-Specific Parameters

| Parameter | default | actorder | mse | mse+actorder |
|---|---|---|---|---|
| actorder | false | true | false | true |
| mse | false | false | true | true |

### Evaluation

- Perplexity (WikiText-2)
- Accuracy (lm-eval-harness)

Both are computed for the original (unquantized) model and all dequantized models.

## Usage

Requires [Hydra](https://hydra.cc/) (see [benchmark/README.md](../README.md) for installation).

```bash
# default
python quant_benchmark.py model_path=/path/to/Qwen3-8B

# actorder
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    gptq.actorder=true output_dir=qwen3-8b-actorder

# mse
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    gptq.mse=true output_dir=qwen3-8b-mse

# mse+actorder
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    gptq.actorder=true gptq.mse=true output_dir=qwen3-8b-mse-actorder
```

### Hydra Overrides

You can override any parameter from the command line:

```bash
# Run only 4-bit
python quant_benchmark.py model_path=/path/to/model 'gptq.bits=[4]'

# Change calibration samples
python quant_benchmark.py model_path=/path/to/model num_calibration_samples=512
```

## Results

PPL = perplexity on WikiText-2 (↓ lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (winogrande) (↑ higher is better).

### GPTQ (default)

`actorder=false`, `mse=false`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 9.73 | 0.5640 | 0.8093 | 0.7758 | 0.6772 | — |
| 4 | 128 | 10.33 | 0.5614 | 0.7879 | 0.7639 | 0.6693 | 273.2 |
| 4 | per-channel | 11.04 | 0.5077 | 0.7441 | 0.7557 | 0.6835 | 268.5 |
| 3 | 128 | 11.75 | 0.4881 | 0.7193 | 0.7584 | 0.6527 | 273.0 |
| 3 | per-channel | 19.55 | 0.3208 | 0.4735 | 0.6850 | 0.5391 | 266.7 |

Total elapsed time (including calibration data preparation): 3909.6 s (~65 min).

### GPTQ (actorder)

`actorder=true`, `mse=false`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 9.99 | 0.5418 | 0.7849 | 0.7688 | 0.6859 | 265.5 |
| 4 | per-channel | 10.39 | 0.5060 | 0.7378 | 0.7688 | 0.6519 | 260.5 |
| 3 | 128 | 11.52 | 0.4761 | 0.6932 | 0.7546 | 0.6511 | 268.7 |
| 3 | per-channel | 16.50 | 0.3191 | 0.4895 | 0.6980 | 0.5533 | 262.8 |

Total elapsed time (including calibration data preparation): 3892.6 s (~65 min).

### GPTQ (mse)

`actorder=false`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.36 | 0.5435 | 0.7879 | 0.7715 | 0.6843 | 1701.3 |
| 4 | per-channel | 12.78 | 0.5119 | 0.7454 | 0.7650 | 0.6511 | 422.2 |
| 3 | 128 | 14.09 | 0.4625 | 0.7037 | 0.7448 | 0.6519 | 2033.1 |
| 3 | per-channel | 51.11 | 0.2662 | 0.3636 | 0.6420 | 0.5517 | 426.1 |

Total elapsed time (including calibration data preparation): 7406.1 s (~123 min).

### GPTQ (mse+actorder)

`actorder=true`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 9.93 | 0.5495 | 0.8005 | 0.7780 | 0.6748 | 1718.1 |
| 4 | per-channel | 11.18 | 0.5316 | 0.7740 | 0.7666 | 0.6756 | 424.6 |
| 3 | 128 | 11.42 | 0.5418 | 0.7841 | 0.7682 | 0.6906 | 2041.9 |
| 3 | per-channel | 43.42 | 0.2986 | 0.4310 | 0.6790 | 0.5635 | 430.0 |

Total elapsed time (including calibration data preparation): 7448.1 s (~124 min).

## Environment

- GPU: NVIDIA B200 × 1

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
