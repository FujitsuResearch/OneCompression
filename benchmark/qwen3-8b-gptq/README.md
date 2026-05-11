# Qwen3-8B GPTQ Benchmark

GPTQ benchmark for [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) using OneComp v1.1.0.

All combinations of `bits × group_size` are run in a single pass, sharing calibration data accumulation across quantizers for efficiency.

Two configurations are benchmarked:

1. **GPTQ (default)** — `actorder=false`, `mse=false`
2. **GPTQ (mse+actorder)** — `actorder=true`, `mse=true` (strongest GPTQ setting)

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

This produces **4 quantizers** (2 bits × 2 group sizes) per configuration.

### Configuration-Specific Parameters

| Parameter | default | mse+actorder |
|---|---|---|
| actorder | false | true |
| mse | false | true |

### Evaluation

- Perplexity (WikiText-2)
- Accuracy (lm-eval-harness)

Both are computed for the original (unquantized) model and all dequantized models.

## Usage

Requires [Hydra](https://hydra.cc/) (see [benchmark/README.md](../README.md) for installation).

```bash
# default
python quant_benchmark.py model_path=/path/to/Qwen3-8B

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

### GPTQ (default)

PPL = perplexity on WikiText-2 (↓ lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (winogrande) (↑ higher is better).

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| 4 | 128 | 10.26 | 0.5580 | 0.7934 | 0.7671 | 0.6669 | 259.4 |
| 4 | per-channel | 10.88 | 0.5119 | 0.7466 | 0.7622 | 0.6740 | 252.3 |
| 3 | 128 | 11.75 | 0.4846 | 0.7222 | 0.7481 | 0.6488 | 257.2 |
| 3 | per-channel | 20.02 | 0.3200 | 0.4217 | 0.6703 | 0.5391 | 251.9 |

Total elapsed time (including calibration data preparation): 3888.8 s (~65 min).

### GPTQ (mse+actorder)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| 4 | 128 | 9.91 | 0.5384 | 0.8056 | 0.7791 | 0.6835 | 1517.6 |
| 4 | per-channel | 11.22 | 0.5401 | 0.7828 | 0.7715 | 0.6693 | 395.0 |
| 3 | 128 | 11.24 | 0.5307 | 0.7597 | 0.7601 | 0.6867 | 1794.6 |
| 3 | per-channel | 41.77 | 0.3259 | 0.4524 | 0.6801 | 0.5643 | 401.0 |

Total elapsed time (including calibration data preparation): 6988.5 s (~117 min).

## Reduced Calibration Benchmark (num_calibration_samples=128)

Same configurations as above with `num_calibration_samples` reduced from 1024 to 128.

### GPTQ (default, calib=128)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.53 | 0.5435 | 0.7807 | 0.7628 | 0.6598 | 267.3 |
| 4 | per-channel | 11.10 | 0.5051 | 0.7437 | 0.7622 | 0.6717 | 261.0 |
| 3 | 128 | 12.01 | 0.4548 | 0.6700 | 0.7454 | 0.6346 | 265.9 |
| 3 | per-channel | 21.94 | 0.2927 | 0.4011 | 0.6594 | 0.5249 | 261.2 |

Total elapsed time (including calibration data preparation): 1634.8 s (~27 min).

### GPTQ (mse+actorder, calib=128)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 9.88 | 0.5597 | 0.8026 | 0.7791 | 0.6843 | 1585.6 |
| 4 | per-channel | 11.19 | 0.5350 | 0.7753 | 0.7677 | 0.6661 | 402.3 |
| 3 | 128 | 11.38 | 0.5290 | 0.7748 | 0.7633 | 0.6922 | 1875.2 |
| 3 | per-channel | 44.03 | 0.3046 | 0.4268 | 0.6806 | 0.5635 | 404.3 |

Total elapsed time (including calibration data preparation): 4893.4 s (~82 min).

## Environment

- GPU: NVIDIA B200 × 1

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
