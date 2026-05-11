# Llama-3-8B GPTQ Benchmark

GPTQ benchmark for [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) using OneComp v1.1.0.

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
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B

# mse+actorder
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    gptq.actorder=true gptq.mse=true output_dir=llama3-8b-mse-actorder
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
| — (Original) | — | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| 4 | 128 | 12.55 | 0.4974 | 0.7731 | 0.7938 | 0.7174 | 261.0 |
| 4 | per-channel | 581.81 | 0.3174 | 0.5101 | 0.6828 | 0.6290 | 254.1 |
| 3 | 128 | 47.74 | 0.3029 | 0.4886 | 0.6665 | 0.6369 | 259.0 |
| 3 | per-channel | 1640.28 | 0.2312 | 0.2942 | 0.5365 | 0.5099 | 253.4 |

Total elapsed time (including calibration data preparation): 3638.8 s (~61 min).

### GPTQ (mse+actorder)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| 4 | 128 | 6.55 | 0.5427 | 0.7891 | 0.7971 | 0.7348 | 1334.8 |
| 4 | per-channel | 8.19 | 0.4727 | 0.7391 | 0.7894 | 0.7435 | 351.2 |
| 3 | 128 | 8.06 | 0.4753 | 0.7176 | 0.7661 | 0.7340 | 1571.1 |
| 3 | per-channel | 22.53 | 0.3106 | 0.4819 | 0.6855 | 0.6780 | 356.1 |

Total elapsed time (including calibration data preparation): 6224.4 s (~104 min).

## Reduced Calibration Benchmark (num_calibration_samples=128)

Same configurations as above with `num_calibration_samples` reduced from 1024 to 128.

### GPTQ (default, calib=128)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 16.07 | 0.5085 | 0.7626 | 0.7938 | 0.7301 | 252.0 |
| 4 | per-channel | 1031.66 | 0.3788 | 0.6157 | 0.7144 | 0.6693 | 243.8 |
| 3 | 128 | 46.90 | 0.2619 | 0.4613 | 0.6295 | 0.5959 | 251.1 |
| 3 | per-channel | 4130.83 | 0.2398 | 0.2769 | 0.5272 | 0.5059 | 243.6 |

Total elapsed time (including calibration data preparation): 1566.6 s (~26 min).

### GPTQ (mse+actorder, calib=128)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.58 | 0.5239 | 0.7816 | 0.8047 | 0.7348 | 1577.7 |
| 4 | per-channel | 7.92 | 0.4855 | 0.7588 | 0.7835 | 0.7332 | 404.1 |
| 3 | 128 | 8.70 | 0.4488 | 0.7151 | 0.7601 | 0.7064 | 1873.2 |
| 3 | per-channel | 39.73 | 0.3063 | 0.4651 | 0.6730 | 0.6677 | 411.1 |

Total elapsed time (including calibration data preparation): 4859.7 s (~81 min).

## Environment

- GPU: NVIDIA B200 × 1

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
