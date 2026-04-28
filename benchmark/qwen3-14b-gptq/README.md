# Qwen3-14B GPTQ Benchmark

GPTQ benchmark for [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) using OneComp v1.1.0.

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
python quant_benchmark.py model_path=/path/to/Qwen3-14B

# mse+actorder
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    gptq.actorder=true gptq.mse=true output_dir=qwen3-14b-mse-actorder
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
| — (Original) | — | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| 4 | 128 | 8.84 | 0.5947 | 0.8228 | 0.8003 | 0.7261 | 430.2 |
| 4 | per-channel | 9.11 | 0.5862 | 0.8098 | 0.7862 | 0.6953 | 419.9 |
| 3 | 128 | 10.11 | 0.5384 | 0.7774 | 0.7878 | 0.6890 | 427.2 |
| 3 | per-channel | 13.61 | 0.4138 | 0.5690 | 0.7296 | 0.6069 | 419.5 |

Total elapsed time (including calibration data preparation): 7213.8 s (~120 min).

### GPTQ (mse+actorder)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| 4 | 128 | 8.87 | 0.6195 | 0.8237 | 0.7949 | 0.7285 | 2264.1 |
| 4 | per-channel | 9.26 | 0.5990 | 0.8215 | 0.7960 | 0.7253 | 583.8 |
| 3 | 128 | 9.47 | 0.5853 | 0.8241 | 0.7927 | 0.7190 | 2672.7 |
| 3 | per-channel | 15.73 | 0.4872 | 0.7336 | 0.7650 | 0.6551 | 589.7 |

Total elapsed time (including calibration data preparation): 11586.7 s (~193 min).

## Reduced Calibration Benchmark (num_calibration_samples=128)

Same configurations as above with `num_calibration_samples` reduced from 1024 to 128.

### GPTQ (default, calib=128)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.85 | 0.5973 | 0.8199 | 0.8025 | 0.7040 | 444.4 |
| 4 | per-channel | 9.19 | 0.5648 | 0.7883 | 0.7797 | 0.6890 | 433.9 |
| 3 | 128 | 10.17 | 0.5137 | 0.7614 | 0.7813 | 0.6717 | 445.0 |
| 3 | per-channel | 14.14 | 0.3737 | 0.5732 | 0.7280 | 0.5738 | 436.7 |

Total elapsed time (including calibration data preparation): 2939.2 s (~49 min).

### GPTQ (mse+actorder, calib=128)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.84 | 0.6041 | 0.8262 | 0.8009 | 0.7238 | 2475.4 |
| 4 | per-channel | 9.41 | 0.5947 | 0.8220 | 0.7949 | 0.7285 | 634.0 |
| 3 | 128 | 9.53 | 0.5930 | 0.8106 | 0.7960 | 0.7135 | 2937.0 |
| 3 | per-channel | 15.46 | 0.4940 | 0.7247 | 0.7563 | 0.6377 | 641.1 |

Total elapsed time (including calibration data preparation): 7872.2 s (~131 min).

## Environment

- GPU: NVIDIA B200 × 2

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
