# Qwen3-14B GPTQ Benchmark

GPTQ benchmark for [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) using OneComp v1.1.1.

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
python quant_benchmark.py model_path=/path/to/Qwen3-14B

# actorder
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    gptq.actorder=true output_dir=qwen3-14b-actorder

# mse
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    gptq.mse=true output_dir=qwen3-14b-mse

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

PPL = perplexity on WikiText-2 (↓ lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (winogrande) (↑ higher is better).

### GPTQ (default)

`actorder=false`, `mse=false`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 8.65 | 0.6041 | 0.8279 | 0.7976 | 0.7301 | — |
| 4 | 128 | 8.85 | 0.5956 | 0.8215 | 0.7987 | 0.7301 | 407.4 |
| 4 | per-channel | 9.11 | 0.5904 | 0.8035 | 0.7878 | 0.7024 | 396.9 |
| 3 | 128 | 10.08 | 0.5469 | 0.7866 | 0.7894 | 0.6827 | 406.5 |
| 3 | per-channel | 13.84 | 0.3703 | 0.5686 | 0.7258 | 0.5706 | 398.4 |

Total elapsed time (including calibration data preparation): 6966.8 s (~116 min).

### GPTQ (actorder)

`actorder=true`, `mse=false`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.90 | 0.5930 | 0.8211 | 0.7960 | 0.7174 | 405.1 |
| 4 | per-channel | 9.21 | 0.5896 | 0.8173 | 0.7916 | 0.7024 | 397.9 |
| 3 | 128 | 9.91 | 0.5486 | 0.7689 | 0.7851 | 0.7040 | 406.7 |
| 3 | per-channel | 13.29 | 0.4224 | 0.6279 | 0.7405 | 0.6338 | 399.0 |

Total elapsed time (including calibration data preparation): 7004.7 s (~117 min).

### GPTQ (mse)

`actorder=false`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.79 | 0.6203 | 0.8262 | 0.7982 | 0.7238 | 2602.7 |
| 4 | per-channel | 9.40 | 0.5862 | 0.8253 | 0.7927 | 0.7103 | 648.1 |
| 3 | 128 | 9.65 | 0.5495 | 0.7719 | 0.7851 | 0.7151 | 3092.9 |
| 3 | per-channel | 17.64 | 0.4838 | 0.7197 | 0.7552 | 0.6259 | 656.3 |

Total elapsed time (including calibration data preparation): 12413.1 s (~207 min).

### GPTQ (mse+actorder)

`actorder=true`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.92 | 0.6067 | 0.8279 | 0.7954 | 0.7230 | 2528.9 |
| 4 | per-channel | 9.26 | 0.5896 | 0.8140 | 0.7960 | 0.7245 | 640.2 |
| 3 | 128 | 9.52 | 0.5939 | 0.8165 | 0.7938 | 0.7198 | 3021.2 |
| 3 | per-channel | 15.68 | 0.4787 | 0.7311 | 0.7655 | 0.6456 | 644.1 |

Total elapsed time (including calibration data preparation): 12209.9 s (~204 min).

## Environment

- GPU: NVIDIA B200 × 2

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
