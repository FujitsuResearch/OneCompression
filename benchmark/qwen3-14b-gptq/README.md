# Qwen3-14B GPTQ Benchmark

GPTQ benchmark for [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) using OneComp v1.3.3.

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
| calibration_batch_size | 16 |
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
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    calibration_batch_size=16

# actorder
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    calibration_batch_size=16 \
    gptq.actorder=true output_dir=qwen3-14b-actorder

# mse
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    calibration_batch_size=16 \
    gptq.mse=true output_dir=qwen3-14b-mse

# mse+actorder
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    calibration_batch_size=16 \
    gptq.actorder=true gptq.mse=true output_dir=qwen3-14b-mse-actorder

# 2-bit, group size 32 (all four configurations)
python quant_benchmark_2bit_g32.py model_path=/path/to/Qwen3-14B \
    calibration_batch_size=16 model_device=cuda:0
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

PPL = perplexity on WikiText-2 (lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (WinoGrande and BoolQ) (higher is better). `Accuracy Avg.` is the arithmetic mean of the six accuracy tasks.

### GPTQ (default)

`actorder=false`, `mse=false`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| - (Original) | - | 8.65 | 0.6041 | 0.8279 | 0.7976 | 0.7309 | 0.7877 | 0.8936 | 0.7736 | - |
| 4 | 128 | 8.85 | 0.6015 | 0.8224 | 0.8003 | 0.7301 | 0.7813 | 0.8896 | 0.7709 | 473.2 |
| 4 | per-channel | 9.11 | 0.5887 | 0.8001 | 0.7840 | 0.7040 | 0.7722 | 0.8657 | 0.7525 | 463.9 |
| 3 | 128 | 10.07 | 0.5478 | 0.7862 | 0.7889 | 0.6914 | 0.7489 | 0.8508 | 0.7356 | 461.7 |
| 3 | per-channel | 13.84 | 0.3720 | 0.5669 | 0.7291 | 0.5746 | 0.6839 | 0.7266 | 0.6089 | 455.2 |

Quantization phase elapsed time: 7467.08 s (~2h 4m).

### GPTQ (actorder)

`actorder=true`, `mse=false`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.90 | 0.5939 | 0.8220 | 0.7954 | 0.7182 | 0.7845 | 0.8887 | 0.7671 | 447.6 |
| 4 | per-channel | 9.21 | 0.5879 | 0.8194 | 0.7922 | 0.7064 | 0.7744 | 0.8798 | 0.7600 | 439.6 |
| 3 | 128 | 9.91 | 0.5529 | 0.7652 | 0.7813 | 0.7032 | 0.7555 | 0.8645 | 0.7371 | 440.5 |
| 3 | per-channel | 13.29 | 0.4198 | 0.6301 | 0.7416 | 0.6306 | 0.6969 | 0.7618 | 0.6468 | 434.1 |

Quantization phase elapsed time: 7372.13 s (~2h 2m).

### GPTQ (mse)

`actorder=false`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.79 | 0.6195 | 0.8249 | 0.7971 | 0.7253 | 0.7832 | 0.8884 | 0.7731 | 2300.3 |
| 4 | per-channel | 9.40 | 0.5870 | 0.8262 | 0.7943 | 0.7088 | 0.7709 | 0.8771 | 0.7607 | 646.7 |
| 3 | 128 | 9.65 | 0.5452 | 0.7723 | 0.7845 | 0.7143 | 0.7574 | 0.8768 | 0.7418 | 2725.4 |
| 3 | per-channel | 17.64 | 0.4855 | 0.7193 | 0.7530 | 0.6227 | 0.6256 | 0.8291 | 0.6725 | 642.3 |

Quantization phase elapsed time: 11905.35 s (~3h 18m).

### GPTQ (mse+actorder)

`actorder=true`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.92 | 0.6067 | 0.8279 | 0.7971 | 0.7261 | 0.7825 | 0.8829 | 0.7705 | 2610.6 |
| 4 | per-channel | 9.26 | 0.5913 | 0.8131 | 0.7954 | 0.7222 | 0.7690 | 0.8847 | 0.7626 | 698.2 |
| 3 | 128 | 9.52 | 0.5973 | 0.8169 | 0.7911 | 0.7174 | 0.7619 | 0.8823 | 0.7611 | 3104.6 |
| 3 | per-channel | 15.68 | 0.4753 | 0.7319 | 0.7677 | 0.6433 | 0.6773 | 0.8697 | 0.6942 | 694.0 |

Quantization phase elapsed time: 12708.35 s (~3h 31m).

### GPTQ (2bit/g32)

Additional benchmark for `bits=2`, `group_size=32` using OneComp v1.3.3.

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| default | 23.93 | 0.2944 | 0.4465 | 0.6746 | 0.5351 | 0.5230 | 0.5829 | 0.5094 | 518.4 |
| actorder | 20.48 | 0.2969 | 0.4352 | 0.6583 | 0.5462 | 0.5302 | 0.6544 | 0.5202 | 517.8 |
| mse | 47407691.34 | 0.2645 | 0.2534 | 0.5071 | 0.4988 | 0.2584 | 0.6217 | 0.4006 | 11599.0 |
| mse+actorder | 45569397.23 | 0.2619 | 0.2546 | 0.5076 | 0.4941 | 0.2591 | 0.6217 | 0.3998 | 11619.3 |

Quantization phase elapsed time: 30017.22 s (~8h 20m).

## Environment

- GPU: NVIDIA B200 × 1

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
