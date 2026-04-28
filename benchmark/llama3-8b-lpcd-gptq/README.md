# Llama-3-8B LPCD+GPTQ Benchmark

LPCD (Layer-Projected Coordinate Descent) + GPTQ benchmark for [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) using OneComp v0.3.7.

QEP only supports a single quantizer per run, so each `bits × group_size` combination is launched as a separate SLURM array task.

## Benchmark Configuration

| Parameter | Values |
|---|---|
| bits | 4, 3 |
| group_size | 128 |
| symmetric | true |
| num_calibration_samples | 1024 |
| calibration_strategy | drop_rand |
| max_length | 2048 |

QEP parameters:

| Parameter | Values |
|---|---|
| percdamp | 0.01 |
| perccorr | 0.5 |

LPCD parameters:
| Parameter | Values |
|---|---|
| percdamp | 0.01 |
| perccorr | 0.5 |

This produces **8 array tasks** (2 bits × 4 sub-modules), each with QEP enabled.

| task_id | bits | sub-module |
|---|---|---|
| 0 | 4 | q_proj, k_proj |
| 1 | 4 | v_proj, o_proj |
| 2 | 4 | up_proj, down_proj |
| 3 | 4 | all |
| 4 | 3 | q_proj, k_proj |
| 5 | 3 | v_proj, o_proj |
| 6 | 3 | up_proj, down_proj |
| 7 | 3 | all |

### Evaluation

- Perplexity (WikiText-2)
- Accuracy (lm-eval-harness)

Each task evaluates its own quantized model. Original model metrics are computed only in task 0.

## Usage

Requires [Hydra](https://hydra.cc/) (see [benchmark/README.md](../README.md) for installation).

Specify the path to the model via `model_path` and the task via `task_id`:

```bash
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B task_id=0
```

### Hydra Overrides

You can override any parameter from the command line:

```bash
# Change QEP correction percentage
python quant_benchmark.py model_path=/path/to/model task_id=0 qep.perccorr=0.3

# Change calibration samples
python quant_benchmark.py model_path=/path/to/model task_id=0 num_calibration_samples=512
```

## Results

### Perplexity (WikiText-2, ↓ lower is better)

| Model | bits | sub-module | PPL |
|---|---|---|---|
| Original | - | - | 6.14 |
| QEP+GPTQ | 4 | - | 6.66 |
| LPCD+GPTQ | 4 | residual | 6.60 |
| LPCD+GPTQ | 4 | q_proj, k_proj | 6.71 |
| LPCD+GPTQ | 4 | v_proj, o_proj | 6.59 |
| LPCD+GPTQ | 4 | up_proj, down_proj | 6.63 |
| LPCD+GPTQ | 4 | all | 6.64 |
| QEP+GPTQ | 3 | - | 8.89 |
| LPCD+GPTQ | 4 | residual | 8.75 |
| LPCD+GPTQ | 3 | q_proj, k_proj | 8.58 |
| LPCD+GPTQ | 3 | v_proj, o_proj | 9.11 |
| LPCD+GPTQ | 3 | up_proj, down_proj | 8.60 |
| LPCD+GPTQ | 3 | all | 8.55 |

### Accuracy (0-shot, ↑ higher is better)

Values are `acc_norm` where available, `acc` otherwise (winogrande).

| Model | bits | sub-module | ARC-c | ARC-e | PIQA | WinoGrande |
|---|---|---|---|---|---|---|
| Original | — | — | 0.5401 | 0.7761 | 0.8063 | 0.7380 |
| QEP+GPTQ | 4 | - | 0.5119 | 0.7698 | 0.7943 | 0.7324 |
| LPCD+GPTQ | 4 | residual | 0.5196 | 0.7753 | 0.7949 | 0.7324 |
| LPCD+GPTQ | 4 | q_proj, k_proj | 0.5188 | 0.7849 | 0.7965 | 0.7332 |
| LPCD+GPTQ | 4 | v_proj, o_proj | 0.5145 | 0.7786 | 0.7992 | 0.7214 |
| LPCD+GPTQ | 4 | up_proj, down_proj | 0.5222 | 0.7744 | 0.7982 | 0.7245 |
| LPCD+GPTQ | 4 | all | 0.5145 | 0.7811 | 0.7982 | 0.7253 |
| QEP+GPTQ | 3 | - | 0.4198 | 0.6848 | 0.7633 | 0.7096 |
| LPCD+GPTQ | 3 | residual | 0.4087 | 0.6637 | 0.7612 | 0.7190 |
| LPCD+GPTQ | 3 | q_proj, k_proj | 0.4292 | 0.6734 | 0.7650 | 0.7222 |
| LPCD+GPTQ | 3 | v_proj, o_proj | 0.4283 | 0.6818 | 0.7584 | 0.7024 |
| LPCD+GPTQ | 3 | up_proj, down_proj | 0.4420 | 0.7180 | 0.7742 | 0.7111 |
| LPCD+GPTQ | 3 | all | 0.4394 | 0.6776 | 0.7900 | 0.6985 |

### Quantization Time

| Model | bits | sub-module | Time (s) |
|---|---|---|---|
| QEP+GPTQ | 4 | - | 3238 |
| LPCD+GPTQ | 4 | residual | 6965 |
| LPCD+GPTQ | 4 | q_proj, k_proj | 17446 |
| LPCD+GPTQ | 4 | v_proj, o_proj | 10252 |
| LPCD+GPTQ | 4 | up_proj, down_proj | 11270 |
| LPCD+GPTQ | 4 | all | 24472 |
| QEP+GPTQ | 3 | - | 3219 |
| LPCD+GPTQ | 3 | residual | 6859 |
| LPCD+GPTQ | 3 | q_proj, k_proj | 17074 |
| LPCD+GPTQ | 3 | v_proj, o_proj | 10199 |
| LPCD+GPTQ | 3 | up_proj, down_proj | 11172 |
| LPCD+GPTQ | 3 | all | 24405 |

Each task's total elapsed time (including calibration data preparation and LPCD error propagation) was approximately 7000–24000 s (~117–400 min).

## Environment

- GPU: NVIDIA B200 × 1
