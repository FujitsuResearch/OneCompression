# Llama-3-8B QEP+GPTQ Benchmark

QEP (Quantization Error Propagation) + GPTQ benchmark for [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) using OneComp v1.1.0.

QEP only supports a single quantizer per run, so each `bits × group_size` combination is launched as a separate SLURM array task.

## Benchmark Configuration

| Parameter | Values |
|---|---|
| bits | 4, 3 |
| group_size | 128, per-channel |
| symmetric | true |
| num_calibration_samples | 1024 |
| calibration_strategy | drop_rand |
| max_length | 2048 |

QEP parameters:

| Parameter | Values |
|---|---|
| percdamp | 0.01 |
| perccorr | 0.5 |

This produces **4 array tasks** (2 bits × 2 group sizes), each with QEP enabled.

| task_id | bits | group_size |
|---|---|---|
| 0 | 4 | 128 |
| 1 | 4 | per-channel |
| 2 | 3 | 128 |
| 3 | 3 | per-channel |

### Evaluation

- Perplexity (WikiText-2)
- Accuracy (lm-eval-harness)

Each task evaluates its own dequantized model. Original model metrics are computed only in task 0.

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

PPL = perplexity on WikiText-2 (↓ lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (winogrande) (↑ higher is better).

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| 4 | 128 | 6.64 | 0.5085 | 0.7904 | 0.7938 | 0.7238 | 248.3 |
| 4 | per-channel | 7.70 | 0.5077 | 0.7694 | 0.7840 | 0.7269 | 264.1 |
| 3 | 128 | 8.82 | 0.4275 | 0.6684 | 0.7644 | 0.7190 | 271.8 |
| 3 | per-channel | 17.73 | 0.2841 | 0.4710 | 0.6844 | 0.6409 | 261.0 |

Each task's total elapsed time (including calibration data preparation and QEP error propagation) was approximately 2900–3020 s (~48–50 min).

## Environment

- GPU: NVIDIA B200 × 1
