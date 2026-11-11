# benchmark

Benchmark scripts for OneComp.
Configuration is managed with [Hydra](https://hydra.cc/).

> **Note:** Hydra is not a dependency of OneComp and must be installed separately.

## Installing Hydra

```bash
pip install hydra-core
```

Verify the installation:

```bash
python -c "import hydra; print(hydra.__version__)"
```

## Benchmarks

| Directory | Description |
|---|---|
| [llama2-7b-gptq/](llama2-7b-gptq/) | Llama-2-7B GPTQ (4bit/3bit x gs128/per-channel) |
| [llama3-8b-gptq/](llama3-8b-gptq/) | Llama-3-8B GPTQ (4bit/3bit × gs128/per-channel) |
| [llama3-8b-qep-gptq/](llama3-8b-qep-gptq/) | Llama-3-8B QEP+GPTQ (4bit/3bit × gs128/per-channel) |
| [llama3-8b-various/](llama3-8b-various/) | Llama-3-8B Various quantizers with default parameters (no QEP) |
| [qwen3-8b-gptq/](qwen3-8b-gptq/) | Qwen3-8B GPTQ (4bit/3bit × gs128/per-channel) |
| [qwen3-14b-gptq/](qwen3-14b-gptq/) | Qwen3-14B GPTQ (4bit/3bit × gs128/per-channel) |
| [jointq_benchmark/](jointq_benchmark/) | JointQ benchmark suite across supported models and ablations |

## JointQ Benchmark Suite

The JointQ suite contains self-contained configurations and results for:

- [main/](jointq_benchmark/main/): selected 4-bit and 3-bit configurations
- [2bit_g32/](jointq_benchmark/2bit_g32/): 2-bit, group-size-32 configuration
- [actorder_mse/](jointq_benchmark/actorder_mse/): activation-ordering and GPTQ MSE ablation
- [unregularization/](jointq_benchmark/unregularization/): regularization-disabled ablation
- [calibration_stability/](jointq_benchmark/calibration_stability/): stability with small calibration datasets across JointQ and GPTQ
