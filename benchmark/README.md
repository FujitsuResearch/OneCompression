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
| [llama3-8b-gptq/](llama3-8b-gptq/) | Llama-3-8B GPTQ (4bit/3bit × gs128/per-channel) |
| [llama3-8b-jointq/](llama3-8b-jointq/) | Llama-3-8B JointQ (4bit/3bit × gs128/per-channel) |
| [llama3-8b-qep-gptq/](llama3-8b-qep-gptq/) | Llama-3-8B QEP+GPTQ (4bit/3bit × gs128/per-channel) |
| [llama3-8b-various/](llama3-8b-various/) | Llama-3-8B Various quantizers with default parameters (no QEP) |
| [qwen3-8b-gptq/](qwen3-8b-gptq/) | Qwen3-8B GPTQ (4bit/3bit × gs128/per-channel) |
| [qwen3-8b-jointq/](qwen3-8b-jointq/) | Qwen3-8B JointQ (4bit/3bit × gs128/per-channel) |
| [qwen3-14b-gptq/](qwen3-14b-gptq/) | Qwen3-14B GPTQ (4bit/3bit × gs128/per-channel) |
| [qwen3-14b-jointq/](qwen3-14b-jointq/) | Qwen3-14B JointQ (4bit/3bit × gs128/per-channel) |

## Results Summary

### GPTQ vs. JointQ

- Referenced GPTQ directories:
  - [`llama3-8b-gptq/`](llama3-8b-gptq/)
  - [`qwen3-8b-gptq/`](qwen3-8b-gptq/)
  - [`qwen3-14b-gptq/`](qwen3-14b-gptq/)
- Referenced JointQ directories:
  - [`llama3-8b-jointq/`](llama3-8b-jointq/)
  - [`qwen3-8b-jointq/`](qwen3-8b-jointq/)
  - [`qwen3-14b-jointq/`](qwen3-14b-jointq/)
- GPTQ rows use the `num_calibration_samples=1024`, `max_length=2048` results from each GPTQ benchmark README.
- JointQ diagonal rows use `λ=0.05` (4-bit) / `λ=0.1` (3-bit) for gs128, and `λ=0.01` (4-bit) / `λ=0.1` (3-bit) for per-channel.
- JointQ diagonal+mse+actorder rows use the same `λ` values.
- Values are judged separately within the `gs128` and `per-channel` groups in each table.
- `PPL` / `Time` mark the best value in each group in bold.
- Accuracy columns mark the best value in each group in bold, and values that match or exceed `Original` are marked with `*`.

Comparison tables will be populated after re-running both GPTQ and JointQ benchmarks.
