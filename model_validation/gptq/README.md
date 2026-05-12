# Model Validation (GPTQ 4-bit, groupsize=128)

Validates OneComp's `GPTQ` quantizer on multiple models with a fixed
`wbits=4`, `groupsize=128` configuration and `qep=False`. The
validation has three phases:

1. **Quantization** (`validate_gptq.py`) - quantize, save, and measure
   original / quantized perplexity.
2. **Load validation** (`validate_load.py`) - reload each saved
   directory via Hugging Face `transformers` and run a short greedy
   generation to confirm the load + inference path works end-to-end.
3. **vLLM inference validation** (`validate_vllm.py`) - reload each
   saved directory via vLLM's offline `LLM` interface and run a short
   greedy generation to confirm the vLLM serving path works.

Configuration is managed with [Hydra](https://hydra.cc/).

## Requirements

Hydra is not part of OneComp's runtime dependencies. Install it via the
`hydra` extra:

```bash
# uv
uv sync --extra <cuXXX> --extra hydra

# pip
pip install "onecomp[hydra]"
```

Replace `<cuXXX>` with the CUDA variant matching your environment
(`cpu`, `cu118`, `cu121`, `cu124`, `cu126`, `cu128`, `cu130`).

Phase 3 additionally requires the `vllm` extra:

```bash
# uv
uv sync --extra <cuXXX> --extra hydra --extra vllm

# pip
pip install "onecomp[hydra]" vllm
```

## Validated Models

All three phases share the same model set:

- TinyLlama-1.1B (`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`)
- gemma-4-E2B
- Llama-2-7B
- Llama-3-8B
- Qwen3-8B

---

## 1. Quantization (`validate_gptq.py`)

Quantizes a model with `GPTQ(wbits=4, groupsize=128)`, `qep=False`, and
`CalibrationConfig(max_length=512, num_calibration_samples=128)`,
saves it via `runner.save_quantized_model(...)`, and reports
original / quantized perplexity on `wikitext-2-raw-v1`.

### Usage

Specify a model via either `model_path` (local directory) or
`model_id` (Hugging Face Hub). Exactly one of the two is required.

```bash
# Local model
python validate_gptq.py model_path=/path/to/model

# Hugging Face Hub
python validate_gptq.py model_id=<HF Hub ID>
```

Any field in [conf/validate.yaml](conf/validate.yaml) can be overridden
on the command line, for example:

```bash
python validate_gptq.py model_path=/path/to/model output_dir=outputs/custom
```

### Outputs

For each run, Hydra changes into `output_dir` and the following are
produced:

- `quantized/` - quantized model saved via `runner.save_quantized_model(...)`
- standard Hydra logs (`.hydra/`, `*.log`)
- stdout: original / quantized perplexity

### Results

Perplexity is measured on `wikitext-2-raw-v1` (OneComp default).

| Model | Original PPL | Quantized PPL | Notes |
|---|---:|---:|---|
| TinyLlama-1.1B | 7.77 | 8.69 | OK |
| gemma-4-E2B (base) | 25.99 | 35.03 | warn (see below) |
| Llama-2-7B | 5.47 | 6.59 | OK |
| Llama-3-8B | 6.14 | 27.74 | warn (see below) |
| Qwen3-8B | 9.72 | 10.72 | OK |

#### Notes on gemma-4-E2B and Llama-3-8B

Both models show a larger original-vs-quantized PPL gap than the other
entries on this validation set:

- gemma-4-E2B: `25.99 -> 35.03` (≈35% relative increase)
- Llama-3-8B: `6.14 -> 27.74` (≈4.5x increase)

Llama-2-7B (`+20%`), TinyLlama-1.1B (`+12%`), and Qwen3-8B (`+10%`) all
land in a healthier range under the same setting. A likely contributor
is the compact calibration config (`max_length=512`,
`num_calibration_samples=128`), which may be a tight fit for these
architectures (e.g. Llama-3-8B's 128k-vocab tokenizer).

Worth trying if the result needs to be improved:

- increase calibration to defaults (`max_length=2048`,
  `num_calibration_samples=512`),
- enable QEP (`qep=True`) to leverage quantization-error propagation.

---

## 2. Load Validation (`validate_load.py`)

After phase 1 has produced `outputs/<name>/quantized`,
[`validate_load.py`](validate_load.py) loads each saved directory via
`load_quantized_model` and runs a short greedy generation to confirm
the load + inference path works end-to-end without errors.

### Usage

```bash
# Default (uses load_quantized_model's float16 default)
python validate_load.py quantized_path=outputs/tinyllama/quantized

# Override torch_dtype manually (required for Gemma 3/4)
python validate_load.py \
  quantized_path=outputs/gemma-4-e2b/quantized \
  torch_dtype=bfloat16
```

`torch_dtype` accepts `float16` / `bfloat16` / `float32`, or `null` to
fall back to the loader's default.

### Results

Prompt: `"Fujitsu is"`, greedy decoding, `max_new_tokens=32`.

| Model | torch_dtype | Load + generate | Generated text | Notes |
|---|---|---|---|---|
| TinyLlama-1.1B | (default) | OK | "a Japanese multinational information technology company headquartered in Tokyo, Japan. ..." | |
| gemma-4-E2B | `bfloat16` | warn | "போதும் ancho الحرب RheumatExamination لا ह्या characterise characterise characterise ..." | loads and runs without errors but generated text is degenerate (see below) |
| Llama-2-7B | (default) | OK | "a leading provider of IT services and products for the global marketplace. ..." | |
| Llama-3-8B | (default) | OK | "a leading provider of information and communication technology (ICT) based business solutions. ..." | |
| Qwen3-8B | (default) | OK | "a Japanese multinational technology company that has been in operation since 1936. ..." | |

#### Notes on gemma-4-E2B

- Must be loaded as `bfloat16` instead of `float16`, i.e.
  `load_quantized_model(path, torch_dtype=torch.bfloat16)`.
  `load_quantized_model`'s default is `float16`, but Gemma 4 internally
  uses `bfloat16`, so the default triggers a `Half`/`BFloat16` dtype
  mismatch at `lm_head`.
- Output is broken (random tokens / non-Latin scripts), consistent
  with the larger PPL gap seen in the quantization phase. Improving it
  likely requires a larger calibration config or `qep=True`.

---

## 3. vLLM Inference (`validate_vllm.py`)

After phase 1 has produced `outputs/<name>/quantized`,
[`validate_vllm.py`](validate_vllm.py) loads each saved directory via
vLLM's offline `LLM` interface and runs a short greedy generation to
confirm the vLLM serving path works end-to-end without errors.
[OneComp's vLLM plugin](../../docs/user-guide/vllm-inference.md) is
auto-registered when `onecomp` is installed, so no `quantization=`
argument is needed when constructing `LLM`.

### Usage

```bash
# Default (float16)
python validate_vllm.py quantized_path=outputs/tinyllama/quantized

# Override dtype (required for Gemma 3/4)
python validate_vllm.py \
  quantized_path=outputs/gemma-4-e2b/quantized \
  dtype=bfloat16
```

`dtype` is passed straight to vLLM and accepts `float16` / `bfloat16` /
`float32` / `auto`.

### Outputs

For each run, Hydra changes into `output_dir` and the following are
produced:

- standard Hydra logs (`.hydra/`, `*.log`)
- stdout: prompt and generated text from the vLLM engine

### Results

Prompt: `"Fujitsu is"`, greedy decoding (`temperature=0.0`),
`max_tokens=32`, `enforce_eager=True`, `max_model_len=512`.

| Model | dtype | Load + generate | Generated text | Notes |
|---|---|---|---|---|
| TinyLlama-1.1B | `float16` | pending | | |
| gemma-4-E2B | `bfloat16` | pending | | |
| Llama-2-7B | `float16` | pending | | |
| Llama-3-8B | `float16` | pending | | |
| Qwen3-8B | `float16` | pending | | |

#### Notes

- gemma-4-E2B must be loaded with `dtype=bfloat16`. Gemma 4 internally
  uses `bfloat16`, so passing `dtype=float16` triggers a
  `Half` / `BFloat16` dtype mismatch at `lm_head` inside vLLM, the
  same way the transformers loader does in Phase 2.
- vLLM spawns worker subprocesses that re-import the script.
  `validate_vllm.py` keeps `LLM(...)` and `llm.generate(...)` inside
  `main()` behind `if __name__ == "__main__":` so re-imports do not
  recursively spawn new engines.
