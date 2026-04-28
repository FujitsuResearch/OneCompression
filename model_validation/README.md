# Model Validation

An operational validation suite for OneComp's end-to-end workflow
(quantize → save → load → inference) across a variety of model
architectures and sizes. Each subdirectory exercises one workflow
configuration; what is exercised varies per subdirectory and is listed
below.

Sanity checks reported here:

- Perplexity on `wikitext-2-raw-v1`, only as a check that the quantized
  model is not broken — **not** an accuracy benchmark.
- Greedy generation from a saved + reloaded model, only as a check
  that the load + inference path runs without errors.

## At a Glance

### Quantization

Status legend (quantization step only):

- **OK** — quantize runs end-to-end and the resulting (quantized, or
  dequantized for recipes that don't have a quantized-inference path
  yet) model passes the PPL sanity check.
- **OK\*** — runs, but PPL is somewhat worse than expected for the
  recipe; worth a closer look.
- **NG** — runs, but PPL is clearly broken.
- **pending** — not yet executed.

| Model | GPTQ | QEP+GPTQ | AutoBit | AutoBit+QEP | JointQ |
|---|:---:|:---:|:---:|:---:|:---:|
| TinyLlama-1.1B | OK | OK | OK | OK | OK |
| gemma-4-E2B (base) | OK\* | OK\* | NG | NG | OK |
| Llama-2-7B | OK | OK | OK | OK | OK |
| Llama-3-8B | OK\* | OK | OK | OK | OK |
| Qwen3-8B | OK | OK | OK | OK | OK |

### Save / Inference

Each cell is `(save, transformers inference, vllm inference)`. Each
entry uses OK / OK\* / NG / pending against that step's own success
criterion:

- **save** — save step completes and produces a saved quantized model directory.
- **transformers inference** — the saved model can be loaded and run inference via HF `transformers`.
- **vllm inference** — the saved model can be loaded and run inference via vLLM.
- **n/a** — the recipe has no implementation for that step yet.

| Model | GPTQ | QEP+GPTQ | AutoBit | AutoBit+QEP | JointQ |
|---|:---:|:---:|:---:|:---:|:---:|
| TinyLlama-1.1B | (OK, OK, pending) | (OK, pending, pending) | (OK, pending, pending) | (OK, pending, pending) | (n/a, n/a, n/a) |
| gemma-4-E2B (base) | (OK, OK\*, pending) | (OK, pending, pending) | (OK, pending, pending) | (OK, pending, pending) | (n/a, n/a, n/a) |
| Llama-2-7B | (OK, OK, pending) | (OK, pending, pending) | (OK, pending, pending) | (OK, pending, pending) | (n/a, n/a, n/a) |
| Llama-3-8B | (OK, OK, pending) | (OK, pending, pending) | (OK, pending, pending) | (OK, pending, pending) | (n/a, n/a, n/a) |
| Qwen3-8B | (OK, OK, pending) | (OK, pending, pending) | (OK, pending, pending) | (OK, pending, pending) | (n/a, n/a, n/a) |

JointQ does not yet provide a quantized-inference layer (no
`save_quantized_model` / `create_quantized_model` path), so save and
inference are listed as `n/a` until those are implemented; the
quantization step for JointQ is sanity-checked on the dequantized
model instead.

See per-recipe details in the [Results](#results) section below.

## Subdirectories

| Directory | Quantization recipe | Steps exercised |
|---|---|---|
| [`gptq/`](gptq/) | `GPTQ(wbits=4, groupsize=128)`, `qep=False` | quantize, save, load + greedy generation, PPL sanity check |
| [`qep_gptq/`](qep_gptq/) | `GPTQ(wbits=4, groupsize=128)`, `qep=True` | quantize, save, PPL sanity check |
| [`autobit/`](autobit/) | `AutoBitQuantizer(target_bit=4)`, candidates `GPTQ(wbits=b, groupsize=128) for b in (2,3,4,8)`, `assignment_strategy="activation_aware"`, `qep=False` | quantize, save, PPL sanity check |
| [`autobit_qep/`](autobit_qep/) | `AutoBitQuantizer(target_bit=4)`, candidates `GPTQ(wbits=b, groupsize=128) for b in (2,3,4,8)`, `assignment_strategy="activation_aware"`, `qep=True` | quantize, save, PPL sanity check |
| [`jointq/`](jointq/) | `JointQ(bits=4, group_size=128, symmetric=True)`, `qep=False` | quantize, PPL sanity check (on dequantized model). Save / inference: not yet implemented. |

## Results

### [`gptq/`](gptq/)

Two phases:

1. Quantize + save (`validate_gptq.py`). Calibration: `max_length=512`,
   `num_calibration_samples=128`.
2. Load + greedy generation (`validate_load.py`). Prompt
   `"Fujitsu is"`, `max_new_tokens=32`.

Phase 1 — quantization PPL on `wikitext-2-raw-v1`:

| Model | Original PPL | Quantized PPL | Status |
|---|---:|---:|---|
| TinyLlama-1.1B | 7.77 | 8.69 | OK |
| gemma-4-E2B (base) | 25.99 | 35.03 | OK\* |
| Llama-2-7B | 5.47 | 6.59 | OK |
| Llama-3-8B | 6.14 | 27.74 | OK\* |
| Qwen3-8B | 9.72 | 10.72 | OK |

Phase 2 — load + greedy generation:

| Model | torch_dtype | Status |
|---|---|---|
| TinyLlama-1.1B | (default) | OK |
| gemma-4-E2B | `bfloat16` | OK\* |
| Llama-2-7B | (default) | OK |
| Llama-3-8B | (default) | OK |
| Qwen3-8B | (default) | OK |

Notes:

- gemma-4-E2B must be loaded as `bfloat16`; the loader's default
  `float16` triggers a `Half`/`BFloat16` mismatch at `lm_head`. With
  `bfloat16`, load + generation runs without errors but the output is
  degenerate (random tokens / non-Latin scripts), consistent with the
  warn-level PPL in phase 1.
- All other models load and generate sensibly under the default dtype.

See [`gptq/README.md`](gptq/README.md) for full details, generated
samples, and discussion.

### [`qep_gptq/`](qep_gptq/)

Quantize + save (`validate_gptq.py`) with QEP on. Calibration:
`max_length=1024`, `num_calibration_samples=128`. Load + inference is
not exercised in this subdirectory yet.

| Model | Original PPL | Quantized PPL | Status |
|---|---:|---:|---|
| TinyLlama-1.1B | 7.77 | 8.63 | OK |
| gemma-4-E2B (base) | 25.99 | 37.82 | OK\* |
| Llama-2-7B | 5.47 | 6.10 | OK |
| Llama-3-8B | 6.14 | 7.14 | OK |
| Qwen3-8B | 9.72 | 10.83 | OK |

See [`qep_gptq/README.md`](qep_gptq/README.md) for details.

### [`autobit/`](autobit/)

Quantize + save (`validate_autobit.py`) with QEP off. Calibration:
`max_length=512`, `num_calibration_samples=128`. Load + inference is
not exercised in this subdirectory yet.

| Model | Original PPL | Quantized PPL | Status |
|---|---:|---:|---|
| TinyLlama-1.1B | 7.77 | 8.75 | OK |
| gemma-4-E2B (base) | 25.99 | 2.40e13 | NG |
| Llama-2-7B | 5.47 | 5.94 | OK |
| Llama-3-8B | 6.14 | 7.27 | OK |
| Qwen3-8B | 9.72 | 10.74 | OK |

See [`autobit/README.md`](autobit/README.md) for details.

### [`autobit_qep/`](autobit_qep/)

Quantize + save (`validate_autobit.py`) with QEP on. Calibration:
`max_length=1024`, `num_calibration_samples=128` (reduced from defaults
to keep 7-8B models within the DGX Spark 128 GB UMA budget). Load +
inference is not exercised in this subdirectory yet.

| Model | Original PPL | Quantized PPL | Status |
|---|---:|---:|---|
| TinyLlama-1.1B | 7.77 | 8.67 | OK |
| gemma-4-E2B (base) | 25.99 | 1.64e14 | NG |
| Llama-2-7B | 5.47 | 5.90 | OK |
| Llama-3-8B | 6.14 | 7.24 | OK |
| Qwen3-8B | 9.72 | 10.82 | OK |

See [`autobit_qep/README.md`](autobit_qep/README.md) for details and
discussion.

### [`jointq/`](jointq/)

Quantize only (`validate_jointq.py`) with `qep=False`. Calibration:
`max_length=512`, `num_calibration_samples=128`. Save and inference
are **not yet implemented** for JointQ (no `save_quantized_model` /
`create_quantized_model` path), so quality is sanity-checked on the
dequantized model.

| Model | Original PPL | Dequantized PPL | Status |
|---|---:|---:|---|
| TinyLlama-1.1B | 7.77 | 8.25 | OK |
| gemma-4-E2B (base) | 25.99 | 27.88 | OK |
| Llama-2-7B | 5.47 | 5.64 | OK |
| Llama-3-8B | 6.14 | 6.67 | OK |
| Qwen3-8B | 9.72 | 10.21 | OK |

See [`jointq/README.md`](jointq/README.md) for details.

## Summary

End-to-end execution succeeded on every model attempted under all
five recipes (`gptq/`, `qep_gptq/`, `autobit/`, `autobit_qep/`,
`jointq/`); no crashes or runtime failures observed. In `gptq/`, save
+ load + greedy generation also ran cleanly on every model. For
`jointq/`, save and inference are not yet implemented, so the
quantization step is sanity-checked on the dequantized model.

**Caveat on cross-recipe PPL comparison.** All five recipes use a
compact calibration (`num_calibration_samples=128` and `max_length`
512–1024) that sits well below typical research settings, partly
because the calibration size has to fit the DGX Spark 128 GB UMA
budget for 7–8B models with QEP on. PPL differences between recipes
under this calibration are not large or stable enough to interpret as
which recipe is "better", or to attribute recovery / regression to a
specific mechanism (QEP vs AutoBit vs JointQ, etc.). PPL is reported
here only to confirm that each recipe produces a model that is not
obviously broken on a per-recipe basis. The bullets below describe
per-recipe outcomes only; cross-recipe ordering should not be read
into them.

Per-recipe sanity-check observations (not accuracy claims):

- TinyLlama-1.1B passes the PPL sanity check under every recipe
  attempted; under `gptq/`, load + greedy generation produces
  sensible text.
- gemma-4-E2B (base) is borderline under both GPTQ-only recipes
  (`gptq/` 35.03, `qep_gptq/` 37.82 vs original 25.99; both OK\*).
  Under `gptq/`, the saved model must be loaded as `bfloat16` (the
  default `float16` triggers a `Half`/`BFloat16` mismatch at
  `lm_head`) and greedy generation produces degenerate output
  (random tokens / non-Latin scripts), consistent with the
  warn-level PPL. Under both AutoBit recipes the PPL diverges to
  ~10^13 (`autobit/`) and ~10^14 (`autobit_qep/`); under
  `autobit_qep/` the bit assignment polarized into 8-bit / 2-bit
  halves (observed directly in the run log). The order-of-magnitude
  divergence and the polarized bit assignment are not artifacts of
  the limited calibration: under `autobit_qep/` the same bimodal
  assignment was reproduced when calibration was reduced from
  `max_length=2048, num_calibration_samples=512` to the current
  `max_length=1024, num_calibration_samples=128`. This model is
  therefore NG under AutoBit on this hardware, regardless of QEP.
  Under `jointq/` it passes the PPL sanity check (27.88 vs 25.99,
  OK).
- Llama-2-7B passes the PPL sanity check under every recipe; under
  `gptq/`, load + greedy generation produces sensible text.
- Llama-3-8B is weak under `gptq/` (PPL 27.74 vs original 6.14, OK\*)
  with the compact calibration (`max_length=512`,
  `num_calibration_samples=128`); within range under each of the
  other four recipes (`qep_gptq/` 7.14, `autobit/` 7.27,
  `autobit_qep/` 7.24, `jointq/` 6.67). Load + greedy generation
  under `gptq/` runs without errors and produces sensible text.
- Qwen3-8B passes the PPL sanity check under every recipe; under
  `gptq/`, load + greedy generation produces sensible text.
- Memory: `qep_gptq/` and `autobit_qep/` at the reduced calibration
  (`max_length=1024`, `num_calibration_samples=128`) fit within the
  DGX Spark 128 GB UMA budget for all 5 models (no OOM at
  `mlp.down_proj` for 7–8B); larger calibration will not fit on this
  hardware. The QEP-off recipes (`gptq/`, `autobit/`, `jointq/`) at
  `max_length=512` fit comfortably within the same budget.
- Save / inference: load + inference for the three QEP/AutoBit
  recipes is pending. For `jointq/`, save and inference are not yet
  implemented in OneComp; quality there is reported on the
  dequantized model only.
