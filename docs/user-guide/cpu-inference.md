# CPU Inference (llama.cpp / GGUF)

OneComp can export quantized models to the **GGUF** format and run them on the
CPU with [llama.cpp](https://github.com/ggml-org/llama.cpp) through
`llama-cpp-python`. This is the recommended path for CPU-only / edge deployment,
and complements the GPU [vLLM Inference](vllm-inference.md) path.

## Installation

```bash
export CMAKE_ARGS="-DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON -DGGML_LTO=ON"
pip install 'onecomp[llamacpp]'        # installs gguf + llama-cpp-python
# or with uv:
uv sync --extra cpu --extra llamacpp
```

`llama-cpp-python` provides prebuilt CPU wheels, so no C++ toolchain is required
for inference. The direct export path additionally uses llama.cpp's pure-Python
`convert_hf_to_gguf.py` to build the model metadata/tokenizer; it is fetched
automatically (a shallow `git clone`) or taken from `$LLAMA_CPP_DIR` if set.


### macOS

#### Symptom

If you have installed gcc or clang on macOS using a tool like Homebrew, the OpenMP dynamic library with them can conflict with the OpenMP dynamic library used by OneComp's PyTorch backend.

- OneComp's PyTorch backend is often configured to use an OpenMP library located within the .venv directory.
- Depending on your environment, Llama.cpp is configured to use an OpenMP library associated with gcc or clang.

If the following problems is occurring , this conflict may be occurring.

- Your Python interpreter shows the following warning message.

```python
.../multiprocessing/resource_tracker.py:279: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```

- The following message is shown, when you import PyTorch and Llama.cpp in your script. 

```python
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
```

#### Solution

##### 1. Search for OpenMP associated with PyTorch.

Search for OpenMP library associated with PyTorch backend (using find command).

```bash
$ find $PWD/.venv -type f \( -name 'libomp.dylib' -o -name 'libgomp*.dylib' \) -print | grep torch
/to/path/.venv/lib/python3.<PYVERSION>/site-packages/torch/lib/libomp.dylib
```

##### 2. Create an .env

Create a .env file and add the OpenMP path as an environment variable, as shown below.

```bash
DYLD_LIBRARY_PATH="/to/path/.venv/lib/python3.<PYVERSION>/site-packages/torch/lib"
```

##### 3. Run uv

Pass the following `--env-file` options when running uv run.

```bash
uv run --env-file /path/to/.env ... python your_script.py
```

## One entry point: `export_to_gguf`

You do not need to know which path a checkpoint requires. `export_to_gguf`
reads `quantization_config` and routes every supported family automatically:

```python
from onecomp.cpu import export_to_gguf, plan_export

print(plan_export("./model"))            # {'path': 'direct'|'mixed'|'fallback', 'reason': ...}
export_to_gguf("./model", "./model.gguf")  # mode="auto" by default
```

| `quant_method`            | Layout                         | Route    | Lossless |
|---------------------------|--------------------------------|----------|----------|
| `gptq` (incl. **QEP**)    | AutoGPTQ `qweight/qzeros/scales` | direct   | yes |
| `jointq`, `rtn`           | same AutoGPTQ layout           | direct   | yes |
| `mixed_gptq`              | per-layer bit-widths           | mixed    | 4/8-bit yes, 2/3-bit no |
| `dbf`, `autobit`          | binary factorization / mixed   | fallback | no (re-quantized) |
| `gptq`/`mixed_gptq` + `rotated=true` | online Hadamard on down_proj | fallback | no (re-quantized) |
| `onebit`                  | —                              | unsupported (by request) |
| `mdbf`                    | —                              | unsupported (not implemented yet) |

**QEP** only changes the GPTQ *integer codes* (via pre-quantization weight
adjustment), so QEP-corrected checkpoints export through the very same lossless
direct/mixed paths as plain GPTQ — nothing extra is needed for CPU inference.

## Direct GPTQ → GGUF (lossless, recommended)

OneComp's GPTQ weights (default group size 128) map **losslessly** onto GGUF
legacy block types because the GPTQ group size is a multiple of the GGUF block
size (32). The integer codes and scales are written directly — no
re-quantization — so the accuracy gained from GPTQ and
[QEP](../algorithms/qep.md) is fully preserved.

| GPTQ layer        | GGUF block | Dequantization        |
|-------------------|------------|-----------------------|
| 4-bit symmetric   | `Q4_0`     | `d · (q − 8)`         |
| 4-bit asymmetric  | `Q4_1`     | `d · q + m`           |
| 8-bit symmetric   | `Q8_0`     | `d · q8`              |

Constraints: `actorder=False` and group size a multiple of 32 (e.g. 128) or
`-1` (per-channel). Layers that do not qualify are kept as fp16 in the output;
use the fallback path below if you need them quantized.

```python
from onecomp.cpu import convert_gptq_to_gguf

summary = convert_gptq_to_gguf(
    quantized_dir="./model-gptq-4bit",   # produced by Runner.save_quantized_model
    out_gguf="./model.gguf",
    original_model=None,                 # optional FP model for skeleton metadata
)
print(summary)   # {"out_gguf": ..., "replaced": N, "skipped": {...}}
```

### RoPE layout (llama architecture)

llama.cpp stores `attn_q` / `attn_k` of **llama**-architecture models in the
interleaved ("NORM") RoPE layout, while Hugging Face checkpoints use the
half-split (`rotate_half`) convention. Both export paths handle this
automatically: the stitched GPTQ codes (and their scales/zeros) are row-permuted
with the same permutation `convert_hf_to_gguf.py` applies, so the export stays
lossless. NEOX-style architectures (Qwen2, Gemma, ...) are not permuted.

## Mixed-precision GGUF (llama.cpp "plugin")

For `mixed_gptq` checkpoints (per-module bit-widths produced by
`GPTQ(mlp_wbits=…, module_wbits=…)` or `AutoBitQuantizer`), OneComp ships the
`llamacpp_plugins` package — the llama.cpp counterpart of the vLLM
[`mixed_gptq`](vllm-inference.md) plugin. llama.cpp has no run-time plugin
mechanism for new quantization types (they are compiled into ggml), so the unit
of extensibility is the **GGUF file itself**: every tensor stores its own type
and llama.cpp dispatches the matching kernel per tensor. The plugin therefore
reads the same `quantization_bits` table and writes each module with the GGUF
type that matches its bit-width:

| GPTQ module        | Route   | GGUF type | Lossless? |
|--------------------|---------|-----------|-----------|
| 4-bit sym / asym   | direct  | `Q4_0` / `Q4_1` | yes (GPTQ codes preserved) |
| 8-bit sym          | direct  | `Q8_0`    | yes |
| 2-bit              | kquant  | `Q2_K`    | no (re-quantized from dequantized weights) |
| 3-bit              | kquant  | `Q3_K`    | no |
| act-order layers   | kquant  | `Q4_K` … `Q6_K` | no |

The 2/3-bit (and act-order) layers have no lossless legacy GGUF type, so they
are re-quantized from their **dequantized** weights via `llama-quantize` (which
needs the binary; `pip install cmake ninja` then build llama.cpp). The 4/8-bit
layers are still packed bit-exactly. The result is a single GGUF that runs
natively on llama.cpp with genuinely mixed per-layer precision.

```python
from llamacpp_plugins.gptq import export_mixed_gptq_gguf, plan_mixed_export

# Preview the per-module routing (no packing):
for p in plan_mixed_export("./model-mixed-gptq")[:4]:
    print(p.name, p.bits, p.route, p.ggml_type)

summary = export_mixed_gptq_gguf("./model-mixed-gptq", "./model-mixed.gguf")
print(summary["plan"]["by_type"])   # {'Q4_0': .., 'Q8_0': .., 'Q3_K': .., 'Q2_K': ..}
```

```bash
onecomp-gguf export --quantized-dir ./model-mixed-gptq --out ./model-mixed.gguf --mode mixed
```

## Fallback: dequantize → llama-quantize

For checkpoints that cannot be mapped directly (2/3-bit, `actorder=True`, or
mixed bitwidths), reconstruct fp16 weights and quantize with llama.cpp:

```python
from onecomp.cpu import export_via_dequantize

export_via_dequantize("./model", "./model.gguf", qtype="Q4_K_M")
```

This **re-quantizes** the weights, so the GPTQ/QEP error correction is lost and
quality is comparable to a stock `Q4_K_M` GGUF. It requires the
`llama-quantize` binary (set `$LLAMA_QUANTIZE_BIN` or put it on `PATH`).

## Running inference

```python
from onecomp.cpu import LlamaCppModel

model = LlamaCppModel("./model.gguf", n_ctx=2048, n_threads=8)
print(model.generate("Fujitsu is", max_tokens=64, temperature=0.0))

# Streaming
for piece in model.stream("Fujitsu is", max_tokens=64):
    print(piece, end="", flush=True)
```

## Serving (one command, OpenAI-compatible)

To remove the deployment barrier, `onecomp-gguf serve` turns **any** GGUF *or*
**packed OneComp checkpoint** into an OpenAI-compatible HTTP API. If you point it
at a packed GPTQ/mixed checkpoint it auto-exports a cached `.gguf` (no
re-quantization) on first launch, then serves it. The server uses only the
Python standard library plus `llama-cpp-python` — no FastAPI/uvicorn — so any
environment that can run inference can also serve.

```bash
# Serve a packed quantized checkpoint directly (auto-exports to GGUF once):
onecomp-gguf serve --model ./model-gptq-4bit --port 8080

# …or an existing GGUF:
onecomp-gguf serve --model ./model.gguf --host 0.0.0.0 --port 8080
```

```bash
# Chat completions
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'

# Text completions (set "stream": true for SSE token streaming)
curl http://localhost:8080/v1/completions \
  -d '{"prompt":"Fujitsu is","max_tokens":32}'
```

Routes: `GET /v1/models`, `GET /health`, `POST /v1/completions`,
`POST /v1/chat/completions` (streaming via `"stream": true`). Chat uses the chat
template embedded in the GGUF. From Python:

```python
from onecomp.cpu import serve, resolve_to_gguf

gguf = resolve_to_gguf("./model-gptq-4bit")   # packed checkpoint -> cached GGUF
serve("./model-gptq-4bit", host="0.0.0.0", port=8080)
```

## Evaluation & inspection (CPU)

The `onecomp.cpu.eval` package provides CPU-only tools to validate a quantized
GGUF model — no GPU required.

```python
from onecomp.cpu import inspect_gguf, perplexity, benchmark
from onecomp.cpu.eval.inspect_gguf import format_report

# 1. Inspect per-tensor quant types, size and effective bits/weight
report = inspect_gguf("./model-mixed.gguf")
print(format_report(report))
print(report.per_block_types()[0])   # {'attn_q.weight': 'Q4_0', 'ffn_down.weight': 'Q2_K', ...}

# 2. Perplexity on held-out text
print(perplexity("./model.gguf", open("wiki.txt").read(), n_ctx=512))

# 3. CPU prefill / decode throughput
print(benchmark("./model.gguf", gen_tokens=64)[0])   # prefill / decode tok/s
```

**Parity vs PyTorch.** Because the direct export repacks the *same* GPTQ integer
codes, the GGUF and HF models run identical weights; the only gap is the kernel
(llama.cpp quantizes activations to 8-bit + fp32 accumulation vs HF float
matmuls). Validate a lossless export by feeding the **same token ids** to both
engines and comparing token-level agreement (it should be ~100%):

```python
from onecomp.cpu.eval.parity import (
    gguf_logits_for_tokens, teacher_forced_parity, gguf_greedy,
)
lc_logits = gguf_logits_for_tokens(model, token_ids)   # model built logits_all=True
parity = teacher_forced_parity(hf_logits, lc_logits)   # top-1 agreement, Pearson, MSE
```

Exact bit-for-bit logit equality between a CPU integer kernel and a GPU/float
kernel is not physically attainable; token-level agreement is the meaningful
target. Empirically a 1.5B 4-bit model reaches **100% top-1 agreement** and
identical greedy output between the HF-GPTQ and GGUF engines.

## Command line

```bash
# Export (direct, lossless)
onecomp-gguf export  --quantized-dir ./model-gptq-4bit --out ./model.gguf

# Export (mixed precision)
onecomp-gguf export  --quantized-dir ./model-mixed-gptq --out ./model.gguf --mode mixed

# Export (fallback)
onecomp-gguf export  --quantized-dir ./model --out ./model.gguf \
    --mode dequantize --qtype Q4_K_M

# Inference (add --stream to stream tokens)
onecomp-gguf run     --gguf ./model.gguf --prompt "Fujitsu is"

# Inspect / perplexity / benchmark
onecomp-gguf inspect --gguf ./model.gguf
onecomp-gguf ppl     --gguf ./model.gguf --text-file ./wiki.txt
onecomp-gguf bench   --gguf ./model.gguf --gen-tokens 64
```

## Rotation (QuaRot / SpinQuant) support

Rotation pre-processing fuses R1/R2/scaling into the weights **offline**, but
keeps an online Hadamard pre-hook on `down_proj` that llama.cpp cannot apply.
Because that Hadamard is orthonormal, the CPU exporter folds its **inverse**
into the `down_proj` weight (`onecomp.cpu.export.rotation.defold_down_proj_hadamard`),
so the resulting GGUF is mathematically equivalent **without any online op** and
runs correctly on stock llama.cpp. Rotated checkpoints are routed through the
dequantize fallback automatically (they are re-quantized, so the GPTQ codes are
not preserved, but the rotation is reproduced exactly).

## DBF support

DBF (Double Binary Factorization) has no GGUF block equivalent, so it is
exported through the fallback path: the dense weight is reconstructed from the
`DoubleBinaryLinear` factors (matching the PyTorch forward exactly) and then
converted/quantized. Use `qtype=None` to keep f16 (no extra loss) or e.g.
`Q4_K_M` for a smaller file.

## Notes

- Tied-embedding models (Qwen2.5, Gemma, …): the exporter re-ties `lm_head` to
  `embed_tokens` after dequantization, so the output projection is correct.
- **OneBit** is intentionally **not** supported.
- ARB / CQ / QBB / QUIP do not implement packed checkpoint saving in OneComp, so
  there is no checkpoint to export/serve for them.
