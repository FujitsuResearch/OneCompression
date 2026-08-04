# `onecomp.cpu` — CPU inference via llama.cpp / GGUF

This subpackage exports OneComp-quantized models to **GGUF** and runs them on
the CPU with [llama.cpp](https://github.com/ggml-org/llama.cpp) (via
`llama-cpp-python`).

## Layout

```
onecomp/cpu/
├── export/            # GGUF export
│   ├── blocks.py        lossless GPTQ-code -> GGUF legacy-block packing
│   ├── checkpoint.py    read an OneComp GPTQ checkpoint -> GPTQLayer
│   ├── dequantize.py    GPTQ checkpoint -> dense fp16 HF model
│   ├── skeleton.py      build metadata/tokenizer skeleton GGUF + stitch tensors
│   ├── direct.py        direct, lossless GPTQ -> GGUF  (preferred)
│   └── fallback.py      dequantize -> llama-quantize   (universal, re-quantizes)
├── eval/              # CPU-side evaluation
│   ├── inspect_gguf.py  per-tensor quant types / size / effective bit-width
│   ├── perplexity.py    CPU perplexity on text
│   ├── parity.py        HF (PyTorch) vs GGUF (llama.cpp) output agreement
│   └── benchmark.py     CPU prefill / decode throughput
├── inference.py       LlamaCppModel (generate / stream / chat / logits)
├── serve.py           one-command OpenAI-compatible CPU server (stdlib only)
├── llama_tooling.py   locate & run convert_hf_to_gguf.py / llama-quantize
└── cli.py             `onecomp-gguf` CLI (export / run / inspect / ppl / bench / serve)
```

Mixed-precision GGUF export lives in the top-level **`llamacpp_plugins/`** package
(the llama.cpp counterpart of `vllm_plugins/`).

## Install

```bash
export CMAKE_ARGS="-DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON -DGGML_LTO=ON"
pip install 'onecomp[llamacpp]'     # gguf + llama-cpp-python
# or, with uv:
uv sync --extra cpu --extra llamacpp
```

`llama-cpp-python` ships prebuilt CPU wheels (no C++ build needed). The direct
export path only additionally needs llama.cpp's pure-Python
`convert_hf_to_gguf.py`, which is fetched automatically (shallow `git clone`) or
taken from `$LLAMA_CPP_DIR`.

## Two export paths

### 1. Direct GPTQ → GGUF (recommended; lossless)

`convert_gptq_to_gguf` unpacks the GPTQ integer codes and writes them straight
into GGUF legacy blocks **without re-quantizing**, so the GPTQ/QEP error
correction is preserved:

| GPTQ layer            | GGUF block |
|-----------------------|------------|
| 4-bit symmetric       | `Q4_0`     |
| 4-bit asymmetric      | `Q4_1`     |
| 8-bit symmetric       | `Q8_0`     |

For **llama**-architecture models the Q/K rows (and their scales/zeros) are
additionally permuted to llama.cpp's interleaved RoPE layout — the same
permutation `convert_hf_to_gguf.py` applies — so the packing stays lossless.

Constraints: `actorder=False`, group size a multiple of 32 (e.g. 128) or `-1`.
Layers that do not qualify (2/3-bit, actorder, mixed unsupported bitwidths) are
left in fp16 in the skeleton; use the fallback path for those.

```python
from onecomp.cpu import convert_gptq_to_gguf
convert_gptq_to_gguf("./model-gptq-4bit", "./model.gguf")
```

### 2. Dequantize → llama-quantize (fallback; re-quantizes)

`export_via_dequantize` reconstructs fp16 weights and uses
`convert_hf_to_gguf.py` + `llama-quantize`. Works for any GPTQ checkpoint but
discards the GPTQ error correction (quality ≈ stock `Q4_K_M`). Needs the
`llama-quantize` binary (`$LLAMA_QUANTIZE_BIN` / PATH).

```python
from onecomp.cpu import export_via_dequantize
export_via_dequantize("./model", "./model.gguf", qtype="Q4_K_M")
```

## CPU inference

```python
from onecomp.cpu import LlamaCppModel
model = LlamaCppModel("./model.gguf", n_ctx=2048)
print(model.generate("Fujitsu is", max_tokens=64))
```

## Serve (OpenAI-compatible, one command)

Point `serve` at a GGUF *or* a packed OneComp checkpoint (auto-exported to a
cached GGUF on first use) to expose `/v1/chat/completions`,
`/v1/completions` and `/v1/models`. Standard-library only — no FastAPI/uvicorn.

```bash
onecomp-gguf serve --model ./model-gptq-4bit --port 8080   # packed checkpoint or .gguf
```


## Evaluation (CPU)

```python
from onecomp.cpu import inspect_gguf, perplexity, benchmark
from onecomp.cpu.eval.inspect_gguf import format_report

print(format_report(inspect_gguf("./model.gguf")))   # per-tensor types & sizes
print(perplexity("./model.gguf", open("wiki.txt").read(), n_ctx=512))
print(benchmark("./model.gguf", gen_tokens=64)[0])    # prefill/decode tok/s
```

Parity between the PyTorch (HF GPTQ) and llama.cpp (GGUF) engines — the honest
way to validate a *lossless* export (token-level agreement should be ~100%):

```python
from onecomp.cpu.eval.parity import (
    gguf_logits_for_tokens, teacher_forced_parity, gguf_greedy,
)
# feed the SAME token ids to both engines and compare argmax / correlation
```

## CLI

```bash
onecomp-gguf export  --quantized-dir ./model-gptq-4bit --out ./model.gguf
onecomp-gguf export  --quantized-dir ./model-mixed     --out ./model.gguf --mode mixed
onecomp-gguf run     --gguf ./model.gguf --prompt "Fujitsu is" --stream
onecomp-gguf inspect --gguf ./model.gguf
onecomp-gguf ppl     --gguf ./model.gguf --text-file ./wiki.txt
onecomp-gguf bench   --gguf ./model.gguf --gen-tokens 64
```
