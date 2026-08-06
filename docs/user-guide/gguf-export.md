# GGUF Export & Hugging Face Hub

OneComp can convert Hugging Face checkpoints (including OneComp save
directories with dequantized FP16 weights) into the
[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) v3 format
used by llama.cpp-based runtimes such as [Ollama](https://ollama.com/), and
publish save directories to the Hugging Face Hub with a generated model card.

The GGUF writer is implemented inside OneComp with the standard library and
numpy only, so no extra dependency (such as the `gguf` package) is required.

## Supported architectures

| Architecture | Tokenizer | Status |
|--------------|-----------|--------|
| Llama (e.g. TinyLlama, Llama-2) | SentencePiece (`tokenizer.model`) | ✅ Verified (TinyLlama-1.1B, Llama-2-7B) |
| Llama-3 style | BPE (`tokenizer.json`, `pre` = `llama-bpe`) | ✅ Supported (tested with synthetic checkpoints) |
| Qwen2 (e.g. Qwen2.5) | BPE (`tokenizer.json`) | ✅ Verified (Qwen2.5-0.5B-Instruct) |

Llama-family checkpoints use `tokenizer.model` (SentencePiece) when it is
present next to the weights; otherwise the BPE vocabulary and merges are
extracted from `tokenizer.json` (Llama-3 style checkpoints ship only the
fast tokenizer).  Qwen2 always uses the `tokenizer.json` path.  The
SentencePiece and Qwen2 paths have been validated against the original
Hugging Face tokenizers (vocabulary, merges, and token-id round trips).

When `config.json` declares a list of `eos_token_id` (e.g. Llama-3.1's
`<|end_of_text|>` / `<|eom_id|>` / `<|eot_id|>`), the extra stop ids are
mapped onto `tokenizer.ggml.eot_token_id` / `eom_token_id` so llama.cpp
still terminates turns correctly.  Tokenizers that define more tokens than
the model's embedding table (`config.json`'s `vocab_size`) are rejected
instead of producing a GGUF that mis-tokenizes.

Only FP16/FP32 (and BF16, converted to F16) safetensors checkpoints can be
exported.  For quantized OneComp models, save dequantized weights first
(e.g. with `Runner.save_dequantized_model`); checkpoints containing packed
quantized tensors are rejected.

## Installation

```bash
pip install 'onecomp[llamacpp]'        # installs gguf + llama-cpp-python
# or with uv:
uv sync --extra cpu --extra llamacpp
```

`llama-cpp-python` provides prebuilt CPU wheels, so no C++ toolchain is required
for inference. The direct export path additionally uses llama.cpp's pure-Python
`convert_hf_to_gguf.py` to build the model metadata/tokenizer; it is fetched
automatically (a shallow `git clone`) or taken from `$LLAMA_CPP_DIR` if set.

## Python API

```python
from onecomp import GGUFExportConfig, export_gguf

export_gguf(
    "./TinyLlama-1.1B-Chat-v1.0",           # HF model directory
    GGUFExportConfig(out_path="./tinyllama-f16.gguf"),
)
```

`GGUFExportConfig` fields:

| Field | Default | Description |
|-------|---------|-------------|
| `out_path` | (required) | Output GGUF file path |
| `dtype` | `"f16"` | Weight matrix dtype (`"f16"` or `"f32"`; 1-D tensors are always F32) |
| `architecture` | `None` | `"llama"` / `"qwen2"`, auto-detected from `config.json` when `None` |
| `name` | `None` | `general.name` metadata, defaults to the directory name |

## CLI

```bash
# Quantize, save, and additionally export a GGUF F16 file into the save dir
onecomp TinyLlama/TinyLlama-1.1B-Chat-v1.0 --save-dir ./tinyllama-quant --format gguf

# Quantize, save, and push to the Hugging Face Hub (private repo)
onecomp TinyLlama/TinyLlama-1.1B-Chat-v1.0 --save-dir ./tinyllama-quant \
    --push-to-hub your-name/tinyllama-onecomp
```

Both options require an explicit `--save-dir`.  The GGUF file is written as
`<save-dir>/<save-dir-name>-f16.gguf`.

With `--format gguf`, the CLI first writes dequantized FP16 weights to a
temporary directory (the save directory itself contains packed quantized
tensors, which GGUF F16 export cannot consume), converts them to GGUF, and
removes the temporary directory again.  Expect transient disk usage of one
extra FP16 copy of the model during the conversion.

## Using the exported model with vLLM

vLLM can serve the exported single-file GGUF directly.  Since vLLM
0.24 the GGUF support lives in the out-of-tree
[vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin)
(`pip install vllm-gguf-plugin`); earlier versions bundle it in-tree.
Pass the original Hugging Face model directory as the tokenizer (the
tokenizer conversion from GGUF metadata is slow and less reliable),
and use `dtype="float16"` — the GGUF quantization method supports
FP16/FP32 only:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="./tinyllama-f16.gguf",              # absolute path recommended
    tokenizer="./TinyLlama-1.1B-Chat-v1.0",    # original HF directory
    dtype="float16",
)
outputs = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0))
```

Or with the OpenAI-compatible server:

```bash
vllm serve ./tinyllama-f16.gguf \
    --tokenizer ./TinyLlama-1.1B-Chat-v1.0 --dtype float16
```

Verified on vLLM 0.24.0 + vllm-gguf-plugin 0.0.2 (B200): TinyLlama-1.1B
F16 GGUF greedy generation matches the original HF checkpoint token for
token; Qwen2.5-0.5B F16 loads and generates equivalent-quality text
(exact token match is not expected across the vLLM and HF runtimes).
Known plugin issue (0.0.2, with vLLM 0.24): models whose embedding
layer is constructed without a module prefix (Llama, Qwen2) fail with
``KeyError: 'embed_tokens.weight'`` because the F16 ``token_embd``
tensor cannot be matched against ``unquantized_modules``; force the
unquantized embedding method until the fix lands upstream:

```python
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm_gguf_plugin.quantization.config import GGUFConfig

_original = GGUFConfig.get_quant_method

def _patched(self, layer, prefix):
    if isinstance(layer, VocabParallelEmbedding) and not prefix:
        return UnquantizedEmbeddingMethod()
    return _original(self, layer, prefix)

GGUFConfig.get_quant_method = _patched
```

Second plugin limitation (0.0.2, with vLLM 0.24), llama architecture
only: llama GGUF files store the Q/K projection rows in the interleaved
("NORM") RoPE order — the same permutation `convert_hf_to_gguf.py` and
`export_gguf` apply — but vLLM hardcodes neox-style RoPE for llama and
the plugin does not un-permute the rows on load, which degrades
generations.  Switch the rotary embedding to interleaved style until
the plugin handles the permutation (qwen2 uses NEOX RoPE and needs no
patch):

```python
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models import llama

def _rotary_patched(self, config, quant_config):
    self.rotary_emb = get_rope(
        self.head_dim,
        max_position=self.max_position_embeddings,
        rope_parameters=getattr(config, "rope_parameters", None),
        is_neox_style=False,  # match the GGUF interleaved layout
    )

llama.LlamaAttention._init_rotary_emb = _rotary_patched
```

With both patches applied, TinyLlama-1.1B F16 GGUF greedy generation
under vLLM matches the original HF checkpoint token for token
(verified on vLLM 0.24.0 + vllm-gguf-plugin 0.0.2, B200).

## Using the exported model with Ollama

Create a `Modelfile` next to the exported GGUF file:

```text
FROM ./tinyllama-f16.gguf
```

Then register and run the model:

```bash
ollama create tinyllama-onecomp -f Modelfile
ollama run tinyllama-onecomp "Hello!"
```

## Publishing to the Hugging Face Hub

```python
from onecomp import generate_model_card, push_to_hub

card = generate_model_card(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    recipe={"method": "AutoBit + QEP", "wbits": 4, "groupsize": 128},
    results={"perplexity (wikitext2)": 8.12, "accuracy (lambada)": 0.65},
)
push_to_hub("./tinyllama-quant", "your-name/tinyllama-onecomp", model_card=card)
```

`generate_model_card` produces a Markdown card with YAML frontmatter
(`license`, `base_model`, `tags: [onecomp, quantized, gptq]`), a quantization
recipe table, and an evaluation results table.  `push_to_hub` writes the card
as `README.md` (unless one already exists), creates the repository (private
by default), and uploads the whole save directory.  Authentication uses the
`token` argument, the cached `huggingface-cli login`, or the `HF_TOKEN`
environment variable.

## Validating an exported file

A lightweight reader is included for validation:

```python
from onecomp.export import GGUFReader

reader = GGUFReader("./tinyllama-f16.gguf")
print(reader.metadata["general.architecture"])   # "llama"
print(len(reader.tensors))                       # e.g. 201
print(reader.tensor("token_embd.weight").shape)  # (32000, 2048)
```

## Loading a GGUF file back into transformers

F16/F32 GGUF files produced by OneComp can be restored to a Hugging
Face state dict, e.g. to double-check a conversion end to end:

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from onecomp.export import load_gguf_state_dict

state_dict = load_gguf_state_dict("./tinyllama-f16.gguf")
config = AutoConfig.from_pretrained("./TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_config(config).to(torch.float16)
model.load_state_dict(state_dict, strict=False)  # lm_head is absent when tied
```

The restore is exact: every tensor is bit-identical to the original
checkpoint cast to float16 (verified for TinyLlama-1.1B and
Qwen2.5-0.5B, including greedy-generation token equality).  For models
with `tie_word_embeddings` the file has no `output.weight`, so load
with `strict=False` and call `model.tie_weights()`.

## Implementation Notes

This section documents the internals of the export pipeline for
contributors and for anyone auditing the produced files.

### GGUF v3 file layout

The writer produces the standard little-endian GGUF v3 layout
([specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)):

```text
offset 0
+----------------------------------------------------------+
| magic "GGUF" (u32 = 0x46554747) | version (u32 = 3)      |
| tensor_count (u64)              | metadata_kv_count (u64) |
+----------------------------------------------------------+
| metadata KV pairs (repeated kv_count times)               |
|   key: string (u64 length + UTF-8 bytes, no NUL)          |
|   value_type (u32) + payload                              |
|   arrays: element_type (u32) + count (u64) + elements     |
+----------------------------------------------------------+
| tensor infos (repeated tensor_count times)                |
|   name: string                                            |
|   n_dims (u32) + dims[n_dims] (u64, ne order:             |
|     fastest-varying dimension first)                      |
|   ggml_type (u32: 0 = F32, 1 = F16)                       |
|   offset (u64, relative to the data section start,        |
|     multiple of general.alignment)                        |
+----------------------------------------------------------+
| zero padding up to the next multiple of general.alignment |
+----------------------------------------------------------+
| tensor data, each tensor padded to general.alignment      |
+----------------------------------------------------------+
```

Key points that are easy to get wrong and are covered by tests:

- Metadata value-type codes follow the specification exactly
  (`UINT8 = 0` ... `FLOAT64 = 12`; notably `BOOL = 7`, `STRING = 8`,
  `ARRAY = 9`, and the 64-bit types 10-12 after `ARRAY`).
- Strings are a `u64` byte length followed by UTF-8 bytes with **no**
  NUL terminator, both for metadata keys/values and tensor names.
- Tensor dimensions are stored in `ne` order (fastest-varying first),
  i.e. the *reverse* of the numpy/PyTorch shape: a `(32000, 2048)` HF
  embedding is written as `dims = [2048, 32000]`.
- Tensor offsets are relative to the start of the data section, which
  begins at the first multiple of `general.alignment` (default 32)
  after the last tensor info.  Every tensor is zero-padded to the
  alignment.

### Why a built-in writer and reader

The writer (`onecomp/export/gguf_writer.py`, ~280 lines) and the
validation reader (`onecomp/export/gguf_reader.py`) only use `struct`
and numpy, so `onecomp` gains GGUF interoperability without adding the
`gguf` package (and its version churn) as a dependency.  Only the F32
and F16 tensor types are implemented because OneComp exports
dequantized checkpoints; quantized GGUF types (`Q4_K`, i-quants, ...)
are produced by re-quantizing the F16 file with `llama-quantize`.

The reader is intentionally minimal (header, metadata, tensor
directory, on-demand tensor loads with offset/alignment validation) and
is used by the test-suite; it is not a general-purpose GGUF loader.

### SentencePiece parser coverage

Llama-family vocabularies are read directly from the serialized
`sentencepiece.ModelProto` (`tokenizer.model`) with a minimal protobuf
wire-format parser, so the `sentencepiece` package is not required.
The parser understands all four wire types used by protobuf
(varint, fixed64, length-delimited, fixed32), skips unknown fields, and
decodes only `ModelProto.pieces` (field 1) with its `piece` (string),
`score` (float, default 0.0), and `type` (varint, default `NORMAL`)
subfields.  Piece types map 1:1 to GGUF token types
(`NORMAL = 1`, `UNKNOWN = 2`, `CONTROL = 3`, `USER_DEFINED = 4`,
`UNUSED = 5`, `BYTE = 6`), so byte-fallback tokens (`<0x00>`...`<0xFF>`)
keep type `BYTE`.

Added tokens that extend the base SentencePiece vocabulary
(`added_tokens` in `tokenizer.json`, or `added_tokens.json`) are merged
by id before padding; ids already covered by the SentencePiece model
are left untouched.  Remaining ids up to `config.vocab_size` are filled
with `[PAD<n>]` placeholders of type `UNUSED`.

### Tokenizer metadata by architecture

| Metadata key | Llama (SentencePiece) | Qwen2 (BPE) |
|--------------|----------------------|-------------|
| `tokenizer.ggml.model` | `llama` | `gpt2` |
| `tokenizer.ggml.pre` | (not written) | `qwen2` |
| `tokenizer.ggml.tokens` | pieces by id, padded to `vocab_size` | vocab by id + added tokens, padded to `vocab_size` (e.g. 151936 vs 151665 real tokens for Qwen2.5) |
| `tokenizer.ggml.scores` | piece scores | (not written) |
| `tokenizer.ggml.token_type` | piece types from the model | `NORMAL`, added specials `CONTROL`, others `USER_DEFINED` |
| `tokenizer.ggml.merges` | (not written) | `"left right"` strings in `tokenizer.json` order |
| `tokenizer.ggml.unknown_token_id` | id of the `UNKNOWN` piece | (not written) |
| `tokenizer.ggml.bos_token_id` / `eos_token_id` | from `config.json` | from `config.json` (first entry if `eos_token_id` is a list) |
| `tokenizer.ggml.add_bos_token` / `add_eos_token` | from `tokenizer_config.json` when boolean | same |
| `tokenizer.chat_template` | from `tokenizer_config.json` when present | same |

### Tensor name mapping

| Hugging Face name | GGUF name |
|-------------------|-----------|
| `model.embed_tokens.weight` | `token_embd.weight` |
| `model.norm.weight` | `output_norm.weight` |
| `lm_head.weight` | `output.weight` (omitted when `tie_word_embeddings` is true; llama.cpp falls back to `token_embd.weight`) |
| `model.layers.<i>.self_attn.{q,k,v}_proj.{weight,bias}` | `blk.<i>.attn_{q,k,v}.{weight,bias}` |
| `model.layers.<i>.self_attn.o_proj.weight` | `blk.<i>.attn_output.weight` |
| `model.layers.<i>.mlp.{gate,up,down}_proj.weight` | `blk.<i>.ffn_{gate,up,down}.weight` |
| `model.layers.<i>.input_layernorm.weight` | `blk.<i>.attn_norm.weight` |
| `model.layers.<i>.post_attention_layernorm.weight` | `blk.<i>.ffn_norm.weight` |
| `*rotary_emb.inv_freq` | skipped (recomputed by the runtime) |

Any other tensor name raises a `ValueError` instead of being silently
dropped, so unsupported architectures fail fast.  1-D tensors (norms,
biases) are always stored as F32; 2-D weight matrices use the
configured `dtype`.  BF16 sources are clamped to the finite F16 range
(±65504) before the narrowing cast so that no ±inf values reach the
file.

### Known limitations and future work

- Only F16/F32 output types; no direct quantized GGUF output
  (`Q4_K_M`, `Q8_0`, i-quants, ...).  Use `llama-quantize` on the
  exported F16 file.
- Architectures: `llama` and `qwen2` only.  Notably Qwen3
  (`q_norm`/`k_norm` tensors) and multimodal models are not mapped yet.
- RoPE scaling metadata (`rope.scaling.*`) is not exported; models
  relying on YaRN/linear scaling will use their base context window
  (a warning is logged when `config.json` declares `rope_scaling`).
- Packed OneComp checkpoints (GPTQ `qweight`/`qzeros` etc.) are
  rejected with a pointer to `Runner.save_dequantized_model`; the CLI
  handles this automatically via a temporary dequantized copy.
