# GPT-OSS (mixed_gptq)

GPT-OSS models (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`) are Mixture-of-Experts
(MoE) models with `swigluoai` activation and per-expert bias. They are served in
vLLM through the OneComp **`mixed_gptq`** plugin, but need a few extra steps beyond
the standard GPTQ + vLLM flow:

1. Quantize the MoE **experts as 4-bit GPTQ**, with `group_size=64`.
2. Apply two vLLM **runtime patches** before serving.
3. Disable the FP8 DeepGEMM warmup.

This page covers the full workflow. For the generic GPTQ + vLLM flow, see
[vLLM Inference](vllm-inference.md).

## Why GPT-OSS needs special handling

- **`hidden_size` = 2880 is not divisible by 128.** vLLM's Marlin MoE kernel is
  unavailable for this shape, so the experts are routed through the WNA16
  (`MoeWNA16Method`) grouped-weight path instead. That path halves the group
  size to 64 internally, so the experts **must be quantized with
  `group_size=64`** (2880 / 64 = 45) for the on-disk scales to match.
- **Experts carry gate/up/down bias**, which the stock WNA16 fused-MoE kernel
  does not support.
- **Experts use `swigluoai`**, not SiLU, which the stock `MoeWNA16Method`
  hard-asserts against.

OneComp provides a `MoeWNA16Method` subclass (`GptOssWNA16MoEMethod`) plus two
source patches to close these gaps.

## Requirements

Install vLLM with the `vllm` extra so `conch-triton-kernels` is present:

```bash
# uv users
uv sync --extra vllm

# pip users
pip install vllm conch-triton-kernels
```

On NVIDIA Blackwell (B200, sm100), Conch is required for the `mixed_gptq` linear
layers when the Marlin kernel cannot handle GPT-OSS weight shapes.

## 1. Quantize with 4-bit MoE experts

Quantize with `GPTQ(wbits=4)` and keep the experts 4-bit by passing
`moe_quant_experts=True` to `Runner`. `group_size=64` is required because
`hidden_size=2880` is not divisible by 128:

```python
from onecomp import Runner
from onecomp.model_config import ModelConfig
from onecomp.quantizer import GPTQ

runner = Runner(
    model_config=ModelConfig(model_id="openai/gpt-oss-20b"),  # or gpt-oss-120b
    quantizer=GPTQ(wbits=4, groupsize=64),  # required: 2880 % 128 != 0
    moe_quant_experts=True,                 # keep experts GPTQ INT4
)
runner.run()
runner.save_quantized_model("./gpt-oss-20b-mixed_gptq")
```

With `moe_quant_experts=True`, the experts are saved as per-expert
`GPTQLinear` tensors — the same on-disk layout as the attention projections:

```
model.layers.{L}.mlp.experts.{i}.gate_proj.{qweight,qzeros,scales,g_idx}
model.layers.{L}.mlp.experts.{i}.up_proj.{qweight,qzeros,scales,g_idx}
model.layers.{L}.mlp.experts.{i}.down_proj.{qweight,qzeros,scales,g_idx}
```

and remain listed in the saved `quantization_config`, so the `mixed_gptq` plugin
serves them 4-bit. Without the flag (the default), experts are dequantized and
fused into dense bf16 tensors — larger on disk and served full-precision.

!!! note "group_size"
    `group_size=64` is mandatory for the 4-bit expert path. `group_size=128` (or
    the channelwise `-1`) will reload with garbage text because the WNA16 kernel
    expects group size 64 for `hidden_size=2880`.


## 2. Save / load (Hugging Face format)

The checkpoint is a standard Hugging Face directory (sharded `*.safetensors` +
`config.json` + `quantization_config`). No special save call is required; the
runner writes it to `ONECOMP_SAVE_DIR`.

To sanity-check that the experts were saved as 4-bit (per-expert GPTQ) rather
than dense fused tensors:

```python
from onecomp.utils.unfuse_moe import verify_saved_moe_quant_checkpoint

n = verify_saved_moe_quant_checkpoint("./gpt-oss-20b-mixed_gptq")
print(f"per-expert qweight tensors: {n}")
```

It raises if the checkpoint accidentally contains dense fused `gate_up_proj`
tensors or deduplicated `$` keys.

## 3. Apply vLLM runtime patches

Before constructing `LLM(...)`, patch the installed vLLM:

```bash
python -m vllm_plugins.patches.apply_all
```

This applies two **idempotent, marker-guarded** source patches (safe to run
repeatedly; re-running is a no-op once patched):

| Patch | What it does |
|-------|--------------|
| `gpt_oss_gptq_moe` | Injects a `_load_weights_gptq_moe` per-expert GPTQ weight-loading path into vLLM's `GptOssModel.load_weights`, so per-expert `qweight/qzeros/scales/g_idx` tensors load into the fused `w13_*` / `w2_*` MoE params. |
| `gpt_oss_wna16_bias` | Adds per-expert bias to the WNA16 fused-MoE Triton kernel (`fused_moe_kernel_gptq_awq`), threads the bias through the dispatch, and forces the Triton path when a bias tensor is present (the compiled CUDA WNA16 kernel has no bias). |

!!! warning "Patch-verified vLLM version"
    These source patches anchor on specific source strings in vLLM's
    `fused_moe.py` / `gpt_oss.py`. They are **verified against vLLM 0.20.2**.
    The `--extra vllm` range (`vllm>=0.10,<0.22`) is broader than the verified
    set: on other versions the anchors may either fail to match (`apply_all`
    raises a `RuntimeError`, which is safe) or match a subtly changed block and
    produce unintended behavior. `apply_all` logs the installed vLLM version and
    emits a warning when it is outside the verified set — check serving output
    if you see that warning.

Use `--dry-run` to validate the patterns without writing files:

```bash
python -m vllm_plugins.patches.apply_all --dry-run
```

The expert routing itself (wrapping vLLM's `MoeWNA16Method` in
`GptOssWNA16MoEMethod`) is registered automatically by the `mixed_gptq` plugin
entry point — no extra step is needed.

## 4. Serve

Disable the FP8 DeepGEMM warmup (OneComp checkpoints do not need it) and load
the checkpoint through vLLM's offline `LLM` interface:

```bash
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
```

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="./gpt-oss-20b-mixed_gptq",
    dtype="float16",
    enforce_eager=True,
)
out = llm.generate(
    ["The capital of France is"],
    SamplingParams(max_tokens=16, temperature=0.0),
)
print(out[0].outputs[0].text)
```
