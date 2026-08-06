# `onecomp-vllm-v0-24-0-rocm`

Opt-in patch for ROCm + vLLM `0.24.x`. 
Kept out of the main `onecomp` `pyproject.toml`; it only activates when explicitly installed in this venv via the `vllm.general_plugins` entry point.

## Setup (`onecomp-lab` repo root)

```bash
VENV=envs/vllm/v0_24_0_rocm/.venv
uv venv --python 3.12 "$VENV"
source "$VENV/bin/activate"
uv pip install --python "$VENV/bin/python" \
    -e . -e envs/vllm/v0_24_0_rocm \
    vllm==0.24.0+rocm723 --extra-index-url https://wheels.vllm.ai/rocm/0.24.0/rocm723
# vLLM pulls torchaudio==2.9.0+eaa9e4e, which is ABI-incompatible with
# torch==2.11.0+gitd0c8b1f and breaks transformers import.  Remove it;
# GPTQ inference does not need audio.
uv pip uninstall --python "$VENV/bin/python" torchaudio flash-attn

VLLM_USE_DEEP_GEMM=0 VLLM_PLUGINS=onecomp_vllm_v0_24_0_rocm \
    python example/vllm_inference/example_gptq_vllm_inference.py
```

- `--python "$VENV/bin/python"` is required (avoids picking the top-level `.venv/`)
- Uninstall `torchaudio` and `flash-attn`: ABI-incompatible with the ROCm `torch` wheel
- `VLLM_USE_DEEP_GEMM=0` : disable FP8 DeepGEMM warmup (not needed for GPTQ; see `docs/user-guide/vllm-inference.md`)
- Add `VLLM_PLUGINS=onecomp_vllm_v0_24_0_rocm` only to suppress the broken `mixed_gptq` plugin warning

## What the patch does (matches `patch.py`)

Active only when `vllm` is importable, `is_rocm()` is true, and the base version is `0.24.*`
(`+rocm723` etc. stripped). Otherwise `apply()` no-ops with a single DEBUG line.

| # | Target | Change | Symptom without patch |
|---|--------|--------|----------------------|
| 1 | `AutoGPTQLinearMethod.process_weights_after_loading` | Add `+1 mod 16` to every packed nibble of `qzeros` when kernel is `TritonW4A16LinearKernel` | Token repetition (`sym=True`) or wrong outputs (`sym=False`) |
| 2 | `TritonW4A16LinearKernel.process_weights_after_loading` | Inject `permute_param_layout_(zp, input_dim=1, output_dim=0, packed_dim=0)` on `qzeros` | `AssertionError: qzeros shape mismatch` (AutoGPTQ layout flipped by unconditional `.t()`) |
| 3 | `AutoGPTQConfig.TYPE_MAP` | `setdefault((4, False), uint4)` and `setdefault((8, False), uint8)` | `ValueError` loading `sym=False` checkpoints |

All three wrappers are idempotent (`_onecomp_vllm_0_24_0_rocm_applied_*` markers skip re-wrapping).

## Verification

After setup, the TinyLlama-1.1B GPTQ example should produce coherent text. Without the patch,
output often collapses into repetition after 10–30 tokens on ROCm.

For Triton attention, pass `attention_config={"backend": "TRITON_ATTN"}` to `LLM()`.
`VLLM_ATTENTION_BACKEND` is not read by v0.24 (logged as unknown env var).

## Known blockers (outside this patch)

1. **flash-attn**: HIP ABI mismatch → uninstall; RoPE falls back to native PyTorch
2. **torchaudio**: `transformers` import fails with undefined symbols → uninstall (not needed for GPTQ)
3. **mixed_gptq plugin**: main `pyproject.toml` references `vllm...quantization.gptq` removed in v0.23+ → startup warning only
4. **venv mix-up**: always pass `--python` or `--active` to `uv`

## Layout

```
envs/vllm/v0_24_0_rocm/
├── README.md
├── pyproject.toml          # entry-point: onecomp_vllm_v0_24_0_rocm = ...patch:apply
└── src/onecomp_vllm_v0_24_0_rocm/
    ├── __init__.py
    └── patch.py
```
