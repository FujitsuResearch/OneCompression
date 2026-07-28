# QuantizedModelLoader

Loader for quantized models saved by OneComp.

On macOS, `load_quantized_model()` places the model on MPS when available
(CUDA > MPS > CPU via `get_default_device()`). Use Transformers `generate()` for
inference; vLLM requires Linux with an NVIDIA GPU. See the
[macOS / MPS guide](../user-guide/mps.md#inference-with-transformers).

::: onecomp.quantized_model_loader.QuantizedModelLoader
    options:
      show_source: false

## Convenience Functions

The top-level aliases provide shortcuts for both formats:

```python
from onecomp import load_quantized_model, load_quantized_model_pt

# Load a safetensors model (standard quantized, no LoRA)
model, tokenizer = load_quantized_model("./saved_model")

# Load a PyTorch .pt model (post-processed, e.g. LoRA-applied)
# Requires explicit opt-in: the .pt loader uses torch.load(weights_only=False),
# which can execute code from a malicious file (CWE-502). Only enable this for
# model.pt files from a fully trusted source.
model, tokenizer = load_quantized_model_pt(
    "./saved_model_lora", allow_unsafe_deserialization=True
)
```

!!! warning "Unsafe deserialization (.pt loader)"
    `load_quantized_model_pt()` loads `model.pt` with
    `torch.load(..., weights_only=False)`. Because PyTorch `.pt` checkpoints use
    Python `pickle`, a maliciously crafted `model.pt` can execute arbitrary code
    during loading (CWE-502). The method refuses to load unless you pass
    `allow_unsafe_deserialization=True`. Only opt in for models you produced
    yourself or obtained from a fully trusted source. For untrusted or
    third-party models, prefer the safetensors-based `load_quantized_model()`,
    which does not execute code.


!!! note "Research/development use only"
    `load_quantized_model_pt()` (and the `.pt` save/load path in general)
    is intended for **research and development** only -- for example, to
    quickly experiment with a new post-process before it has a
    safetensors-compatible `load_quantized_model()` implementation. It is
    **not recommended** for general or production use; prefer the
    safetensors-based `load_quantized_model()`, which is HF-compatible and
    does not execute code.
