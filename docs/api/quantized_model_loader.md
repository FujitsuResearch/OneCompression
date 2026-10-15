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

# Load a safetensors model, including BlockWisePTQ / GlobalPTQ / GlobalPTQDistributed
# outputs. If a LoRA adapter sidecar (lora_adapter/) is present it is auto-detected
# and applied, so LoRA models saved with save_quantized_model() load here too.
model, tokenizer = load_quantized_model("./saved_model")

# Keep the loaded model on CPU before running additional post-processes
model, tokenizer = load_quantized_model("./saved_model", device_map=None)

# Load a legacy PyTorch .pt model (whole-object torch.save; prefer the
# safetensors load_quantized_model() above, including for LoRA).
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

## Low-memory loading

`load_quantized_model()` loads supported checkpoints without a dense copy of the
model in host memory. It plans the load from the safetensors headers on an empty
(meta) model, applying the same key resolution, checks and dtype rules as
before, and then reads the tensors one at a time onto their final device.
Existing calls need no change.

A checkpoint is loaded this way (streamed) when all of the following hold:

- `model_type` is `qwen3` or `llama`, and its config and model classes are the
  built-in Transformers classes (no custom registration).
- `quant_method` is `gptq` or `mixed_gptq`, at least one layer is quantized,
  and the model is not rotation-preprocessed.
- No LoRA adapter sidecar is present.
- No tensor is stored in more than one root `*.safetensors` file. As before,
  the root `*.safetensors` files are read and `model.safetensors.index.json`
  is ignored.
- The checkpoint holds exactly the tensors of the model; only a tied
  `lm_head.weight` may be omitted.
- `device_map` is `None` or `""` (keep the model on CPU), or another string for
  which automatic placement puts the whole model on the CPU or on one CUDA
  device.

`load_strategy` decides what happens to the other checkpoints:

| `load_strategy` | Supported checkpoints | Other checkpoints |
| --- | --- | --- |
| `"auto"` (default) | Streamed | Loaded by the legacy loader after a `WARNING` that gives the reason |
| `"streaming"` | Streamed | `StreamingUnsupportedError` before any tensor value is read |
| `"legacy"` | Loaded by the legacy loader | Loaded by the legacy loader |

The legacy loader builds a dense model on CPU and reads the whole checkpoint,
so it needs as much host RAM as in earlier versions. An error raised after
streaming has started is passed to the caller; the load is not retried with the
legacy loader.

```python
from onecomp import StreamingUnsupportedError, load_quantized_model

try:
    model, tokenizer = load_quantized_model("./saved_model", load_strategy="streaming")
except StreamingUnsupportedError as error:  # a ValueError subclass, so catch it first
    print(f"Needs the legacy loader: {error.reason}")
except ValueError as error:
    print(f"Invalid checkpoint or arguments: {error}")
```

::: onecomp.quantized_model_loader.StreamingUnsupportedError
    options:
      show_source: false

!!! note "Memory requirements"
    When the whole model is placed on a GPU, streaming keeps the host memory
    used for loading to about the largest single tensor on top of the
    process's baseline. When the model is kept on CPU (`device_map=None`), the
    final weights stay in host memory. `load_strategy="streaming"` selects the
    loading method; it does not guarantee a memory limit or prevent
    out-of-memory errors, for example during inference.
