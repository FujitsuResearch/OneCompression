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

# Load a safetensors model, including BlockWisePTQ / GlobalPTQ / GlobalPTQDistributed outputs
model, tokenizer = load_quantized_model("./saved_model")

# Keep the loaded model on CPU before running additional post-processes
model, tokenizer = load_quantized_model("./saved_model", device_map=None)

# Load a PyTorch .pt model with custom modules, e.g. LoRA-applied
model, tokenizer = load_quantized_model_pt("./saved_model_lora")
```
