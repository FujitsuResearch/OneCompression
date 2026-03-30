# QuantizedModelLoader

Loader for quantized models saved by OneComp.

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
model, tokenizer = load_quantized_model_pt("./saved_model_lora")
```
