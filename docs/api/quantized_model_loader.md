# QuantizedModelLoader

Loader for quantized models saved by OneComp.

::: onecomp.quantized_model_loader.QuantizedModelLoader
    options:
      show_source: false

## Convenience Function

The top-level `load_quantized_model` is a shortcut:

```python
from onecomp import load_quantized_model

# Equivalent to QuantizedModelLoader.load_quantized_model(...)
model, tokenizer = load_quantized_model("./saved_model")
```
