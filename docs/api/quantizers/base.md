# Base Classes

## Quantizer

Abstract base class for all quantizers. Defines the common interface and shared functionality.

::: onecomp.quantizer._quantizer.Quantizer
    options:
      show_source: false
      members:
        - __post_init__
        - setup
        - quantize
        - quantize_layer
        - quantize_with_qep
        - save_results
        - load_results
        - apply_results_to_model
        - get_quant_config
        - create_inference_layer

## QuantizationResult

Base dataclass for quantization results returned by each layer.

::: onecomp.quantizer._quantizer.QuantizationResult
    options:
      show_source: false

## ResultLoader

Loader for reading saved quantization results without performing quantization.

::: onecomp.quantizer._quantizer.ResultLoader
    options:
      show_source: false
