# API Reference

This section provides detailed API documentation for all public classes and functions in Fujitsu One Compression (OneComp).

## Top-level Exports

The following are available directly from `import onecomp`:

| Name                   | Type     | Description                              |
|------------------------|----------|------------------------------------------|
| `Runner`               | Class    | Main entry point for quantization        |
| `ModelConfig`          | Class    | Model and tokenizer configuration        |
| `QEPConfig`            | Class    | QEP configuration                        |
| `GPTQ`                 | Class    | GPTQ quantizer                           |
| `RTN`                  | Class    | RTN quantizer                            |
| `DBF`                  | Class    | DBF quantizer                            |
| `JointQ`               | Class    | JointQ quantizer                         |
| `QUIP`                 | Class    | QuIP quantizer                           |
| `ARB`                  | Class    | ARB quantizer                            |
| `CQ`                   | Class    | CQ quantizer                             |
| `QBB`                  | Class    | QBB quantizer                            |
| `Onebit`               | Class    | 1-bit quantizer                          |
| `QuantizedModelLoader` | Class    | Loader for saved quantized models        |
| `load_quantized_model` | Function | Shortcut for `QuantizedModelLoader.load_quantized_model` |
| `setup_logger`         | Function | Configure logging output                 |

## Module Structure

```
onecomp/
    runner.py              # Runner class
    model_config.py        # ModelConfig class
    qep/                   # QEP module
        _qep_config.py     # QEPConfig dataclass
    quantizer/             # Quantizer implementations
        _quantizer.py      # Quantizer base class, QuantizationResult
        gptq/              # GPTQ
        dbf/               # DBF
        rtn/               # RTN
        jointq/            # JointQ
        ...
    quantized_model_loader.py  # QuantizedModelLoader
    log.py                 # setup_logger
    utils/                 # Calibration, perplexity, accuracy utilities
```
