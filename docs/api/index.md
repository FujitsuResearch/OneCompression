# API Reference

This section provides detailed API documentation for all public classes and functions in Fujitsu One Compression (OneComp).

## Top-level Exports

The following are available directly from `import onecomp`:

| Name                   | Type     | Description                              |
|------------------------|----------|------------------------------------------|
| `Runner`               | Class    | Main entry point for quantization        |
| `ModelConfig`          | Class    | Model and tokenizer configuration        |
| `QEPConfig`            | Class    | QEP configuration                        |
| `AutoBitQuantizer`     | Class    | AutoBit mixed-precision quantizer (ILP)  |
| `GPTQ`                 | Class    | GPTQ quantizer                           |
| `RTN`                  | Class    | RTN quantizer                            |
| `DBF`                  | Class    | DBF quantizer                            |
| `JointQ`               | Class    | JointQ quantizer                         |
| `QUIP`                 | Class    | QuIP quantizer                           |
| `ARB`                  | Class    | ARB quantizer                            |
| `CQ`                   | Class    | CQ quantizer                             |
| `QBB`                  | Class    | QBB quantizer                            |
| `Onebit`               | Class    | 1-bit quantizer                          |
| `RotatedModelConfig`   | Class    | ModelConfig for rotation-preprocessed models |
| `QuantizedModelLoader` | Class    | Loader for saved quantized models        |
| `load_quantized_model` | Function | Shortcut for `QuantizedModelLoader.load_quantized_model` |
| `load_quantized_model_pt` | Function | Shortcut for `QuantizedModelLoader.load_quantized_model_pt` |
| `prepare_rotated_model`| Function | Rotation preprocessing pipeline          |
| `setup_logger`         | Function | Configure logging output                 |

## Module Structure

```
onecomp/
    runner.py              # Runner class (includes auto_run)
    cli.py                 # CLI entry point (onecomp command)
    __main__.py            # python -m onecomp support
    model_config.py        # ModelConfig class
    rotated_model_config.py # RotatedModelConfig class
    pre_process/           # Rotation preprocessing pipeline
        prepare_rotated_model.py  # prepare_rotated_model()
    qep/                   # QEP module
        _qep_config.py     # QEPConfig dataclass
    quantizer/             # Quantizer implementations
        _quantizer.py      # Quantizer base class, QuantizationResult
        autobit/           # AutoBit (ILP mixed-precision)
        gptq/              # GPTQ
        dbf/               # DBF
        rtn/               # RTN
        jointq/            # JointQ
        ...
    quantized_model_loader.py  # QuantizedModelLoader
    log.py                 # setup_logger
    utils/                 # Calibration, perplexity, accuracy utilities
```
