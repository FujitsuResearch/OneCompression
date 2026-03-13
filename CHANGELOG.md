# Change log

## [v0.3.6] 2026-03-12

### Completion of Save/Load Pipeline

- Added new `QuantizedModelLoader` class (`quantized_model_loader.py`)
  - Automatically detects quantization config (GPTQ/DBF) from `config.json` and loads the model
  - Reads state_dict from safetensors, replaces layers with quantized layers, and loads into an empty model
  - Supports automatic device placement via `accelerate`
  - Top-level API: exported as `onecomp.load_quantized_model()`
- Added `GPTQLinear.from_saved_state()` (reconstructs layer from safetensors state_dict)
- Added `DoubleBinaryLinear.from_saved_state()` (same as above)
- Revised `config.json` output format to enable direct inference with vLLM
  - Added list of quantized layer names to `modules_in_block_to_quantize`

### Forward Implementation for `DoubleBinaryLinear` and `GPTQLinear`

- `GPTQLinear.forward()`: Unpacks bit-packed weights → dequantizes → infers via `F.linear()` (fast path when using GemLite)
- `DoubleBinaryLinear.forward()`: Implements 5-stage pipeline (scaling0 → binary_B → scaling2 → binary_A → scaling4) (GemLite compatible)

### Expansion of Unit Tests

- Added new common test base class `BaseQuantizeSpec` (`test_module.py`)
  - `test_quantize_layer_returns`: Validates type, shape, device, and dtype of quantization results (CPU/CUDA)
  - `test_quantize_layer_reproducibility`: Validates reproducibility with the same seed
  - `test_parameters_boundary`: Confirms correct behavior with boundary parameter values
  - `test_parameters_abnormal_values_raise`: Confirms exceptions are raised for abnormal parameters
  - `test_cpu_gpu_output_match`: Validates that CPU/GPU quantization results match
  - `test_quantize_error`: Validates quantization error is within tolerance on a 2-layer model
  - `test_forward_error`: Validates forward accuracy of inference layer (dequantized output vs inference layer output)
- Added dedicated test classes for GPTQ and DBF (`test_gptq.py`, `test_dbf.py`)

### Fixes to `DBF` and `GPTQ` Quantizers

- Added parameter validation mechanism via `validate_params()` during `setup()` for `DBF` and `GPTQ`
- Unified and revised dtype (FP16/INT32) and device (CPU) of quantization results

### Build System Updates

- Migrated package and project management to `uv` and `pyproject.toml`.
- Applied `black` linter to scripts.

### QEP Module Refactoring

- Added `QEPConfig` dataclass (`onecomp/qep/_qep_config.py`)
- Extracted `quantize_with_qep` logic into standalone function (`onecomp/qep/_quantize_with_qep.py`)
- Added `general` flag to `QEPConfig` for dispatching between generic and architecture-aware implementations
- Added stub for architecture-aware QEP quantization (`onecomp/qep/_quantize_with_qep_arch.py`)
- Implemented architecture-aware QEP quantization with block-wise sequential pipeline (`onecomp/qep/_quantize_with_qep_arch.py`)
  - Added helper functions: `_get_blocks`, `get_blocks_and_inputs`, `make_grouped_module`, `compute_hessian_and_crossterm`, `forward_input`
  - Added `Catcher` class for capturing input activations of transformer blocks
  - Groups layers sharing the same input activations for efficient Hessian/cross-term computation
- Extended `Quantizer.quantize_with_qep()` and `adjust_weight()` to accept precomputed `hessian` and `delta_hatX` (`onecomp/quantizer/_quantizer.py`)
- Fixed `_record_quantization_error` to handle `quant_input_activation=None` for architecture-aware QEP (`onecomp/quantizer/_quantizer.py`)
- Fixed architecture-aware QEP to respect `num_layers` and layer selection by checking `quantizer.module_to_name` (`onecomp/qep/_quantize_with_qep_arch.py`)
- Fixed architecture-aware QEP to support `exclude_layer_keywords`: excluded layers are quantized without weight correction (`onecomp/qep/_quantize_with_qep_arch.py`)
- Added consistency test between generic and architecture-aware QEP implementations (`tests/onecomp/test_qep_general_consistency.py`)
- Changed `QEPConfig.general` default from `True` to `False` (architecture-aware implementation is now the default)

## [v0.3.5] 2026-03-05

- Based on v0.3.4 codebase
- Difference from v0.3.4: Changed comments to English

