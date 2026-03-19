# Change log

## [v0.3.8] 2026-03-19

### Python Version Constraint

- Restricted `requires-python` to `">=3.12, <3.14"` in `pyproject.toml`
  - PyTorch does not yet provide wheels for Python 3.14, causing `uv sync` to fail when uv auto-selects CPython 3.14
- Updated `uv.lock` to reflect the new Python version constraint

### Documentation

- Added "Building Documentation Locally" section to `README.md`
  - Previously this information was only available inside the built docs, creating a chicken-and-egg problem

### Bug Fix: Onebit Quantizer

- Fixed `Onebit` to declare `flag_calibration=True` and `flag_hessian=True` (`onecomp/quantizer/onebit/_onebit.py`)
  - Previously, Onebit computed the Hessian internally from `input` despite declaring all flags as `False`, causing a crash when used through `quantize_without_calibration` or chunked quantization paths
  - Now uses the Hessian provided by the Runner, consistent with other calibration-based quantizers (GPTQ, DBF, QUIP)

### Quantizer Signature Consistency

- Added `input=None` default to `quantize_layer` in `RTN`, `CQ`, `QBB` (`onecomp/quantizer/{rtn,cq,qbb}/`)
  - Aligns with the base `Quantizer.quantize_layer(self, module, input=None, hessian=None)` signature
  - Enables these quantizers to be used in `Runner(quantizers=[...])` via the chunked quantization path
- Added `input=None, hessian=None` defaults to `Onebit.quantize_layer` for the same reason

## [v0.3.7] 2026-03-16

### GPU Memory Optimization for Architecture-aware QEP

- Added `device` field to `QEPConfig` (`onecomp/qep/_qep_config.py`)
  - Specifies the GPU device for block-wise QEP computation (default: `"cuda:0"`)
  - Eliminates dependency on `model_config.device` and supports multi-GPU environments
- Added `device_map` parameter to `ModelConfig.load_model()` (`onecomp/model_config.py`)
  - Allows overriding the device placement at load time without affecting existing callers
- Optimized `run_quantize_with_qep_arch` to avoid loading the entire model onto GPU (`onecomp/qep/_quantize_with_qep_arch.py`)
  - Model is now loaded on CPU via `load_model(device_map="cpu")`
  - Calibration data is prepared on CPU
  - Only individual transformer blocks are moved to GPU during processing
- Added `StopForward` exception and modified `Catcher` to halt the forward pass immediately after capturing first-block inputs, avoiding unnecessary computation through remaining layers (`onecomp/qep/_quantize_with_qep_arch.py`)
- Added `move_kwargs_to_device` helper to recursively move keyword arguments to the target device (`onecomp/qep/_quantize_with_qep_arch.py`)
- Fixed `UnboundLocalError` when a module in a group is not registered in `quantizer.module_to_name` (`onecomp/qep/_quantize_with_qep_arch.py`)

## [v0.3.6] 2026-03-12

### Completion of Save/Load Pipeline

- **Added new `QuantizedModelLoader` class (`quantized_model_loader.py`)**
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

- **Migrated package and project management to `uv` and `pyproject.toml`.**
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
- **BREAKING: Changed `QEPConfig.general` default from `True` to `False` (architecture-aware implementation is now the default)**

### GPTQ Refactoring (`onecomp/quantizer/gptq/_gptq.py`)

- **BREAKING: Changed default `sym` from `False` to `True` (symmetric quantization) for both `GPTQ` class and `run_gptq()` function. Code relying on the previous asymmetric default must now explicitly pass `sym=False`.**
- Expanded `GPTQ` class docstring with full attribute descriptions and usage examples
- Renamed `H` parameter to `hessian` in `run_gptq()` for clarity
- Renamed local variable `W` to `matrix_W` in `run_gptq()` for clarity
- Changed imports to `from` style (`from torch import nn`, `from transformers import Conv1D`)
- Refactored `GPTQExcecutor.__init__`: replaced `register_buffer` with explicit `None` initialization for all attributes
- Added docstrings to `GPTQExcecutor.quantize()`, `enabled()`, and `ready()` methods
- Updated `test_gptq.py` boundary/abnormal parameters to reflect new `sym=True` default

## [v0.3.5] 2026-03-05

- Based on v0.3.4 codebase
- Difference from v0.3.4: Changed comments to English

