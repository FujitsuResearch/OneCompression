# Change log

## [v0.4.3] 2026-03-26

### Implement AutoBit to automatically determine bit-allocation

- Add `AutoBitQuantizer` (`onecomp/quantizer/autobit/_autobit.py`) that automatically assigns optimal bit-width per module using ILP with considering activation-aware error (`onecomp/quantizer/autobit/ilp.py`)  and DBF fallback (`onecomp/quantizer/autobit/dbf_fallback.py`) for ultra-low-bit targets ( <= target bit 2bit) 
  - [SCIP](https://www.scipopt.org) solver was utilized to solve ILP (`onecomp/quantizer/autobit/ilp.py`)
  - Sequentially load and forward each layer to collect activation and curvature statistics (`onecomp/quantizer/autobit/activation_stats.py`, `onecomp/utils/blockwise.py`)
  - Usage example is shown in (`example/example3.py`)
- Add VRAM auto-estimation utility to derive target bit-width from available GPU memory (`onecomp/utils/vram_estimator.py`)

### VLM and Multi-Architecture Support for Architecture-aware QEP

- Extended `_get_blocks` to detect `language_model` sub-module and restrict block search to the text decoder (`onecomp/qep/_quantize_with_qep_arch.py`)
  - VLMs (Qwen3-VL, Gemma3, etc.) no longer return vision-encoder blocks
  - CausalLM behaviour is unchanged (falls back to full-model search)
- Added `__getattr__` proxy to `Catcher` to forward attribute access to the wrapped module (`onecomp/qep/_quantize_with_qep_arch.py`)
  - Prevents `AttributeError` when model code reads decoder-layer attributes (e.g. `attention_type`) before `forward()`
- Changed `get_blocks_and_inputs` to capture block-level kwargs with batch=1 (`onecomp/qep/_quantize_with_qep_arch.py`)
  - Internally generated kwargs (position_embeddings, attention_mask, etc.) are now batch-size-independent
  - Avoids shape mismatches when reused with varying batch sizes in downstream functions
- Added `expand_kwargs_batch` helper to expand batch=1 kwargs via `Tensor.expand` (zero-copy view) (`onecomp/qep/_quantize_with_qep_arch.py`)
  - Used in `compute_hessian_and_crossterm` and `forward_input` before each block forward call
  - Resolves failures on models requiring exact batch-dimension matching (e.g. Gemma3 sliding-window attention)
- Added early termination and group skipping to `run_quantize_with_qep_arch` (`onecomp/qep/_quantize_with_qep_arch.py`)
  - Groups with no quantization targets are skipped (avoids unnecessary Hessian/cross-term computation)
  - Block loop exits once all target layers are quantized

### End-to-end CLI tests

- Added `tests/onecomp/test_cli.py`: end-to-end tests that verify `onecomp TinyLlama/...` CLI runs without errors
  - `test_default_full_run`: full default pipeline (AutoBit + QEP + eval + save) on GPU
  - Variant tests for individual options (`--wbits`, `--no-qep`, `--total-vram-gb`, `--groupsize`, `--save-dir`, etc.) on CPU
  - Variant tests are skipped by default; enable with `RUN_CLI_VARIANT_TESTS=1`
  - Uses `python -m onecomp` to avoid implicit `uv sync` that could modify the environment

### Fixes

- Fixed crash when DBF quantization fails with NaN/Inf (`onecomp/quantizer/dbf/_dbf.py`, `onecomp/qep/_quantize_with_qep_arch.py`)
  - `_quantize_with_qep_arch.py`: Catch `ValueError`/`NotImplementedError` from `compute_dequantized_weight()`, log the error, and keep QEP-adjusted weights for the failed layer
- Fixed GemLite import crash when PyTorch version is incompatible (`onecomp/quantizer/gemlite.py`)
  - Broadened `except ImportError` to `except (ImportError, AttributeError)` so that GemLite gracefully falls back when `torch` lacks newer dtypes (e.g. `float8_e8m0fnu`)
- Fixed `test_dbf_gemlite.py` to skip when GemLite is unavailable instead of crashing (`tests/vllm-plugins/dbf/test_dbf_gemlite.py`)

### Dependency and documentation updates

- Added `vllm` as an optional dependency (`--extra vllm`) in `pyproject.toml`
  - Prevents environment corruption caused by `uv pip install vllm` being overwritten by subsequent `uv sync`/`uv run`
- Added `torchvision` to CUDA extras and `[tool.uv.sources]` in `pyproject.toml` to prevent CUDA version mismatch
- Updated installation docs to reflect new extras (`README.md`, `docs/getting-started/installation.md`, `docs/user-guide/vllm-inference.md`)
- Updated `uv.lock`

## [v0.4.2] 2026-03-25

### Unit tests for additional quantizers

- **Added unit tests for QBB, RTN, QUIP, ONEBIT, CQ, ARB, and JOINTQ**
  - New test modules under `tests/onecomp/quantizer/`: `test_qbb.py`, `test_rtn.py`, `test_quip.py`, `test_onebit.py`, `test_cq.py`, `test_arb.py`, `test_jointq.py`
  - Shared test base and helpers updated in `tests/onecomp/quantizer/test_module.py`
  - Quantizer implementations adjusted for test compatibility: `onecomp/quantizer/qbb/`, `onecomp/quantizer/rtn/`, `onecomp/quantizer/quip/`, `onecomp/quantizer/onebit/`, `onecomp/quantizer/arb/`, `onecomp/quantizer/jointq/` (and related `*_impl.py`); minor updates in `onecomp/quantizer/dbf/_dbf.py`, `onecomp/quantizer/gptq/_gptq.py`

### vLLM plugin integration (DBF, Mixed-GPTQ)

- **Added vLLM plugin implementation for DBF and Mixed-GPTQ**
  - New `vllm_plugins` package: `vllm_plugins/__init__.py`, DBF and GPTQ plugin entry points (`vllm_plugins/dbf/`, `vllm_plugins/gptq/`)
  - DBF: `vllm_plugins/dbf/vllm_plugin.py` and modules (`vllm_plugins/dbf/modules/gemlite_linear.py`, `vllm_plugins/dbf/modules/naive.py`); shared utilities in `vllm_plugins/utils/module.py`
  - GPTQ: `vllm_plugins/gptq/vllm_plugin.py` for Mixed-GPTQ inference
  - Tests: `tests/vllm-plugins/dbf/test_dbf_gemlite.py`, `tests/vllm-plugins/dbf/test_dbf_naive.py`
  - Package and dependency wiring in `pyproject.toml`

### Fixes

- **Mixed-GPTQ:** raise an error when quantization bit widths differ within the same shard (align with DBF behavior) (`vllm_plugins/gptq/vllm_plugin.py`)

## [v0.4.1] 2026-03-19

### Mixed GPTQ/DBF Save/Load

- **Extended Save/Load for mixed GPTQ and mixed DBF**
  - `QuantizedModelLoader` now loads models with `quant_method` `mixed_gptq` or `mixed_dbf` (`onecomp/quantized_model_loader.py`)
  - `effective_method` treats mixed_* as the same tensor format as the base method (gptq/dbf) and resolves per-layer bit-width via `quantization_bits`
  - Load validates `quant_method`, `quantization_bits`, and `modules_in_block_to_quantize` from `config.json`'s `quantization_config`
- **GPTQ**
  - Added `get_quant_config()` to return save-time `quantization_config` with vLLM-compatible keys (`onecomp/quantizer/gptq/_gptq.py`)
  - Sets `quant_method` to `mixed_gptq` when `module_wbits` or `mlp_wbits` is present
  - New `onecomp/quantizer/gptq/config.py`: `resolve_gptq_layer_wbits()` resolves per-layer bit-width from `quantization_config` (priority: quantization_bits → module_wbits → mlp_wbits → bits/wbits)
  - `GPTQLinear`: extended to accept bit-width when restoring from saved state (`onecomp/quantizer/gptq/gptq_layer.py`)
- **DBF**
  - Added `get_quant_config()` to return save-time `quantization_config` (`onecomp/quantizer/dbf/_dbf.py`)
  - New `onecomp/quantizer/dbf/config.py`: `resolve_dbf_layer_bits()` resolves per-layer bit-width from `quantization_config` (priority: quantization_bits → module_target_bits → mlp_target_bits → bits)
  - `DoubleBinaryLinear`: added argument for target bit-width (for mixed_dbf) (`onecomp/quantizer/dbf/dbf_layer.py`)
- **Shared**
  - `onecomp/utils/quant_config.py`: added common helper `get_quant_param()` for `quantization_config` schema (fetch params by alias keys)
  - `Quantizer.finalize_quant_config_for_save()` hook added; subclasses (GPTQ/DBF) inject method-specific metadata (`onecomp/quantizer/_quantizer.py`)
  - `runner`: set `quantization_config` when saving (`onecomp/runner.py`)

### Evaluation and benchmark (Runner and accuracy utils)

- **Runner:** unified perplexity/accuracy evaluation via `_calculate_evaluation()` and added optional `dequantized_model` evaluation (`onecomp/runner.py`)
- **BREAKING: `calculate_perplexity()` / `calculate_accuracy()` now return a 3-tuple `(original, dequantized, quantized)` instead of 2-tuple `(original, quantized)`.** Existing code using `orig, quant = runner.calculate_perplexity()` must be updated to unpack three values. (`onecomp/runner.py`)
- **BREAKING: `calculate_perplexity()` / `calculate_accuracy()` default for `original_model` changed from `True` to `False`.** To evaluate the original model, pass `original_model=True` explicitly. (`onecomp/runner.py`)
- **Benchmark:** `benchmark_perplexity()` / `benchmark_accuracy()` now accept `dequantized_model` and `quantized_model` arguments. When `dequantized_model=True`, the result dict includes `"{name}_dequantized"` keys. (`onecomp/runner.py`)
- **lm_eval:** added helper to create `HFLM` while temporarily disabling `model.config.quantization_config` for compatibility (`onecomp/utils/accuracy.py`)

### Dequantized-weight API and compatibility fixes

- Implemented `compute_dequantized_weight()` for GPTQ and DBF quantizers (`onecomp/quantizer/gptq/_gptq.py`, `onecomp/quantizer/dbf/_dbf.py`)
- Removed `dequantized_weight` from Result classes and switched call sites to compute it via `compute_dequantized_weight()` (`onecomp/quantizer/_quantizer.py`, `onecomp/runner_methods/*`)
- Fixed compatibility for quantization methods other than DBF/GPTQ in runner and QEP paths (`onecomp/runner.py`, `onecomp/qep/_quantize_with_qep*.py`)
- Updated unit tests accordingly (`tests/onecomp/test_qep_general_consistency.py`)

### `auto_run` / CLI improvements

- **`Runner.auto_run()`:** added `eval_original_model` parameter to optionally evaluate the original (unquantized) model's perplexity and accuracy (default: `False`) (`onecomp/runner.py`)
- **`Runner.auto_run()`:** evaluation now only computes quantized model metrics by default; pass `eval_original_model=True` to include original model metrics
- **CLI:** added `--eval-original` flag to `onecomp` command (`onecomp/cli.py`)

### GPU memory optimization for model saving

- **`save_quantized_model()` / `save_dequantized_model()`** now load the base model on CPU (`device_map="cpu"`) instead of GPU when building the save artifact (`onecomp/runner.py`). Previously the full original model was loaded onto GPU, which was unnecessary for saving and could cause OOM on memory-constrained setups.

### Bug fix: Architecture-aware QEP group alignment

- Fixed non-deterministic crash in `compute_hessian_and_crossterm` caused by `groups_q` and `groups_f` being ordered differently (`onecomp/qep/_quantize_with_qep_arch.py`). `make_grouped_module` groups modules by tensor identity (`id()` + `data_ptr()`), but after `copy.deepcopy` the CUDA memory allocator can assign different addresses, causing group misalignment between the quantized and full-precision blocks. Now `groups_f` is derived from `groups_q` by module name lookup instead of independent grouping.

### Other fixes in this release

- Refactored runner evaluation paths and fixed benchmark-based evaluation behavior (`onecomp/runner.py`, `onecomp/utils/accuracy.py`)
- Examples: updated to pass `original_model=True` and `quantized_model=True` explicitly, and to unpack the new triple return value (`example/example1.py`, `example/example2.py`)

## [v0.4.0] 2026-03-20

### New Feature: `Runner.auto_run()` Classmethod

- Added `Runner.auto_run()` classmethod for one-liner quantization (`onecomp/runner.py`)
  - Handles model loading, GPTQ quantization with QEP, evaluation (perplexity + accuracy), and model saving in a single call
  - Parameters: `model_id`, `wbits` (default: 4), `groupsize` (default: 128), `device`, `qep` (default: True), `evaluate` (default: True), `save_dir` (default: "auto")
  - Returns the configured `Runner` instance for further analysis
- Made `model_config` parameter optional in `Runner.__init__()` (default: `None`) to allow `Runner()` without arguments

### New Feature: `onecomp` CLI Command

- Added `onecomp` CLI command for terminal-based quantization (`onecomp/cli.py`)
  - Usage: `onecomp <model_id> [--wbits N] [--groupsize N] [--device DEV] [--no-qep] [--no-eval] [--save-dir DIR]`
  - Thin wrapper around `Runner.auto_run()`
- Added `onecomp/__main__.py` for `python -m onecomp` support
- Registered `console_scripts` entry point in `pyproject.toml`

### New Example

- Added `example/example_auto_run.py` demonstrating one-liner quantization with `Runner.auto_run()`

### Documentation

- Updated `docs/index.md`: Quick Example now shows `auto_run` and CLI with tabbed view
- Restructured `docs/getting-started/quickstart.md`: `auto_run` / CLI as the fastest path, step-by-step workflow below
- Updated `docs/getting-started/installation.md`: Added `onecomp` command examples to Running Commands section
- Updated `docs/user-guide/basic-usage.md`: Added "Quick Path: `Runner.auto_run()`" section
- Updated `docs/user-guide/examples.md`: Added `auto_run` and CLI examples at the top
- Added `docs/user-guide/cli.md`: Full CLI reference with all options and usage examples
- Updated `docs/api/runner.md`: Added `auto_run` to mkdocstrings members
- Updated `docs/api/index.md`: Added `cli.py` and `__main__.py` to Module Structure
- Updated `mkdocs.yml`: Added CLI page to navigation
- Added "Building Documentation Locally" section to `README.md`

### Python Version Constraint

- Restricted `requires-python` to `">=3.12, <3.14"` in `pyproject.toml`
  - PyTorch does not yet provide wheels for Python 3.14, causing `uv sync` to fail when uv auto-selects CPython 3.14
- Updated `uv.lock` to reflect the new Python version constraint

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

