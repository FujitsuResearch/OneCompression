# Change log

## [v1.0.1] 2026-03-31

### Packaging

- Moved `matplotlib` from `dev` extra to new `visualize` extra in `pyproject.toml`
- Made `visualize_bit_assignment` import lazy in `onecomp/quantizer/autobit/__init__.py` to avoid requiring matplotlib at import time
- Updated installation instructions in `README.md` and `docs/getting-started/installation.md` to reflect the new `visualize` extra
- Updated `uv.lock`

## [v1.0.0] 2026-03-31

### PyPI Publishing Setup

- Added PyPI metadata to `pyproject.toml`: `keywords`, `classifiers`, and `project.urls` (Homepage, Documentation, Repository, Bug Tracker, Changelog)
- Removed `gemlite` optional-dependency extra that used direct git URLs (PEP 440 violation); equivalent packages are already in main `dependencies`
- Added `.github/workflows/publish.yml`: automated PyPI publishing via Trusted Publishers (OIDC) on GitHub Release
- Updated `README.md`: installation command changed from `pip install git+<URL>` to `pip install onecomp`
- Added `dist/` and `build/` to `.gitignore`
- Updated `uv.lock`

### Default Parameter Changes

- Changed `Runner.__init__` default values for calibration parameters:
  - `max_length`: `512` → `2048`
  - `num_calibration_samples`: `128` → `512`
- Pinned old default values explicitly in all `example/` and `tests/` files that previously relied on the defaults

### Documentation

- Updated `docs/user-guide/configuration.md` to reflect the new default values for `max_length` and `num_calibration_samples`
- Added quantizer feature support table to `docs/user-guide/basic-usage.md` and `docs/api/quantizers/base.md`
  - Documents which quantizers support `save_quantized_model()` / `create_quantized_model()` and quantized-model PPL/ACC evaluation
  - Currently supported: **GPTQ**, **DBF**, **AutoBitQuantizer** (requires `get_quant_config()` and `create_inference_layer()`)
  - Unsupported quantizers (RTN, JointQ, QUIP, CQ, ARB, QBB, Onebit): PPL/ACC evaluation automatically falls back to the dequantized (FP16) model
- Updated the perplexity/accuracy evaluation note in `basic-usage.md` to reflect AutoBitQuantizer support and fallback behavior

## [v0.5.0] 2026-03-30

### New Feature: Post-quantization Workflow

- Added `PostQuantizationProcess` abstract base class (`onecomp/post_process/_base.py`)
  - Defines the interface for post-quantization operations (e.g. block-wise PTQ, fine-tuning)
- Added `post_processes` parameter to `Runner.__init__`
  - Accepts a list of `PostQuantizationProcess` instances
  - After quantization, builds a quantized model on CPU and executes each process in order
  - The processed model is stored as `self.quantized_model`
- Updated `Runner.calculate_perplexity` and `Runner.calculate_accuracy` to use `self.quantized_model` if available (GPU transfer is handled automatically; `device="auto"` is resolved to `"cuda"`)
- Added LoRA SFT post-process implementation (`onecomp/post_process/post_process_lora_sft.py`)
  - Provides learning-based post-quantization fine-tuning for GPTQ-quantized models
  - Public API is exposed as `PostProcessLoraSFT`

### New Feature: Rotation Preprocessing Pipeline (`onecomp/pre_process/`)

SpinQuant/OstQuant-based rotation preprocessing that reduces quantization error by learning optimal rotation matrices before quantization. Supports Llama and Qwen3 architectures.

- Added `prepare_rotated_model()` (`onecomp/pre_process/prepare_rotated_model.py`): End-to-end pipeline — model loading → rotation/scaling training → rotation application → saving
  - Memory-optimized: moves model between CPU/GPU to reduce peak memory (e.g. Qwen3-32B: ~128GB → ~64GB)
- Added `RotatedModelConfig` (`onecomp/rotated_model_config.py`): `ModelConfig` subclass that automatically registers Hadamard hooks on `down_proj` layers during `load_model()`
- Added `onecomp/pre_process/` package:
  - `train_rotation.py`: Training pipeline with `PreprocessManager` (R1/R2/S_* tensor management), HF `Trainer` subclass, `apply_preprocess_train` / `apply_preprocess_eval`
  - `optimizer.py`: `SGDG` — SGD on the Stiefel manifold with Cayley-retraction orthogonal updates (ported from SpinQuant)
  - `quant_models.py`: `WeightQuantizer` (RTN proxy) with per-channel / per-tensor / group-wise quantization; quantized decoder layers for Llama and Qwen3
  - `rotation_utils.py`: `fuse_layer_norms`, `rotate_model`, `register_online_hadamard_hooks`
  - `hadamard_utils.py`: Hadamard transform utilities and pre-computed matrices (ported from QuIP#)
  - `modeling_llama.py` / `modeling_qwen3.py`: Custom `ForCausalLM` classes that propagate R1 through the forward pass during training
  - `preprocess_args.py`: `TrainingArguments` subclass with SGDG-specific LR/momentum fields
- Fixed `_PreprocessTrainer` to override `create_optimizer()` instead of `create_optimizer_and_scheduler()` for transformers >= 5.x compatibility (SGDG optimizer was silently replaced by AdamW)
- Updated `Runner.save_dequantized_model()` and `Runner.save_quantized_model()` to warn when saving models loaded with additional preprocessing (e.g., Hadamard hooks)

### Added JointQ Quantizer

- **Added new `JointQ` quantizer (`onecomp/quantizer/jointq/`)**
  - Local-search-based post-training quantization method that minimizes ||Y - hat{W} X^T||_F^2
  - Supports both symmetric and asymmetric quantization (1–4 bits)
  - Group-wise quantization with configurable group size
  - Tikhonov regularization for over-fitting (X^T X + nλI)
  - Three initialization strategies: Clip-Optimize, Clip-Optimize with Error Propagation, and GPTQ

### AutoBitQuantizer vLLM-compatible quantization_config

- `AutoBitQuantizer` now emits `mixed_gptq`-compatible `quantization_config` (`onecomp/quantizer/autobit/_autobit.py`)
- ILP solver now enforces fused-layer equality constraints (`onecomp/quantizer/autobit/ilp.py`)
  - vLLM fuses q/k/v → `qkv_proj` and gate/up → `gate_up_proj`

### API changes

- Made `Runner.create_quantized_model()` a public method (renamed from `_create_quantized_model`)
  - Builds a quantized model with quantized inference layers from `quantizer.results`
  - Returns `(model, tokenizer)` for use in evaluation, saving, and post-process workflows
- Added `Runner.save_quantized_model_pt()` for saving post-processed models (e.g. LoRA-applied) as PyTorch `.pt` files
  - Uses `torch.save` to preserve custom module types such as `LoRAGPTQLinear`
  - Saves tokenizer files alongside the model
- Added `QuantizedModelLoader.load_quantized_model_pt()` for loading `.pt`-format models
  - Counterpart to `save_quantized_model_pt`; uses `torch.load` to restore models with custom modules
  - Also available as `onecomp.load_quantized_model_pt()` convenience alias

### Bug Fix: Onebit Quantizer

- Fixed `Onebit` to declare `flag_calibration=True` and `flag_hessian=True` (`onecomp/quantizer/onebit/_onebit.py`)
  - Previously, Onebit computed the Hessian internally from `input` despite declaring all flags as `False`, causing a crash when used through `quantize_without_calibration` or chunked quantization paths
  - Now uses the Hessian provided by the Runner, consistent with other calibration-based quantizers (GPTQ, DBF, QUIP)

### Quantizer Signature Consistency

- Added `input=None` default to `quantize_layer` in `RTN`, `CQ`, `QBB` (`onecomp/quantizer/{rtn,cq,qbb}/`)
  - Aligns with the base `Quantizer.quantize_layer(self, module, input=None, hessian=None)` signature
  - Enables these quantizers to be used in `Runner(quantizers=[...])` via the chunked quantization path
- Added `input=None, hessian=None` defaults to `Onebit.quantize_layer` for the same reason

### Examples

- Added `example/post_process/example_lora_sft.py`: End-to-end demo — GPTQ 4-bit quantization + LoRA SFT (WikiText-2) + PPL evaluation + save/load with `save_quantized_model_pt` / `load_quantized_model_pt`
- Added `example/post_process/example_lora_sft_knowledge.py`: Knowledge injection demo — teaches the quantized model about "OneCompression" via LoRA SFT and compares generation before/after
- Added `example/post_process/onecomp_knowledge.jsonl`: Training data describing OneCompression for the knowledge injection example
- Added `example/example_jointq.py`: JointQ 4-bit (groupsize=128) quantization example with dequantized model PPL evaluation
- Added `example/pre_process/example_llama_preprocess_rtn.py`: Rotation preprocessing + RTN quantization (TinyLlama-1.1B)
- Added `example/pre_process/example_preprocess_save_load.py`: Rotation preprocessing + GPTQ quantization → save → load → PPL verification
- Added `example/vllm_inference/example_gptq_vllm_inference.py`: GPTQ + QEP quantization and vLLM inference end-to-end example
- Added `example/vllm_inference/example_autobit_vllm_inference.py`: AutoBit mixed-precision quantization and vLLM inference example

### Documentation

- Added `docs/user-guide/post-process.md`: LoRA SFT user guide covering accuracy recovery, knowledge injection, save/load, key parameters, teacher distillation, intermediate block alignment, and vLLM limitations
- Added `docs/api/post_process.md`: API reference for `PostQuantizationProcess`, `PostProcessLoraSFT`, and convenience variants
- Updated `docs/user-guide/examples.md` with LoRA SFT code examples (accuracy recovery, knowledge injection, save/load)
- Updated `docs/api/runner.md` to include `create_quantized_model` and `save_quantized_model_pt`
- Updated `docs/api/quantized_model_loader.md` to include `load_quantized_model_pt`
- Updated `mkdocs.yml` navigation with new post-process pages
- Added `docs/user-guide/pre-process.md`: Rotation preprocessing user guide covering workflow, key parameters, save/load, and limitations
- Added `docs/api/pre_process.md`: API reference for `prepare_rotated_model` and `RotatedModelConfig`
- Updated `docs/user-guide/examples.md` with rotation preprocessing code examples (RTN, GPTQ with save/load)
- Updated `docs/api/index.md` with `RotatedModelConfig`, `prepare_rotated_model`, and `pre_process/` module structure
- Updated `docs/index.md` Key Features with rotation preprocessing

### Tests

- Added smoke test for `PostProcessLoraSFT` (`tests/onecomp/post_process/test_post_process_lora_sft.py`)
  - Verifies that `PostProcessLoraSFT.run()` completes without error on TinyLlama with minimal settings
  - Checks LoRA layer injection, CPU placement, and eval mode after run
  - Includes Runner end-to-end integration test with `post_processes` parameter
- Expanded and updated unit tests for DBF quantizer (`tests/onecomp/quantizer/dbf/test_dbf.py`)
  - Extended boundary and abnormal parameter cases; aligned with `BaseQuantizeSpec` and current DBF API
- Expanded and updated unit tests for GPTQ quantizer (`tests/onecomp/quantizer/gptq/test_gptq.py`)
  - Extended boundary and abnormal parameter cases; aligned with `BaseQuantizeSpec` and current GPTQ API
- Adjusted DBF and GPTQ quantizer implementations for test compatibility and consistency (`onecomp/quantizer/dbf/_dbf.py`, `onecomp/quantizer/gptq/_gptq.py`)
- Fixed and improved JointQ unit tests (`tests/onecomp/quantizer/jointq/test_jointq.py`)
  - Use `compute_dequantized_weight()` instead of direct `dequantized_weight` access
  - Override boundary test to use CUDA with 128×128 layers for group_size compatibility
  - Skip CPU-only tests (JointQ is GPU-based)
  - Fix `batch_size` validation: `>= 0` → `>= 1` (`onecomp/quantizer/jointq/_jointq.py`)
- Improved JointQ regression test (`tests/onecomp/quantizer/jointq/test_quantize_regression.py`)
  - Replaced exact tensor match with MSE-based quality check for environment portability
  - Hardcoded expected MSE in helper; removed `.pth` baseline file

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
