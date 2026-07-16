# Post-Process (Global PTQ / Block-wise PTQ / LoRA SFT)

OneComp supports **post-quantization processing** — additional steps applied to a quantized model to improve accuracy or inject domain-specific knowledge. Three implementations are available:

- **Global PTQ** — Globally optimises quantization parameters (scales, zeros, scaling factors) via KL distillation from a full-precision teacher model
- **Block-wise PTQ** — Minimises intermediate-representation MSE against an FP16 teacher model at Transformer-block granularity. No training data labelling required.
- **LoRA SFT** — Fine-tunes quantized models using Low-Rank Adaptation (LoRA) adapters with SFT loss, optional teacher distillation, and intermediate block alignment.

## Overview

The post-process framework integrates into the `Runner` pipeline via the `post_processes` parameter. After quantization completes, `Runner` builds a quantized model on CPU and executes each process in order. The processed model is stored as `runner.quantized_model` and is automatically used by subsequent evaluation and save operations.

```
Quantize ──► Build Model ──► Post-Process 1 ──► Post-Process 2 ──► Evaluate / Save
                              (e.g. GlobalPTQ)   (e.g. LoRA SFT)
```

## Load -> Post-process -> Re-save

Saved quantized checkpoints can be loaded, refined with additional
structure-preserving post-processes, and saved again. Load with
`device_map=None` to keep the model on CPU before `run_post_processes()`:

```python
from onecomp import BlockWisePTQ, CalibrationConfig, ModelConfig, Runner, load_quantized_model

model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    device="cuda:0",
)

model, _ = load_quantized_model(
    "./tinyllama-gptq4-initial",
    device_map=None,
)

runner = Runner(
    model_config=model_config,
    quantizer=None,
    post_processes=[
        BlockWisePTQ(
            lr=1e-4,
            epochs=10,
            cbq_enable=True,
            calibration_config=CalibrationConfig(num_calibration_samples=128),
        )
    ],
)
runner.quantized_model = model
runner.run_post_processes()
runner.save_quantized_model("./tinyllama-gptq4-blockwise-resaved")
```

`quantizer=None` is valid for this flow because `runner.quantized_model` is
assigned before calling `run_post_processes()`. Each successful post-process
appends an entry to
`quantization_config["onecomp_post_processes"]`, so the audit metadata
accumulates across repeated load -> post-process -> re-save cycles.

!!! tip
    A complete working example is available at
    [`example/post_process/example_reload_post_process_resave.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/post_process/example_reload_post_process_resave.py).

---

## Global PTQ: Parameter Optimisation

Global PTQ improves quantized model accuracy by globally optimising continuous quantization parameters (scales and zeros for GPTQLinear-backed quantizers such as GPTQ, RTN, and JointQ; scaling factors for DBF) using KL-divergence distillation from a full-precision teacher model.

### Single-GPU (GlobalPTQ)

```python
from onecomp import CalibrationConfig, DBF, GPTQ, JointQ, RTN, ModelConfig, Runner, GlobalPTQ, setup_logger

setup_logger()

model_config = ModelConfig(
    model_id="meta-llama/Llama-2-7b-hf",
    device="cuda:0",
)
quantizer = GPTQ(wbits=4, groupsize=128)
# quantizer = RTN(wbits=4, groupsize=128)
# quantizer = JointQ(bits=4, group_size=128)
# quantizer = DBF(target_bits=1.5)

global_ptq = GlobalPTQ(
    epochs=5,
    gptq_lr=1e-5,
    calibration_config=CalibrationConfig(
        num_calibration_samples=128,
        max_length=2048,
    ),
)

runner = Runner(
    model_config=model_config,
    quantizer=quantizer,
    post_processes=[global_ptq],
)
runner.run()
```

To invoke GlobalPTQ directly on a model instance, use the following pattern
instead of passing it through `post_processes`:

```python
runner = Runner(
    model_config=model_config,
    quantizer=quantizer,
)
runner.run()

model, _ = runner.create_quantized_model(use_gemlite=False)
post_process = global_ptq
post_process.run(model, model_config)
runner.quantized_model = model
```

The direct path uses the same post-process metadata recording as
`run_post_processes()`. For explicit unpacked buffers, use
`runner.create_quantized_model(pack_weights=False, use_gemlite=False)`; this is
only needed when GPTQLinear bit packing cannot represent the quantizer output
(e.g. 1-bit JointQ, or GPTQ/RTN bit widths outside `{2, 3, 4, 8}`), and is
documented in the `GlobalPTQ` API reference.

!!! tip
    A complete working example of the Runner-managed packed/default path (GPTQ
    by default; uncomment `DBF(target_bits=1.5)` in the script to run DBF) is
    available at
    [`example/post_process/example_global_ptq.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/post_process/example_global_ptq.py).

### Multi-GPU with DeepSpeed (GlobalPTQDistributed)

For large models that do not fit on a single GPU, use `GlobalPTQDistributed` with DeepSpeed ZeRO-2.

!!! note "Installation"
    Multi-GPU training requires DeepSpeed. Install it via the `distributed` extra:
    
    - **uv**: `uv sync --extra <cuda-extra> --extra distributed`
    - **pip**: `pip install "onecomp[distributed]"`

```python
from onecomp import CalibrationConfig, GPTQ, ModelConfig, Runner, GlobalPTQDistributed, setup_logger

setup_logger()

model_config = ModelConfig(
    model_id="meta-llama/Llama-2-7b-hf",
    device="cuda:0",
)
gptq = GPTQ(wbits=4, groupsize=128)

global_ptq = GlobalPTQDistributed(
    epochs=5,
    gptq_lr=1e-5,
    deepspeed_config="ds_zero2.json",
    calibration_config=CalibrationConfig(
        num_calibration_samples=128,
        max_length=2048,
    ),
)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    post_processes=[global_ptq],
)
runner.run()
```

Launch with `torchrun`:

```bash
torchrun --nproc_per_node=2 my_script.py
```

### GlobalPTQ Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | `5` | Number of distillation epochs |
| `gptq_lr` | `1e-5` | Learning rate for GPTQ scales/zeros |
| `dbf_lr` | `5e-5` | Learning rate for DBF scaling parameters |
| `temperature` | `1.0` | Softmax temperature for KL divergence |
| `calibration_config` | `CalibrationConfig(num_calibration_samples=128)` | Calibration data configuration (see [CalibrationConfig](../api/calibration_config.md)) |
| `use_gradient_checkpointing` | `True` | Reduce GPU memory at the cost of recomputation |
| `early_stopping_patience` | `0` | Stop early if KL does not improve for N epochs (0 = disabled) |
| `use_mixed_precision` | `False` | Enable BF16 autocast to reduce memory |
| `grad_accum_steps` | `1` | Gradient accumulation steps |

> **Note — DBF vs GPTQ training differences:**
> When optimising **DBF** scaling factors, Global PTQ uses plain Adam (not AdamW) without a learning-rate scheduler.
> For **GPTQ** scales/zeros, it uses AdamW with a cosine-warmup LR schedule.
> Adjust `dbf_lr` and `gptq_lr` independently for best results.

> **Note — Mixed GPTQ + DBF models:**
> Global PTQ currently optimises a single quantization method per run.
> If a model contains both GPTQ and DBF layers (e.g. from AutoBit fallback), only GPTQ layers are optimised and a warning is logged.
> Joint GPTQ + DBF optimisation is planned for a future release.

### GlobalPTQDistributed Additional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `deepspeed_config` | `None` | Path to DeepSpeed config JSON |
| `w_distill` | `1.0` | Weight for KL distillation loss |
| `w_ntp` | `0.0` | Weight for next-token prediction loss |
| `bf16` | `True` | Enable bfloat16 training |
| `per_device_train_batch_size` | `1` | Batch size per GPU |
| `gradient_accumulation_steps` | `1` | Gradient accumulation steps |

See the [API Reference](../api/post_process.md) for the full parameter list.

!!! info "Save / Load"
    GlobalPTQ and GlobalPTQDistributed update quantized-layer buffers without
    introducing custom module types, so optimised models can be saved with
    `save_quantized_model()` and reloaded with `load_quantized_model()`. They
    also record an entry under
    `quantization_config["onecomp_post_processes"]`, so the saved checkpoint
    retains the post-process audit trail across load → post-process → re-save
    cycles. The `.pt` path remains available when whole-object PyTorch
    serialization is explicitly needed.

---

## Block-wise PTQ

Block-wise PTQ improves quantized model accuracy by minimising intermediate-representation MSE against an FP16 teacher at Transformer-block granularity. It supports GPTQLinear-backed quantizers (GPTQ, RTN, JointQ), DBF, and Onebit.

### How it works

1. **Phase 1 (Greedy per-block distillation)** — For each Transformer block, optimise quantization parameters (scales, zeros, binary matrices) so the block's output matches the FP16 teacher block's output.
2. **Phase 2 CBQ (Cross-Block Quantisation)** — Jointly optimise pairs of adjacent blocks with a sliding window (K=2) to reduce error accumulation from the greedy Phase 1.

Only 1–2 blocks are loaded onto GPU at a time, so large models can be processed without loading the entire model into GPU memory.

### Usage via Runner

```python
from onecomp import DBF, GPTQ, JointQ, Onebit, RTN, BlockWisePTQ, ModelConfig, Runner, setup_logger

setup_logger()

model_config = ModelConfig(
    model_id="meta-llama/Llama-2-7b-hf",
    device="cuda:0",
)
quantizer = GPTQ(wbits=4, groupsize=128)
# quantizer = RTN(wbits=4, groupsize=128)
# quantizer = JointQ(bits=4, group_size=128)
# quantizer = DBF(target_bits=1.5)
# quantizer = Onebit()

blockwise_ptq = BlockWisePTQ(
    lr=1e-4,
    epochs=10,
    cbq_enable=True,
    gptq_lr=1e-3,
)

runner = Runner(
    model_config=model_config,
    quantizer=quantizer,
    post_processes=[blockwise_ptq],
)
runner.run()

original_ppl, _, quantized_ppl = runner.calculate_perplexity(
    original_model=True, quantized_model=True,
)
print(f"Original PPL:                    {original_ppl:.4f}")
print(f"Quantized + BlockWisePTQ PPL:    {quantized_ppl:.4f}")
```

### Direct invocation

You can also call BlockWisePTQ directly on an existing quantized model to compare before/after PPL without re-running quantization:

```python
runner = Runner(model_config=model_config, quantizer=quantizer)
runner.run()

# Baseline PPL
_, _, baseline_ppl = runner.calculate_perplexity(quantized_model=True)

# Apply BlockWisePTQ
# pack_weights defaults to True (matches run_post_processes); use_gemlite=False
# keeps the plain PyTorch path the block optimiser needs.
model, _ = runner.create_quantized_model(use_gemlite=False)
blockwise_ptq.run(model, model_config)
runner.quantized_model = model

# Improved PPL
_, _, improved_ppl = runner.calculate_perplexity(quantized_model=True)
```

!!! tip
    A complete working example of the Runner-managed packed/default path is
    available at
    [`example/post_process/example_blockwise_ptq.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/post_process/example_blockwise_ptq.py).
    The explicit unpacked path (`create_quantized_model(pack_weights=False)`
    → `BlockWisePTQ.run()`) is only needed when GPTQLinear bit packing cannot
    represent the quantizer output (e.g. 1-bit JointQ, or GPTQ/RTN bit widths
    outside `{2, 3, 4, 8}`); see the `BlockWisePTQ` API reference for that path.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `1e-4` | Learning rate for block-wise optimisation (DBF / OneBit / generic) |
| `epochs` | `10` | Number of optimisation epochs per block |
| `cbq_enable` | `False` | Enable Phase 2 Cross-Block Quantisation |
| `gptq_lr` | `1e-3` | Learning rate for GPTQ scales/zeros optimisation |
| `gptq_optimize_intweight` | `False` | Optimise integer weights via Smooth STE (GPTQ) |
| `gptq_intweight_lr` | `1e-4` | Learning rate for integer weight optimisation |
| `grad_clip` | `1.0` | Gradient clipping norm |
| `optimize_binary` | `True` | Optimise binary/sign matrices (DBF / OneBit) |
| `k_smooth` | `100.0` | SmoothSign STE temperature |
| `num_calibration_samples` | `128` | Number of calibration samples |
| `max_length` | `2048` | Sequence length for calibration data |

See the [API Reference](../api/post_process.md) for the full parameter list.

!!! info "Save / Load"
    BlockWisePTQ-optimised models can be saved with `save_quantized_model()` and
    reloaded with `load_quantized_model()`. On the standard Runner path
    (`run()` → `run_post_processes()` → `save_quantized_model()`) the quantized
    model is built with packed buffers by default, so the saved checkpoint keeps
    the packed layout. A saved checkpoint can also be reloaded, refined with
    further post-processes, and saved again; the applied post-processes are
    recorded under `quantization_config["onecomp_post_processes"]` as an audit
    trail across save/load cycles.

---

## LoRA SFT: Accuracy Recovery

The most common use case is recovering accuracy lost during quantization. Provide a general-purpose dataset (e.g., WikiText-2) to fine-tune the quantized model:

```python
from onecomp import GPTQ, ModelConfig, Runner, PostProcessLoraSFT, setup_logger

setup_logger()

model_config = ModelConfig(
    model_id="meta-llama/Llama-2-7b-hf",
    device="cuda:0",
)
gptq = GPTQ(wbits=4, groupsize=128)

post_process = PostProcessLoraSFT(
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    train_split="train",
    text_column="text",
    max_train_samples=256,
    max_length=512,
    epochs=4,
    batch_size=2,
    gradient_accumulation_steps=8,
    lr=1e-4,
    lora_r=16,
    lora_alpha=32,
)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    post_processes=[post_process],
)
runner.run()

# Evaluate: PPL should be lower than without LoRA SFT
original_ppl, _, quantized_ppl = runner.calculate_perplexity(
    original_model=True, quantized_model=True,
)
print(f"Original PPL:              {original_ppl:.4f}")
print(f"Quantized + LoRA SFT PPL:  {quantized_ppl:.4f}")
```

!!! tip
    A complete working example is available at
    [`example/post_process/example_lora_sft.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/post_process/example_lora_sft.py).

---

## LoRA SFT: Knowledge Injection

LoRA SFT also supports injecting new knowledge into a quantized model using custom training data. Provide a JSONL file where each line has a `"text"` field:

```json
{"text": "OneCompression (OneComp) is an open-source Python library for LLM quantization developed by Fujitsu."}
{"text": "OneComp supports GPTQ, DBF, RTN, and AutoBit quantization methods."}
```

Then pass the file path to `data_files`:

```python
post_process = PostProcessLoraSFT(
    data_files="./my_knowledge.jsonl",
    max_length=256,
    epochs=20,
    batch_size=2,
    lr=3e-4,
    lora_r=16,
    lora_alpha=32,
)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    post_processes=[post_process],
)
runner.run()
```

After training, the model can generate responses based on the injected knowledge.

!!! tip
    A complete working example with before/after comparison is available at
    [`example/post_process/example_lora_sft_knowledge.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/post_process/example_lora_sft_knowledge.py).

---

## Saving and Loading LoRA Models

LoRA-applied models contain custom module types (`LoRAGPTQLinear`) that are not compatible with the standard safetensors format. Use the dedicated PyTorch `.pt` save/load methods instead:

### Save

```python
# After quantization + LoRA SFT
runner.run()

# Save the LoRA-applied model (PyTorch .pt format)
runner.save_quantized_model_pt("./my_model_lora")
```

### Load

```python
from onecomp import load_quantized_model_pt

# The .pt loader uses torch.load(weights_only=False) and requires an explicit
# opt-in. Only enable it for models from a fully trusted source (see warning).
model, tokenizer = load_quantized_model_pt(
    "./my_model_lora", allow_unsafe_deserialization=True
)
```

!!! warning "Unsafe deserialization (.pt loader)"
    `load_quantized_model_pt()` deserializes `model.pt` with
    `torch.load(..., weights_only=False)`, which uses Python `pickle` and can
    execute arbitrary code from a malicious file (CWE-502). It refuses to load
    unless you pass `allow_unsafe_deserialization=True`. Only opt in for models
    you produced yourself or trust completely; otherwise use the safetensors
    `load_quantized_model()`, which does not execute code.

!!! info "save_quantized_model vs save_quantized_model_pt"
    | Method | Format | Use Case |
    |--------|--------|----------|
    | `save_quantized_model()` | safetensors (HF-compatible) | Quantized models, including post-processes that keep the quantized layer structure (`BlockWisePTQ`, `GlobalPTQ`, `GlobalPTQDistributed`) |
    | `save_quantized_model_pt()` | PyTorch `.pt` | Post-processes that introduce custom module types not representable in safetensors (e.g. LoRA adapters / `LoRAGPTQLinear`) |

    Similarly, use `load_quantized_model()` for safetensors and `load_quantized_model_pt()` for `.pt` files.

---

## Data Sources

`PostProcessLoraSFT` supports two ways to provide training data:

### Hugging Face Datasets

```python
PostProcessLoraSFT(
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    train_split="train",
    text_column="text",
)
```

### Local Files

Supported formats: JSON, JSONL, CSV, TXT, Parquet.

```python
PostProcessLoraSFT(
    data_files="./train_data.jsonl",
    text_column="text",
)
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | `4` | Number of training epochs |
| `lr` | `1e-4` | Learning rate |
| `batch_size` | `1` | Training batch size |
| `gradient_accumulation_steps` | `16` | Gradient accumulation steps |
| `max_length` | `1024` | Maximum sequence length for tokenization |
| `max_train_samples` | `None` | Cap on number of training samples (unlimited if `None`) |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | LoRA scaling factor (effective scaling = `alpha / r`) |
| `lora_dropout` | `0.05` | LoRA dropout rate |
| `target_modules` | `None` | Module name suffixes to wrap with LoRA. Defaults to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `warmup_ratio` | `0.03` | Learning rate warmup ratio |
| `weight_decay` | `0.0` | Weight decay |
| `use_bf16` | `None` | Use bfloat16 training. Auto-detected from GPU capability if `None` |

See the [API Reference](../api/post_process.md) for the full parameter list.

---

## Advanced: Teacher Distillation

Teacher distillation aligns the quantized model's output distribution with a full-precision teacher model. This can improve accuracy beyond what SFT alone achieves:

```python
post_process = PostProcessLoraSFT(
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    train_split="train",
    text_column="text",
    sft_loss_weight=1.0,
    teacher_loss_weight=0.5,           # Enable teacher distillation
    teacher_loss_type="kl",            # "kl" or "mse"
    teacher_temperature=1.0,
    teacher_model_id="meta-llama/Llama-2-7b-hf",  # Full-precision teacher
    cache_teacher_outputs=True,        # Pre-compute teacher logits for speed
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sft_loss_weight` | `1.0` | Weight for causal LM (SFT) loss |
| `teacher_loss_weight` | `0.0` | Weight for teacher distillation loss (0 = disabled) |
| `teacher_loss_type` | `"kl"` | `"kl"` (KL divergence) or `"mse"` (mean squared error) on logits |
| `teacher_temperature` | `1.0` | Temperature for softening teacher logits |
| `teacher_model_id` | `None` | Hugging Face model ID for the teacher |
| `teacher_model_path` | `None` | Local path for the teacher model |
| `cache_teacher_outputs` | `False` | Pre-compute and cache teacher outputs on CPU |

---

## Advanced: Intermediate Block Alignment

Intermediate block alignment adds a loss term that aligns hidden states at selected transformer blocks between the teacher and student models:

```python
post_process = PostProcessLoraSFT(
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    train_split="train",
    text_column="text",
    teacher_model_id="meta-llama/Llama-2-7b-hf",
    intermediate_block_loss_weight=0.1,
    intermediate_block_indices=[8, 16, 24],
    cache_intermediate_outputs=True,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `intermediate_block_loss_weight` | `0.0` | Weight for intermediate alignment loss (0 = disabled) |
| `intermediate_block_indices` | `None` | Transformer block indices to align |
| `cache_intermediate_outputs` | `False` | Pre-compute and cache teacher block outputs |

---

## Limitations

!!! warning "vLLM Inference"
    LoRA-applied models saved with `save_quantized_model_pt()` are **not currently supported** by the vLLM plugins. vLLM integration for LoRA post-processed models is planned for a future release.

    For standard quantized models, and for post-processes that keep the
    quantized layer structure such as `BlockWisePTQ`, `GlobalPTQ`, and
    `GlobalPTQDistributed`, use `save_quantized_model()`. Models whose saved
    `quant_method` is supported by vLLM can then be served as described in the
    [vLLM Inference guide](vllm-inference.md).

!!! note "Supported Quantizers"
    LoRA SFT currently supports **GPTQ**-quantized models only. Support for other quantization methods (DBF, RTN) may be added in the future.
