# Post-Process (LoRA SFT)

OneComp supports **post-quantization processing** — additional steps applied to a quantized model to improve accuracy or inject domain-specific knowledge. The main implementation is **LoRA SFT**, which fine-tunes quantized models using Low-Rank Adaptation (LoRA) adapters.

## Overview

The post-process framework integrates into the `Runner` pipeline via the `post_processes` parameter. After quantization completes, `Runner` builds a quantized model on CPU and executes each process in order. The processed model is stored as `runner.quantized_model` and is automatically used by subsequent evaluation and save operations.

```
Quantize ──► Build Model ──► Post-Process 1 ──► Post-Process 2 ──► Evaluate / Save
                              (e.g. LoRA SFT)
```

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

model, tokenizer = load_quantized_model_pt("./my_model_lora")
```

!!! info "save_quantized_model vs save_quantized_model_pt"
    | Method | Format | Use Case |
    |--------|--------|----------|
    | `save_quantized_model()` | safetensors (HF-compatible) | Standard quantized models (no post-processing) |
    | `save_quantized_model_pt()` | PyTorch `.pt` | Post-processed models (e.g. LoRA adapters) |

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

    For standard quantized models (without LoRA), use `save_quantized_model()` and serve via vLLM as described in the [vLLM Inference guide](vllm-inference.md).

!!! note "Supported Quantizers"
    LoRA SFT currently supports **GPTQ**-quantized models only. Support for other quantization methods (DBF, RTN) may be added in the future.
