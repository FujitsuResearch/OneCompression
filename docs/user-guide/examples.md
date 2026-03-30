# Examples

This page demonstrates common usage patterns beyond the basic workflow.

## One-liner with `auto_run`

The simplest way to quantize a model:

```python
from onecomp import Runner

# Default: AutoBit (VRAM auto-estimation, ILP mixed-precision) + QEP
Runner.auto_run(model_id="meta-llama/Llama-2-7b-hf")

# Specify VRAM budget
Runner.auto_run(model_id="meta-llama/Llama-2-7b-hf", total_vram_gb=8)

# Fixed 4-bit, custom save directory
Runner.auto_run(
    model_id="meta-llama/Llama-2-7b-hf",
    wbits=4,
    save_dir="./llama2-7b-gptq-4bit",
)

# Without QEP, skip evaluation
Runner.auto_run(
    model_id="meta-llama/Llama-2-7b-hf",
    qep=False,
    evaluate=False,
)

# Also evaluate the original model for comparison
Runner.auto_run(
    model_id="meta-llama/Llama-2-7b-hf",
    eval_original_model=True,
)
```

## CLI

The `onecomp` command provides the same functionality from the terminal:

```bash
# Default (AutoBit with VRAM auto-estimation + QEP)
onecomp meta-llama/Llama-2-7b-hf

# Specify VRAM budget
onecomp meta-llama/Llama-2-7b-hf --total-vram-gb 8

# Fixed 4-bit, custom save directory
onecomp meta-llama/Llama-2-7b-hf --wbits 4 --save-dir ./llama2-7b-gptq-4bit

# Without QEP, skip evaluation
onecomp meta-llama/Llama-2-7b-hf --no-qep --no-eval

# Also evaluate the original model
onecomp meta-llama/Llama-2-7b-hf --eval-original

# Skip saving
onecomp meta-llama/Llama-2-7b-hf --save-dir none
```

See the [CLI Reference](cli.md) for all options.

---

## GPTQ with QEP (3-bit)

Quantize a model using GPTQ at 3-bit precision with QEP to improve quality:

```python
from onecomp import ModelConfig, Runner, GPTQ, setup_logger

setup_logger()

model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    device="cuda:0",
)
gptq = GPTQ(wbits=3)

runner = Runner(model_config=model_config, quantizer=gptq, qep=True)
runner.run()

_, _, quantized_ppl = runner.calculate_perplexity()
print(f"Quantized model perplexity: {quantized_ppl}")
```

## GPTQ without QEP

Standard GPTQ quantization without error propagation:

```python
gptq = GPTQ(wbits=3)
runner = Runner(model_config=model_config, quantizer=gptq, qep=False)
runner.run()
```

## Chunked Calibration (Large-scale Data)

When using large calibration datasets that don't fit in GPU memory, use chunked calibration.
The `calibration_batch_size` parameter splits the forward pass into smaller batches while
accumulating statistics exactly:

```python
gptq = GPTQ(wbits=4, groupsize=128)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    max_length=2048,
    num_calibration_samples=1024,
    calibration_batch_size=128,
)
runner.run()
```

!!! info
    Chunked calibration is mathematically exact -- it accumulates \(X^T X\) across batches without approximation.

## Multi-GPU Quantization

Distribute layer-wise quantization across multiple GPUs:

```python
runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    multi_gpu=True,
)
runner.run()

# Or specify particular GPUs
runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    multi_gpu=True,
    gpu_ids=[0, 2, 3],
)
runner.run()
```

## Comparing Multiple Quantizers

Run multiple quantizers in a single session with shared calibration data:

```python
from onecomp import GPTQ
from onecomp.quantizer.jointq import JointQ
import torch

gptq = GPTQ(wbits=4, groupsize=128, calc_quant_error=True)
jointq = JointQ(bits=4, group_size=128, calc_quant_error=True,
                device=torch.device(0))

runner = Runner(
    model_config=model_config,
    quantizers=[gptq, jointq],
    max_length=2048,
    num_calibration_samples=1024,
    calibration_batch_size=128,
)
runner.run()

# Benchmark perplexity across all quantizers
ppl_dict = runner.benchmark_perplexity()
print(ppl_dict)
# {'original': 5.47, 'GPTQ': 5.72, 'JointQ': 5.68}
```

## Saving and Loading Quantized Models

### Save the quantized model

```python
# Save with packed integer weights (compatible with vLLM)
runner.save_quantized_model("./output/my_quantized_model")

# Or save dequantized FP16 weights
runner.save_dequantized_model("./output/my_dequantized_model")
```

### Load a saved quantized model

```python
from onecomp import load_quantized_model

model, tokenizer = load_quantized_model("./output/my_quantized_model")

# Use like any Hugging Face model
inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## LoRA SFT: Accuracy Recovery

Quantize a model and apply LoRA SFT to recover accuracy lost during quantization:

```python
from onecomp import GPTQ, ModelConfig, Runner, PostProcessLoraSFT, setup_logger

setup_logger()

model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
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
    epochs=2,
    batch_size=2,
    gradient_accumulation_steps=4,
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

original_ppl, _, quantized_ppl = runner.calculate_perplexity(
    original_model=True, quantized_model=True,
)
print(f"Original PPL:              {original_ppl:.4f}")
print(f"Quantized + LoRA SFT PPL:  {quantized_ppl:.4f}")
```

## LoRA SFT: Knowledge Injection

Inject custom knowledge into a quantized model using a JSONL file:

```python
from onecomp import GPTQ, ModelConfig, Runner, PostProcessLoraSFT

model_config = ModelConfig(model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", device="cuda:0")
gptq = GPTQ(wbits=4, groupsize=128)

post_process = PostProcessLoraSFT(
    data_files="./my_knowledge.jsonl",
    max_length=256,
    epochs=20,
    batch_size=2,
    lr=3e-4,
    lora_r=16,
    lora_alpha=32,
)

runner = Runner(model_config=model_config, quantizer=gptq, post_processes=[post_process])
runner.run()
```

See [Post-Process (LoRA SFT)](post-process.md) for the full guide including teacher distillation and save/load.

## Saving and Loading LoRA Models

LoRA-applied models use a dedicated save/load API:

```python
# Save after LoRA SFT
runner.save_quantized_model_pt("./my_model_lora")

# Load
from onecomp import load_quantized_model_pt
model, tokenizer = load_quantized_model_pt("./my_model_lora")
```

!!! note
    For standard quantized models (without LoRA), use `save_quantized_model()` / `load_quantized_model()` instead.

## Analyzing Cumulative Error

Analyze how quantization error accumulates across layers:

```python
runner.run()

results = runner.analyze_cumulative_error(
    layer_keywords=["mlp.down_proj"],
    plot_path="cumulative_error.png",
    json_path="cumulative_error.json",
)
```

## Saving Quantization Statistics

```python
runner.run()
runner.print_quantization_results()
runner.save_quantization_statistics("stats.json")
runner.save_quantization_results("results.pt")
```

## Layer Selection

### Quantize only specific layers

```python
gptq = GPTQ(
    wbits=4,
    include_layer_names=["model.layers.0.self_attn.q_proj"],
)
```

### Quantize layers matching keywords

```python
gptq = GPTQ(
    wbits=4,
    include_layer_keywords=["q_proj", "k_proj", "v_proj"],
)
```

### Exclude layers by keyword

```python
gptq = GPTQ(
    wbits=4,
    exclude_layer_keywords=["down_proj", "gate_proj"],
)
```
