## GlobalPTQ

`GlobalPTQ` globally optimises quantization parameters via KL-divergence distillation from a full-precision teacher model.
It supports both continuous parameters (scales, zeros, scaling factors) and discrete parameters (GPTQ integer weights, DBF binary matrices) with Straight-Through Estimator (STE) gradients.

### Single-GPU

```python
from onecomp import Runner, ModelConfig, GPTQ
from onecomp_globalptq import GlobalPTQ

model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf", device="cuda:0")
gptq = GPTQ(wbits=4, groupsize=128)

global_ptq = GlobalPTQ(
    epochs=5,
    gptq_lr=1e-5,
    gptq_optimize_intweight=True,   # discrete integer-weight optimisation
    use_sam=True,                    # Sharpness-Aware Minimisation
    num_calibration_samples=128,
    max_length=2048,
)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    post_processes=[global_ptq],
)
runner.run()
```

### Multi-GPU with DeepSpeed (GlobalPTQDistributed)

For large models that do not fit on a single GPU:

```python
from onecomp import Runner, ModelConfig, GPTQ
from onecomp_globaptq import GlobalPTQDistributed

model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf", device="cuda:0")
gptq = GPTQ(wbits=4, groupsize=128)

global_ptq = GlobalPTQDistributed(
    epochs=5,
    gptq_lr=1e-5,
    gptq_optimize_intweight=True,
    deepspeed_config="ds_zero2.json",
    num_calibration_samples=128,
    max_length=2048,
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

### Key Parameters

#### Continuous Parameter Optimisation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | `5` | Number of distillation epochs |
| `gptq_lr` | `1e-5` | Learning rate for GPTQ scales/zeros |
| `dbf_lr` | `5e-5` | Learning rate for DBF scaling parameters |
| `temperature` | `1.0` | Softmax temperature for KL divergence |
| `num_calibration_samples` | `128` | Number of calibration samples |
| `max_length` | `2048` | Maximum sequence length |
| `use_gradient_checkpointing` | `True` | Reduce GPU memory via recomputation |
| `early_stopping_patience` | `0` | Stop early if KL does not improve (0 = disabled) |
| `use_mixed_precision` | `False` | Enable BF16 autocast |
| `grad_accum_steps` | `1` | Gradient accumulation steps |

#### Discrete Parameter Optimisation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gptq_optimize_intweight` | `False` | Optimise GPTQ integer weights via STE |
| `gptq_intweight_lr` | `1e-4` | Learning rate for integer-weight parameters |
| `optimize_binary` | `False` | Optimise DBF/MDBF binary matrices via sign-STE |
| `gptq_ste_k` | `100.0` | Smoothness for GPTQ integer-weight STE rounding |
| `dbf_ste_k` | `2.0` | Sharpness for DBF binary sign STE |
| `mdbf_ste_k` | `2.0` | Sharpness for MDBF binary sign STE |

#### Advanced Optimisation Techniques

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_sam` | `False` | Sharpness-Aware Minimisation |
| `sam_rho` | `0.02` | SAM perturbation radius |
| `use_ema` | `False` | Exponential Moving Average of parameters |
| `ema_decay` | `0.99` | EMA decay rate |
| `use_lookahead` | `False` | Lookahead optimiser wrapper |
| `lookahead_k` | `5` | Lookahead sync interval |
| `lookahead_alpha` | `0.5` | Lookahead interpolation weight |
| `use_fisher_lr` | `False` | Fisher-information-adaptive per-layer LR |
| `fisher_n_samples` | `4` | Samples for Fisher diagonal estimation |
| `use_entropy_reg` | `False` | Entropy regularisation on weight distributions |
| `entropy_lambda` | `0.1` | Entropy regularisation strength |
| `use_inter_loss` | `False` | Intermediate-layer cosine alignment loss |
| `lambda_inter` | `10.0` | Intermediate loss weight |
| `use_progressive_unfreeze` | `False` | Gradually unfreeze layers from output to input |

#### GlobalPTQDistributed Additional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `deepspeed_config` | `None` | Path to DeepSpeed config JSON |
| `w_distill` | `1.0` | Weight for KL distillation loss |
| `w_ntp` | `0.0` | Weight for next-token prediction loss |
| `bf16` | `True` | Enable bfloat16 training |
| `per_device_train_batch_size` | `1` | Batch size per GPU |
| `gradient_accumulation_steps` | `1` | Gradient accumulation steps |

## Development

```bash
pip install -e ".[dev]"
# or, if using uv
uv pip install -e ".[dev]"
```

## License

Proprietary. See LICENSE for details.
