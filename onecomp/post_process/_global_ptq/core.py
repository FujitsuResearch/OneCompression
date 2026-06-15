"""Core KL-distillation training loop for global PTQ.

Includes the main distillation loop, evaluation helpers, and LR schedule.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

"""

import contextlib
import gc
import math
from logging import getLogger
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .dbf_adapter import (
    load_dbf_state,
    restore_dbf_original,
    save_dbf_state,
    setup_dbf_differentiable,
    setup_dbf_forwards_only,
    write_back_dbf_scaling,
)
from .gptq_adapter import (
    load_gptq_state,
    restore_gptq_original,
    save_gptq_state,
    setup_gptq_differentiable,
    setup_gptq_forwards_only,
    write_back_gptq_params,
)
from .helpers import (
    detect_quantization_method,
    disable_gradient_checkpointing,
    enable_gradient_checkpointing,
    get_logits,
    remove_input_require_grads,
)
from .losses import compute_kl_loss

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def cosine_warmup_lr_lambda(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.01,
) -> float:
    """Warmup-then-cosine-decay multiplier for ``LambdaLR``."""
    if step < warmup_steps:
        return max(step / max(warmup_steps, 1), 1e-6)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_kl(
    model: nn.Module,
    teacher_model: nn.Module,
    dataloader: List[torch.Tensor],
    dev: torch.device,
    temperature: float = 1.0,
) -> float:
    """Mean KL divergence over *dataloader* batches."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for input_ids in dataloader:
        input_ids = input_ids.to(dev)
        logits_s = get_logits(model(input_ids))
        logits_t = get_logits(teacher_model(input_ids))
        total += compute_kl_loss(logits_t, logits_s, temperature).item()
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)


def clip_and_step(
    optimizer: torch.optim.Optimizer,
    param_groups: list,
    grad_clip: float,
) -> None:
    """Gradient clipping followed by an optimizer step."""
    if grad_clip > 0:
        all_params = [p for pg in param_groups for p in pg["params"]]
        nn.utils.clip_grad_norm_(all_params, grad_clip)
    optimizer.step()


# ---------------------------------------------------------------------------
# Calibration data → simple batch list
# ---------------------------------------------------------------------------


def _prepare_dataloader(
    model_config,
    model: nn.Module,
    calibration_config,
) -> List[torch.Tensor]:
    """Build a list of ``input_ids`` tensors from onecomp calibration utils.

    Each element is a ``(1, seq_len)`` tensor on CPU.
    """
    from ...calibration import CalibrationConfig, prepare_calibration_dataset

    tokenizer = model_config.load_tokenizer()
    cal = prepare_calibration_dataset(
        tokenizer=tokenizer,
        device="cpu",
        calibration_config=calibration_config,
        model=model,
        logger=logger,
    )
    input_ids = cal["input_ids"]
    return [input_ids[i : i + 1] for i in range(input_ids.size(0))]


# ---------------------------------------------------------------------------
# NaN / Inf guard
# ---------------------------------------------------------------------------


def _has_nan_grad(param_groups: list) -> bool:
    """Return True if any trainable parameter has NaN or Inf gradients."""
    for pg in param_groups:
        for p in pg["params"]:
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                return True
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_kl_distillation(
    quantized_model: nn.Module,
    model_config,
    *,
    epochs: int = 5,
    gptq_lr: float = 1e-5,
    dbf_lr: float = 5e-5,
    temperature: float = 1.0,
    grad_clip: float = 1.0,
    calibration_config=None,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.01,
    eval_interval: int = 1,
    use_gradient_checkpointing: bool = True,
    early_stopping_patience: int = 0,
    use_mixed_precision: bool = False,
    grad_accum_steps: int = 1,
) -> Dict:
    """Run KL-distillation global PTQ on a GPTQ or DBF quantized model.

    The model is modified **in-place**.  Returns a results dict.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Detect method
    # ------------------------------------------------------------------
    method, detected_modules = detect_quantization_method(quantized_model)
    if method is None:
        logger.warning("No quantized layers detected — skipping global PTQ.")
        return {"global_executed": False, "reason": "not_quantized"}

    if method not in ("gptq", "dbf"):
        logger.info("Method '%s' detected — not supported.", method)
        return {"global_executed": False, "reason": f"unsupported_method_{method}"}

    logger.info("[Global PTQ] method=%s, modules=%d", method, len(detected_modules))

    # ------------------------------------------------------------------
    # 2. Calibration data
    # ------------------------------------------------------------------
    from ...calibration import CalibrationConfig

    if calibration_config is None:
        calibration_config = CalibrationConfig(num_calibration_samples=128)

    logger.info(
        "Loading calibration data (n=%d, len=%d)...",
        calibration_config.num_calibration_samples,
        calibration_config.max_length,
    )
    dataloader = _prepare_dataloader(
        model_config,
        quantized_model,
        calibration_config,
    )

    # ------------------------------------------------------------------
    # 3. Teacher model (FP16)
    # ------------------------------------------------------------------
    logger.info("Loading FP16 teacher model...")
    teacher_model = model_config.load_model(device_map="cpu")
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.to(dev)

    # ------------------------------------------------------------------
    # 4. Move student to GPU and set up differentiable parameters
    # ------------------------------------------------------------------
    quantized_model.to(dev)
    _, detected_modules = detect_quantization_method(quantized_model)

    gptq_modules: list = []
    dbf_modules: list = []
    original_forwards: Dict[str, object] = {}
    param_groups: list = []

    if method == "gptq":
        gptq_modules = detected_modules
        original_forwards, scaling_params = setup_gptq_differentiable(
            gptq_modules,
            dev,
        )
        param_groups = [{"params": scaling_params, "lr": gptq_lr}]

        logger.info("Trainable: %d scales/zeros", len(scaling_params))

    elif method == "dbf":
        dbf_modules = detected_modules
        original_forwards, scaling_params = setup_dbf_differentiable(
            dbf_modules,
        )
        param_groups = [{"params": list(scaling_params), "lr": dbf_lr}]

        logger.info("Trainable: %d scaling", len(scaling_params))

    total_trainable = sum(len(pg["params"]) for pg in param_groups)
    if total_trainable == 0:
        logger.warning("No trainable parameters — skipping.")
        if method == "gptq":
            restore_gptq_original(gptq_modules, original_forwards)
        elif method == "dbf":
            restore_dbf_original(dbf_modules, original_forwards)
        quantized_model.cpu()
        del teacher_model
        gc.collect()
        torch.cuda.empty_cache()
        return {"global_executed": False, "reason": "no_params"}

    # ------------------------------------------------------------------
    # 5. Optimizer and LR scheduler
    # ------------------------------------------------------------------
    use_adamw = method == "gptq"
    for pg in param_groups:
        if use_adamw:
            pg.setdefault("weight_decay", 0.01)
        else:
            pg.setdefault("weight_decay", 0.0)
    optimizer = (torch.optim.AdamW if use_adamw else torch.optim.Adam)(param_groups)

    grad_accum_steps = max(1, grad_accum_steps)

    total_batches_all = epochs * len(dataloader)
    effective_total_steps = max(1, total_batches_all // grad_accum_steps)
    warmup_steps = int(effective_total_steps * warmup_ratio)
    use_lr_schedule = method == "gptq"
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    if use_lr_schedule:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_warmup_lr_lambda(
                step, effective_total_steps, warmup_steps, min_lr_ratio
            ),
        )

    # Gradient checkpointing
    grad_ckpt_enabled = False
    original_use_cache = getattr(getattr(quantized_model, "config", None), "use_cache", None)
    if use_gradient_checkpointing:
        if original_use_cache is not None:
            quantized_model.config.use_cache = False
        grad_ckpt_enabled = enable_gradient_checkpointing(quantized_model)
        if grad_ckpt_enabled:
            logger.info("Gradient checkpointing enabled")
            if hasattr(quantized_model, "enable_input_require_grads"):
                quantized_model.enable_input_require_grads()

    # Mixed precision context
    amp_ctx: contextlib.AbstractContextManager = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_mixed_precision and dev.type == "cuda"
        else contextlib.nullcontext()
    )
    if use_mixed_precision and dev.type == "cuda":
        logger.info("Mixed precision enabled (BF16 autocast)")

    # ------------------------------------------------------------------
    # 6. Initial state (for rollback)
    # ------------------------------------------------------------------
    if method == "gptq":
        initial_state = save_gptq_state(gptq_modules)
        restore_gptq_original(gptq_modules, original_forwards)
    else:
        initial_state = save_dbf_state(dbf_modules)
        restore_dbf_original(dbf_modules, original_forwards)

    initial_kl = eval_kl(quantized_model, teacher_model, dataloader, dev, temperature)
    logger.info("Initial KL = %.6f", initial_kl)

    if method == "gptq":
        setup_gptq_forwards_only(gptq_modules, original_forwards)
    elif method == "dbf":
        setup_dbf_forwards_only(dbf_modules, original_forwards)

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    best_kl = initial_kl
    best_state = None
    eval_interval = max(1, eval_interval)
    patience_counter = 0
    stopped_early = False
    total_batches = len(dataloader)

    try:
        for epoch in range(epochs):
            quantized_model.train()
            epoch_kl, epoch_loss = 0.0, 0.0
            n_batches = 0

            for batch_idx, input_ids in enumerate(dataloader):
                input_ids = input_ids.to(dev)
                is_accum_boundary = (
                    batch_idx + 1
                ) % grad_accum_steps == 0 or batch_idx == total_batches - 1

                if batch_idx % grad_accum_steps == 0:
                    optimizer.zero_grad()

                with amp_ctx:
                    logits_s = get_logits(quantized_model(input_ids))
                    with torch.no_grad():
                        logits_t = get_logits(teacher_model(input_ids))

                loss = compute_kl_loss(logits_t, logits_s, temperature)

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue

                (loss / grad_accum_steps).backward()

                if is_accum_boundary:
                    if _has_nan_grad(param_groups):
                        optimizer.zero_grad()
                        continue

                    clip_and_step(optimizer, param_groups, grad_clip)

                    if scheduler is not None:
                        scheduler.step()

                epoch_kl += loss.item()
                epoch_loss += loss.item()
                n_batches += 1

            avg_kl = epoch_kl / max(n_batches, 1)

            # Periodic evaluation
            do_eval = ((epoch + 1) % eval_interval == 0) or (epoch == epochs - 1)
            if do_eval:
                if method == "gptq":
                    write_back_gptq_params(gptq_modules)
                    restore_gptq_original(gptq_modules, original_forwards)
                elif method == "dbf":
                    write_back_dbf_scaling(dbf_modules)
                    restore_dbf_original(dbf_modules, original_forwards)

                current_kl = eval_kl(quantized_model, teacher_model, dataloader, dev, temperature)

                if current_kl < best_kl:
                    best_kl = current_kl
                    patience_counter = 0
                    if method == "gptq":
                        best_state = save_gptq_state(gptq_modules)
                    else:
                        best_state = save_dbf_state(dbf_modules)
                else:
                    patience_counter += 1

                if method == "gptq":
                    setup_gptq_forwards_only(gptq_modules, original_forwards)
                elif method == "dbf":
                    setup_dbf_forwards_only(dbf_modules, original_forwards)

                logger.info(
                    "Epoch %d/%d: train_KL=%.6f | eval_KL=%.6f (best=%.6f)",
                    epoch + 1,
                    epochs,
                    avg_kl,
                    current_kl,
                    best_kl,
                )

                # Early stopping
                if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                    logger.info("Early stopping (patience=%d)", early_stopping_patience)
                    stopped_early = True
                    break
            else:
                logger.info("Epoch %d/%d: train_KL=%.6f", epoch + 1, epochs, avg_kl)

    finally:
        if grad_ckpt_enabled:
            disable_gradient_checkpointing(quantized_model)
            remove_input_require_grads(quantized_model)
        if original_use_cache is not None:
            quantized_model.config.use_cache = original_use_cache

    # ------------------------------------------------------------------
    # 8. Finalize
    # ------------------------------------------------------------------
    if best_state is not None and best_kl < initial_kl:
        if method == "gptq":
            load_gptq_state(gptq_modules, best_state)
        else:
            load_dbf_state(dbf_modules, best_state)
        logger.info("Loaded best state (KL=%.6f)", best_kl)
    elif best_kl >= initial_kl:
        logger.info("No improvement — rolling back to initial state.")
        if method == "gptq":
            load_gptq_state(gptq_modules, initial_state)
        else:
            load_dbf_state(dbf_modules, initial_state)
        best_kl = initial_kl
    else:
        if method == "gptq":
            write_back_gptq_params(gptq_modules)
        elif method == "dbf":
            write_back_dbf_scaling(dbf_modules)

    if method == "gptq":
        restore_gptq_original(gptq_modules, original_forwards, cleanup=True)
    elif method == "dbf":
        restore_dbf_original(dbf_modules, original_forwards)

    # Final evaluation
    quantized_model.eval()
    final_kl = eval_kl(quantized_model, teacher_model, dataloader, dev, temperature)

    # Cleanup
    del teacher_model
    gc.collect()
    torch.cuda.empty_cache()

    # Move model back to CPU (PostQuantizationProcess contract)
    quantized_model.cpu()
    for param in quantized_model.parameters():
        param.requires_grad = False

    improvement = ((initial_kl - final_kl) / max(initial_kl, 1e-10)) * 100.0
    logger.info("KL: %.6f -> %.6f (%.2f%%)", initial_kl, final_kl, improvement)

    return {
        "global_executed": True,
        "method": method,
        "initial_kl": initial_kl,
        "final_kl": final_kl,
        "best_kl": best_kl,
        "improvement_pct": improvement,
        "epochs": epoch + 1 if stopped_early else epochs,
        "stopped_early": stopped_early,
        "gradient_checkpointing": grad_ckpt_enabled,
        "mixed_precision": use_mixed_precision,
        "grad_accum_steps": grad_accum_steps,
    }
