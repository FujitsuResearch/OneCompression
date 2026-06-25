"""Core KL-distillation training loop for global PTQ.

Includes the main distillation loop, evaluation helpers, optimiser
wrappers (SAM, EMA, Lookahead), Fisher-adaptive LR, and progressive
layer unfreezing.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

"""

import contextlib
import gc
import math
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import (
    detect_quantization_method,
    disable_gradient_checkpointing,
    enable_gradient_checkpointing,
    remove_input_require_grads,
    get_language_model_backbone,
    get_logits,
)
from .losses import (
    clear_hooks,
    compute_entropy_loss,
    compute_intermediate_loss,
    compute_kl_loss,
    remove_hooks,
    setup_intermediate_hooks,
)
from .gptq_adapter import (
    find_gptq_modules,
    load_gptq_state,
    restore_gptq_original,
    save_gptq_state,
    setup_gptq_differentiable,
    setup_gptq_forwards_only,
    write_back_gptq_params,
)
from .dbf_adapter import (
    load_dbf_state,
    restore_dbf_original,
    save_dbf_state,
    setup_dbf_differentiable,
    setup_dbf_forwards_only,
    write_back_dbf_binary,
    write_back_dbf_scaling,
)

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM (Sharpness-Aware Minimisation)
# ---------------------------------------------------------------------------


class SAMOptimizer:
    """Sharpness-Aware Minimisation wrapper.

    Each step consists of (1) ascending towards the steepest direction
    within a *rho*-ball, (2) computing gradients at the perturbed point,
    and (3) restoring parameters and stepping.  The resulting flat
    minima tend to be more robust to quantisation noise.
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05):
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.rho = rho
        self._epsilon: Dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def first_step(self) -> None:
        """Perturb parameters towards steepest ascent."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = self.rho * p.grad / (grad_norm + 1e-12)
                p.add_(e_w)
                self._epsilon[id(p)] = e_w

    @torch.no_grad()
    def second_step(self, grad_clip: float = 0.0) -> None:
        """Undo perturbation, clip gradients, and step."""
        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid in self._epsilon:
                    p.sub_(self._epsilon[pid])
        if grad_clip > 0:
            all_p = [p for g in self.param_groups for p in g["params"]]
            nn.utils.clip_grad_norm_(all_p, grad_clip)
        self.base_optimizer.step()
        self._epsilon = {}

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad()

    @torch.no_grad()
    def undo_first_step(self) -> None:
        """Revert perturbation without stepping (for NaN recovery)."""
        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid in self._epsilon:
                    p.sub_(self._epsilon[pid])
        self._epsilon = {}

    def _grad_norm(self) -> torch.Tensor:
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm())
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms))


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------


class EMATracker:
    """Exponential moving average of parameters (Polyak averaging).

    Maintains a shadow copy that is a running exponential average of
    the optimised parameters.  Use :meth:`apply` before evaluation to
    swap in the averaged weights, and :meth:`restore` to return to the
    live training weights.
    """

    def __init__(self, params: list, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[int, torch.Tensor] = {}
        self._backup: Dict[int, torch.Tensor] = {}
        for p in params:
            self.shadow[id(p)] = p.data.clone()

    @torch.no_grad()
    def update(self, params: list) -> None:
        """Update shadow weights with current parameters."""
        for p in params:
            pid = id(p)
            if pid in self.shadow:
                self.shadow[pid].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, params: list) -> None:
        """Swap model parameters with EMA shadow (call before eval)."""
        for p in params:
            pid = id(p)
            if pid in self.shadow:
                self._backup[pid] = p.data.clone()
                p.data.copy_(self.shadow[pid])

    @torch.no_grad()
    def restore(self, params: list) -> None:
        """Restore original parameters (call after eval)."""
        for p in params:
            pid = id(p)
            if pid in self._backup:
                p.data.copy_(self._backup[pid])
        self._backup = {}


# ---------------------------------------------------------------------------
# Lookahead Optimiser
# ---------------------------------------------------------------------------


class LookaheadOptimizer:
    """Lookahead optimiser wrapper (slow-fast weight interpolation).

    Every *k* inner optimiser steps the *slow weights* are updated
    towards the current *fast weights* via linear interpolation, then
    the fast weights are reset to the new slow position.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        k: int = 5,
        alpha: float = 0.5,
    ):
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        self._slow_weights: Dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            for p in group["params"]:
                self._slow_weights[id(p)] = p.data.clone()

    def step(self) -> None:
        """Perform one inner step and sync slow weights every *k* steps."""
        self.base_optimizer.step()
        self._step_count += 1
        if self._step_count % self.k == 0:
            self._update_slow()

    @torch.no_grad()
    def _update_slow(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid in self._slow_weights:
                    slow = self._slow_weights[pid]
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad()

    @torch.no_grad()
    def sync_to_slow(self) -> None:
        """Copy slow weights to parameters (call at end of training)."""
        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid in self._slow_weights:
                    p.data.copy_(self._slow_weights[pid])


# ---------------------------------------------------------------------------
# Fisher-adaptive per-layer learning rate
# ---------------------------------------------------------------------------


def build_param_to_layer_map(
    modules: List[Tuple[str, nn.Module]],
) -> Dict[int, int]:
    """Map ``id(param)`` to the transformer layer index it belongs to.

    Layer index is extracted from the module name by looking for
    patterns like ``model.layers.12.self_attn.q_proj``.
    """
    param_to_layer: Dict[int, int] = {}
    for name, mod in modules:
        parts = name.split(".")
        layer_idx = -1
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    break
                except ValueError:
                    continue

        for p in mod.parameters():
            if p.requires_grad:
                param_to_layer[id(p)] = layer_idx
        for attr in ("_opt_scales", "_opt_zeros", "_opt_intweight",
                     "_opt_bp1", "_opt_bp3"):
            if hasattr(mod, attr):
                param_to_layer[id(getattr(mod, attr))] = layer_idx

    return param_to_layer


def compute_fisher_diagonal(
    model: nn.Module,
    dataloader: List[Dict[str, torch.Tensor]],
    dev: torch.device,
    param_to_layer: Dict[int, int],
    n_samples: int = 4,
) -> Dict[int, float]:
    """Estimate per-layer Fisher trace from squared gradients on NLL.

    A higher Fisher trace indicates that the layer is more sensitive to
    parameter changes, and should therefore receive a *smaller* learning
    rate to avoid destabilisation.
    """
    was_training = model.training
    model.eval()
    fisher_acc: Dict[int, float] = {}
    n_processed = 0

    for batch in dataloader:
        if n_processed >= n_samples:
            break
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        model.zero_grad()
        logits = get_logits(model(input_ids))

        if attention_mask is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous().view(-1)
            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            nll = (nll * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)
        else:
            nll = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                input_ids[:, 1:].contiguous().view(-1),
                reduction="mean",
            )
        nll.backward()

        for p in model.parameters():
            if p.grad is None:
                continue
            pid = id(p)
            if pid in param_to_layer:
                fisher_acc[param_to_layer[pid]] = (
                    fisher_acc.get(param_to_layer[pid], 0.0)
                    + p.grad.float().pow(2).sum().item()
                )

        model.zero_grad()
        n_processed += 1

    if n_processed > 0:
        for k in fisher_acc:
            fisher_acc[k] /= n_processed

    if was_training:
        model.train()
    return fisher_acc


def build_fisher_lr_multipliers(
    fisher_per_layer: Dict[int, float],
    min_mult: float = 0.1,
    max_mult: float = 10.0,
) -> Dict[int, float]:
    """Convert Fisher trace to per-layer LR multipliers.

    High Fisher (sensitive) layers get a *small* multiplier and vice
    versa:  ``mult = mean(Fisher) / Fisher_i``, clamped to
    ``[min_mult, max_mult]``.
    """
    if not fisher_per_layer:
        return {}
    values = list(fisher_per_layer.values())
    f_mean = sum(values) / len(values)
    if f_mean < 1e-20:
        return {k: 1.0 for k in fisher_per_layer}

    multipliers: Dict[int, float] = {}
    for layer_idx, f_val in fisher_per_layer.items():
        mult = max_mult if f_val < 1e-20 else f_mean / f_val
        multipliers[layer_idx] = max(min_mult, min(max_mult, mult))
    return multipliers


# ---------------------------------------------------------------------------
# Progressive layer unfreezing
# ---------------------------------------------------------------------------


def set_layer_grad(
    modules: List[Tuple[str, nn.Module]],
    layer_idx_set: set,
    enable: bool,
) -> None:
    """Enable or disable gradients for quantisation parameters in specific layers."""
    for name, mod in modules:
        parts = name.split(".")
        lidx = -1
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    lidx = int(parts[i + 1])
                    break
                except ValueError:
                    continue
        if lidx in layer_idx_set:
            for p in mod.parameters():
                p.requires_grad_(enable)
            for attr in ("_opt_scales", "_opt_zeros", "_opt_intweight",
                         "_opt_bp1", "_opt_bp3"):
                if hasattr(mod, attr):
                    getattr(mod, attr).requires_grad_(enable)


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
    dataloader: List[Dict[str, torch.Tensor]],
    dev: torch.device,
    temperature: float = 1.0,
) -> float:
    """Mean KL divergence over *dataloader* batches."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        logits_s = get_logits(model(input_ids))
        logits_t = get_logits(teacher_model(input_ids))
        total += compute_kl_loss(
            logits_t, logits_s, temperature, attention_mask=attention_mask,
        ).item()
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)


@torch.no_grad()
def eval_approx_ppl(
    model: nn.Module,
    dataloader: List[Dict[str, torch.Tensor]],
    dev: torch.device,
    max_batches: int = 8,
) -> float:
    """Cheap perplexity approximation on the first *max_batches* of data."""
    was_training = model.training
    model.eval()
    total_loss, n_tokens = 0.0, 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        logits = get_logits(model(input_ids))
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous().view(-1)
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            total_loss += (loss * shift_mask).sum().item()
            n_tokens += shift_mask.sum().item()
        else:
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            n_tokens += shift_labels.numel()

    if was_training:
        model.train()
    avg_loss = total_loss / max(n_tokens, 1)
    return math.exp(min(avg_loss, 100.0))


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
    num_samples: int,
    max_length: int,
    strategy: str,
    seed: int,
    calibration_dataset=None,
    model=None,
) -> List[Dict[str, torch.Tensor]]:
    """Build a list of calibration batches from onecomp calibration utils.

    Each element is a dict with ``input_ids`` and ``attention_mask``
    tensors on CPU.
    """
    from onecomp import CalibrationConfig
    from onecomp.calibration import prepare_calibration_dataset

    tokenizer = model_config.load_tokenizer()
    calib_config = CalibrationConfig(
        calibration_dataset=calibration_dataset or "c4",
        max_length=max_length,
        num_calibration_samples=num_samples,
        strategy=strategy,
        seed=seed,
    )
    cal = prepare_calibration_dataset(
        tokenizer=tokenizer,
        device="cpu",
        calibration_config=calib_config,
        model=model,
        logger=logger,
    )
    input_ids = cal["input_ids"]
    attention_mask = cal.get("attention_mask")

    batches = []
    for i in range(input_ids.size(0)):
        batch = {"input_ids": input_ids[i : i + 1]}
        if attention_mask is not None:
            batch["attention_mask"] = attention_mask[i : i + 1]
        batches.append(batch)
    return batches


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
    gptq_optimize_intweight: bool = False,
    gptq_intweight_lr: float = 1e-4,
    optimize_binary: bool = False,
    ste_k: float = 100.0,
    calibration_dataset=None,
    num_calibration_samples: int = 128,
    max_length: int = 2048,
    calibration_strategy: str = "drop_rand",
    calibration_seed: int = 0,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.01,
    eval_interval: int = 1,
    use_sam: bool = False,
    sam_rho: float = 0.02,
    use_ema: bool = False,
    ema_decay: float = 0.99,
    use_lookahead: bool = False,
    lookahead_k: int = 5,
    lookahead_alpha: float = 0.5,
    use_fisher_lr: bool = False,
    fisher_n_samples: int = 4,
    fisher_min_mult: float = 0.1,
    fisher_max_mult: float = 10.0,
    use_entropy_reg: bool = False,
    entropy_lambda: float = 0.1,
    entropy_temperature: float = 1.0,
    use_inter_loss: bool = False,
    lambda_inter: float = 10.0,
    use_progressive_unfreeze: bool = False,
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
    # 1b. Validation
    # ------------------------------------------------------------------
    if use_ema and use_lookahead:
        raise ValueError(
            "use_ema and use_lookahead cannot be enabled simultaneously. "
            "Both perform parameter averaging and will conflict."
        )

    # ------------------------------------------------------------------
    # 2. Calibration data
    # ------------------------------------------------------------------
    logger.info("Loading calibration data (n=%d, len=%d)...", num_calibration_samples, max_length)
    dataloader = _prepare_dataloader(
        model_config, num_calibration_samples, max_length,
        calibration_strategy, calibration_seed,
        calibration_dataset=calibration_dataset,
        model=quantized_model,
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
    binary_params: list = []

    if method == "gptq":
        gptq_modules = detected_modules
        original_forwards, scaling_params, intweight_params = setup_gptq_differentiable(
            gptq_modules, dev, gptq_optimize_intweight, ste_k,
        )
        param_groups = [{"params": scaling_params, "lr": gptq_lr}]
        if intweight_params:
            param_groups.append({"params": intweight_params, "lr": gptq_intweight_lr})

        logger.info(
            "Trainable: %d scales/zeros%s",
            len(scaling_params),
            f", {len(intweight_params)} intweight" if intweight_params else "",
        )

    elif method == "dbf":
        dbf_modules = detected_modules
        original_forwards, scaling_params, binary_params = setup_dbf_differentiable(
            dbf_modules, optimize_binary,
        )
        all_dbf_params = list(scaling_params)
        if binary_params:
            all_dbf_params += binary_params
        param_groups = [{"params": all_dbf_params, "lr": dbf_lr}]

        logger.info(
            "Trainable: %d scaling%s",
            len(scaling_params),
            f", {len(binary_params)} binary" if binary_params else "",
        )

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
    # 4b. Fisher-adaptive per-layer LR
    # ------------------------------------------------------------------
    if use_fisher_lr and detected_modules:
        logger.info("Computing Fisher diagonal for per-layer LR...")
        ptl_map = build_param_to_layer_map(detected_modules)
        fisher_diag = compute_fisher_diagonal(
            quantized_model, dataloader, dev, ptl_map, n_samples=fisher_n_samples,
        )
        fisher_mult = build_fisher_lr_multipliers(fisher_diag, fisher_min_mult, fisher_max_mult)
        if fisher_mult:
            logger.info(
                "Fisher LR multipliers: min=%.3f, max=%.3f, layers=%d",
                min(fisher_mult.values()), max(fisher_mult.values()), len(fisher_mult),
            )
            # Determine default weight decay based on method (AdamW vs Adam)
            default_wd = 0.01 if method == "gptq" else 0.0
            
            new_groups: list = []
            for pg in param_groups:
                for p in pg["params"]:
                    pid = id(p)
                    mult = fisher_mult.get(ptl_map.get(pid, -1), 1.0)
                    new_groups.append({
                        "params": [p],
                        "lr": pg["lr"] * mult,
                        "weight_decay": pg.get("weight_decay", default_wd),
                    })
            param_groups = new_groups

    # ------------------------------------------------------------------
    # 4c. Intermediate-layer hooks
    # ------------------------------------------------------------------
    student_hooks: Optional[dict] = None
    teacher_hooks: Optional[dict] = None
    if use_inter_loss:
        language_model = get_language_model_backbone(quantized_model)
        teacher_lm = get_language_model_backbone(teacher_model)
        student_hooks = setup_intermediate_hooks(language_model)
        teacher_hooks = setup_intermediate_hooks(teacher_lm)
        logger.info("Intermediate-layer hooks registered.")

    # ------------------------------------------------------------------
    # 4d. Progressive layer unfreezing setup
    # ------------------------------------------------------------------
    language_model_bb = get_language_model_backbone(quantized_model)
    total_layers = 0
    layers_per_epoch = 0
    if use_progressive_unfreeze:
        for attr_path in ("layers", "model.layers"):
            obj = language_model_bb
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                if hasattr(obj, "__len__"):
                    total_layers = len(obj)
                    break
            except AttributeError:
                continue
        if total_layers > 1:
            layers_per_epoch = max(1, total_layers // epochs)
            all_layer_indices = set(range(total_layers))
            initial_active = set(range(total_layers - layers_per_epoch, total_layers))
            set_layer_grad(detected_modules, all_layer_indices - initial_active, False)
            logger.info(
                "Progressive unfreeze: %d/%d layers active initially",
                len(initial_active), total_layers,
            )

    # ------------------------------------------------------------------
    # 5. Optimizer and LR scheduler
    # ------------------------------------------------------------------
    use_adamw = method == "gptq"
    for pg in param_groups:
        if use_adamw:
            pg.setdefault("weight_decay", 0.01)
        else:
            pg.setdefault("weight_decay", 0.0)
    base_optimizer = (torch.optim.AdamW if use_adamw else torch.optim.Adam)(param_groups)

    sam_optimizer: Optional[SAMOptimizer] = None
    if use_sam:
        sam_optimizer = SAMOptimizer(base_optimizer, rho=sam_rho)
        logger.info("SAM enabled (rho=%.4f)", sam_rho)

    lookahead: Optional[LookaheadOptimizer] = None
    if use_lookahead:
        lookahead = LookaheadOptimizer(base_optimizer, k=lookahead_k, alpha=lookahead_alpha)
        logger.info("Lookahead enabled (k=%d, alpha=%.2f)", lookahead_k, lookahead_alpha)

    optimizer = base_optimizer

    grad_accum_steps = max(1, grad_accum_steps)
    if use_sam and grad_accum_steps > 1:
        logger.warning(
            "Gradient accumulation is incompatible with SAM; "
            "falling back to grad_accum_steps=1."
        )
        grad_accum_steps = 1

    total_batches_all = epochs * len(dataloader)
    effective_total_steps = max(1, total_batches_all // grad_accum_steps)
    warmup_steps = int(effective_total_steps * warmup_ratio)
    use_lr_schedule = method == "gptq"
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    if use_lr_schedule:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_warmup_lr_lambda(step, effective_total_steps, warmup_steps, min_lr_ratio),
        )

    # EMA tracker
    all_opt_params = [p for pg in param_groups for p in pg["params"]]
    ema_tracker: Optional[EMATracker] = None
    if use_ema:
        ema_tracker = EMATracker(all_opt_params, decay=ema_decay)
        logger.info("EMA enabled (decay=%.4f)", ema_decay)

    if use_entropy_reg:
        logger.info("Entropy regularisation enabled (lambda=%.4f)", entropy_lambda)

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
        setup_gptq_forwards_only(gptq_modules, original_forwards, gptq_optimize_intweight)
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

    for epoch in range(epochs):
        quantized_model.train()
        epoch_kl, epoch_inter, epoch_loss = 0.0, 0.0, 0.0
        n_batches = 0

        # Progressive unfreeze: expand active layers
        if use_progressive_unfreeze and total_layers > 1:
            active_start = max(0, total_layers - (epoch + 1) * layers_per_epoch)
            active_set = set(range(active_start, total_layers))
            set_layer_grad(detected_modules, active_set, True)
            set_layer_grad(detected_modules, set(range(active_start)), False)

        n_accum_steps = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)

            n_accum_steps += 1
            is_accum_boundary = (
                (batch_idx + 1) % grad_accum_steps == 0
                or batch_idx == total_batches - 1
            )
            # Actual number of steps in this accumulation cycle
            current_accum_steps = n_accum_steps if is_accum_boundary else grad_accum_steps

            # --- Forward + loss computation ---
            def _forward_and_loss() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if use_inter_loss and student_hooks and teacher_hooks:
                    clear_hooks(student_hooks)
                    clear_hooks(teacher_hooks)

                with amp_ctx:
                    logits_s = get_logits(quantized_model(input_ids))
                    with torch.no_grad():
                        logits_t = get_logits(teacher_model(input_ids))

                kl = compute_kl_loss(
                    logits_t, logits_s, temperature, attention_mask=attention_mask,
                )

                inter = torch.tensor(0.0, device=dev)
                if use_inter_loss and student_hooks and teacher_hooks:
                    inter = compute_intermediate_loss(student_hooks, teacher_hooks)

                total = kl + (lambda_inter * inter if use_inter_loss else 0.0)

                if use_entropy_reg:
                    ent_loss = compute_entropy_loss(logits_s, entropy_temperature)
                    total = total + entropy_lambda * ent_loss

                return total, kl, inter

            # --- SAM path (two-step) ---
            if sam_optimizer is not None:
                optimizer.zero_grad()
                loss, kl_loss, inter_loss = _forward_and_loss()

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    continue

                (loss / current_accum_steps).backward()

                if _has_nan_grad(param_groups):
                    optimizer.zero_grad()
                    n_accum_steps = 0
                    if scheduler is not None:
                        scheduler.step()
                    continue

                sam_optimizer.first_step()
                loss2, _, _ = _forward_and_loss()

                if torch.isnan(loss2) or torch.isinf(loss2):
                    sam_optimizer.undo_first_step()
                    sam_optimizer.zero_grad()
                    n_accum_steps = 0
                    if scheduler is not None:
                        scheduler.step()
                    continue

                (loss2 / current_accum_steps).backward()

                if is_accum_boundary:
                    if lookahead:
                        sam_optimizer.second_step(grad_clip=grad_clip)
                        lookahead._step_count += 1
                        if lookahead._step_count % lookahead.k == 0:
                            lookahead._update_slow()
                    else:
                        sam_optimizer.second_step(grad_clip=grad_clip)

                    sam_optimizer.zero_grad()
                    n_accum_steps = 0
                    if scheduler is not None:
                        scheduler.step()

            # --- Standard path ---
            else:
                if (batch_idx % grad_accum_steps == 0):
                    optimizer.zero_grad()

                loss, kl_loss, inter_loss = _forward_and_loss()

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    n_accum_steps = 0
                    continue

                (loss / current_accum_steps).backward()

                if is_accum_boundary:
                    if _has_nan_grad(param_groups):
                        optimizer.zero_grad()
                        n_accum_steps = 0
                        continue

                    if lookahead:
                        if grad_clip > 0:
                            all_p = [p for pg in param_groups for p in pg["params"]]
                            nn.utils.clip_grad_norm_(all_p, grad_clip)
                        lookahead.step()
                    else:
                        clip_and_step(optimizer, param_groups, grad_clip)

                    if scheduler is not None:
                        scheduler.step()
                    
                    optimizer.zero_grad()
                    n_accum_steps = 0

            # EMA update (only at accumulation boundaries)
            if ema_tracker is not None and is_accum_boundary:
                ema_tracker.update(all_opt_params)

            epoch_kl += kl_loss.item()
            epoch_inter += inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss
            epoch_loss += loss.item()
            n_batches += 1

        avg_kl = epoch_kl / max(n_batches, 1)

        # Periodic evaluation
        do_eval = ((epoch + 1) % eval_interval == 0) or (epoch == epochs - 1)
        if do_eval:
            # Swap to EMA params for evaluation
            if ema_tracker is not None:
                ema_tracker.apply(all_opt_params)

            if method == "gptq":
                write_back_gptq_params(gptq_modules, gptq_optimize_intweight)
                restore_gptq_original(gptq_modules, original_forwards)
            elif method == "dbf":
                write_back_dbf_binary(dbf_modules)
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
                setup_gptq_forwards_only(gptq_modules, original_forwards, gptq_optimize_intweight)
            elif method == "dbf":
                setup_dbf_forwards_only(dbf_modules, original_forwards)

            # Restore non-EMA params for continued training
            if ema_tracker is not None:
                ema_tracker.restore(all_opt_params)

            log_parts = [f"train_KL={avg_kl:.6f}"]
            if use_inter_loss:
                log_parts.append(f"L_inter={epoch_inter / max(n_batches, 1):.6f}")
            log_parts.append(f"eval_KL={current_kl:.6f} (best={best_kl:.6f})")
            logger.info("Epoch %d/%d: %s", epoch + 1, epochs, " | ".join(log_parts))

            # Early stopping
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                logger.info("Early stopping (patience=%d)", early_stopping_patience)
                stopped_early = True
                break
        else:
            logger.info("Epoch %d/%d: train_KL=%.6f", epoch + 1, epochs, avg_kl)

    # ------------------------------------------------------------------
    # 8. Finalize
    # ------------------------------------------------------------------
    if lookahead is not None:
        lookahead.sync_to_slow()

    if ema_tracker is not None:
        ema_tracker.apply(all_opt_params)

    if use_progressive_unfreeze and total_layers > 1:
        set_layer_grad(detected_modules, set(range(total_layers)), True)

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
            write_back_gptq_params(gptq_modules, gptq_optimize_intweight)
        elif method == "dbf":
            write_back_dbf_binary(dbf_modules)
            write_back_dbf_scaling(dbf_modules)

    if method == "gptq":
        restore_gptq_original(gptq_modules, original_forwards, cleanup=False)
    elif method == "dbf":
        restore_dbf_original(dbf_modules, original_forwards, cleanup=False)

    # Cleanup hooks
    if use_inter_loss:
        if student_hooks is not None:
            remove_hooks(student_hooks)
        if teacher_hooks is not None:
            remove_hooks(teacher_hooks)

    if grad_ckpt_enabled:
        disable_gradient_checkpointing(quantized_model)
        remove_input_require_grads(quantized_model)
    if original_use_cache is not None:
        quantized_model.config.use_cache = original_use_cache

    # Final evaluation
    quantized_model.eval()
    final_kl = eval_kl(quantized_model, teacher_model, dataloader, dev, temperature)

    # Cleanup
    if method == "gptq":
        # Final cleanup of differentiable parameters
        restore_gptq_original(gptq_modules, original_forwards, cleanup=True)
    elif method == "dbf":
        restore_dbf_original(dbf_modules, original_forwards, cleanup=True)
    
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
        "features": {
            "sam": use_sam,
            "ema": use_ema,
            "lookahead": use_lookahead,
            "fisher_lr": use_fisher_lr,
            "entropy_reg": use_entropy_reg,
            "inter_loss": use_inter_loss,
            "progressive_unfreeze": use_progressive_unfreeze,
            "gradient_checkpointing": grad_ckpt_enabled,
            "mixed_precision": use_mixed_precision,
            "grad_accum_steps": grad_accum_steps,
        },
    }
