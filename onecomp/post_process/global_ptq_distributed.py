"""Trainer-based Global Post-Training Quantization process.

Provides ``GlobalPTQDistributed``, a Trainer-based alternative to
``GlobalPTQ`` that supports multi-GPU training via DeepSpeed.

Compared to ``GlobalPTQ``:
- Uses ``transformers.Trainer`` for the training loop
- Supports DeepSpeed ZeRO-2 via ``deepspeed_config``
- Supports combined KL + NTP loss (enabling QAT mode)

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii

"""

import gc
import math
import os
import tempfile
from dataclasses import dataclass, field
from logging import getLogger
from typing import Optional

import torch
import torch.nn as nn

from ..calibration import CalibrationConfig
from ..model_config import ModelConfig
from ._base import PostQuantizationProcess

logger = getLogger(__name__)


def _remove_deepspeed_hooks(model: nn.Module) -> None:
    """Remove DeepSpeed-injected forward (pre/post) hooks from every submodule.

    DeepSpeed registers hooks whose callables are local closures defined inside
    ``DeepSpeedEngine`` (e.g. ``_module_forward_post_hook``). They are not
    picklable, so they must be removed before ``torch.save(model)``.
    """
    hook_dict_names = (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks_with_kwargs",
    )
    for module in model.modules():
        forward_hooks = getattr(module, "_forward_hooks", None)
        pre_hooks = getattr(module, "_forward_pre_hooks", None)
        stale_ids = set()
        for hook_dict in (forward_hooks, pre_hooks):
            if not hook_dict:
                continue
            for handle_id, hook in list(hook_dict.items()):
                if "DeepSpeedEngine" in getattr(hook, "__qualname__", ""):
                    stale_ids.add(handle_id)
        if not stale_ids:
            continue
        for hook_dict_name in hook_dict_names:
            hook_dict = getattr(module, hook_dict_name, None)
            if not hook_dict:
                continue
            for handle_id in stale_ids:
                hook_dict.pop(handle_id, None)


@dataclass
class GlobalPTQDistributed(PostQuantizationProcess):
    """Global PTQ via Trainer-based KL distillation.

    Trainer-based implementation of global post-training quantization
    that supports single-GPU and multi-GPU (DeepSpeed) execution.

    Args:
        temperature (float):
            Softmax temperature for KL divergence.  Default is 1.0.
        w_distill (float):
            Weight for KL distillation loss.  Default is 1.0.
        w_ntp (float):
            Weight for next-token prediction loss.  Default is 0.0.
            Setting ``w_distill=0, w_ntp=1`` enables pure QAT mode;
            the teacher model is **not** loaded in this case.
        gptq_lr (float):
            Learning rate for GPTQ scales / zeros.  Default is 1e-5.
        dbf_lr (float):
            Learning rate for DBF scaling parameters.
            Default is 5e-5.
        calibration_config (CalibrationConfig or None):
            Calibration data configuration.  When ``None`` (default),
            a :class:`CalibrationConfig` is created with
            ``num_calibration_samples=128``.
            See :class:`~onecomp.calibration.CalibrationConfig`.
        epochs (int):
            Number of distillation epochs.  Default is 5.
        per_device_train_batch_size (int):
            Batch size per device.  Default is 1.
        gradient_accumulation_steps (int):
            Gradient accumulation steps.  Default is 1.
        warmup_ratio (float):
            Fraction of total steps for LR warm-up.  Default is 0.1.
        max_grad_norm (float):
            Gradient clipping norm.  Default is 1.0.
        lr_scheduler_type (str):
            Learning-rate scheduler type (any value accepted by
            ``transformers.TrainingArguments``).  Default is
            ``"cosine"``.
        use_gradient_checkpointing (bool):
            Enable gradient checkpointing.  Default is True.
        bf16 (bool):
            Enable BF16 mixed-precision.  Default is True.
        deepspeed_config (str or None):
            Path to a DeepSpeed JSON config file.  ``None`` disables
            DeepSpeed (single-GPU Trainer mode).  Default is None.
        output_dir (str or None):
            Directory for Trainer outputs (logs, checkpoints).
            ``None`` (default) creates a temporary directory.
        logging_steps (int):
            Log training metrics every N steps.  Default is 1.
        report_to (str or list or None):
            Logging integrations (e.g. ``"wandb"``, ``"tensorboard"``).
            Default is ``"none"`` (no external logging).
        save_strategy (str):
            Checkpoint saving strategy passed to
            ``TrainingArguments``.  One of ``"no"``, ``"epoch"``,
            ``"steps"``.  Default is ``"no"``.
        save_steps (int or None):
            Save checkpoint every N steps (when
            ``save_strategy="steps"``).  Default is None.
        eval_interval (int):
            Evaluate every N epochs.  Default is 1.

    Examples:
        Single GPU:

        >>> from onecomp import Runner, ModelConfig, GPTQ, GlobalPTQDistributed
        >>> runner = Runner(
        ...     model_config=ModelConfig(model_id="Qwen/Qwen3-0.6B"),
        ...     quantizer=GPTQ(wbits=4, groupsize=128),
        ...     post_processes=[GlobalPTQDistributed(epochs=5, gptq_lr=1e-5)],
        ... )
        >>> runner.run()

        With DeepSpeed ZeRO-2:

        >>> GlobalPTQDistributed(
        ...     epochs=5,
        ...     deepspeed_config="configs/ds_zero2.json",
        ...     per_device_train_batch_size=2,
        ... )

        Pure QAT (no teacher model):

        >>> GlobalPTQDistributed(
        ...     w_distill=0.0, w_ntp=1.0,
        ...     epochs=3,
        ... )

        Custom calibration config:

        >>> from onecomp import CalibrationConfig
        >>> GlobalPTQDistributed(
        ...     calibration_config=CalibrationConfig(
        ...         calibration_dataset="wikitext2",
        ...         num_calibration_samples=64,
        ...     ),
        ... )

    """

    # --- Loss ---
    temperature: float = 1.0
    w_distill: float = 1.0
    w_ntp: float = 0.0

    # --- GPTQ ---
    gptq_lr: float = 1e-5

    # --- DBF ---
    dbf_lr: float = 5e-5

    # --- Calibration ---
    calibration_config: Optional[CalibrationConfig] = None

    # --- Training ---
    epochs: int = 5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    use_gradient_checkpointing: bool = True
    bf16: bool = True

    # --- Distributed ---
    deepspeed_config: Optional[str] = None

    # --- Output / Logging ---
    output_dir: Optional[str] = None
    logging_steps: int = 1
    report_to: Optional[str] = "none"
    save_strategy: str = "no"
    save_steps: Optional[int] = None

    # --- Evaluation ---
    eval_interval: int = 1

    # --- Post-run diagnostics (set by run(), not user-configurable) ---
    _last_train_loss: Optional[float] = field(
        default=None,
        init=False,
        repr=False,
    )
    _last_log_history: list = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self):
        super().__post_init__()
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.w_distill == 0 and self.w_ntp == 0:
            raise ValueError(
                "Both w_distill and w_ntp are 0. At least one loss weight " "must be positive."
            )
        if self.calibration_config is None:
            self.calibration_config = CalibrationConfig(
                num_calibration_samples=128,
            )
        if self.save_steps is not None and self.save_strategy != "steps":
            logger.warning(
                "save_steps=%d is ignored because save_strategy=%r "
                "(only effective with save_strategy='steps').",
                self.save_steps,
                self.save_strategy,
            )

    def run(
        self,
        quantized_model: nn.Module,
        model_config: ModelConfig,
    ) -> None:
        """Execute global PTQ via Trainer.

        Modifies *quantized_model* in-place.  The model is returned on
        CPU in eval mode per the ``PostQuantizationProcess`` contract.

        Args:
            quantized_model (nn.Module):
                Quantized model on CPU (GPTQLinear / DoubleBinaryLinear).
            model_config (ModelConfig):
                Model configuration (provides tokenizer, model path, etc.).
        """
        from transformers import TrainingArguments, default_data_collator

        from ..calibration import prepare_calibration_dataset
        from ._global_ptq.dbf_adapter import (
            load_dbf_state,
            restore_dbf_original,
            save_dbf_state,
            setup_dbf_differentiable,
            write_back_dbf_scaling,
        )
        from ._global_ptq.gptq_adapter import (
            load_gptq_state,
            restore_gptq_original,
            save_gptq_state,
            setup_gptq_differentiable,
            write_back_gptq_params,
        )
        from ._global_ptq.helpers import detect_quantization_method
        from ._global_ptq.trainer import _GlobalPTQTrainer, _KDDataset

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dev = torch.device(f"cuda:{local_rank}")
        else:
            dev = torch.device("cpu")

        # ------------------------------------------------------------------
        # 1. Detect method (cf. core.py section 1)
        # ------------------------------------------------------------------
        method, detected_modules = detect_quantization_method(quantized_model)
        if method is None:
            logger.warning("No quantized layers detected — skipping.")
            return
        if method not in ("gptq", "dbf"):
            logger.info("Method '%s' not supported — skipping.", method)
            return

        logger.info(
            "[GlobalPTQDistributed] method=%s, modules=%d",
            method,
            len(detected_modules),
        )

        # ------------------------------------------------------------------
        # 2. Calibration data (cf. core.py section 2)
        # ------------------------------------------------------------------
        logger.info(
            "Loading calibration data (n=%d, len=%d)...",
            self.calibration_config.num_calibration_samples,
            self.calibration_config.max_length,
        )
        tokenizer = model_config.load_tokenizer()
        cal = prepare_calibration_dataset(
            tokenizer=tokenizer,
            device="cpu",
            calibration_config=self.calibration_config,
            model=quantized_model,
            logger=logger,
        )
        train_dataset = _KDDataset(cal["input_ids"])

        # ------------------------------------------------------------------
        # 3. Move student to GPU + differentiable setup (cf. core.py section 4)
        # ------------------------------------------------------------------
        quantized_model.to(dev)
        _, detected_modules = detect_quantization_method(quantized_model)

        gptq_modules = []
        dbf_modules = []
        original_forwards = {}
        param_groups = []

        if method == "gptq":
            gptq_modules = detected_modules
            original_forwards, scaling_params = setup_gptq_differentiable(gptq_modules, dev)
            param_groups = [{"params": scaling_params, "lr": self.gptq_lr}]
            logger.info(
                "Trainable: %d scales/zeros",
                len(scaling_params),
            )

        elif method == "dbf":
            dbf_modules = detected_modules
            original_forwards, scaling_params = setup_dbf_differentiable(dbf_modules)
            param_groups = [
                {
                    "params": list(scaling_params),
                    "lr": self.dbf_lr,
                    "weight_decay": 0.0,
                }
            ]
            logger.info(
                "Trainable: %d scaling",
                len(scaling_params),
            )

        # DeepSpeed ZeRO requires contiguous tensors for all-reduce.
        for pg in param_groups:
            for p in pg["params"]:
                if isinstance(p, torch.nn.Parameter) and not p.data.is_contiguous():
                    p.data = p.data.contiguous()

        total_trainable = sum(len(pg["params"]) for pg in param_groups)
        if total_trainable == 0:
            logger.warning("No trainable parameters — skipping.")
            if method == "gptq":
                restore_gptq_original(gptq_modules, original_forwards)
            elif method == "dbf":
                restore_dbf_original(dbf_modules, original_forwards)
            quantized_model.cpu()
            return

        # ------------------------------------------------------------------
        # 4. Gradient checkpointing (delegated to Trainer via TrainingArguments)
        # ------------------------------------------------------------------
        original_use_cache = getattr(quantized_model.config, "use_cache", None)
        quantized_model.config.use_cache = False

        teacher_model = None
        try:
            # ------------------------------------------------------------------
            # 5. Teacher model (cf. core.py section 3)
            # ------------------------------------------------------------------
            need_teacher = self.w_distill > 0
            if need_teacher:
                logger.info("Loading FP16 teacher model...")
                teacher_model = model_config.load_model(device_map="cpu")
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False
                teacher_model.to(dev)
            else:
                logger.info("w_distill=0 — skipping teacher model load (pure QAT mode).")
            # ------------------------------------------------------------------
            # 6. TrainingArguments (cf. preprocess_args.py defaults)
            # ------------------------------------------------------------------
            lr = self.gptq_lr if method == "gptq" else self.dbf_lr
            resolved_output_dir = (
                self.output_dir
                if self.output_dir is not None
                else tempfile.mkdtemp(prefix="global_ptq_distributed_")
            )

            eval_epoch_interval = max(1, self.eval_interval)

            training_args = TrainingArguments(
                output_dir=resolved_output_dir,
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.per_device_train_batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                learning_rate=lr,
                lr_scheduler_type=self.lr_scheduler_type,
                warmup_ratio=self.warmup_ratio,
                bf16=self.bf16 and dev.type == "cuda",
                gradient_checkpointing=self.use_gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                logging_steps=self.logging_steps,
                eval_strategy="epoch" if eval_epoch_interval == 1 else "steps",
                eval_steps=(
                    None
                    if eval_epoch_interval == 1
                    else eval_epoch_interval
                    * max(
                        1,
                        len(train_dataset)
                        // self.per_device_train_batch_size
                        // self.gradient_accumulation_steps,
                    )
                ),
                save_strategy=self.save_strategy,
                save_steps=self.save_steps,
                report_to=self.report_to,
                deepspeed=self.deepspeed_config,
                max_grad_norm=self.max_grad_norm,
                disable_tqdm=False,
            )

            # Save initial state for rollback if training degrades quality
            if method == "gptq":
                _initial_state = save_gptq_state(gptq_modules)
            else:
                _initial_state = save_dbf_state(dbf_modules)

            # ------------------------------------------------------------------
            # 7. Train (cf. core.py section 7)
            # ------------------------------------------------------------------
            trainer = _GlobalPTQTrainer(
                model=quantized_model,
                teacher_model=teacher_model,
                method=method,
                gptq_modules=gptq_modules,
                dbf_modules=dbf_modules,
                original_forwards=original_forwards,
                temperature=self.temperature,
                w_distill=self.w_distill,
                w_ntp=self.w_ntp,
                custom_param_groups=param_groups,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=train_dataset,
                processing_class=tokenizer,
                data_collator=default_data_collator,
            )

            train_output = trainer.train()
            self._last_train_loss = getattr(train_output, "training_loss", None)
            self._last_log_history = list(trainer.state.log_history)

            # ------------------------------------------------------------------
            # 8. Finalize — rollback if training degraded quality
            # ------------------------------------------------------------------
            eval_losses = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
            rollback_happened = len(eval_losses) >= 2 and (
                eval_losses[-1] >= eval_losses[0] or math.isnan(eval_losses[-1])
            )
            if rollback_happened:
                logger.info(
                    "eval_loss did not improve (%.6f -> %.6f) " "— rolling back to initial state.",
                    eval_losses[0],
                    eval_losses[-1],
                )
                if method == "gptq":
                    load_gptq_state(gptq_modules, _initial_state)
                else:
                    load_dbf_state(dbf_modules, _initial_state)

            if method == "gptq":
                if not rollback_happened:
                    write_back_gptq_params(gptq_modules)
                restore_gptq_original(gptq_modules, original_forwards, cleanup=True)
            elif method == "dbf":
                write_back_dbf_scaling(dbf_modules)
                restore_dbf_original(dbf_modules, original_forwards)
        finally:
            if original_use_cache is not None:
                quantized_model.config.use_cache = original_use_cache

            if teacher_model is not None:
                del teacher_model
            gc.collect()
            torch.cuda.empty_cache()

            quantized_model.cpu()
            for param in quantized_model.parameters():
                param.requires_grad = False
            quantized_model.eval()

            # HF Trainer enables gradient checkpointing by attaching a
            # non-picklable forward hook (``make_inputs_require_grads``, a
            # local closure) via ``enable_input_require_grads``. Leaving it on
            # the model breaks ``torch.save(model)`` in
            # ``save_quantized_model_pt`` with an AttributeError. Remove it
            # here (no-op when no hook is registered).
            if hasattr(quantized_model, "gradient_checkpointing_disable"):
                quantized_model.gradient_checkpointing_disable()
            if hasattr(quantized_model, "disable_input_require_grads"):
                quantized_model.disable_input_require_grads()

            # DeepSpeed wraps the module and injects non-picklable forward
            # (pre/post) hook closures (e.g. ``_module_forward_post_hook`` from
            # ``DeepSpeedEngine``) onto every submodule. These persist after
            # training and likewise break ``torch.save(model)``. Strip any hook
            # whose closure originates from DeepSpeed.
            _remove_deepspeed_hooks(quantized_model)

        logger.info("GlobalPTQDistributed complete.")
