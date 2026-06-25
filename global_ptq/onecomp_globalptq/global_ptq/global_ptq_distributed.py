"""Trainer-based Global Post-Training Quantization process.

Provides ``GlobalPTQDistributed``, a Trainer-based alternative to
``GlobalPTQ`` that supports multi-GPU training via DeepSpeed.

Compared to ``GlobalPTQ``:
- Uses ``transformers.Trainer`` for the training loop
- Supports DeepSpeed ZeRO-2 via ``deepspeed_config``
- Supports combined KL + NTP loss (enabling QAT mode)
- Does NOT include SAM, EMA, Lookahead, or Fisher-adaptive LR

For advanced single-GPU optimisation with SAM/EMA/Lookahead,
use ``GlobalPTQ`` instead.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii

"""

import gc
import math
import os
import tempfile
from dataclasses import dataclass, field
from logging import getLogger
from typing import List, Optional

import torch
import torch.nn as nn

from onecomp import PostQuantizationProcess, ModelConfig

logger = getLogger(__name__)

_VALID_CALIBRATION_STRATEGIES = (
    "concat_chunk",
    "concat_chunk_align",
    "drop_head",
    "drop_rand",
)


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
        gptq_optimize_intweight (bool):
            Whether to optimise integer weights via Smooth STE.
            Default is False.
        gptq_intweight_lr (float):
            Learning rate for integer weight optimisation.
            Default is 1e-4.
        ste_k (float):
            Smoothness parameter for GPTQ integer-weight Smooth STE
            rounding.  Only used when ``gptq_optimize_intweight=True``.
            DBF binary STE uses a fixed internal sharpness (k=2).
            Default is 100.0.
        dbf_lr (float):
            Learning rate for DBF scaling parameters.
            Default is 5e-5.
        optimize_binary (bool):
            Whether to optimise DBF binary matrices via sign STE.
            Default is False.
        calibration_dataset (list or None):
            List of text strings to use as calibration data.
            If ``None`` (default), the AllenAI C4 dataset is
            downloaded automatically.
        num_calibration_samples (int):
            Number of calibration samples.  Default is 128.
        max_length (int):
            Sequence length for calibration data.  Default is 2048.
        calibration_strategy (str):
            Calibration strategy.  One of ``"drop_rand"`` (default),
            ``"drop_head"``, ``"concat_chunk"``,
            ``"concat_chunk_align"``.
        calibration_seed (int):
            Random seed for calibration.  Default is 0.
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
        eval_interval (int):
            Evaluate every N epochs.  Default is 1.

    Examples:
        Single GPU:

        >>> from onecomp import Runner, ModelConfig, GPTQ
        >>> from onecomp_globalptq import GlobalPTQDistributed
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

        Custom calibration data:

        >>> GlobalPTQDistributed(
        ...     calibration_dataset=["Text sample 1...", "Text sample 2..."],
        ...     num_calibration_samples=64,
        ... )

    """

    # --- Loss ---
    temperature: float = 1.0
    w_distill: float = 1.0
    w_ntp: float = 0.0

    # --- GPTQ ---
    gptq_lr: float = 1e-5
    gptq_optimize_intweight: bool = False
    gptq_intweight_lr: float = 1e-4
    ste_k: float = 100.0

    # --- DBF ---
    dbf_lr: float = 5e-5
    optimize_binary: bool = False

    # --- Calibration ---
    calibration_dataset: Optional[List[str]] = None
    num_calibration_samples: int = 128
    max_length: int = 2048
    calibration_strategy: str = "drop_rand"
    calibration_seed: int = 0

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

    # --- Output / Logging / Checkpointing ---
    output_dir: Optional[str] = None
    logging_steps: int = 1
    report_to: Optional[str] = "none"
    save_strategy: str = "no"
    save_steps: Optional[int] = None

    # --- Evaluation ---
    eval_interval: int = 1

    # --- Post-run diagnostics (set by run(), not user-configurable) ---
    _last_train_loss: Optional[float] = field(
        default=None, init=False, repr=False,
    )
    _last_log_history: list = field(
        default_factory=list, init=False, repr=False,
    )

    def __post_init__(self):
        super().__post_init__()
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.num_calibration_samples < 1:
            raise ValueError(
                f"num_calibration_samples must be >= 1, got "
                f"{self.num_calibration_samples}"
            )
        if self.w_distill == 0 and self.w_ntp == 0:
            raise ValueError(
                "Both w_distill and w_ntp are 0. At least one loss weight "
                "must be positive."
            )
        if self.calibration_strategy not in _VALID_CALIBRATION_STRATEGIES:
            raise ValueError(
                f"Unknown calibration_strategy: {self.calibration_strategy!r}. "
                f"Available: {list(_VALID_CALIBRATION_STRATEGIES)}"
            )
        if (self.save_steps is not None
                and self.save_strategy != "steps"):
            logger.warning(
                "save_steps=%d is ignored because save_strategy=%r "
                "(only effective with save_strategy='steps').",
                self.save_steps, self.save_strategy,
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
        from ._core.trainer import _GlobalPTQTrainer, _KDDataset
        from ._core.helpers import (
            detect_quantization_method,
            remove_input_require_grads,
        )
        from ._core.gptq_adapter import (
            load_gptq_state,
            save_gptq_state,
            setup_gptq_differentiable,
            write_back_gptq_params,
            restore_gptq_original,
        )
        from ._core.dbf_adapter import (
            load_dbf_state,
            save_dbf_state,
            setup_dbf_differentiable,
            write_back_dbf_binary,
            write_back_dbf_scaling,
            restore_dbf_original,
        )
        from onecomp import CalibrationConfig
        from onecomp.calibration import prepare_calibration_dataset
        from transformers import TrainingArguments, default_data_collator

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dev = torch.device(f"cuda:{local_rank}")
        else:
            dev = torch.device("cpu")

        # ------------------------------------------------------------------
        # 1. Detect method
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
            method, len(detected_modules),
        )

        # ------------------------------------------------------------------
        # 2. Calibration data
        # ------------------------------------------------------------------
        logger.info(
            "Loading calibration data (n=%d, len=%d)...",
            self.num_calibration_samples, self.max_length,
        )
        tokenizer = model_config.load_tokenizer()
        calib_config = CalibrationConfig(
            calibration_dataset=self.calibration_dataset or "c4",
            max_length=self.max_length,
            num_calibration_samples=self.num_calibration_samples,
            strategy=self.calibration_strategy,
            seed=self.calibration_seed,
        )
        cal = prepare_calibration_dataset(
            tokenizer=tokenizer,
            device="cpu",
            calibration_config=calib_config,
            model=quantized_model,
            logger=logger,
        )
        train_dataset = _KDDataset(cal)

        # ------------------------------------------------------------------
        # 3. Move student to GPU + differentiable setup
        # ------------------------------------------------------------------
        quantized_model.to(dev)
        _, detected_modules = detect_quantization_method(quantized_model)

        gptq_modules = []
        dbf_modules = []
        original_forwards = {}
        param_groups = []

        if method == "gptq":
            gptq_modules = detected_modules
            original_forwards, scaling_params, intweight_params = (
                setup_gptq_differentiable(
                    gptq_modules, dev,
                    self.gptq_optimize_intweight, self.ste_k,
                )
            )
            param_groups = [{"params": scaling_params, "lr": self.gptq_lr}]
            if intweight_params:
                param_groups.append({
                    "params": intweight_params,
                    "lr": self.gptq_intweight_lr,
                })
            logger.info(
                "Trainable: %d scales/zeros%s",
                len(scaling_params),
                f", {len(intweight_params)} intweight" if intweight_params else "",
            )

        elif method == "dbf":
            dbf_modules = detected_modules
            original_forwards, scaling_params, binary_params = (
                setup_dbf_differentiable(dbf_modules, self.optimize_binary)
            )
            all_dbf_params = list(scaling_params)
            if binary_params:
                all_dbf_params += binary_params
            param_groups = [{
                "params": all_dbf_params,
                "lr": self.dbf_lr,
                "weight_decay": 0.0,
            }]
            logger.info(
                "Trainable: %d scaling%s",
                len(scaling_params),
                f", {len(binary_params)} binary" if binary_params else "",
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
                restore_gptq_original(gptq_modules, original_forwards, cleanup=True)
            elif method == "dbf":
                restore_dbf_original(dbf_modules, original_forwards)
            quantized_model.cpu()
            return

        # ------------------------------------------------------------------
        # 4. Gradient checkpointing
        # ------------------------------------------------------------------
        original_use_cache = getattr(quantized_model.config, "use_cache", None)
        quantized_model.config.use_cache = False

        teacher_model = None
        try:
            # ------------------------------------------------------------------
            # 5. Teacher model
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
                logger.info(
                    "w_distill=0 — skipping teacher model load (pure QAT mode)."
                )
            # ------------------------------------------------------------------
            # 6. TrainingArguments
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
                    None if eval_epoch_interval == 1
                    else eval_epoch_interval * max(
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
            # 7. Train
            # ------------------------------------------------------------------
            trainer = _GlobalPTQTrainer(
                model=quantized_model,
                teacher_model=teacher_model,
                method=method,
                gptq_modules=gptq_modules,
                dbf_modules=dbf_modules,
                original_forwards=original_forwards,
                optimize_intweight=self.gptq_optimize_intweight,
                optimize_binary=self.optimize_binary,
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
            eval_losses = [
                e["eval_loss"] for e in trainer.state.log_history
                if "eval_loss" in e
            ]

            # Find the best eval loss and its index
            best_eval_loss = float("inf")
            best_step = -1
            for e in trainer.state.log_history:
                if "eval_loss" in e:
                    if e["eval_loss"] < best_eval_loss:
                        best_eval_loss = e["eval_loss"]
                        best_step = e["step"]

            initial_eval_loss = eval_losses[0] if eval_losses else float("inf")

            # Rollback if the best loss is not better than initial, or if it's NaN
            rollback_happened = (
                not eval_losses
                or math.isnan(best_eval_loss)
                or best_eval_loss >= initial_eval_loss
            )

            if rollback_happened:
                logger.info(
                    "No improvement in eval_loss (initial=%.6f, best=%.6f) "
                    "— rolling back to initial state.",
                    initial_eval_loss, best_eval_loss,
                )
                if method == "gptq":
                    load_gptq_state(gptq_modules, _initial_state)
                else:
                    load_dbf_state(dbf_modules, _initial_state)
            else:
                logger.info(
                    "Best eval_loss: %.6f (initial=%.6f) at step %d.",
                    best_eval_loss, initial_eval_loss, best_step
                )

            if method == "gptq":
                if not rollback_happened:
                    write_back_gptq_params(gptq_modules, self.gptq_optimize_intweight)
                restore_gptq_original(gptq_modules, original_forwards, cleanup=True)
            elif method == "dbf":
                write_back_dbf_binary(dbf_modules)
                write_back_dbf_scaling(dbf_modules)
                restore_dbf_original(dbf_modules, original_forwards)
        finally:
            if original_use_cache is not None:
                quantized_model.config.use_cache = original_use_cache

            remove_input_require_grads(quantized_model)

            if teacher_model is not None:
                del teacher_model
            gc.collect()
            torch.cuda.empty_cache()

            quantized_model.cpu()
            for param in quantized_model.parameters():
                param.requires_grad = False
            quantized_model.eval()

        logger.info("GlobalPTQDistributed complete.")
