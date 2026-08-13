"""
Global Post-Training Quantization process.

Optimises quantization parameters model-wide by minimising KL divergence
between a frozen FP16 teacher and the quantized student.

Compared to the OSS ``onecomp.GlobalPTQ`` (continuous parameters only),
this version additionally supports:
- Discrete parameter optimisation (integer weights via Smooth STE,
  binary matrices via Sign STE)
- SAM (Sharpness-Aware Minimisation)
- EMA (Exponential Moving Average)
- Lookahead optimiser
- Fisher-adaptive per-layer learning rates
- Entropy regularisation
- Intermediate-layer cosine similarity loss
- Progressive layer unfreezing

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

"""

from dataclasses import dataclass
from logging import getLogger
from typing import List, Optional

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
class GlobalPTQ(PostQuantizationProcess):
    """Global Post-Training Quantization via KL distillation.

    After layer-wise PTQ (GPTQ / DBF) quantises each linear layer
    independently, global PTQ minimises the KL divergence between an
    FP16 teacher model and the quantized student model across the
    entire sequence, fine-tuning quantization parameters (scales,
    zeros, and optionally integer weights / binary matrices).

    Args:
        epochs (int):
            Number of distillation epochs.  Default is 5.
        gptq_lr (float):
            Learning rate for GPTQ scales / zeros.
            Default is 1e-5.
        temperature (float):
            Softmax temperature for KL divergence.
            Default is 1.0.
        grad_clip (float):
            Gradient clipping norm.  Default is 1.0.
        gptq_optimize_intweight (bool):
            Whether to optimise integer weights via Smooth STE.
            Default is False.
        gptq_intweight_lr (float):
            Learning rate for integer weight optimisation.
            Default is 1e-4.
        dbf_lr (float):
            Learning rate for DBF scaling parameters.
            Default is 5e-5.
        optimize_binary (bool):
            Whether to optimise DBF binary matrices via sign STE.
            Default is False.
        ste_k (float):
            Smoothness parameter for GPTQ integer-weight Smooth STE
            rounding.  Only used when ``gptq_optimize_intweight=True``.
            Default is 100.0.
        mdbf_ste_k (float):
            Sharpness for MDBF binary sign STE (``tanh(k*x)`` backward).
            Default is 2.0.
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
        warmup_ratio (float):
            Fraction of total steps used for LR warm-up.
            Default is 0.1.
        min_lr_ratio (float):
            Minimum LR as a fraction of peak LR (cosine decay floor).
            Default is 0.01.
        eval_interval (int):
            Evaluate every N epochs.  Default is 1.
        use_sam (bool):
            Enable Sharpness-Aware Minimisation.  Default is False.
            Incompatible with ``grad_accum_steps > 1``; when both are
            set, ``grad_accum_steps`` is silently forced to 1 at runtime
            because SAM requires two forward-backward passes per step.
        sam_rho (float):
            SAM perturbation radius.  Default is 0.02.
        use_ema (bool):
            Enable Exponential Moving Average of parameters.
            Default is False.
        ema_decay (float):
            EMA decay factor.  Default is 0.99.
        use_lookahead (bool):
            Enable Lookahead optimiser wrapper.  Default is False.
        lookahead_k (int):
            Lookahead inner-step interval.  Default is 5.
        lookahead_alpha (float):
            Lookahead interpolation factor.  Default is 0.5.
        use_fisher_lr (bool):
            Enable Fisher-adaptive per-layer learning rate.
            Default is False.
        fisher_n_samples (int):
            Number of samples for Fisher diagonal estimation.
            Default is 4.
        fisher_min_mult (float):
            Minimum Fisher LR multiplier.  Default is 0.1.
        fisher_max_mult (float):
            Maximum Fisher LR multiplier.  Default is 10.0.
        use_entropy_reg (bool):
            Enable entropy regularisation (prevents overconfidence).
            Default is False.
        entropy_lambda (float):
            Weight for entropy regularisation loss.  Default is 0.1.
        entropy_temperature (float):
            Temperature for entropy loss computation.  Default is 1.0.
        use_inter_loss (bool):
            Enable intermediate-layer cosine similarity loss.
            Default is False.
        lambda_inter (float):
            Weight for intermediate-layer loss.  Default is 10.0.
        use_progressive_unfreeze (bool):
            Enable progressive layer unfreezing (output-to-input).
            Default is False.
        use_gradient_checkpointing (bool):
            Enable gradient checkpointing to reduce GPU memory at the
            cost of recomputing activations during backpropagation.
            Default is True.
        early_stopping_patience (int):
            Stop training if eval KL does not improve for this many
            consecutive evaluations.  0 disables early stopping.
            Default is 0.
        use_mixed_precision (bool):
            Enable BF16 mixed-precision (``torch.amp.autocast``).
            Default is False.
        grad_accum_steps (int):
            Number of gradient accumulation steps before each
            optimiser update.  Default is 1 (no accumulation).
            Incompatible with ``use_sam=True``; when both are set,
            this value is silently forced to 1.
    Examples:
        >>> from onecomp import Runner, ModelConfig, GPTQ
        >>> from onecomp_globalptq import GlobalPTQ
        >>> model_config = ModelConfig(model_id="Qwen/Qwen3-0.6B")
        >>> quantizer = GPTQ(wbits=4, groupsize=128)
        >>> runner = Runner(
        ...     model_config=model_config,
        ...     quantizer=quantizer,
        ...     post_processes=[GlobalPTQ(epochs=5, gptq_lr=1e-5)],
        ... )
        >>> runner.run()

    """

    # --- Basic distillation parameters ---
    epochs: int = 5
    gptq_lr: float = 1e-5
    temperature: float = 1.0
    grad_clip: float = 1.0
    gptq_optimize_intweight: bool = False
    gptq_intweight_lr: float = 1e-4
    dbf_lr: float = 5e-5
    optimize_binary: bool = False
    ste_k: float = 100.0
    mdbf_ste_k: float = 2.0
    calibration_dataset: Optional[List[str]] = None
    num_calibration_samples: int = 128
    max_length: int = 2048
    calibration_strategy: str = "drop_rand"
    calibration_seed: int = 0
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.01
    eval_interval: int = 1

    # --- SAM ---
    use_sam: bool = False
    sam_rho: float = 0.02

    # --- EMA ---
    use_ema: bool = False
    ema_decay: float = 0.99

    # --- Lookahead ---
    use_lookahead: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5

    # --- Fisher-Adaptive LR ---
    use_fisher_lr: bool = False
    fisher_n_samples: int = 4
    fisher_min_mult: float = 0.1
    fisher_max_mult: float = 10.0

    # --- Entropy Regularisation ---
    use_entropy_reg: bool = False
    entropy_lambda: float = 0.1
    entropy_temperature: float = 1.0

    # --- Intermediate Layer Loss ---
    use_inter_loss: bool = False
    lambda_inter: float = 10.0

    # --- Progressive Layer Unfreezing ---
    use_progressive_unfreeze: bool = False

    # --- Gradient Checkpointing ---
    use_gradient_checkpointing: bool = True

    # --- Early Stopping ---
    early_stopping_patience: int = 0

    # --- Mixed Precision ---
    use_mixed_precision: bool = False

    # --- Gradient Accumulation ---
    grad_accum_steps: int = 1

    # --- Device placement (multi-GPU / CPU teacher) ---
    student_device: Optional[str] = None
    teacher_device: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.num_calibration_samples < 1:
            raise ValueError(
                f"num_calibration_samples must be >= 1, got "
                f"{self.num_calibration_samples}"
            )
        if self.calibration_strategy not in _VALID_CALIBRATION_STRATEGIES:
            raise ValueError(
                f"Unknown calibration_strategy: {self.calibration_strategy!r}. "
                f"Available: {list(_VALID_CALIBRATION_STRATEGIES)}"
            )

    # OneComp < 1.3.1 made ``run`` abstract.  Newer versions provide a
    # validated/audited public ``run`` wrapper and make ``_run`` abstract.
    # Define the legacy entry point only when the installed base requires it,
    # so current OneComp can retain its wrapper instead of being overridden.
    if "_run" not in getattr(PostQuantizationProcess, "__abstractmethods__", set()):

        def run(self, quantized_model: nn.Module, model_config: ModelConfig) -> None:
            return self._run(quantized_model, model_config)

    def _run(
        self,
        quantized_model: nn.Module,
        model_config: ModelConfig,
    ) -> dict:
        """Execute global PTQ on the quantized model.

        Modifies *quantized_model* in-place.  The model is returned on
        CPU in eval mode per the ``PostQuantizationProcess`` contract.

        Args:
            quantized_model (nn.Module):
                Quantized model on CPU (GPTQLinear / DoubleBinaryLinear).
            model_config (ModelConfig):
                Model configuration (provides tokenizer, model path, etc.).
        """
        from ._core.core import run_kl_distillation

        original_use_cache = getattr(
            getattr(quantized_model, "config", None), "use_cache", None,
        )

        try:
            results = run_kl_distillation(
            quantized_model,
            model_config,
            epochs=self.epochs,
            gptq_lr=self.gptq_lr,
            dbf_lr=self.dbf_lr,
            temperature=self.temperature,
            grad_clip=self.grad_clip,
            gptq_optimize_intweight=self.gptq_optimize_intweight,
            gptq_intweight_lr=self.gptq_intweight_lr,
            optimize_binary=self.optimize_binary,
            ste_k=self.ste_k,
            mdbf_ste_k=self.mdbf_ste_k,
            calibration_dataset=self.calibration_dataset,
            num_calibration_samples=self.num_calibration_samples,
            max_length=self.max_length,
            calibration_strategy=self.calibration_strategy,
            calibration_seed=self.calibration_seed,
            warmup_ratio=self.warmup_ratio,
            min_lr_ratio=self.min_lr_ratio,
            eval_interval=self.eval_interval,
            use_sam=self.use_sam,
            sam_rho=self.sam_rho,
            use_ema=self.use_ema,
            ema_decay=self.ema_decay,
            use_lookahead=self.use_lookahead,
            lookahead_k=self.lookahead_k,
            lookahead_alpha=self.lookahead_alpha,
            use_fisher_lr=self.use_fisher_lr,
            fisher_n_samples=self.fisher_n_samples,
            fisher_min_mult=self.fisher_min_mult,
            fisher_max_mult=self.fisher_max_mult,
            use_entropy_reg=self.use_entropy_reg,
            entropy_lambda=self.entropy_lambda,
            entropy_temperature=self.entropy_temperature,
            use_inter_loss=self.use_inter_loss,
            lambda_inter=self.lambda_inter,
            use_progressive_unfreeze=self.use_progressive_unfreeze,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            early_stopping_patience=self.early_stopping_patience,
            use_mixed_precision=self.use_mixed_precision,
            grad_accum_steps=self.grad_accum_steps,
            student_device=self.student_device,
            teacher_device=self.teacher_device,
        )

        except Exception:
            logger.exception("GlobalPTQ training failed — restoring model.")
            quantized_model.cpu()
            for p in quantized_model.parameters():
                p.requires_grad = False
            quantized_model.eval()
            if original_use_cache is not None:
                quantized_model.config.use_cache = original_use_cache
            raise

        if results.get("global_executed"):
            logger.info(
                "Global PTQ complete: KL %.6f -> %.6f (%.2f%%)",
                results["initial_kl"],
                results["final_kl"],
                results["improvement_pct"],
            )
        else:
            logger.info(
                "Global PTQ skipped: %s", results.get("reason", "unknown"),
            )
