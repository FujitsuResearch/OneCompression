"""
Global Post-Training Quantization process.

Optimises quantization parameters model-wide by minimising KL divergence
between a frozen FP16 teacher and the quantized student.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

"""

from dataclasses import dataclass
from logging import getLogger
from typing import Optional

import torch.nn as nn

from ..calibration import CalibrationConfig
from ..model_config import ModelConfig
from ._base import PostQuantizationProcess

logger = getLogger(__name__)


@dataclass
class GlobalPTQ(PostQuantizationProcess):
    """Global Post-Training Quantization via KL distillation.

    After layer-wise PTQ (GPTQ / DBF) quantises each linear layer
    independently, global PTQ minimises the KL divergence between an
    FP16 teacher model and the quantized student model across the
    entire sequence, fine-tuning continuous quantization parameters
    (scales and zeros for GPTQ; scaling factors for DBF).

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
        dbf_lr (float):
            Learning rate for DBF scaling parameters.
            Default is 5e-5.
        calibration_config (CalibrationConfig or None):
            Calibration data configuration.  When ``None`` (default),
            a :class:`CalibrationConfig` is created with
            ``num_calibration_samples=128``.
            See :class:`~onecomp.calibration.CalibrationConfig`.
        warmup_ratio (float):
            Fraction of total steps used for LR warm-up.
            Default is 0.1.
        min_lr_ratio (float):
            Minimum LR as a fraction of peak LR (cosine decay floor).
            Default is 0.01.
        eval_interval (int):
            Evaluate every N epochs.  Default is 1.
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

    Buffer layout:
        ``GlobalPTQ`` accepts a quantized model with either packed or unpacked
        ``GPTQLinear`` buffers.  The recommended Runner-managed path
        (``post_processes=[GlobalPTQ(...)]`` + ``Runner.run()``) builds packed
        buffers by default, and packed outputs are what the save / load and vLLM
        workflows expect, so prefer it for reusable checkpoints.

        Unpacked buffers are required only when ``GPTQLinear`` bit packing cannot
        represent the quantizer output:

        - ``JointQ(bits=1, ...)`` — 1-bit JointQ output is not supported by the
          ``GPTQLinear`` packing helpers.
        - ``GPTQ`` / ``RTN`` bit widths in ``{1, 5, 6, 7}`` — the packing helpers
          currently support only ``{2, 3, 4, 8}``.

        For those cases, build the post-process input with ``pack_weights=False``
        and call :meth:`run` directly instead of going through ``Runner.run()``.

    Examples:
        Recommended Runner-managed path (packed buffers by default):

        >>> from onecomp import Runner, ModelConfig, GPTQ, GlobalPTQ, CalibrationConfig
        >>> model_config = ModelConfig(model_id="Qwen/Qwen3-0.6B")
        >>> quantizer = GPTQ(wbits=4, groupsize=128)
        >>> runner = Runner(
        ...     model_config=model_config,
        ...     quantizer=quantizer,
        ...     post_processes=[GlobalPTQ(epochs=5, gptq_lr=1e-5)],
        ... )
        >>> runner.run()

        Explicit unpacked path (e.g. 1-bit JointQ or GPTQ/RTN bit widths
        outside ``{2, 3, 4, 8}``):

        >>> runner = Runner(model_config=model_config, quantizer=quantizer)
        >>> runner.run()
        >>> global_ptq = GlobalPTQ(epochs=5, gptq_lr=1e-5)
        >>> model, _ = runner.create_quantized_model(
        ...     pack_weights=False, use_gemlite=False
        ... )
        >>> global_ptq.run(model, model_config)
        >>> runner.quantized_model = model

    """

    # --- Basic distillation parameters ---
    epochs: int = 5
    gptq_lr: float = 1e-5
    temperature: float = 1.0
    grad_clip: float = 1.0
    dbf_lr: float = 5e-5
    calibration_config: Optional[CalibrationConfig] = None
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.01
    eval_interval: int = 1

    # --- Gradient Checkpointing ---
    use_gradient_checkpointing: bool = True

    # --- Early Stopping ---
    early_stopping_patience: int = 0

    # --- Mixed Precision ---
    use_mixed_precision: bool = False

    # --- Gradient Accumulation ---
    grad_accum_steps: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.calibration_config is None:
            self.calibration_config = CalibrationConfig(
                num_calibration_samples=128,
            )
        if self.calibration_config.num_calibration_samples < 1:
            raise ValueError(
                f"num_calibration_samples must be >= 1, got "
                f"{self.calibration_config.num_calibration_samples}"
            )

    def _run(
        self,
        quantized_model: nn.Module,
        model_config: ModelConfig,
    ) -> dict:
        """Execute global PTQ on the quantized model.

        Modifies *quantized_model* in-place.  The model is returned on
        CPU in eval mode per the ``PostQuantizationProcess`` contract.
        Returns the GlobalPTQ result dict from ``run_kl_distillation``.

        Args:
            quantized_model (nn.Module):
                Quantized model on CPU (GPTQLinear / DoubleBinaryLinear).
            model_config (ModelConfig):
                Model configuration (provides tokenizer, model path, etc.).
        """
        from ._global_ptq.core import run_kl_distillation

        original_use_cache = getattr(
            getattr(quantized_model, "config", None),
            "use_cache",
            None,
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
                calibration_config=self.calibration_config,
                warmup_ratio=self.warmup_ratio,
                min_lr_ratio=self.min_lr_ratio,
                eval_interval=self.eval_interval,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                early_stopping_patience=self.early_stopping_patience,
                use_mixed_precision=self.use_mixed_precision,
                grad_accum_steps=self.grad_accum_steps,
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
                "Global PTQ skipped: %s",
                results.get("reason", "unknown"),
            )

        return results
