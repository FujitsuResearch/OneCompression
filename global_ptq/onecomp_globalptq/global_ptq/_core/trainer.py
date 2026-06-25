"""Trainer-based KL-distillation training for global PTQ.

Provides ``_GlobalPTQTrainer`` (a ``transformers.Trainer`` subclass) and
``_KDDataset`` (a calibration data wrapper).

The Trainer overrides follow patterns established in the onecomp codebase
(``_PreprocessTrainer`` in ``onecomp.pre_process.train_rotation``) and
the LittleBit ``KDTrainer`` reference implementation.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii

"""

from logging import getLogger

import torch
import torch.nn as nn
from transformers import Trainer

from .helpers import get_logits
from .losses import compute_kl_loss, compute_ntp_loss
from .gptq_adapter import (
    restore_gptq_original,
    setup_gptq_forwards_only,
    write_back_gptq_params,
)
from .dbf_adapter import (
    restore_dbf_original,
    setup_dbf_forwards_only,
    write_back_dbf_binary,
)

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class _KDDataset(torch.utils.data.Dataset):
    """Calibration dataset for knowledge distillation.

    Wraps a dict of tensors (e.g. from ``prepare_calibration_dataset``)
    into a format expected by ``transformers.Trainer``.

    Args:
        calibration_data: dict mapping keys (``input_ids``,
            ``attention_mask``, etc.) to tensors on CPU.
    """

    def __init__(self, calibration_data: dict):
        self.keys = list(calibration_data.keys())
        self.num_samples = calibration_data["input_ids"].size(0)
        self.data = []
        for i in range(self.num_samples):
            item = {}
            for k in self.keys:
                val = calibration_data[k][i]
                if isinstance(val, torch.Tensor):
                    item[k] = val.tolist() if val.dim() > 0 else val.item()
                else:
                    item[k] = val
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class _GlobalPTQTrainer(Trainer):
    """KL-distillation trainer for global PTQ.

    Overrides:
        compute_loss  — KL distillation + optional NTP loss
        create_optimizer — quantization-parameter-only param_groups
        evaluate — write-back / forward-restore cycle around eval
        log — route metrics through module logger (not stdout)
    """

    def __init__(
        self,
        *,
        teacher_model: nn.Module,
        method: str,
        gptq_modules: list,
        dbf_modules: list,
        original_forwards: dict,
        optimize_intweight: bool,
        optimize_binary: bool,
        temperature: float,
        w_distill: float,
        w_ntp: float,
        custom_param_groups: list,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.method = method
        self.gptq_modules = gptq_modules
        self.dbf_modules = dbf_modules
        self.original_forwards = original_forwards
        self.optimize_intweight = optimize_intweight
        self.optimize_binary = optimize_binary
        self.temperature = temperature
        self.w_distill = w_distill
        self.w_ntp = w_ntp
        self._custom_param_groups = custom_param_groups

    # -- log (same as _PreprocessTrainer in train_rotation.py) ---------------

    def log(self, logs, start_time=None):
        """Route training metrics through logger instead of print."""
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch

        self.state.log_history.append({**logs, "step": self.state.global_step})

        if self.state.is_local_process_zero:
            display = {
                k: (f"{v:.4g}" if isinstance(v, float) else v)
                for k, v in logs.items()
                if k != "total_flos"
            }
            logger.info(display)

    # -- create_optimizer (same pattern as _PreprocessTrainer) ---------------

    def create_optimizer(self, model=None):
        """Create AdamW with custom quantization-parameter groups."""
        if self.optimizer is not None:
            return self.optimizer

        self.optimizer = torch.optim.AdamW(self._custom_param_groups)
        return self.optimizer

    # -- compute_loss (KL + NTP, following LittleBit KDTrainer) --------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute L = w_distill * KL + w_ntp * NTP."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        
        # Forward pass with all available inputs
        outputs = model(**inputs)
        logits_s = get_logits(outputs)

        loss = torch.tensor(0.0, device=logits_s.device)
        if self.w_distill > 0 and self.teacher_model is not None:
            with torch.no_grad():
                # Teacher also gets all available inputs
                teacher_outputs = self.teacher_model(**inputs)
                logits_t = get_logits(teacher_outputs)
            
            loss = loss + self.w_distill * compute_kl_loss(
                logits_t, logits_s, self.temperature,
                attention_mask=attention_mask,
            )
        if self.w_ntp > 0:
            loss = loss + self.w_ntp * compute_ntp_loss(logits_s, input_ids)

        return (loss, logits_s) if return_outputs else loss

    # -- prediction_step (ensure eval_loss is computed) ----------------------

    def prediction_step(self, model, inputs, prediction_loss_only,
                        ignore_keys=None):
        """Always compute loss even when the dataset has no ``labels`` key."""
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    # -- evaluate (write-back cycle from core.py L970-993) -------------------

    def evaluate(self, eval_dataset=None, ignore_keys=None,
                 metric_key_prefix="eval"):
        """Write-back quantization params before eval, restore after."""
        if self.method == "gptq":
            write_back_gptq_params(self.gptq_modules, self.optimize_intweight)
            restore_gptq_original(self.gptq_modules, self.original_forwards)
        elif self.method == "dbf":
            write_back_dbf_binary(self.dbf_modules)
            restore_dbf_original(self.dbf_modules, self.original_forwards)

        result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self.method == "gptq":
            setup_gptq_forwards_only(
                self.gptq_modules, self.original_forwards,
                self.optimize_intweight,
            )
        elif self.method == "dbf":
            setup_dbf_forwards_only(
                self.dbf_modules, self.original_forwards,
            )

        return result
