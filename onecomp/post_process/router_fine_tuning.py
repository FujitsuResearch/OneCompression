"""Router-only fine-tuning for quantized mixture-of-experts models.

Copyright 2025-2026 Fujitsu Ltd.

"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader

from ..model_config import ModelConfig
from ._base import PostQuantizationProcess
from .post_process_lora_sft import (
    _capture_gptq_pack_state,
    _infer_dataset_loader,
    _restore_gptq_pack_state,
    _unpack_gptq_linears_in_place,
)

logger = getLogger(__name__)

_DEFAULT_ROUTER_MODULES = ("router", "gate", "shared_expert_gate")


def _extract_logits(outputs: Any) -> torch.Tensor:
    """Extract logits from common causal-LM output formats."""
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise TypeError("The model forward result does not contain `logits`.")


def _compute_next_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute shifted causal-LM cross entropy, ignoring padded labels."""
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    valid_targets = shift_labels.ne(-100).sum()
    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum / valid_targets.clamp_min(1)


@dataclass
class RouterFineTuning(PostQuantizationProcess):
    """Fine-tune only MoE routers after quantization.

    All model parameters are frozen before modules named ``router``, ``gate``,
    or ``shared_expert_gate`` are made trainable. Exact path-component matching
    avoids selecting expert projections such as ``gate_proj``. Training uses
    standard shifted next-token prediction loss and modifies the model in
    place.

    Args:
        dataset_name: Hugging Face dataset identifier.
        dataset_config_name: Optional Hugging Face dataset configuration.
        data_files: Local JSON/JSONL/CSV/TXT/Parquet training files.
        train_split: Dataset split used for training.
        text_column: Column containing training text.
        router_modules: Exact module-name components considered routers.
        max_train_samples: Optional maximum number of training examples.
        max_length: Tokenized sequence length.
        lr: AdamW learning rate.
        epochs: Number of training epochs.
        batch_size: Per-step batch size.
        gradient_accumulation_steps: Number of backward passes per update.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of optimizer updates used for linear warmup.
        max_grad_norm: Gradient clipping norm. Set to ``None`` to disable.
        use_bf16: Use bfloat16 autocast on CUDA. Auto-detected when ``None``.

    Examples:
        >>> from onecomp import GPTQ, ModelConfig, RouterFineTuning, Runner
        >>> runner = Runner(
        ...     model_config=ModelConfig(model_id="Qwen/Qwen3-30B-A3B"),
        ...     quantizer=GPTQ(wbits=4, groupsize=128),
        ...     post_processes=[
        ...         RouterFineTuning(data_files="train.jsonl", epochs=1)
        ...     ],
        ... )
        >>> runner.run()
    """

    dataset_name: str | None = None
    dataset_config_name: str | None = None
    data_files: str | list[str] | dict[str, str] | None = None
    train_split: str = "train"
    text_column: str = "text"
    router_modules: tuple[str, ...] = _DEFAULT_ROUTER_MODULES
    max_train_samples: int | None = None
    max_length: int = 1024
    shuffle_seed: int = 42

    lr: float = 1e-5
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    max_grad_norm: float | None = 1.0
    logging_steps: int = 10
    use_bf16: bool | None = None

    def _validate_config(self) -> None:
        if self.dataset_name is None and self.data_files is None:
            raise ValueError("Either `dataset_name` or `data_files` must be specified.")
        if not self.router_modules:
            raise ValueError("`router_modules` must contain at least one module name.")
        for field_name in ("epochs", "batch_size", "gradient_accumulation_steps", "max_length"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"`{field_name}` must be > 0, but got {value}.")
        if self.lr <= 0.0:
            raise ValueError(f"`lr` must be > 0, but got {self.lr}.")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(f"`warmup_ratio` must be in [0, 1], but got {self.warmup_ratio}.")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError(f"`max_grad_norm` must be > 0 or None, but got {self.max_grad_norm}.")

    def _resolve_train_device(self, model_config: ModelConfig) -> torch.device:
        requested = model_config.device if model_config.device not in (None, "auto") else None
        if requested is None:
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA is unavailable; falling back to CPU for router fine-tuning.")
            return torch.device("cpu")
        return torch.device(requested)

    def _load_train_dataset(self) -> Dataset:
        if self.dataset_name is not None:
            dataset_or_dict = load_dataset(
                path=self.dataset_name,
                name=self.dataset_config_name,
                data_files=self.data_files,
            )
        else:
            dataset_or_dict = load_dataset(
                _infer_dataset_loader(self.data_files),
                data_files=self.data_files,
            )

        if isinstance(dataset_or_dict, DatasetDict):
            if self.train_split not in dataset_or_dict:
                raise ValueError(
                    f"`train_split`={self.train_split!r} was not found. "
                    f"Available splits: {sorted(dataset_or_dict.keys())}."
                )
            dataset = dataset_or_dict[self.train_split]
        else:
            dataset = dataset_or_dict

        if self.text_column not in dataset.column_names:
            raise ValueError(
                f"`text_column`={self.text_column!r} was not found in "
                f"dataset columns {dataset.column_names}."
            )
        if self.max_train_samples is not None:
            if self.max_train_samples <= 0:
                raise ValueError(
                    "`max_train_samples` must be > 0, " f"but got {self.max_train_samples}."
                )
            dataset = dataset.shuffle(seed=self.shuffle_seed).select(
                range(min(self.max_train_samples, len(dataset)))
            )
        else:
            dataset = dataset.shuffle(seed=self.shuffle_seed)
        if len(dataset) == 0:
            raise ValueError("Training dataset is empty.")
        return dataset

    def _tokenize_dataset(self, dataset: Dataset, tokenizer) -> Dataset:
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer has neither pad_token nor eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize_batch(batch: dict[str, Any]) -> dict[str, Any]:
            texts = [
                text if isinstance(text, str) else str(text) for text in batch[self.text_column]
            ]
            tokenized = tokenizer(
                texts,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
            )
            tokenized["labels"] = [
                [token_id if mask else -100 for token_id, mask in zip(input_ids, attention_mask)]
                for input_ids, attention_mask in zip(
                    tokenized["input_ids"], tokenized["attention_mask"]
                )
            ]
            return tokenized

        tokenized = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=list(dataset.column_names),
            desc="Tokenizing router fine-tuning dataset",
        )
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized

    @staticmethod
    def _collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            key: torch.stack([item[key] for item in batch], dim=0)
            for key in ("input_ids", "attention_mask", "labels")
        }

    def _select_router_parameters(self, model: nn.Module) -> list[tuple[str, nn.Parameter]]:
        targets = set(self.router_modules)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        selected = []
        for name, parameter in model.named_parameters():
            if any(component in targets for component in name.split(".")):
                parameter.requires_grad_(True)
                selected.append((name, parameter))

        if not selected:
            raise ValueError(
                "No MoE router parameters matched " f"router_modules={self.router_modules!r}."
            )
        return selected

    def _run(self, quantized_model: nn.Module, model_config: ModelConfig) -> dict:
        self._validate_config()
        tokenizer = model_config.load_tokenizer()
        train_dataset = self._tokenize_dataset(self._load_train_dataset(), tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_batch,
        )
        trainable = self._select_router_parameters(quantized_model)
        trainable_parameters = [parameter for _, parameter in trainable]
        train_device = self._resolve_train_device(model_config)
        use_bf16 = self.use_bf16
        if use_bf16 is None:
            use_bf16 = bool(train_device.type == "cuda" and torch.cuda.is_bf16_supported())

        total_updates = max(
            1,
            math.ceil(len(train_loader) * self.epochs / self.gradient_accumulation_steps),
        )
        warmup_steps = int(total_updates * self.warmup_ratio)
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            return max(
                0.0,
                float(total_updates - step) / float(max(1, total_updates - warmup_steps)),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        pack_state = None
        original_use_cache = None
        optimizer_step = 0
        accumulation_step = 0
        skipped_batches = 0
        try:
            pack_state = _capture_gptq_pack_state(quantized_model)
            unpacked_count = _unpack_gptq_linears_in_place(quantized_model)
            if hasattr(quantized_model, "config") and hasattr(quantized_model.config, "use_cache"):
                original_use_cache = quantized_model.config.use_cache
                quantized_model.config.use_cache = False

            logger.info(
                "RouterFineTuning started: parameters=%d, samples=%d, epochs=%d, "
                "device=%s, unpacked_layers=%d",
                len(trainable),
                len(train_dataset),
                self.epochs,
                train_device,
                unpacked_count,
            )
            quantized_model.to(train_device)
            optimizer.zero_grad(set_to_none=True)
            quantized_model.train()
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if train_device.type == "cuda" and use_bf16
                else nullcontext()
            )
            for epoch in range(self.epochs):
                for batch_index, batch in enumerate(train_loader):
                    batch = {key: value.to(train_device) for key, value in batch.items()}
                    if not batch["labels"][:, 1:].ne(-100).any():
                        skipped_batches += 1
                        continue
                    with autocast_context:
                        outputs = quantized_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                        loss = _compute_next_token_loss(_extract_logits(outputs), batch["labels"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            "Router fine-tuning produced a non-finite loss at "
                            f"epoch={epoch + 1}, batch={batch_index + 1}."
                        )
                    (loss / self.gradient_accumulation_steps).backward()
                    accumulation_step += 1

                    is_update = accumulation_step % self.gradient_accumulation_steps == 0
                    if is_update:
                        if not all(
                            parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in trainable_parameters
                        ):
                            raise FloatingPointError(
                                "Router fine-tuning produced non-finite gradients at "
                                f"epoch={epoch + 1}, batch={batch_index + 1}."
                            )
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                trainable_parameters,
                                self.max_grad_norm,
                            )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_step += 1
                        if self.logging_steps > 0 and optimizer_step % self.logging_steps == 0:
                            logger.info(
                                "RouterFineTuning epoch=%d step=%d loss=%.6f",
                                epoch + 1,
                                optimizer_step,
                                loss.item(),
                            )
            if accumulation_step % self.gradient_accumulation_steps != 0:
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in trainable_parameters
                ):
                    raise FloatingPointError("Router fine-tuning produced non-finite gradients.")
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
        finally:
            if original_use_cache is not None:
                quantized_model.config.use_cache = original_use_cache
            try:
                quantized_model.to("cpu")
            finally:
                if pack_state is not None:
                    _restore_gptq_pack_state(quantized_model, pack_state)

        if skipped_batches:
            logger.info(
                "RouterFineTuning skipped %d batch(es) without next-token targets.",
                skipped_batches,
            )

        if optimizer_step == 0:
            return {
                "executed": False,
                "reason": "no_valid_next_token_targets",
                "optimizer_steps": 0,
            }

        return {
            "executed": True,
            "trainable_parameters": [name for name, _ in trainable],
            "optimizer_steps": optimizer_step,
        }
