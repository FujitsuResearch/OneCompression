"""
LoRA SFT post-process for quantized GPTQ models.

Copyright 2025-2026 Fujitsu Ltd.

Author: Genki Shikada

"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from logging import getLogger
import json
import math
import os
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..model_config import ModelConfig
from ..quantizer.gptq.gptq_layer import GPTQLinear
from ._base import PostQuantizationProcess

logger = getLogger(__name__)

_DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRAGPTQLinear(nn.Module):
    """LoRA wrapper for a GPTQLinear layer."""

    def __init__(
        self,
        base_layer: GPTQLinear,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        super().__init__()
        if lora_r <= 0:
            raise ValueError(f"`lora_r` must be > 0, but got {lora_r}.")

        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.lora_r = int(lora_r)
        self.scaling = float(lora_alpha) / float(lora_r)
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(self.in_features, self.lora_r, bias=False)
        self.lora_B = nn.Linear(self.lora_r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for param in self.base_layer.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)

        lora_input = self.dropout(x).to(self.lora_A.weight.dtype)
        lora_update = self.lora_B(self.lora_A(lora_input))
        return base_output + lora_update.to(base_output.dtype) * self.scaling


def _resolve_submodule(root_module: nn.Module, module_path: str) -> nn.Module:
    """Resolve submodule by dotted path (supports ModuleList numeric keys)."""
    module = root_module
    for key in module_path.split("."):
        if key in module._modules:
            module = module._modules[key]
        else:
            module = getattr(module, key)
    return module


def _replace_submodule(root_module: nn.Module, module_path: str, new_module: nn.Module) -> None:
    """Replace a submodule by dotted path in-place."""
    parent_path, _, child_name = module_path.rpartition(".")
    parent = _resolve_submodule(root_module, parent_path) if parent_path else root_module
    if child_name in parent._modules:
        parent._modules[child_name] = new_module
    else:
        setattr(parent, child_name, new_module)


def _is_lm_head_module(module_name: str) -> bool:
    leaf_name = module_name.rsplit(".", maxsplit=1)[-1]
    return leaf_name == "lm_head"


def _normalize_file_paths(
    data_files: str | list[str] | dict[str, str] | None,
) -> list[str]:
    if data_files is None:
        return []
    if isinstance(data_files, str):
        return [data_files]
    if isinstance(data_files, list):
        return data_files
    if isinstance(data_files, dict):
        return list(data_files.values())
    raise TypeError(
        "`data_files` must be str | list[str] | dict[str, str] | None, "
        f"but got {type(data_files).__name__}."
    )


def _infer_dataset_loader(
    data_files: str | list[str] | dict[str, str] | None,
) -> str:
    paths = _normalize_file_paths(data_files)
    if not paths:
        raise ValueError("`data_files` is empty. Please provide at least one file path.")

    extensions = {os.path.splitext(path)[1].lower().lstrip(".") for path in paths}
    extensions.discard("")
    if not extensions:
        raise ValueError(
            "Could not infer dataset format from `data_files`. "
            "Use file extensions such as .json, .jsonl, .csv, .parquet, or .txt."
        )
    if len(extensions) != 1:
        raise ValueError(
            "All `data_files` must share the same extension. "
            f"Found extensions: {sorted(extensions)}."
        )

    extension = next(iter(extensions))
    if extension in {"json", "jsonl"}:
        return "json"
    if extension == "csv":
        return "csv"
    if extension in {"txt", "text"}:
        return "text"
    if extension == "parquet":
        return "parquet"

    raise ValueError(
        f"Unsupported file extension '.{extension}' for `data_files`. "
        "Supported extensions: json, jsonl, csv, txt, parquet."
    )


# ---------------------------------------------------------------------------
# Intermediate block-output alignment helpers
# ---------------------------------------------------------------------------


def _get_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Return the nn.ModuleList of transformer decoder blocks.

    Supports standard CausalLMs and VLMs with a ``language_model``
    sub-module (e.g. Qwen3-VL, Gemma3).
    """
    # Standard LLM: model.model.layers (Llama, Mistral, Qwen, Gemma text-only)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # VLM with language_model.layers (Qwen3-VL, Gemma3)
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
    # GPT-NeoX (Pythia)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    # OPT
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder.layers
    raise ValueError(
        f"Cannot find transformer blocks in {type(model).__name__}. "
        "If this is a new architecture, extend _get_transformer_blocks()."
    )


def _register_block_output_hooks(
    model: nn.Module,
    capture: dict,
    block_indices: tuple,
) -> list:
    """Register forward hooks on the specified transformer blocks.

    The captured value for block *i* is the first element of the output
    tuple (i.e. the hidden-state tensor), or the output itself if it is
    already a tensor.  Hooks are returned so the caller can remove them.
    """
    blocks = _get_transformer_blocks(model)
    hooks = []
    for idx in block_indices:
        block = blocks[idx]

        def _hook(module, inp, out, _idx=idx):  # noqa: ARG001
            h = out[0] if isinstance(out, tuple) else out
            capture[_idx] = h

        hooks.append(block.register_forward_hook(_hook))
    return hooks


def _compute_intermediate_block_loss(
    student_hiddens: dict,
    teacher_hiddens: dict,
    device: torch.device,
) -> torch.Tensor:
    """Average MSE across aligned transformer-block outputs."""
    total = torch.zeros((), device=device)
    count = 0
    for idx, s_h in student_hiddens.items():
        if idx not in teacher_hiddens:
            continue
        t_h = teacher_hiddens[idx].float().to(device)
        total = total + F.mse_loss(s_h.float(), t_h)
        count += 1
    if count == 0:
        return total
    return total / count


@dataclass
class PostProcessLoraSFT(PostQuantizationProcess):
    """LoRA-based SFT post-process on a GPTQ-quantized model.

    This post-process improves a GPTQ-quantized causal language model by
    injecting LoRA adapters into selected ``GPTQLinear`` layers and training
    only those adapters while keeping the quantized base weights frozen.
    The given ``quantized_model`` is modified in-place.

    Algorithm overview:
        1. Load an SFT training dataset from ``dataset_name`` or ``data_files``.
        2. Tokenize the dataset and build causal LM labels.
        3. Find target ``GPTQLinear`` modules such as ``q_proj`` / ``k_proj`` /
           ``v_proj`` / ``o_proj`` / ``gate_proj`` / ``up_proj`` / ``down_proj``.
        4. Replace each target module with ``LoRAGPTQLinear``, which keeps the
           original GPTQ layer as the frozen base path and adds trainable LoRA
           low-rank updates.
        5. Optimize the LoRA parameters with SFT loss and, optionally, an
           additional teacher distillation loss against an FP teacher model.
        6. Move the post-processed model back to CPU at the end so it can be
           reused by ``Runner`` for perplexity / accuracy evaluation.

    Training objective:
        - ``sft_loss_weight > 0`` enables the standard causal LM loss.
        - ``teacher_loss_weight > 0`` enables teacher-guided distillation.
        - ``teacher_loss_type`` selects the distillation loss on logits
          (currently ``"kl"`` or ``"mse"``).
        - ``cache_teacher_outputs=True`` precomputes teacher logits on CPU to
          reduce repeated teacher forward passes during multi-epoch training.

    Typical usage:
        - Use ``data_files=...`` for local JSON/JSONL/CSV/TXT/Parquet files.
        - Use ``dataset_name=...`` when loading from Hugging Face Datasets.
        - Pass this class to ``Runner(post_processes=[...])`` after GPTQ
          quantization, or call ``run()`` directly on a previously saved
          quantized model loaded with ``torch.load(..., weights_only=False)``.

    LoRA implementations:
        - ``PostProcessLoraSFT``:
          Standard LoRA SFT with only the causal LM objective by default.
        - ``PostProcessLoraTeacherSFT``:
          LoRA SFT with teacher distillation enabled by default.
          Use this when combining SFT loss and teacher loss.
        - ``PostProcessLoraTeacherOnlySFT``:
          LoRA training with only teacher distillation.

    Examples:
        Via Runner:
            >>> from onecomp import Runner, ModelConfig, GPTQ, PostProcessLoraSFT
            >>> model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf")
            >>> quantizer = GPTQ(wbits=4, groupsize=128)
            >>> runner = Runner(
            ...     model_config=model_config,
            ...     quantizer=quantizer,
            ...     post_processes=[
            ...         PostProcessLoraSFT(data_files="train.jsonl")
            ...     ],
            ... )
            >>> runner.run()

        Direct execution on a saved quantized model:
            >>> import torch
            >>> from onecomp import ModelConfig, PostProcessLoraSFT
            >>> model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf")
            >>> quantized_model = torch.load(
            ...     "quantized_model.pt",
            ...     map_location="cpu",
            ...     weights_only=False,
            ... )
            >>> post_process = PostProcessLoraSFT(data_files="train.jsonl")
            >>> post_process.run(quantized_model, model_config)

        With teacher distillation enabled:
            >>> from onecomp import Runner, ModelConfig, GPTQ, PostProcessLoraTeacherSFT
            >>> model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf")
            >>> quantizer = GPTQ(wbits=4, groupsize=128)
            >>> runner = Runner(
            ...     model_config=model_config,
            ...     quantizer=quantizer,
            ...     post_processes=[
            ...         PostProcessLoraTeacherSFT(
            ...             data_files="train.jsonl",
            ...             teacher_model_id="meta-llama/Llama-2-7b-hf",
            ...         )
            ...     ],
            ... )
            >>> runner.run()

        With teacher-logit caching enabled:
            >>> from onecomp import Runner, ModelConfig, GPTQ, PostProcessLoraSFT
            >>> model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf")
            >>> quantizer = GPTQ(wbits=4, groupsize=128)
            >>> runner = Runner(
            ...     model_config=model_config,
            ...     quantizer=quantizer,
            ...     post_processes=[
            ...         PostProcessLoraSFT(
            ...             data_files="train.jsonl",
            ...             teacher_loss_weight=1.0,
            ...             cache_teacher_outputs=True,
            ...         )
            ...     ],
            ... )
            >>> runner.run()

    """

    dataset_name: str | None = None
    dataset_config_name: str | None = None
    data_files: str | list[str] | dict[str, str] | None = None
    train_split: str = "train"
    text_column: str = "text"
    max_train_samples: int | None = None
    max_length: int = 1024
    shuffle_seed: int = 42

    lr: float = 1e-4
    epochs: int = 4
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] | None = None

    output_dir: str | None = None
    use_bf16: bool | None = None
    sft_loss_weight: float = 1.0
    teacher_loss_weight: float = 0.0
    teacher_loss_type: str = "kl"
    teacher_temperature: float = 1.0
    teacher_model_id: str | None = None
    teacher_model_path: str | None = None
    teacher_dtype: str | None = None
    teacher_device: str | None = None
    cache_teacher_outputs: bool = False
    teacher_cache_dtype: str | None = None

    # ------------------------------------------------------------------
    # Intermediate block-output alignment (Pattern 1 / Pattern 3)
    # ------------------------------------------------------------------
    # intermediate_block_loss_weight > 0 enables aligning the hidden-state
    # output of selected transformer blocks between the teacher (FP) model
    # and the student (quantized + LoRA) model.
    #
    # Pattern 1 (default when enabled):
    #   Both teacher and student models are on the same device; teacher
    #   forward is executed for every training batch.
    #
    # Pattern 3 (cache_intermediate_outputs=True):
    #   Teacher block outputs are precomputed over the entire training set
    #   and stored on CPU.  Only the student model occupies the GPU during
    #   the training loop.
    intermediate_block_loss_weight: float = 0.0
    cache_intermediate_outputs: bool = False
    intermediate_block_indices: tuple[int, ...] | None = None
    intermediate_cache_dtype: str | None = None

    def _resolve_train_device(self, model_config: ModelConfig) -> torch.device:
        requested = model_config.device if model_config.device not in (None, "auto") else None
        if requested is None:
            requested = "cuda" if torch.cuda.is_available() else "cpu"

        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA device %s was requested, but CUDA is unavailable. Falling back to CPU.",
                requested,
            )
            return torch.device("cpu")
        return torch.device(requested)

    def _normalize_runner_eval_device(self, model_config: ModelConfig) -> None:
        # Runner forces "auto" to "cuda" when evaluating self.quantized_model.
        # Normalize to CPU here so CPU-only post-process flows remain evaluable.
        if model_config.device == "auto" and not torch.cuda.is_available():
            logger.info(
                "CUDA is unavailable; setting model_config.device='cpu' so "
                "post-process evaluation runs on CPU."
            )
            model_config.device = "cpu"

    def _resolve_use_bf16(self, device: torch.device) -> bool:
        if self.use_bf16 is not None:
            return bool(self.use_bf16)
        return bool(device.type == "cuda" and torch.cuda.is_bf16_supported())

    @property
    def teacher_distillation_enabled(self) -> bool:
        return self.teacher_loss_weight > 0.0

    def _resolve_optional_device(
        self,
        requested: str | None,
        fallback_device: torch.device,
    ) -> torch.device:
        if requested in (None, "auto"):
            return fallback_device

        resolved = torch.device(requested)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA device %s was requested, but CUDA is unavailable. Falling back to %s.",
                requested,
                fallback_device,
            )
            return fallback_device
        return resolved

    def _resolve_teacher_dtype(
        self,
        model_config: ModelConfig,
        teacher_device: torch.device,
    ) -> str:
        if self.teacher_dtype is not None:
            return self.teacher_dtype
        if teacher_device.type == "cpu" and model_config.dtype in {"float16", "bfloat16"}:
            return "float32"
        return model_config.dtype

    def _build_teacher_model_config(
        self,
        model_config: ModelConfig,
        teacher_device: torch.device,
    ) -> ModelConfig:
        if self.teacher_model_id is not None and self.teacher_model_path is not None:
            raise ValueError("Specify only one of `teacher_model_id` or `teacher_model_path`.")

        teacher_model_id = self.teacher_model_id
        teacher_model_path = self.teacher_model_path
        if teacher_model_id is None and teacher_model_path is None:
            if model_config.model_id is not None:
                teacher_model_id = model_config.model_id
            else:
                teacher_model_path = model_config.path

        return ModelConfig(
            model_id=teacher_model_id,
            path=teacher_model_path,
            dtype=self._resolve_teacher_dtype(model_config, teacher_device),
            device="cpu",
        )

    def _load_teacher_model(
        self,
        model_config: ModelConfig,
        teacher_device: torch.device,
    ) -> nn.Module:
        teacher_model_config = self._build_teacher_model_config(model_config, teacher_device)
        teacher_model = teacher_model_config.load_model(device_map="cpu")
        teacher_model.to(teacher_device)
        teacher_model.eval()
        return teacher_model

    def _resolve_teacher_cache_dtype(self) -> torch.dtype:
        dtype_name = self.teacher_cache_dtype or "float16"
        try:
            return getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(
                "`teacher_cache_dtype` must be a valid torch dtype name such as "
                "'float16', 'bfloat16', or 'float32', "
                f"but got {dtype_name!r}."
            ) from exc

    def _build_teacher_cache(
        self,
        teacher_model: nn.Module,
        train_dataset: Dataset,
        teacher_device: torch.device,
    ) -> torch.Tensor:
        cache_dtype = self._resolve_teacher_cache_dtype()
        cache_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_batch,
        )
        teacher_cache = None

        logger.info(
            "Precomputing teacher outputs for %d samples (cache_dtype=%s).",
            len(train_dataset),
            cache_dtype,
        )

        for batch in cache_loader:
            sample_indices = batch["sample_idx"].to(dtype=torch.long)
            teacher_inputs = {
                "input_ids": batch["input_ids"].to(teacher_device),
                "attention_mask": batch["attention_mask"].to(teacher_device),
            }
            with torch.no_grad():
                teacher_outputs = teacher_model(**teacher_inputs)

            logits_cpu = teacher_outputs.logits.detach().to(device="cpu", dtype=cache_dtype)
            if teacher_cache is None:
                teacher_cache = torch.empty(
                    (len(train_dataset),) + tuple(logits_cpu.shape[1:]),
                    dtype=cache_dtype,
                    device="cpu",
                )
            teacher_cache.index_copy_(0, sample_indices, logits_cpu)

        if teacher_cache is None:
            raise RuntimeError("Teacher cache is empty. Training dataset may be empty.")

        cache_size_mb = teacher_cache.numel() * teacher_cache.element_size() / (1024.0 * 1024.0)
        logger.info("Teacher output cache is ready (%.2f MiB on CPU).", cache_size_mb)
        return teacher_cache

    # ------------------------------------------------------------------
    # Intermediate block-output helpers
    # ------------------------------------------------------------------

    def _resolve_target_block_indices(self, model: nn.Module) -> tuple:
        """Return the block indices to align.

        If ``intermediate_block_indices`` is set, those indices are used
        as-is.  Otherwise all blocks are returned.
        """
        if self.intermediate_block_indices is not None:
            return tuple(self.intermediate_block_indices)
        blocks = _get_transformer_blocks(model)
        return tuple(range(len(blocks)))

    def _resolve_intermediate_cache_dtype(self) -> torch.dtype:
        dtype_name = self.intermediate_cache_dtype or "float16"
        try:
            return getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(
                "`intermediate_cache_dtype` must be a valid torch dtype name "
                f"such as 'float16', 'bfloat16', or 'float32', "
                f"but got {dtype_name!r}."
            ) from exc

    def _build_intermediate_cache(
        self,
        teacher_model: nn.Module,
        train_dataset,
        teacher_device: torch.device,
        block_indices: tuple,
    ) -> dict:
        """Precompute teacher transformer-block outputs and cache on CPU.

        Returns a dict mapping block index to a CPU tensor of shape
        ``(N_samples, seq_len, hidden_size)`` in ``intermediate_cache_dtype``.
        """
        cache_dtype = self._resolve_intermediate_cache_dtype()
        cache: dict = {idx: None for idx in block_indices}
        cache_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_batch,
        )

        logger.info(
            "Precomputing teacher intermediate block outputs for %d samples "
            "(blocks=%s, dtype=%s).",
            len(train_dataset),
            list(block_indices),
            cache_dtype,
        )

        for batch in cache_loader:
            sample_indices = batch["sample_idx"].to(dtype=torch.long)
            teacher_inputs = {
                "input_ids": batch["input_ids"].to(teacher_device),
                "attention_mask": batch["attention_mask"].to(teacher_device),
            }
            captured: dict = {}
            hooks = _register_block_output_hooks(teacher_model, captured, block_indices)
            with torch.no_grad():
                teacher_model(**teacher_inputs)
            for h in hooks:
                h.remove()

            for idx in block_indices:
                if idx not in captured:
                    continue
                h_cpu = captured[idx].detach().to(device="cpu", dtype=cache_dtype)
                if cache[idx] is None:
                    cache[idx] = torch.empty(
                        (len(train_dataset),) + tuple(h_cpu.shape[1:]),
                        dtype=cache_dtype,
                        device="cpu",
                    )
                cache[idx].index_copy_(0, sample_indices, h_cpu)

        total_mb = sum(t.numel() * t.element_size() for t in cache.values() if t is not None) / (
            1024.0 * 1024.0
        )
        logger.info("Intermediate block output cache ready (%.2f MiB on CPU).", total_mb)
        return cache

    def _compute_teacher_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        shift_student_logits = student_logits[..., :-1, :].float()
        shift_teacher_logits = teacher_logits[..., :-1, :].float().to(shift_student_logits.device)
        shift_labels = labels[..., 1:]
        valid_mask = shift_labels.ne(-100)

        if not torch.any(valid_mask):
            return shift_student_logits.new_zeros(())

        teacher_loss_type = self.teacher_loss_type.lower()
        if teacher_loss_type == "kl":
            temperature = float(self.teacher_temperature)
            student_log_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(shift_teacher_logits / temperature, dim=-1)
            per_token_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="none",
            ).sum(dim=-1)
            return per_token_loss.masked_select(valid_mask).mean() * (temperature**2)

        if teacher_loss_type == "mse":
            per_token_loss = F.mse_loss(
                shift_student_logits,
                shift_teacher_logits,
                reduction="none",
            ).mean(dim=-1)
            return per_token_loss.masked_select(valid_mask).mean()

        raise ValueError(
            "`teacher_loss_type` must be either 'kl' or 'mse', "
            f"but got {self.teacher_loss_type!r}."
        )

    def _load_train_dataset(self) -> Dataset:
        if self.dataset_name is None and self.data_files is None:
            raise ValueError(
                "Either `dataset_name` or `data_files` must be specified for PostProcessLoraSFT."
            )

        if self.dataset_name is not None:
            dataset_or_dict = load_dataset(
                path=self.dataset_name,
                name=self.dataset_config_name,
                data_files=self.data_files,
            )
        else:
            loader_name = _infer_dataset_loader(self.data_files)
            dataset_or_dict = load_dataset(loader_name, data_files=self.data_files)

        if isinstance(dataset_or_dict, DatasetDict):
            if self.train_split not in dataset_or_dict:
                available_splits = sorted(dataset_or_dict.keys())
                raise ValueError(
                    f"`train_split`={self.train_split!r} was not found. "
                    f"Available splits: {available_splits}."
                )
            dataset = dataset_or_dict[self.train_split]
        else:
            dataset = dataset_or_dict

        if self.text_column not in dataset.column_names:
            raise ValueError(
                f"`text_column`={self.text_column!r} was not found in dataset columns "
                f"{dataset.column_names}."
            )

        if self.max_train_samples is not None:
            if self.max_train_samples <= 0:
                raise ValueError(
                    f"`max_train_samples` must be > 0, but got {self.max_train_samples}."
                )
            sample_count = min(self.max_train_samples, len(dataset))
            dataset = dataset.shuffle(seed=self.shuffle_seed).select(range(sample_count))
        else:
            dataset = dataset.shuffle(seed=self.shuffle_seed)

        if len(dataset) == 0:
            raise ValueError("Training dataset is empty after filtering/shuffling.")

        return dataset

    def _tokenize_dataset(self, dataset: Dataset, tokenizer) -> Dataset:
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer has neither pad_token nor eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

        sample_indices = list(range(len(dataset)))
        dataset = dataset.add_column("_sample_idx", sample_indices)

        def tokenize_batch(batch: dict[str, Any]) -> dict[str, Any]:
            raw_texts = batch[self.text_column]
            texts = [text if isinstance(text, str) else str(text) for text in raw_texts]
            tokenized = tokenizer(
                texts,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
            )
            labels = []
            for input_ids, attention_mask in zip(
                tokenized["input_ids"], tokenized["attention_mask"]
            ):
                labels.append(
                    [
                        token_id if mask_value == 1 else -100
                        for token_id, mask_value in zip(input_ids, attention_mask)
                    ]
                )
            tokenized["labels"] = labels
            tokenized["sample_idx"] = batch["_sample_idx"]
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=list(dataset.column_names),
            desc="Tokenizing SFT dataset",
        )
        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels", "sample_idx"],
        )
        return tokenized_dataset

    @staticmethod
    def _collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        collated = {
            "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "labels": torch.stack([item["labels"] for item in batch], dim=0),
        }
        if "sample_idx" in batch[0]:
            collated["sample_idx"] = torch.stack(
                [item["sample_idx"].to(dtype=torch.long) for item in batch],
                dim=0,
            )
        return collated

    def _find_target_layer_names(self, quantized_model: nn.Module) -> list[str]:
        requested_targets = self.target_modules or _DEFAULT_TARGET_MODULES
        candidate_names: list[str] = []
        fallback_names: list[str] = []

        for name, module in quantized_model.named_modules():
            if not isinstance(module, GPTQLinear):
                continue
            if _is_lm_head_module(name):
                continue

            fallback_names.append(name)
            leaf_name = name.rsplit(".", maxsplit=1)[-1]
            if leaf_name in requested_targets:
                candidate_names.append(name)

        if candidate_names:
            return candidate_names

        if fallback_names:
            logger.warning(
                "No GPTQLinear layers matched target_modules=%s. "
                "Falling back to all GPTQLinear layers except lm_head.",
                requested_targets,
            )
            return fallback_names

        raise ValueError(
            "No GPTQLinear layers were found in `quantized_model`. "
            "PostProcessLoraSFT requires a GPTQ-quantized model."
        )

    def _inject_lora_layers(self, quantized_model: nn.Module) -> int:
        target_names = self._find_target_layer_names(quantized_model)
        for layer_name in target_names:
            base_layer = _resolve_submodule(quantized_model, layer_name)
            wrapped_layer = LoRAGPTQLinear(
                base_layer=base_layer,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
            )
            _replace_submodule(quantized_model, layer_name, wrapped_layer)
        return len(target_names)

    def _collect_lora_state(self, quantized_model: nn.Module) -> dict[str, torch.Tensor]:
        state_dict: dict[str, torch.Tensor] = {}
        for name, module in quantized_model.named_modules():
            if not isinstance(module, LoRAGPTQLinear):
                continue
            state_dict[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            state_dict[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
        return state_dict

    def _save_adapter(self, quantized_model: nn.Module) -> None:
        if self.output_dir is None:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        state_dict = self._collect_lora_state(quantized_model)
        if not state_dict:
            logger.warning("No LoRA adapter state found to save.")
            return

        adapter_path = os.path.join(self.output_dir, "adapter_model.bin")
        config_path = os.path.join(self.output_dir, "adapter_config.json")
        torch.save(state_dict, adapter_path)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "lora_r": self.lora_r,
                    "lora_alpha": self.lora_alpha,
                    "lora_dropout": self.lora_dropout,
                    "target_modules": list(self.target_modules or _DEFAULT_TARGET_MODULES),
                    "sft_loss_weight": self.sft_loss_weight,
                    "teacher_loss_weight": self.teacher_loss_weight,
                    "teacher_loss_type": self.teacher_loss_type,
                    "teacher_temperature": self.teacher_temperature,
                    "teacher_model_id": self.teacher_model_id,
                    "teacher_model_path": self.teacher_model_path,
                    "teacher_dtype": self.teacher_dtype,
                    "teacher_device": self.teacher_device,
                    "cache_teacher_outputs": self.cache_teacher_outputs,
                    "teacher_cache_dtype": self.teacher_cache_dtype,
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
        logger.info("Saved LoRA adapter to %s", self.output_dir)

    def run(
        self,
        quantized_model: nn.Module,
        model_config: ModelConfig,
    ) -> None:
        """Run LoRA SFT on the GPTQ-quantized model in-place."""
        if self.epochs <= 0:
            raise ValueError(f"`epochs` must be > 0, but got {self.epochs}.")
        if self.batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, but got {self.batch_size}.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "`gradient_accumulation_steps` must be > 0, "
                f"but got {self.gradient_accumulation_steps}."
            )
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(f"`lora_dropout` must be in [0, 1), but got {self.lora_dropout}.")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(f"`warmup_ratio` must be in [0, 1], but got {self.warmup_ratio}.")
        if self.sft_loss_weight < 0.0:
            raise ValueError(f"`sft_loss_weight` must be >= 0, but got {self.sft_loss_weight}.")
        if self.teacher_loss_weight < 0.0:
            raise ValueError(
                f"`teacher_loss_weight` must be >= 0, but got {self.teacher_loss_weight}."
            )
        if (
            self.sft_loss_weight == 0.0
            and self.teacher_loss_weight == 0.0
            and self.intermediate_block_loss_weight == 0.0
        ):
            raise ValueError(
                "At least one of `sft_loss_weight`, `teacher_loss_weight`, "
                "or `intermediate_block_loss_weight` must be > 0."
            )
        if self.teacher_distillation_enabled and self.teacher_temperature <= 0.0:
            raise ValueError(
                "`teacher_temperature` must be > 0 when teacher distillation is enabled, "
                f"but got {self.teacher_temperature}."
            )

        self._normalize_runner_eval_device(model_config)

        tokenizer = model_config.load_tokenizer()
        train_dataset = self._tokenize_dataset(self._load_train_dataset(), tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_batch,
        )

        for param in quantized_model.parameters():
            param.requires_grad_(False)

        replaced_layers = self._inject_lora_layers(quantized_model)
        trainable_parameters = [
            param for param in quantized_model.parameters() if param.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters found after LoRA layer injection.")

        train_device = self._resolve_train_device(model_config)
        use_bf16 = self._resolve_use_bf16(train_device)
        teacher_device = self._resolve_optional_device(self.teacher_device, train_device)
        teacher_model = None
        teacher_cache = None
        do_intermediate = self.intermediate_block_loss_weight > 0.0
        target_block_indices: tuple | None = None
        teacher_intermediate_cache: dict | None = None
        logger.info(
            "PostProcessLoraSFT started: layers=%d, samples=%d, epochs=%d, "
            "device=%s, bf16=%s, sft_weight=%.3f, teacher_weight=%.3f, "
            "intermediate_weight=%.3f, teacher_device=%s, "
            "cache_teacher_outputs=%s, cache_intermediate_outputs=%s",
            replaced_layers,
            len(train_dataset),
            self.epochs,
            train_device,
            use_bf16,
            self.sft_loss_weight,
            self.teacher_loss_weight,
            self.intermediate_block_loss_weight,
            teacher_device if (self.teacher_distillation_enabled or do_intermediate) else None,
            self.cache_teacher_outputs if self.teacher_distillation_enabled else False,
            self.cache_intermediate_outputs if do_intermediate else False,
        )

        if do_intermediate:
            target_block_indices = self._resolve_target_block_indices(quantized_model)
            logger.info(
                "Intermediate block alignment: %d block(s) %s",
                len(target_block_indices),
                target_block_indices,
            )

        quantized_model.to(train_device)
        original_use_cache = None
        if hasattr(quantized_model, "config") and hasattr(quantized_model.config, "use_cache"):
            original_use_cache = bool(quantized_model.config.use_cache)
            quantized_model.config.use_cache = False
        teacher_original_use_cache = None
        if self.teacher_distillation_enabled:
            teacher_model = self._load_teacher_model(model_config, teacher_device)
            if hasattr(teacher_model, "config") and hasattr(teacher_model.config, "use_cache"):
                teacher_original_use_cache = bool(teacher_model.config.use_cache)
                teacher_model.config.use_cache = False
            if self.cache_teacher_outputs:
                teacher_cache = self._build_teacher_cache(
                    teacher_model=teacher_model,
                    train_dataset=train_dataset,
                    teacher_device=teacher_device,
                )
                if teacher_original_use_cache is not None:
                    teacher_model.config.use_cache = teacher_original_use_cache
                    teacher_original_use_cache = None
                teacher_model.to("cpu")
                del teacher_model
                teacher_model = None

        # ------------------------------------------------------------------
        # Pattern 1: teacher needed for per-batch intermediate forward.
        # Load here if not already loaded by the logit-distillation path.
        # ------------------------------------------------------------------
        if do_intermediate and not self.cache_intermediate_outputs and teacher_model is None:
            teacher_model = self._load_teacher_model(model_config, teacher_device)
            if hasattr(teacher_model, "config") and hasattr(teacher_model.config, "use_cache"):
                teacher_original_use_cache = bool(teacher_model.config.use_cache)
                teacher_model.config.use_cache = False

        # ------------------------------------------------------------------
        # Intermediate block cache (Pattern 3)
        # ------------------------------------------------------------------
        if do_intermediate and self.cache_intermediate_outputs:
            # Load teacher temporarily if not already available.
            _temp_teacher: nn.Module | None = None
            if teacher_model is None:
                _temp_teacher = self._load_teacher_model(model_config, teacher_device)
                if hasattr(_temp_teacher, "config") and hasattr(_temp_teacher.config, "use_cache"):
                    _temp_teacher.config.use_cache = False
            _teacher_for_icache = _temp_teacher if _temp_teacher is not None else teacher_model
            teacher_intermediate_cache = self._build_intermediate_cache(
                teacher_model=_teacher_for_icache,
                train_dataset=train_dataset,
                teacher_device=teacher_device,
                block_indices=target_block_indices,
            )
            if _temp_teacher is not None:
                _temp_teacher.to("cpu")
                del _temp_teacher

        total_updates = math.ceil(
            (len(train_loader) * self.epochs) / self.gradient_accumulation_steps
        )
        total_updates = max(total_updates, 1)
        warmup_steps = int(total_updates * self.warmup_ratio)

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress_steps = current_step - warmup_steps
            remaining_steps = max(total_updates - warmup_steps, 1)
            return max(0.0, 1.0 - (float(progress_steps) / float(remaining_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        optimizer.zero_grad(set_to_none=True)
        global_step = 0

        try:
            quantized_model.train()
            if teacher_model is not None:
                teacher_model.eval()

            autocast_enabled = train_device.type == "cuda" and use_bf16

            for epoch in range(self.epochs):
                epoch_total_loss = 0.0
                epoch_sft_loss = 0.0
                epoch_teacher_loss = 0.0
                epoch_intermediate_loss = 0.0
                for step, batch in enumerate(train_loader):
                    sample_indices = batch.pop("sample_idx", None)
                    batch = {key: value.to(train_device) for key, value in batch.items()}
                    student_inputs = {
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                    }
                    if self.sft_loss_weight > 0.0:
                        student_inputs["labels"] = batch["labels"]

                    # --------------------------------------------------
                    # Pattern 1: run teacher forward (no cache) to capture
                    # intermediate block outputs before the student forward.
                    # --------------------------------------------------
                    teacher_icaptured: dict = {}
                    if (
                        do_intermediate
                        and not self.cache_intermediate_outputs
                        and teacher_model is not None
                    ):
                        _t_in = {
                            "input_ids": batch["input_ids"].to(teacher_device),
                            "attention_mask": batch["attention_mask"].to(teacher_device),
                        }
                        _t_hooks = _register_block_output_hooks(
                            teacher_model, teacher_icaptured, target_block_indices
                        )
                        with torch.no_grad():
                            teacher_model(**_t_in)
                        for _h in _t_hooks:
                            _h.remove()

                    # Register student intermediate hooks.
                    student_icaptured: dict = {}
                    student_ihooks = []
                    if do_intermediate:
                        student_ihooks = _register_block_output_hooks(
                            quantized_model, student_icaptured, target_block_indices
                        )

                    with (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if autocast_enabled
                        else nullcontext()
                    ):
                        outputs = quantized_model(**student_inputs)

                        # Remove student hooks now that forward is complete.
                        for _h in student_ihooks:
                            _h.remove()
                        student_ihooks = []

                        total_loss = outputs.logits.new_zeros(())

                        if self.sft_loss_weight > 0.0:
                            sft_loss = getattr(outputs, "loss", None)
                            if sft_loss is None:
                                raise RuntimeError(
                                    "Model output does not include `loss`. "
                                    "Expected a CausalLM model output when `labels` are provided."
                                )
                            total_loss = total_loss + (self.sft_loss_weight * sft_loss)
                        else:
                            sft_loss = outputs.logits.new_zeros(())

                        if teacher_cache is not None:
                            if sample_indices is None:
                                raise RuntimeError(
                                    "Teacher cache is enabled, but batch does not include "
                                    "`sample_idx`."
                                )
                            teacher_logits = teacher_cache.index_select(
                                0,
                                sample_indices.to(dtype=torch.long, device="cpu"),
                            ).to(train_device)
                            teacher_loss = self._compute_teacher_loss(
                                student_logits=outputs.logits,
                                teacher_logits=teacher_logits,
                                labels=batch["labels"],
                            )
                            total_loss = total_loss + (self.teacher_loss_weight * teacher_loss)
                        elif teacher_model is not None and self.teacher_loss_weight > 0.0:
                            teacher_inputs = {
                                "input_ids": batch["input_ids"],
                                "attention_mask": batch["attention_mask"],
                            }
                            if teacher_device != train_device:
                                teacher_inputs = {
                                    key: value.to(teacher_device)
                                    for key, value in teacher_inputs.items()
                                }
                            with torch.no_grad():
                                teacher_outputs = teacher_model(**teacher_inputs)
                            teacher_loss = self._compute_teacher_loss(
                                student_logits=outputs.logits,
                                teacher_logits=teacher_outputs.logits,
                                labels=batch["labels"],
                            )
                            total_loss = total_loss + (self.teacher_loss_weight * teacher_loss)
                        else:
                            teacher_loss = outputs.logits.new_zeros(())

                        # ----------------------------------------------
                        # Intermediate block alignment loss
                        # ----------------------------------------------
                        if do_intermediate:
                            if teacher_intermediate_cache is not None:
                                # Pattern 3: load teacher hiddens from cache.
                                if sample_indices is None:
                                    raise RuntimeError(
                                        "Intermediate cache is enabled, but batch does not "
                                        "include `sample_idx`."
                                    )
                                _sidx = sample_indices.to(dtype=torch.long, device="cpu")
                                teacher_icaptured = {
                                    idx: teacher_intermediate_cache[idx].index_select(0, _sidx)
                                    for idx in target_block_indices
                                    if teacher_intermediate_cache.get(idx) is not None
                                }
                            intermediate_loss = _compute_intermediate_block_loss(
                                student_icaptured, teacher_icaptured, train_device
                            )
                            total_loss = total_loss + (
                                self.intermediate_block_loss_weight * intermediate_loss
                            )
                        else:
                            intermediate_loss = outputs.logits.new_zeros(())

                        scaled_loss = total_loss / self.gradient_accumulation_steps

                    scaled_loss.backward()
                    # Ensure hooks are removed even if an exception occurred.
                    for _h in student_ihooks:
                        _h.remove()
                    student_ihooks = []

                    epoch_total_loss += float(total_loss.detach().item())
                    epoch_sft_loss += float(sft_loss.detach().item())
                    epoch_teacher_loss += float(teacher_loss.detach().item())
                    epoch_intermediate_loss += float(intermediate_loss.detach().item())

                    should_update = (step + 1) % self.gradient_accumulation_steps == 0 or (
                        step + 1
                    ) == len(train_loader)
                    if should_update:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                            logger.info(
                                "PostProcessLoraSFT step %d/%d - total_loss=%.6f, "
                                "sft_loss=%.6f, teacher_loss=%.6f, "
                                "intermediate_loss=%.6f",
                                global_step,
                                total_updates,
                                float(total_loss.detach().item()),
                                float(sft_loss.detach().item()),
                                float(teacher_loss.detach().item()),
                                float(intermediate_loss.detach().item()),
                            )

                logger.info(
                    "PostProcessLoraSFT epoch %d/%d - avg_total_loss=%.6f, "
                    "avg_sft_loss=%.6f, avg_teacher_loss=%.6f, "
                    "avg_intermediate_loss=%.6f",
                    epoch + 1,
                    self.epochs,
                    epoch_total_loss / float(max(len(train_loader), 1)),
                    epoch_sft_loss / float(max(len(train_loader), 1)),
                    epoch_teacher_loss / float(max(len(train_loader), 1)),
                    epoch_intermediate_loss / float(max(len(train_loader), 1)),
                )
        finally:
            if original_use_cache is not None:
                quantized_model.config.use_cache = original_use_cache
            if teacher_model is not None:
                if teacher_original_use_cache is not None:
                    teacher_model.config.use_cache = teacher_original_use_cache
                teacher_model.to("cpu")
                del teacher_model
            quantized_model.eval()
            quantized_model.to("cpu")
            self._save_adapter(quantized_model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@dataclass
class PostProcessLoraTeacherSFT(PostProcessLoraSFT):
    """LoRA SFT with teacher-guided distillation enabled by default."""

    teacher_loss_weight: float = 1.0


@dataclass
class PostProcessLoraTeacherOnlySFT(PostProcessLoraSFT):
    """LoRA SFT variant that optimizes only the teacher distillation loss."""

    sft_loss_weight: float = 0.0
    teacher_loss_weight: float = 1.0
