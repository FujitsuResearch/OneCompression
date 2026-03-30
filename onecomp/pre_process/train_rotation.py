"""Rotation matrix training and application orchestration.

Contains ``PreprocessManager`` for managing R1/R2/S_* tensors, the training
pipeline (``apply_preprocess_train``) and the evaluation/application pipeline
(``apply_preprocess_eval``).

Copyright 2025-2026 Fujitsu Ltd.

Author: Yusei Kawakami
"""

from logging import getLogger

import torch
import torch.nn as nn
import transformers
from transformers import default_data_collator

from .quant_models import (
    QuantEmbedding,
    QuantLlamaDecoderLayer,
    QuantLinear,
    QuantQwen3DecoderLayer,
    QuantRMSNorm,
    RotateModule,
    ScalingModule,
    WeightQuantizer,
)
from .rotation_utils import (
    cleanup_memory,
    find_linear_layers,
    fuse_layer_norms,
    register_online_hadamard_hooks,
    rotate_model,
    untie_word_embeddings,
)
from .optimizer import SGDG
from .preprocess_args import TrainingArguments, get_training_arguments

logger = getLogger(__name__)


# ============================================================
# Calibration dataset for training
# ============================================================


class _PreprocessTrainDataset(torch.utils.data.Dataset):
    """Calibration samples for HuggingFace Trainer rotation training.

    Each item is a dict with ``input_ids``, ``attention_mask``, and ``labels``
    (labels match ``input_ids`` for causal language modeling).

    Args:
        inputs: Output of ``prepare_calibration_dataset`` — a dict containing
            ``input_ids`` and ``attention_mask`` tensors of shape
            ``(num_examples, sequence_length)``.
    """

    def __init__(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.data = [
            dict(
                input_ids=input_ids[i].tolist(),
                attention_mask=attention_mask[i].tolist(),
                labels=input_ids[i].tolist(),
            )
            for i in range(input_ids.shape[0])
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


# ============================================================
# PreprocessManager
# ============================================================


class PreprocessManager:
    """Manages rotation (R1, R2) and scaling (S_*) tensors.

    Args:
        config: ``model.config`` of the target model.
        rotation: Whether to generate rotation matrices.
        scaling: Whether to generate scaling vectors.
        rotation_mode: ``"random"`` | ``"identity"``.
        scaling_mode: ``"identity"`` | ``"random_ones"`` | ``"random"``.
        seed: Random seed for tensor generation.
        dev: Device for intermediate computation.
    """

    def __init__(
        self,
        config,
        *,
        rotation=True,
        scaling=False,
        rotation_mode="random",
        scaling_mode="identity",
        seed=0,
        dev="cpu",
    ):
        self.config = config
        self.rotation = rotation
        self.scaling = scaling
        self.rotation_mode = rotation_mode
        self.scaling_mode = scaling_mode
        self.seed = seed
        self.dev = dev
        self.save_dict: dict = {}
        self._generate_tensors()

    def _generate_tensors(self):
        """Initialize ``save_dict`` with ``RotateModule`` and ``ScalingModule`` entries.

        Seeds RNG with ``self.seed`` for reproducibility, fills ``R1``, per-layer
        ``R2``, and optional ``S_*`` tensors from ``config`` dimensions, then
        restores the previous CPU/CUDA RNG state.
        """
        rng_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        cfg = self.config
        hidden = cfg.hidden_size
        head_dim = cfg.head_dim
        n_kv = cfg.num_key_value_heads
        intermediate = cfg.intermediate_size
        n_layers = cfg.num_hidden_layers

        self.save_dict["R1"] = RotateModule(self._ortho(hidden)) if self.rotation else None
        self.save_dict["R2"] = (
            {i: RotateModule(self._ortho(head_dim)) for i in range(n_layers)}
            if self.rotation
            else None
        )

        self.save_dict["S_attn"] = (
            {i: ScalingModule(self._scale(hidden)) for i in range(n_layers)}
            if self.scaling
            else None
        )
        if self.scaling and getattr(cfg, "model_type", "") != "qwen3":
            self.save_dict["S_qk"] = {
                i: ScalingModule(self._scale((head_dim // 2) * n_kv)) for i in range(n_layers)
            }
        else:
            self.save_dict["S_qk"] = None
        self.save_dict["S_ov"] = (
            {i: ScalingModule(self._scale(head_dim * n_kv)) for i in range(n_layers)}
            if self.scaling
            else None
        )
        self.save_dict["S_mlp"] = (
            {i: ScalingModule(self._scale(hidden)) for i in range(n_layers)}
            if self.scaling
            else None
        )
        self.save_dict["S_up_down"] = (
            {i: ScalingModule(self._scale(intermediate)) for i in range(n_layers)}
            if self.scaling
            else None
        )

        torch.random.set_rng_state(rng_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        cleanup_memory()

    def _ortho(self, size):
        """Sample or build an orthogonal matrix used inside ``RotateModule``.

        Args:
            size: Side length of the square matrix (e.g. ``hidden_size`` or
                ``head_dim``).

        Returns:
            torch.Tensor: ``(size, size)`` float64 orthogonal matrix. In
            ``"random"`` mode, QR factorization of a Gaussian on ``self.dev``,
            returned on CPU; in ``"identity"`` mode, the identity matrix.
        """
        mode = self.rotation_mode
        if mode == "random":
            M = torch.randn(size, size, dtype=torch.float64, device=self.dev)
            q, r = torch.linalg.qr(M)
            q *= torch.sign(torch.diag(r)).unsqueeze(0)
            return q.cpu()
        if mode == "identity":
            return torch.eye(size, dtype=torch.float64)
        raise ValueError(f"Unknown rotation_mode: {mode}")

    def _scale(self, size):
        """Build a 1-D scaling vector for ``ScalingModule``.

        Args:
            size: Length of the scaling vector for the target submodule.

        Returns:
            torch.Tensor: 1-D float64 tensor of length ``size``: all ones
            (``identity``), random ±1 (``random_ones``), or random magnitude
            with random sign (``random``), depending on ``scaling_mode``.
        """
        mode = self.scaling_mode
        if mode == "identity":
            return torch.ones(size, dtype=torch.float64)
        if mode == "random_ones":
            return torch.randint(0, 2, (size,), dtype=torch.float64) * 2 - 1
        if mode == "random":
            mag = torch.rand(size, dtype=torch.float64) + 0.5
            sign = torch.sign(torch.randn(size, dtype=torch.float64))
            sign[sign == 0] = 1.0
            return mag * sign
        raise ValueError(f"Unknown scaling_mode: {mode}")

    # --- Extract raw tensors for rotate_model() ---

    def get_R1(self):
        """Tensor form of the global input embedding rotation, if enabled.

        Returns:
            torch.Tensor or None: ``R1.weight`` data, or ``None`` when
            ``rotation`` was False.
        """
        if self.save_dict["R1"] is None:
            return None
        return self.save_dict["R1"].weight.data

    def get_R2_per_layer(self):
        """Per-layer head-dimension rotations for attention, if enabled.

        Returns:
            dict[int, torch.Tensor] or None: Maps layer index to ``R2`` weight
            data, or ``None`` when ``rotation`` was False.
        """
        if self.save_dict["R2"] is None:
            return None
        return {i: m.weight.data for i, m in self.save_dict["R2"].items()}

    def get_S_per_layer(self, key):
        """Per-layer scaling vectors for a named slot in ``save_dict``.

        Args:
            key: ``"S_attn"``, ``"S_qk"``, ``"S_ov"``, ``"S_mlp"``, or
                ``"S_up_down"``.

        Returns:
            dict[int, torch.Tensor] or None: Maps layer index to the 1-D
            scaling tensor, or ``None`` if that slot is absent (e.g. scaling
            disabled, or ``S_qk`` omitted for Qwen3).
        """
        if self.save_dict[key] is None:
            return None
        return {i: m.weight.data for i, m in self.save_dict[key].items()}


# ============================================================
# Training pipeline helpers
# ============================================================


def _convert_model_structure(config, model_type, model_path, use_sdpa=False):
    """Re-load the checkpoint as a Quant-wrapped causal LM for rotation training.

    The original model does not need to be in memory; only its ``config`` and
    ``model_type`` are required to select the correct architecture.  Weights
    are loaded from *model_path* via ``from_pretrained``.

    Args:
        config: ``PretrainedConfig`` of the target model (will be deep-copied).
        model_type: Architecture identifier (``"llama"`` or ``"qwen3"``).
        model_path: Hugging Face model id or local path passed to
            ``from_pretrained`` for the quant model.
        use_sdpa: If True, sets ``config._attn_implementation`` to ``"sdpa"``;
            otherwise ``"eager"``.

    Returns:
        nn.Module: Custom ``LlamaForCausalLM`` or ``Qwen3ForCausalLM`` (from
        ``modeling_llama`` / ``modeling_qwen3``) with decoder blocks replaced
        by ``Quant*`` layers, fused-norm-ready embeddings and head, and
        untied embeddings.
    """
    from .modeling_llama import LlamaForCausalLM as QLlamaFC
    from .modeling_qwen3 import Qwen3ForCausalLM as QQwen3FC

    import copy

    config = copy.deepcopy(config)
    config._attn_implementation = "sdpa" if use_sdpa else "eager"

    is_llama = model_type == "llama"
    is_qwen3 = model_type == "qwen3"

    if is_llama:
        quant_model = QLlamaFC.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cpu",
            low_cpu_mem_usage=True,
            config=config,
        )
    elif is_qwen3:
        quant_model = QQwen3FC.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cpu",
            low_cpu_mem_usage=True,
            config=config,
        )
    else:
        raise ValueError(
            f"Unsupported model type for rotation training: {model_type}. "
            "Only LlamaForCausalLM and Qwen3ForCausalLM are supported."
        )

    quant_model.seqlen = 2048
    untie_word_embeddings(quant_model)

    layers = quant_model.model.layers
    for i in range(len(layers)):
        if is_llama:
            layers[i] = QuantLlamaDecoderLayer(config, i, layers[i])
        else:
            layers[i] = QuantQwen3DecoderLayer(config, i, layers[i])

    quant_model.model.embed_tokens = QuantEmbedding(quant_model.model.embed_tokens)
    quant_model.model.norm = QuantRMSNorm(quant_model.model.norm)
    quant_model.lm_head = QuantLinear(quant_model.lm_head, name="head")
    cleanup_memory()
    quant_model.eval()
    return quant_model


def _register_preprocess_modules(model, manager):
    """Attach ``R1`` and per-layer ``R_S_modules`` from ``manager`` onto ``model``.

    Args:
        model: Quant causal LM from ``_convert_model_structure`` (must expose
            ``model.model.layers``).
        manager: ``PreprocessManager`` whose ``save_dict`` holds ``R1``, ``R2``,
            and optional ``S_*`` modules indexed by layer.
    """
    model.R1 = manager.save_dict["R1"]
    for i, layer in enumerate(model.model.layers):
        layer.R_S_modules = nn.ModuleDict(
            {
                k: manager.save_dict[k][i] if manager.save_dict[k] is not None else None
                for k in ("R2", "S_attn", "S_qk", "S_ov", "S_mlp", "S_up_down")
            }
        )
    cleanup_memory()


def _insert_weight_quantizer(model, wbits=4, sym=False, groupsize=-1):
    """Install RTN-style ``WeightQuantizer`` on every ``QuantLinear`` in decoder layers.

    Args:
        model: Quant causal LM whose layers contain ``QuantLinear`` submodules.
        wbits: Bit width forwarded to ``WeightQuantizer.configure``.
        sym: Whether to use symmetric quantization in the proxy.
        groupsize: Weight group size for the proxy (-1 to disable grouping).
    """
    for layer in model.model.layers:
        subset = find_linear_layers(layer, layers=[QuantLinear])
        for name, mod in subset.items():
            wq = WeightQuantizer()
            wq.configure(wbits, perchannel=True, sym=sym, weight_groupsize=groupsize)
            mod.quantizer = wq
    cleanup_memory()


class _PreprocessTrainer(transformers.Trainer):
    """``Trainer`` subclass for rotation/scaling preprocessing optimization.

    Routes metrics to the module logger instead of stdout, and builds an
    ``SGDG`` optimizer that treats 2-D parameters as Stiefel rotations and
    1-D parameters as Euclidean scaling factors, using learning rates and
    momentum values from ``TrainingArguments``.
    """

    def log(self, logs, start_time=None):
        """Override to route training metrics through logger instead of print.

        Preserves ``log_history`` and ``epoch`` from the base implementation
        but replaces callback-based printing with ``logger.info``.
        """
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

    def create_optimizer(self, model=None):
        """Create ``SGDG`` with rotation vs. scaling parameter groups.

        Overrides ``Trainer.create_optimizer`` (not the deprecated
        ``create_optimizer_and_scheduler``) so that the custom optimizer
        is actually used in transformers >= 5.x where the training loop
        calls ``create_optimizer()`` and ``create_scheduler()`` separately.

        Args:
            model: Optional model override (used by the Trainer when the
                model is wrapped after optimizer creation).

        Returns:
            torch.optim.Optimizer: The SGDG optimizer instance.
        """
        if self.optimizer is not None:
            return self.optimizer

        args = self.args
        opt_model = self.model if model is None else model
        params_rot, params_scale = [], []
        for p in opt_model.parameters():
            if p.requires_grad:
                (params_rot if len(p.size()) == 2 else params_scale).append(p)

        groups = [
            {
                "params": params_rot,
                "lr": args.rotation_lr,
                "momentum": args.rotation_momentum,
                "stiefel": True,
            },
            {
                "params": params_scale,
                "lr": args.scaling_lr,
                "momentum": args.scaling_momentum,
                "stiefel": False,
                "nesterov": False,
            },
        ]
        self.optimizer = SGDG(groups, weight_decay=args.weight_decay)
        return self.optimizer


# ============================================================
# Public API
# ============================================================


def apply_preprocess_train(
    config,
    model_type,
    manager: PreprocessManager,
    model_path: str,
    tokenizer,
    calibration_inputs,
    *,
    wbits=4,
    sym=False,
    groupsize=-1,
    use_sdpa=False,
    fp32_had=False,
    training_args_override=None,
):
    """Train rotation/scaling matrices via the SpinQuant/OstQuant training pipeline.

    Args:
        config: ``PretrainedConfig`` of the target model.
        model_type: Architecture identifier (``"llama"`` or ``"qwen3"``).
        manager: ``PreprocessManager`` holding R1/R2/S_* tensors.
        model_path: Model id or path (for ``from_pretrained``).
        tokenizer: Tokenizer (from ``ModelConfig.load_tokenizer()``).
        calibration_inputs: Output of ``prepare_calibration_dataset``
            (dict with ``"input_ids"`` tensor of shape ``(N, seqlen)``).
        wbits: RTN proxy bit-width.
        sym: Symmetric quantisation for RTN proxy.
        groupsize: Group size for RTN proxy.
        use_sdpa: Use SDPA attention during training.
        fp32_had: FP32 online Hadamard.
        training_args_override: Override ``TrainingArguments`` fields.
    """
    with torch.no_grad():
        logger.info("Step 1/6: Converting model structure for training ...")
        quant_model = _convert_model_structure(config, model_type, model_path, use_sdpa=use_sdpa)

        logger.info("Step 2/6: Fusing layer norms ...")
        fuse_layer_norms(quant_model)

        logger.info("Step 3/6: Registering preprocess modules ...")
        _register_preprocess_modules(quant_model, manager)

        logger.info("Step 4/6: Registering online Hadamard hooks on down_proj ...")
        register_online_hadamard_hooks(
            quant_model,
            fp32_had=fp32_had,
            layers_cls=[nn.Linear, QuantLinear],
        )
        cleanup_memory()

        logger.info("Step 5/6: Inserting weight quantizers ...")
        _insert_weight_quantizer(quant_model, wbits=wbits, sym=sym, groupsize=groupsize)

        logger.info("Step 6/6: Preparing trainer ...")
        for p in quant_model.parameters():
            p.requires_grad = False
        factor_keys = {"R1", "R2", "S_attn", "S_qk", "S_ov", "S_mlp", "S_up_down"}
        for pname, p in quant_model.named_parameters():
            if any(k in pname for k in factor_keys):
                p.requires_grad = True
        cleanup_memory()

        quant_model.config.use_cache = False
        train_dataset = _PreprocessTrainDataset(calibration_inputs)

        train_cfg = training_args_override or {}
        training_args = get_training_arguments(train_cfg)

        if getattr(training_args, "gradient_checkpointing", False) and hasattr(
            quant_model, "enable_input_require_grads"
        ):
            quant_model.enable_input_require_grads()

        trainer = _PreprocessTrainer(
            model=quant_model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=default_data_collator,
        )

    # Training runs outside no_grad — Trainer enables gradients internally
    trainer.train()

    del quant_model
    cleanup_memory()


def apply_preprocess_eval(model, manager: PreprocessManager, *, fp32_had=False, dev="cpu"):
    """Apply trained rotation/scaling to model weights in-place and register hooks.

    Args:
        model: HuggingFace causal LM whose weights ``rotate_model`` will update.
        manager: ``PreprocessManager`` providing ``get_R1`` / ``get_R2_per_layer``
            / ``get_S_per_layer`` tensors after training.
        fp32_had: If True, online Hadamard transforms on ``down_proj`` use FP32.
        dev: Device for tensor ops inside ``rotate_model`` (e.g. ``"cpu"``).
    """
    with torch.no_grad():
        logger.info("Step 1/3: Fusing layer norms ...")
        untie_word_embeddings(model)
        fuse_layer_norms(model)

        logger.info("Step 2/3: Applying rotation/scaling to weights ...")
        rotate_model(
            model,
            R1=manager.get_R1(),
            R2_per_layer=manager.get_R2_per_layer(),
            S_attn_per_layer=manager.get_S_per_layer("S_attn"),
            S_qk_per_layer=manager.get_S_per_layer("S_qk"),
            S_ov_per_layer=manager.get_S_per_layer("S_ov"),
            S_mlp_per_layer=manager.get_S_per_layer("S_mlp"),
            S_up_down_per_layer=manager.get_S_per_layer("S_up_down"),
            dev=dev,
        )

        logger.info("Step 3/3: Registering online Hadamard hooks on down_proj ...")
        register_online_hadamard_hooks(model, fp32_had=fp32_had)
        cleanup_memory()
        logger.info("Rotation preprocessing completed.")
