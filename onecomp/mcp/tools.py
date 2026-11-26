"""MCP tool implementations for OneComp.

Each tool is a plain Python function paired with a JSON Schema
definition, exposed through :class:`onecomp.mcp.protocol.MCPServer`.
Memory estimation reuses the formulas in
:mod:`onecomp.utils.vram_estimator` and reads only ``config.json``
(no model weights are loaded).

Classes:
    ModelProfile: Lightweight per-model facts used by the estimation tools.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments
import os
from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, List, Optional

from ..utils.vram_estimator import effective_bits_per_param
from .compaction_tools import build_compaction_tools
from .protocol import ToolDefinition

_BYTES_PER_GB = 1e9
_FP16_BITS = 16.0
_DEFAULT_GROUPSIZE = 128
_QUANTIZABLE_RATIO = 0.95
_VRAM_RATIO = 0.80
_WBITS_CANDIDATES = (8, 6, 5, 4, 3, 2)


def _trust_remote_code() -> bool:
    """Whether custom modeling code from the Hub may be executed.

    MCP tools receive ``model_id`` from a calling agent, so loading a config
    with ``trust_remote_code=True`` would let any tool call execute arbitrary
    Python shipped with a Hub repository.  The server operator must opt in
    explicitly by setting ``ONECOMP_MCP_TRUST_REMOTE_CODE=1``.
    """
    return os.environ.get("ONECOMP_MCP_TRUST_REMOTE_CODE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


@dataclass
class ModelProfile:
    """Facts about a model derived from its ``config.json`` only.

    Args:
        model_id: Hugging Face model ID or local path.
        num_params: Total number of parameters (counted on the ``meta``
            device, so no weight data is ever materialised).
        num_hidden_layers: Number of transformer layers.
        num_key_value_heads: Number of KV heads (GQA-aware).
        head_dim: Dimension of one attention head.
    """

    model_id: str
    num_params: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int


def load_model_profile(model_id: str) -> ModelProfile:
    """Build a :class:`ModelProfile` from ``config.json`` without weights.

    Works with both local paths and Hugging Face model IDs.  The model
    architecture is instantiated on the ``meta`` device to obtain an
    exact parameter count, so no weight memory is allocated and no
    weight files are downloaded.

    Custom modeling code from the Hub is NOT executed unless the server
    operator sets ``ONECOMP_MCP_TRUST_REMOTE_CODE=1`` (``model_id`` comes
    from the calling agent and must be treated as untrusted input).
    """
    _non_empty_string("model_id", model_id)
    import torch  # pylint: disable=import-outside-toplevel
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoConfig,
        AutoModelForCausalLM,
    )

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=_trust_remote_code())
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    num_params = sum(p.numel() for p in model.parameters())

    num_heads = getattr(config, "num_attention_heads", 1)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = getattr(config, "hidden_size", num_heads) // num_heads

    return ModelProfile(
        model_id=model_id,
        num_params=num_params,
        num_hidden_layers=getattr(config, "num_hidden_layers", 0),
        num_key_value_heads=getattr(config, "num_key_value_heads", num_heads),
        head_dim=head_dim,
    )


def _weight_memory_gb(num_params: int, wbits: Optional[float], groupsize: int) -> Dict[str, float]:
    """Weight memory (GB) for a partially quantised model.

    ``quantizable_ratio`` of the parameters are stored at the effective
    bit-width (weights + per-group scale/zero metadata, AutoGPTQ
    convention: scale FP16, zero packed at ``wbits``); the remainder
    stays in FP16.  ``wbits=None`` means no quantisation (FP16).
    """
    if wbits is None or wbits >= _FP16_BITS:
        return {
            "effective_bits_per_param": _FP16_BITS,
            "weight_gb": num_params * _FP16_BITS / 8 / _BYTES_PER_GB,
        }
    eff = effective_bits_per_param(wbits=wbits, group_size=groupsize)
    quantizable = int(num_params * _QUANTIZABLE_RATIO)
    non_quantizable = num_params - quantizable
    weight_bits = quantizable * eff + non_quantizable * _FP16_BITS
    return {
        "effective_bits_per_param": eff,
        "weight_gb": weight_bits / 8 / _BYTES_PER_GB,
    }


def _kv_cache_gb(
    profile: ModelProfile,
    context_len: int,
    batch_size: int,
    kv_bytes: int = 2,
) -> float:
    """KV-cache memory (GB): K and V per layer per KV head per token."""
    elements = (
        2
        * profile.num_hidden_layers
        * profile.num_key_value_heads
        * profile.head_dim
        * context_len
        * batch_size
    )
    return elements * kv_bytes / _BYTES_PER_GB


def estimate_memory(
    model_id: str,
    wbits: Optional[float] = None,
    context_len: int = 4096,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Estimate weight + KV-cache memory (GB) from ``config.json`` only.

    Args:
        model_id: Hugging Face model ID or local path.
        wbits: Quantisation bit-width.  ``None`` means FP16 (no
            quantisation).
        context_len: Context length for the KV-cache estimate.
        batch_size: Batch size for the KV-cache estimate.

    Returns:
        Dict with ``num_params``, ``weight_gb``, ``kv_cache_gb``,
        ``total_gb``, and the config facts used for the estimate.
    """
    _non_empty_string("model_id", model_id)
    if wbits is not None:
        _bounded_positive_number("wbits", wbits, maximum=16)
    _positive_int("context_len", context_len)
    _positive_int("batch_size", batch_size)
    profile = load_model_profile(model_id)
    weights = _weight_memory_gb(profile.num_params, wbits, _DEFAULT_GROUPSIZE)
    kv_gb = _kv_cache_gb(profile, context_len, batch_size)
    return {
        "model_id": model_id,
        "num_params": profile.num_params,
        "wbits": _FP16_BITS if wbits is None else wbits,
        "groupsize": _DEFAULT_GROUPSIZE,
        "effective_bits_per_param": weights["effective_bits_per_param"],
        "weight_gb": weights["weight_gb"],
        "kv_cache_gb": kv_gb,
        "total_gb": weights["weight_gb"] + kv_gb,
        "context_len": context_len,
        "batch_size": batch_size,
        "num_hidden_layers": profile.num_hidden_layers,
        "num_key_value_heads": profile.num_key_value_heads,
        "head_dim": profile.head_dim,
    }


def _recipe_for_bits(wbits: int) -> Dict[str, Any]:
    """Rule-based recipe parameters for a given bit-width."""
    if wbits >= 4:
        return {"groupsize": 128, "method": "gptq+qep", "kv_cache_precision": "fp16"}
    if wbits == 3:
        return {"groupsize": 64, "method": "gptq+qep", "kv_cache_precision": "fp8"}
    return {"groupsize": 32, "method": "gptq+qep", "kv_cache_precision": "fp8"}


def recommend_recipe(
    model_id: str,
    total_vram_gb: float,
    context_len: int = 4096,
    target: str = "quality",
) -> Dict[str, Any]:
    """Recommend a quantisation recipe that fits a VRAM budget.

    Only ``_VRAM_RATIO`` of the total VRAM is budgeted for weights and
    KV cache; the rest is reserved for activations and runtime
    overhead.  Candidate bit-widths are evaluated from high to low and:

    - ``target="quality"`` picks the **largest** bit-width that fits
      (best accuracy within budget).
    - ``target="memory"`` picks the **smallest** fitting bit-width not
      below 3 bits (aggressive compression while avoiding the extreme
      accuracy loss typical of 2-bit).

    Args:
        model_id: Hugging Face model ID or local path.
        total_vram_gb: Total VRAM of the target GPU in GB.
        context_len: Context length for the KV-cache estimate.
        target: ``"quality"`` (default) or ``"memory"``.

    Returns:
        Dict with the recommended ``wbits``, ``groupsize``, ``method``,
        ``kv_cache_precision``, the memory estimates, and a
        ``rationale`` string explaining the decision.
    """
    _non_empty_string("model_id", model_id)
    _positive_finite("total_vram_gb", total_vram_gb)
    _positive_int("context_len", context_len)
    if target not in {"quality", "memory"}:
        raise ValueError("target must be 'quality' or 'memory'")
    profile = load_model_profile(model_id)
    budget_gb = total_vram_gb * _VRAM_RATIO

    fitting: List[Dict[str, Any]] = []
    estimates: List[Dict[str, Any]] = []
    for bits in _WBITS_CANDIDATES:
        recipe = _recipe_for_bits(bits)
        weights = _weight_memory_gb(profile.num_params, bits, recipe["groupsize"])
        kv_bytes = 1 if recipe["kv_cache_precision"] == "fp8" else 2
        kv_gb = _kv_cache_gb(profile, context_len, 1, kv_bytes=kv_bytes)
        total_gb = weights["weight_gb"] + kv_gb
        entry = {"wbits": bits, "total_gb": total_gb, **recipe}
        estimates.append(entry)
        if total_gb <= budget_gb:
            fitting.append(entry)

    if not fitting:
        chosen = min(estimates, key=lambda e: e["total_gb"])
        rationale = (
            f"No candidate bit-width fits the budget of {budget_gb:.2f} GB "
            f"({_VRAM_RATIO:.0%} of {total_vram_gb:.2f} GB); the smallest "
            f"configuration ({chosen['wbits']}-bit, "
            f"{chosen['total_gb']:.2f} GB) is returned but will not fit. "
            f"Consider a smaller model or a larger GPU."
        )
        fits = False
    elif target == "memory":
        preferred = [e for e in fitting if e["wbits"] >= 3]
        chosen = min(preferred or fitting, key=lambda e: e["wbits"])
        rationale = (
            f"target='memory': smallest bit-width (>= 3-bit when possible) "
            f"fitting the budget of {budget_gb:.2f} GB "
            f"({_VRAM_RATIO:.0%} of {total_vram_gb:.2f} GB) is "
            f"{chosen['wbits']}-bit at {chosen['total_gb']:.2f} GB estimated."
        )
        fits = True
    else:
        chosen = max(fitting, key=lambda e: e["wbits"])
        rationale = (
            f"target='quality': largest bit-width fitting the budget of "
            f"{budget_gb:.2f} GB ({_VRAM_RATIO:.0%} of {total_vram_gb:.2f} GB) "
            f"is {chosen['wbits']}-bit at {chosen['total_gb']:.2f} GB estimated."
        )
        fits = True

    return {
        "model_id": model_id,
        "total_vram_gb": total_vram_gb,
        "budget_gb": budget_gb,
        "wbits": chosen["wbits"],
        "groupsize": chosen["groupsize"],
        "method": chosen["method"],
        "kv_cache_precision": chosen["kv_cache_precision"],
        "estimated_total_gb": chosen["total_gb"],
        "fits": fits,
        "rationale": rationale,
    }


def quantize(
    model_id: str,
    save_dir: str,
    wbits: Optional[float] = None,
    total_vram_gb: Optional[float] = None,
    qep: bool = True,
    evaluate: bool = False,
) -> Dict[str, Any]:
    """Quantise a model via :meth:`onecomp.Runner.auto_run`.

    This is a thin wrapper: all heavy lifting (bit allocation, GPTQ,
    QEP, saving) is delegated to ``Runner.auto_run``.  Note that this
    call can take from minutes to hours depending on the model size.

    Args:
        model_id: Hugging Face model ID or local path.
        save_dir: Directory to save the quantised model to.
        wbits: Target bit-width (``None`` = estimated from VRAM).
        total_vram_gb: VRAM budget in GB for bit-width estimation.
        qep: Whether to apply QEP (default: ``True``).
        evaluate: Whether to evaluate perplexity/accuracy after
            quantisation (default: ``False``).

    Returns:
        Dict with the effective settings and the save directory.
    """
    _non_empty_string("model_id", model_id)
    _non_empty_string("save_dir", save_dir)
    if wbits is not None:
        _bounded_positive_number("wbits", wbits, maximum=16)
    if total_vram_gb is not None:
        _positive_finite("total_vram_gb", total_vram_gb)
    if not isinstance(qep, bool):
        raise ValueError("qep must be a boolean")
    if not isinstance(evaluate, bool):
        raise ValueError("evaluate must be a boolean")
    from ..runner import Runner  # pylint: disable=import-outside-toplevel

    Runner.auto_run(
        model_id=model_id,
        wbits=wbits,
        total_vram_gb=total_vram_gb,
        qep=qep,
        evaluate=evaluate,
        save_dir=save_dir,
    )
    return {
        "model_id": model_id,
        "save_dir": save_dir,
        "wbits": wbits,
        "total_vram_gb": total_vram_gb,
        "qep": qep,
        "evaluated": evaluate,
        "status": "completed",
    }


def evaluate(model_dir: str, num_samples: int = 16) -> Dict[str, Any]:
    """Evaluate the perplexity of a saved quantised model.

    Loads the model with
    :meth:`onecomp.QuantizedModelLoader.load_quantized_model` and
    computes WikiText-2 perplexity with
    :func:`onecomp.utils.calculate_perplexity`.

    Args:
        model_dir: Directory containing the saved quantised model.
        num_samples: Number of evaluation samples (default: 16).

    Returns:
        Dict with ``perplexity`` and the evaluation settings.
    """
    _non_empty_string("model_dir", model_dir)
    _positive_int("num_samples", num_samples)
    # pylint: disable=import-outside-toplevel
    from ..quantized_model_loader import QuantizedModelLoader
    from ..utils import calculate_perplexity

    model, tokenizer = QuantizedModelLoader.load_quantized_model(model_dir)
    ppl = calculate_perplexity(model=model, tokenizer=tokenizer, max_samples=num_samples)
    return {
        "model_dir": model_dir,
        "perplexity": float(ppl),
        "num_samples": num_samples,
        "dataset": "wikitext-2-raw-v1",
    }


def _non_empty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite(name: str, value: Any) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _bounded_positive_number(name: str, value: Any, *, maximum: float) -> float:
    number = _positive_finite(name, value)
    if number > maximum:
        raise ValueError(f"{name} must be at most {maximum:g}")
    return number


_KV_PRECISION_CHOICES = ("auto", "fp16", "fp8")

# quant_method values implemented by OneComp vLLM plugins.  The plugins are
# registered through Python entry points when ``onecomp`` is installed in the
# vLLM environment, so serving these models requires onecomp alongside vLLM.
_PLUGIN_QUANT_METHODS = {
    "mixed_gptq": "vllm_plugins.gptq (Mixed-GPTQ)",
    "dbf": "vllm_plugins.dbf (Double Binary Factorization)",
}

# quant_method values vLLM handles natively (no OneComp plugin needed).
_NATIVE_QUANT_METHODS = ("gptq", "gptq_marlin", "awq", "fp8", "compressed-tensors")


def _summarise_quantization_bits(quant_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Summarise the per-module ``quantization_bits`` block, if present."""
    qbits = quant_config.get("quantization_bits")
    if not qbits:
        return None
    all_bits = set()
    all_group_sizes = set()
    for layer_cfg in qbits:
        for mod_cfg in layer_cfg.values():
            all_bits.add(mod_cfg.get("bits", 0))
            group_size = mod_cfg.get("group_size")
            if group_size is None:
                group_size = mod_cfg.get("params", {}).get("group_size")
            if group_size is None:
                group_size = quant_config.get("group_size")
            all_group_sizes.add(group_size)
    return {
        "num_layers": len(qbits),
        "bits": sorted(all_bits),
        "group_sizes": sorted(g for g in all_group_sizes if g is not None),
    }


def vllm_engine_args(model_dir: str, kv_cache_precision: str = "auto") -> Dict[str, Any]:
    """Derive vLLM launch arguments for a saved (quantised) model directory.

    Read-only: inspects ``config.json`` (``quantization_config`` block) and
    never loads weights.  Reports whether the model needs an OneComp vLLM
    plugin (``mixed_gptq`` / ``dbf``), recommends a ``kv_cache_dtype``, and
    assembles both ``LLM(**engine_kwargs)`` kwargs and a ``vllm serve``
    command line.

    Args:
        model_dir: Directory containing a model saved by OneComp (or any
            HF model directory with a ``config.json``).
        kv_cache_precision: KV-cache precision to serve with: ``"auto"``
            (keep model dtype), ``"fp16"`` (same as auto), or ``"fp8"``
            (vLLM native FP8 KV cache with online scales).

    Returns:
        Dict with ``quant_method``, ``needs_onecomp_plugin``, ``plugin``,
        ``vllm_supported``, ``kv_cache_dtype``, ``engine_kwargs``,
        ``serve_command``, ``quantization_bits_summary``, ``notes`` and
        ``warnings``.
    """
    # pylint: disable=import-outside-toplevel, too-many-branches
    import json
    import os
    import shlex

    model_dir = os.path.abspath(model_dir)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No config.json found in {model_dir}")
    if kv_cache_precision not in _KV_PRECISION_CHOICES:
        raise ValueError(
            f"kv_cache_precision must be one of {_KV_PRECISION_CHOICES}, "
            f"got {kv_cache_precision!r}"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    notes: List[str] = []
    warnings: List[str] = []
    vllm_supported = True

    quant_config = config.get("quantization_config") or {}
    quant_method = quant_config.get("quant_method")
    plugin = _PLUGIN_QUANT_METHODS.get(quant_method)
    needs_plugin = plugin is not None

    if quant_method is None:
        notes.append("No quantization_config: vLLM loads the model unquantised.")
    elif needs_plugin:
        notes.append(
            f"quant_method '{quant_method}' is served by the OneComp plugin "
            f"{plugin}; install onecomp in the vLLM environment so the plugin "
            "entry points register it."
        )
    elif quant_method in _NATIVE_QUANT_METHODS:
        notes.append(f"quant_method '{quant_method}' is supported natively by vLLM.")
    else:
        vllm_supported = False
        warnings.append(
            f"Unknown quant_method '{quant_method}': neither a vLLM native "
            "method nor an OneComp plugin; the model may not load in vLLM."
        )

    if quant_config.get("rotated"):
        vllm_supported = False
        warnings.append(
            "The model was quantised after rotation preprocessing "
            "(rotated=true). vLLM kernels do not apply the online Hadamard "
            "transform on down_proj inputs, so this model cannot be served "
            "with vLLM; use the HF path instead."
        )

    if kv_cache_precision == "fp8":
        kv_cache_dtype = "fp8"
        notes.append(
            "kv_cache_dtype='fp8' with online scale calculation (data-free, "
            "no calibration pass)."
        )
    else:
        kv_cache_dtype = "auto"
        if kv_cache_precision == "fp16":
            notes.append("kv_cache_precision 'fp16' keeps the model dtype (auto).")

    engine_kwargs: Dict[str, Any] = {"model": model_dir}
    serve_parts = ["vllm", "serve", shlex.quote(model_dir)]
    if kv_cache_dtype != "auto":
        engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
        engine_kwargs["calculate_kv_scales"] = True
        serve_parts += ["--kv-cache-dtype", kv_cache_dtype, "--calculate-kv-scales"]

    return {
        "model_dir": model_dir,
        "quant_method": quant_method,
        "needs_onecomp_plugin": needs_plugin,
        "plugin": plugin,
        "vllm_supported": vllm_supported,
        "kv_cache_dtype": kv_cache_dtype,
        "engine_kwargs": engine_kwargs,
        "serve_command": " ".join(serve_parts),
        "quantization_bits_summary": _summarise_quantization_bits(quant_config),
        "notes": notes,
        "warnings": warnings,
    }


def build_tools(
    *,
    require_session_token: bool = True,
    compaction_profile: str = "formal-v1",
) -> List[ToolDefinition]:
    """Return the :class:`ToolDefinition` list served by the MCP server."""
    return [
        ToolDefinition(
            name="estimate_memory",
            description=(
                "Estimate GPU memory (GB) needed by a model: quantized weights "
                "(including scale/zero metadata) plus KV cache. Reads only "
                "config.json; no weights are loaded. Accepts a local path or a "
                "Hugging Face model ID."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Hugging Face model ID or local path",
                    },
                    "wbits": {
                        "type": "number",
                        "description": "Quantization bit-width (omit for FP16)",
                    },
                    "context_len": {
                        "type": "integer",
                        "description": "Context length for the KV-cache estimate",
                        "default": 4096,
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size for the KV-cache estimate",
                        "default": 1,
                    },
                },
                "required": ["model_id"],
            },
            handler=estimate_memory,
        ),
        ToolDefinition(
            name="recommend_recipe",
            description=(
                "Recommend a quantization recipe (bit-width, group size, method, "
                "KV-cache precision) that fits a given VRAM budget, with a "
                "rationale string explaining the decision."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Hugging Face model ID or local path",
                    },
                    "total_vram_gb": {
                        "type": "number",
                        "description": "Total VRAM of the target GPU in GB",
                    },
                    "context_len": {
                        "type": "integer",
                        "description": "Context length for the KV-cache estimate",
                        "default": 4096,
                    },
                    "target": {
                        "type": "string",
                        "enum": ["quality", "memory"],
                        "description": "Optimization target",
                        "default": "quality",
                    },
                },
                "required": ["model_id", "total_vram_gb"],
            },
            handler=recommend_recipe,
        ),
        ToolDefinition(
            name="quantize",
            description=(
                "Quantize a model with OneComp (AutoBit + GPTQ + QEP) and save "
                "it to save_dir. Long-running: may take minutes to hours "
                "depending on model size and hardware."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Hugging Face model ID or local path",
                    },
                    "save_dir": {
                        "type": "string",
                        "description": "Directory to save the quantized model",
                    },
                    "wbits": {
                        "type": "number",
                        "description": "Target bit-width (omit to estimate from VRAM)",
                    },
                    "total_vram_gb": {
                        "type": "number",
                        "description": "VRAM budget in GB for bit-width estimation",
                    },
                    "qep": {
                        "type": "boolean",
                        "description": "Apply QEP error propagation",
                        "default": True,
                    },
                    "evaluate": {
                        "type": "boolean",
                        "description": "Evaluate perplexity/accuracy after quantization",
                        "default": False,
                    },
                },
                "required": ["model_id", "save_dir"],
            },
            handler=quantize,
        ),
        ToolDefinition(
            name="evaluate",
            description=(
                "Evaluate the WikiText-2 perplexity of a quantized model "
                "previously saved by OneComp."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "model_dir": {
                        "type": "string",
                        "description": "Directory containing the saved quantized model",
                    },
                    "num_samples": {
                        "type": "integer",
                        "description": "Number of evaluation samples",
                        "default": 16,
                    },
                },
                "required": ["model_dir"],
            },
            handler=evaluate,
        ),
        ToolDefinition(
            name="vllm_engine_args",
            description=(
                "Derive vLLM launch arguments for a model directory saved by "
                "OneComp: whether an OneComp vLLM plugin (mixed_gptq/dbf) is "
                "required, the recommended kv_cache_dtype, LLM(**kwargs) "
                "engine kwargs, and a vllm serve command line. Read-only: "
                "inspects config.json only, no weights are loaded."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "model_dir": {
                        "type": "string",
                        "description": "Directory containing the saved model",
                    },
                    "kv_cache_precision": {
                        "type": "string",
                        "enum": ["auto", "fp16", "fp8"],
                        "description": "KV-cache precision to serve with",
                        "default": "auto",
                    },
                },
                "required": ["model_dir"],
            },
            handler=vllm_engine_args,
        ),
        *build_compaction_tools(
            require_session_token=require_session_token,
            profile=compaction_profile,
        ),
    ]
