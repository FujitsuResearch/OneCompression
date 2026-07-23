"""vLLM-native checkpoint export (compressed-tensors NVFP4 / MXFP4 / FP8)

Exports FloatQuant results as checkpoints that vLLM executes with real
low-precision kernels (no plugin, no fake-quant FP16 weights), giving
actual memory and inference-speed benefits:

- ``nvfp4``: ``compressed-tensors`` ``nvfp4-pack-quantized`` layout.
  FP4 E2M1 codes packed two per byte, FP8 E4M3 block scales (group 16),
  and one FP32 global scale per layer.  vLLM runs these with the FP4
  Marlin kernel (W4A16), storing weights at ~4.5 bits/element.  When
  per-layer activation global scales are supplied (see
  :func:`collect_input_global_scales`), the checkpoint additionally
  quantizes activations (W4A4): vLLM then executes real FP4 x FP4
  matmuls on Blackwell tensor cores, with activation block scales
  computed dynamically at runtime (``dynamic="local"``).
- ``mxfp4``: ``compressed-tensors`` ``mxfp4-pack-quantized`` layout.
  FP4 E2M1 codes packed two per byte with per-block (32) E8M0 scales
  stored as biased-exponent bytes (W4A16 Marlin).
- ``fp8``: ``compressed-tensors`` ``float-quantized`` layout with
  per-channel weight scales and dynamic per-token FP8 activation
  quantization (W8A8).  Unlike :func:`save_vllm_fp8_model`, this path
  preserves FloatQuant's per-channel scales and Hessian-compensated
  weights bit-exactly.
- mixed NVFP4 / FP8: ``compressed-tensors`` ``mixed-precision`` format
  with one config group per format (see :func:`save_vllm_mixed_model`).
  Sensitive layers keep FP8 W8A8 quality while the rest are stored as
  NVFP4, trading memory against accuracy per layer.

Functions:
    save_vllm_native_model: Export FloatQuant results as a vLLM-native
        compressed-tensors checkpoint (all formats).
    collect_input_global_scales: Calibrate per-layer activation amax for
        NVFP4 W4A4 export.
    diagnose_nvfp4_fused_export_gap: Measure the weight-domain export
        gap introduced by fused-group NVFP4 global-scale unification.
    save_vllm_fp8_model: Re-quantize a model to FP8 W8A8 with per-tensor
        scales in vLLM's plain ``fp8`` layout (legacy path).

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import json
import math
import os
import re
from logging import getLogger
from typing import Optional

import torch
from torch import nn

from onecomp.quantizer.floatquant.formats import (
    E2M1_MAX,
    E4M3_MAX,
    e8m0_scales_to_uint8,
    nvfp4_dequantize,
    nvfp4_quantize,
    pack_fp4_codes,
)

logger = getLogger(__name__)

_LAYER_LINEAR_RE = re.compile(r"\.layers\.\d+\.")

# Leaf-name groups that vLLM fuses into a single matrix.  Fused NVFP4
# layers must share one global scale (vLLM keeps only the max across
# the fused shards), so the exporter unifies them.
_FUSED_LEAF_GROUPS = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
)

# compressed-tensors serialization formats understood by vLLM
_CT_FORMATS = {
    "nvfp4": "nvfp4-pack-quantized",
    "mxfp4": "mxfp4-pack-quantized",
    "fp8": "float-quantized",
}

# Block sizes vLLM's scheme detection is hard-wired to
_CT_REQUIRED_BLOCK_SIZE = {"nvfp4": 16, "mxfp4": 32}


def _collect_target_linears(model: nn.Module) -> list:
    """Return the names of transformer-block Linear layers to quantize."""
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lm_head" in name:
            continue
        if _LAYER_LINEAR_RE.search(name) is None:
            continue
        targets.append(name)
    return targets


def _dedupe_tied_state(state_dict: dict) -> dict:
    """Drop aliases of tensors that share storage (tied weights).

    safetensors rejects shared storage; vLLM re-ties ``lm_head`` from
    ``config.tie_word_embeddings``.  When a tied pair is found, the
    embedding-side name is kept regardless of iteration order.

    Args:
        state_dict (dict): Model state dict (name -> tensor).

    Returns:
        dict: New mapping without storage aliases.
    """
    result: dict = {}
    seen_storages: dict = {}
    for key, tensor in state_dict.items():
        storage_key = (tensor.data_ptr(), tensor.shape, tensor.dtype)
        if storage_key in seen_storages:
            kept = seen_storages[storage_key]
            if kept.startswith("lm_head") and not key.startswith("lm_head"):
                result[key] = result.pop(kept)
                seen_storages[storage_key] = key
                logger.info("Renamed tied tensor %s -> %s", kept, key)
            else:
                logger.info("Skipping %s (shares storage with %s)", key, kept)
            continue
        seen_storages[storage_key] = key
        result[key] = tensor
    return result


def _fused_groups(names) -> list:
    """Group quantized layer names by vLLM fusion (qkv / gate_up).

    Args:
        names: Iterable of quantized module names.

    Returns:
        list[list[str]]: Groups (length >= 2) of module names that vLLM
        fuses into one matrix.
    """
    by_parent: dict = {}
    for name in names:
        parent, _, leaf = name.rpartition(".")
        by_parent.setdefault(parent, {})[leaf] = name
    groups = []
    for leaves in by_parent.values():
        for pattern in _FUSED_LEAF_GROUPS:
            members = [leaves[leaf] for leaf in pattern if leaf in leaves]
            if len(members) >= 2:
                groups.append(members)
    return groups


def _unify_nvfp4_global_scales(results: dict) -> dict:
    """Return per-layer NVFP4 tensors with fused-group global scales unified.

    vLLM fuses q/k/v and gate/up projections and keeps a single global
    scale for the fused matrix, so shards quantized with different
    per-tensor scales would be mis-scaled.  For every fused group this
    re-quantizes the shards whose scale differs from the group maximum,
    starting from the (Hessian-compensated) dequantized weights.

    Args:
        results (dict): Mapping of module name -> FloatQuantResult
            (fmt="nvfp4").

    Returns:
        dict: Mapping of module name ->
        (codes, block_scales, tensor_scale) tensors ready for packing.
    """
    export: dict = {}
    for name, result in results.items():
        export[name] = (result.codes, result.block_scales, result.tensor_scale)

    # The sweep-based re-quantization is heavily parallel; run it on the
    # GPU when one is available (the CPU path can take minutes per large
    # layer on a single core).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for group in _fused_groups(results.keys()):
        scales = [results[name].tensor_scale.float().reshape(()) for name in group]
        shared = torch.stack(scales).max()
        for name, scale in zip(group, scales):
            if torch.equal(scale, shared):
                continue
            weight = results[name].compute_dequantized_weight().float().to(device)
            # scale_search never increases the per-block reconstruction
            # error relative to AbsMax, so the sweep is always enabled
            # for this forced re-quantization.
            codes, block_scales, tensor_scale = nvfp4_quantize(
                weight,
                results[name].block_size,
                tensor_scale=shared.to(device),
                scale_search=True,
            )
            export[name] = (codes.cpu(), block_scales.cpu(), tensor_scale.cpu())
            logger.info(
                "Re-quantized %s with the shared fused global scale " "(%.3e -> %.3e)",
                name,
                float(scale),
                float(shared),
            )
    return export


def _sum_squared_error(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    """Return ``||lhs - rhs||^2`` as a Python float."""
    return float((lhs.float() - rhs.float()).pow(2).sum())


def _error_summary(
    squared_error: float,
    numel: int,
    reference_squared_norm: Optional[float] = None,
) -> dict:
    """Build a JSON-friendly error summary."""
    summary = {
        "squared_error": float(squared_error),
        "mean_squared_error": float(squared_error / max(numel, 1)),
    }
    if reference_squared_norm is not None:
        denom = max(float(reference_squared_norm), torch.finfo(torch.float32).tiny)
        summary["relative_squared_error"] = float(squared_error / denom)
    return summary


def _delta_summary(after: dict, before: dict) -> dict:
    """Return ``after - before`` for matching error summary keys."""
    return {
        key: float(after[key] - before[key])
        for key in ("squared_error", "mean_squared_error", "relative_squared_error")
        if key in after and key in before
    }


def _reference_weights(model: Optional[nn.Module], results: dict) -> Optional[dict]:
    """Return original model weights matching ``results`` or ``None``."""
    if model is None:
        return None

    params = dict(model.named_parameters())
    weights = {}
    missing = []
    shape_mismatch = []
    for name, result in results.items():
        key = f"{name}.weight"
        if key not in params:
            missing.append(name)
            continue
        weight = params[key].detach().float().cpu()
        expected_shape = tuple(result.compute_dequantized_weight().shape)
        if tuple(weight.shape) != expected_shape:
            shape_mismatch.append((name, tuple(weight.shape), expected_shape))
            continue
        weights[name] = weight

    if missing:
        raise ValueError(
            "diagnose_nvfp4_fused_export_gap: model is missing weights for "
            f"{missing[:3]}{'...' if len(missing) > 3 else ''}."
        )
    if shape_mismatch:
        name, actual, expected = shape_mismatch[0]
        raise ValueError(
            "diagnose_nvfp4_fused_export_gap: model weight shape mismatch for "
            f"{name}: got {actual}, expected {expected}."
        )
    return weights


def diagnose_nvfp4_fused_export_gap(
    results: dict,
    model: Optional[nn.Module] = None,
    top_k: int = 8,
) -> dict:
    """Measure NVFP4 export damage from fused global-scale unification.

    vLLM fuses q/k/v and gate/up projections and stores one global scale
    for the fused matrix.  The exporter therefore re-quantizes shards
    whose NVFP4 tensor scale differs from the fused-group maximum.  This
    helper reconstructs the same post-unification tensors in the weight
    domain and returns bounded, JSON-friendly diagnostics.

    When ``model`` is supplied, the report includes pre-export and
    post-unification reconstruction error against the original model
    weights plus their delta.  Without ``model``, it still reports the
    export-induced shift from the pre-export reconstruction.

    Args:
        results (dict): Mapping of module name -> FloatQuantResult with
            ``fmt="nvfp4"`` and ``block_size=16``.
        model (nn.Module, optional): Model carrying original weights,
            named like ``<module>.weight``.  Supplying it enables
            pre/post reconstruction error and ``delta_export`` metrics.
        top_k (int): Maximum number of layer and group detail rows to
            return. Aggregate metrics always cover all fused groups.

    Returns:
        dict: JSON-serializable aggregate metrics plus bounded
        ``worst_layers`` and ``worst_groups`` detail lists.

    Raises:
        ValueError: If ``results`` is empty, not NVFP4, not vLLM-compatible
            block-16, transposed, or if ``model`` is missing reference
            weights for any result.
    """
    if top_k < 0:
        raise ValueError("diagnose_nvfp4_fused_export_gap: top_k must be non-negative.")

    fmt, block_size = _validate_results(results)
    if fmt != "nvfp4":
        raise ValueError(
            "diagnose_nvfp4_fused_export_gap: only supports nvfp4 results, " f"got fmt={fmt!r}."
        )
    groups = _fused_groups(results.keys())
    fused_results = {name: results[name] for group in groups for name in group}
    references = _reference_weights(model, fused_results)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_rows = []
    group_rows = []

    totals = {
        "numel": 0,
        "reference_squared_norm": 0.0,
        "pre_squared_error": 0.0,
        "post_squared_error": 0.0,
        "gap_squared_error": 0.0,
        "pre_squared_norm": 0.0,
        "num_requantized_layers": 0,
    }

    for group in groups:
        scales = [results[name].tensor_scale.float().reshape(()) for name in group]
        shared = torch.stack(scales).max()
        group_totals = {
            "numel": 0,
            "reference_squared_norm": 0.0,
            "pre_squared_error": 0.0,
            "post_squared_error": 0.0,
            "gap_squared_error": 0.0,
            "pre_squared_norm": 0.0,
            "num_requantized_layers": 0,
        }

        for name, scale in zip(group, scales):
            result = results[name]
            pre = result.compute_dequantized_weight().float().cpu()
            if torch.equal(scale, shared):
                post = pre
                requantized = False
            else:
                codes, block_scales, tensor_scale = nvfp4_quantize(
                    pre.to(device),
                    result.block_size,
                    tensor_scale=shared.to(device),
                    scale_search=True,
                )
                post = nvfp4_dequantize(
                    codes.cpu(),
                    block_scales.cpu(),
                    tensor_scale.cpu(),
                    result.block_size,
                ).float()
                requantized = True

            numel = pre.numel()
            pre_norm = float(pre.pow(2).sum())
            gap_sq = _sum_squared_error(post, pre)
            layer = {
                "name": name,
                "group": list(group),
                "numel": numel,
                "tensor_scale": float(scale),
                "shared_tensor_scale": float(shared),
                "scale_ratio": float(shared / scale) if float(scale) != 0.0 else float("inf"),
                "requantized": requantized,
                "gap_to_pre_export": _error_summary(gap_sq, numel, pre_norm),
            }

            group_totals["numel"] += numel
            group_totals["gap_squared_error"] += gap_sq
            group_totals["pre_squared_norm"] += pre_norm
            if requantized:
                group_totals["num_requantized_layers"] += 1

            if references is not None:
                reference = references[name]
                reference_norm = float(reference.pow(2).sum())
                pre_sq = _sum_squared_error(pre, reference)
                post_sq = _sum_squared_error(post, reference)
                pre_export = _error_summary(pre_sq, numel, reference_norm)
                post_export = _error_summary(post_sq, numel, reference_norm)
                layer["pre_export"] = pre_export
                layer["post_export"] = post_export
                layer["delta_export"] = _delta_summary(post_export, pre_export)
                group_totals["reference_squared_norm"] += reference_norm
                group_totals["pre_squared_error"] += pre_sq
                group_totals["post_squared_error"] += post_sq

            layer_rows.append(layer)

        totals["numel"] += group_totals["numel"]
        totals["reference_squared_norm"] += group_totals["reference_squared_norm"]
        totals["pre_squared_error"] += group_totals["pre_squared_error"]
        totals["post_squared_error"] += group_totals["post_squared_error"]
        totals["gap_squared_error"] += group_totals["gap_squared_error"]
        totals["pre_squared_norm"] += group_totals["pre_squared_norm"]
        totals["num_requantized_layers"] += group_totals["num_requantized_layers"]

        group_row = {
            "members": list(group),
            "num_layers": len(group),
            "numel": group_totals["numel"],
            "shared_tensor_scale": float(shared),
            "num_requantized_layers": group_totals["num_requantized_layers"],
            "gap_to_pre_export": _error_summary(
                group_totals["gap_squared_error"],
                group_totals["numel"],
                group_totals["pre_squared_norm"],
            ),
        }
        if references is not None:
            pre_export = _error_summary(
                group_totals["pre_squared_error"],
                group_totals["numel"],
                group_totals["reference_squared_norm"],
            )
            post_export = _error_summary(
                group_totals["post_squared_error"],
                group_totals["numel"],
                group_totals["reference_squared_norm"],
            )
            group_row["pre_export"] = pre_export
            group_row["post_export"] = post_export
            group_row["delta_export"] = _delta_summary(post_export, pre_export)
        group_rows.append(group_row)

    aggregate_gap = _error_summary(
        totals["gap_squared_error"], totals["numel"], totals["pre_squared_norm"]
    )
    report = {
        "format": "nvfp4",
        "block_size": block_size,
        "has_reference": references is not None,
        "num_fused_groups": len(groups),
        "num_fused_layers": sum(len(group) for group in groups),
        "num_requantized_layers": totals["num_requantized_layers"],
        "numel": totals["numel"],
        "gap_to_pre_export": aggregate_gap,
        "pre_export": None,
        "post_export": None,
        "delta_export": None,
    }

    if references is not None:
        pre_export = _error_summary(
            totals["pre_squared_error"], totals["numel"], totals["reference_squared_norm"]
        )
        post_export = _error_summary(
            totals["post_squared_error"], totals["numel"], totals["reference_squared_norm"]
        )
        report["pre_export"] = pre_export
        report["post_export"] = post_export
        report["delta_export"] = _delta_summary(post_export, pre_export)

    score_key = lambda row: (
        row["delta_export"]["squared_error"]
        if row.get("delta_export") is not None
        else row["gap_to_pre_export"]["squared_error"]
    )
    report["worst_layers"] = sorted(layer_rows, key=score_key, reverse=True)[:top_k]
    report["worst_groups"] = sorted(group_rows, key=score_key, reverse=True)[:top_k]
    return report


def collect_input_global_scales(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    model: nn.Module,
    tokenizer,
    module_names,
    calibration_texts,
    device: str = "cuda:0",
    max_length: int = 2048,
    percentile: float = 100.0,
    scale_multiplier: float = 1.0,
) -> dict:
    """Calibrate per-layer NVFP4 activation global scales (for W4A4).

    Runs forward passes over the calibration texts with pre-forward hooks
    on the listed modules, recording a running activation magnitude
    statistic per layer, and converts it to the NVFP4 two-level
    activation scale::

        input_global_scale = (448 * 6) / amax(X)

    which matches the divisor convention (``1 / tensor_scale``) that
    compressed-tensors checkpoints use for global scales. The default
    ``percentile=100`` is AbsMax. Lower percentiles intentionally trade
    clipping for finer resolution and are useful for W4A4 activation
    scale-search ablations.

    Args:
        model (nn.Module): Model with weights loaded (any dtype).
        tokenizer: Tokenizer used to encode the calibration texts.
        module_names: Names of the Linear modules to calibrate (usually
            ``quantizer.results.keys()``).
        calibration_texts: Iterable of calibration strings.
        device (str): Device to run the forward passes on.
        max_length (int): Truncation length per text.
        percentile (float): Activation magnitude percentile in ``(0,100]``.
            ``100`` records the amax.
        scale_multiplier (float): Positive multiplier applied to the
            final divisor-convention scale for log-scale local searches.

    Returns:
        dict: Mapping of module name -> 0-dim FP32 tensor with the
        activation global scale (divisor convention, CPU).
    """
    if not (0.0 < percentile <= 100.0):
        raise ValueError("collect_input_global_scales: percentile must be in (0, 100].")
    if scale_multiplier <= 0.0:
        raise ValueError("collect_input_global_scales: scale_multiplier must be positive.")

    modules = dict(model.named_modules())
    amax: dict = {name: torch.zeros((), dtype=torch.float32) for name in module_names}
    handles = []

    def _make_hook(name):
        def _hook(_module, args):
            x = args[0]
            values = x.detach().abs().float().reshape(-1)
            if percentile == 100.0:
                current = values.amax().cpu()
            else:
                current = torch.quantile(values.cpu(), percentile / 100.0)
            if current > amax[name]:
                amax[name] = current

        return _hook

    for name in module_names:
        handles.append(modules[name].register_forward_pre_hook(_make_hook(name)))
    try:
        model.eval()
        with torch.no_grad():
            for text in calibration_texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                model(enc.input_ids.to(device))
    finally:
        for handle in handles:
            handle.remove()

    scales = {}
    for name, value in amax.items():
        safe = torch.clamp(value, min=torch.finfo(torch.float32).tiny)
        scales[name] = ((E4M3_MAX * E2M1_MAX) / safe) * float(scale_multiplier)
    return scales


def _quant_config_dict(
    fmt: str, block_size: int, ignore: list, quantize_activations: bool = False
) -> dict:
    """Build the compressed-tensors ``quantization_config`` for ``fmt``."""
    if fmt == "nvfp4":
        weights = {
            "num_bits": 4,
            "type": "float",
            "strategy": "tensor_group",
            "group_size": block_size,
            "symmetric": True,
            "dynamic": False,
        }
        input_activations = None
        if quantize_activations:
            # W4A4: block scales are computed at runtime ("local"), the
            # per-layer global scale is static (input_global_scale).
            input_activations = {
                "num_bits": 4,
                "type": "float",
                "strategy": "tensor_group",
                "group_size": block_size,
                "symmetric": True,
                "dynamic": "local",
            }
    elif fmt == "mxfp4":
        weights = {
            "num_bits": 4,
            "type": "float",
            "strategy": "group",
            "group_size": block_size,
            "symmetric": True,
            "dynamic": False,
        }
        input_activations = None
    else:  # fp8
        weights = {
            "num_bits": 8,
            "type": "float",
            "strategy": "channel",
            "symmetric": True,
            "dynamic": False,
        }
        input_activations = {
            "num_bits": 8,
            "type": "float",
            "strategy": "token",
            "symmetric": True,
            "dynamic": True,
        }

    return {
        "quant_method": "compressed-tensors",
        "format": _CT_FORMATS[fmt],
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": weights,
                "input_activations": input_activations,
            }
        },
        "ignore": ignore,
        "producer": {"name": "onecomp", "quantizer": "FloatQuant", "fmt": fmt},
    }


def _format_group_dict(fmt: str, block_size: int, targets: list, w4a4: bool = False) -> dict:
    """Build one mixed-precision config group for ``fmt``."""
    group = _quant_config_dict(fmt, block_size, [], quantize_activations=w4a4)
    return {
        "targets": targets,
        "weights": group["config_groups"]["group_0"]["weights"],
        "input_activations": group["config_groups"]["group_0"]["input_activations"],
        "format": _CT_FORMATS[fmt],
    }


def _prune_dominated_knapsack_states(states: dict[int, tuple[float, tuple[int, ...]]]) -> dict:
    """Keep only Pareto-optimal (memory, gain) states for sparse exact DP."""
    pruned: dict[int, tuple[float, tuple[int, ...]]] = {}
    best_value = float("-inf")
    for cost in sorted(states):
        value, picks = states[cost]
        if value > best_value:
            pruned[cost] = (value, picks)
            best_value = value
    return pruned


def _select_upgrade_units_exact(
    candidates: list[tuple[float, float, float, list]],
    budget_bytes: float,
    cost_scale: int = 16,
) -> set[int]:
    """Solve the FP8-upgrade budget exactly with sparse 0-1 knapsack DP.

    Costs are discretized in sixteenths of a byte by default, which is
    exact for the current NVFP4-to-FP8 extra-memory model
    ``numel * 7 / 16`` bytes. The DP stores only non-dominated
    (memory, gain) states, avoiding a dense array over multi-GB budgets.
    """
    budget_units = int(math.floor(budget_bytes * cost_scale + 1e-6))
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for idx, (_, gain, extra_bytes, _) in enumerate(candidates):
        if gain <= 0:
            continue
        cost_units = int(round(extra_bytes * cost_scale))
        if cost_units <= 0 or cost_units > budget_units:
            continue
        updates = dict(states)
        for spent, (value, picks) in states.items():
            next_spent = spent + cost_units
            if next_spent > budget_units:
                continue
            next_value = value + gain
            if next_spent not in updates or next_value > updates[next_spent][0]:
                updates[next_spent] = (next_value, picks + (idx,))
        states = _prune_dominated_knapsack_states(updates)

    best_value, best_picks = max(states.values(), key=lambda item: item[0])
    if best_value <= 0:
        return set()
    return set(best_picks)


def _select_upgrade_units_greedy(
    candidates: list[tuple[float, float, float, list]],
    budget_bytes: float,
) -> set[int]:
    """Legacy ratio-greedy FP8-upgrade heuristic."""
    selected = set()
    spent = 0.0
    for idx, (_, gain, extra_bytes, _) in sorted(
        enumerate(candidates), reverse=True, key=lambda item: item[1][0]
    ):
        if gain <= 0 or spent + extra_bytes > budget_bytes:
            continue
        selected.add(idx)
        spent += extra_bytes
    return selected


def select_mixed_formats(  # pylint: disable=too-many-locals
    model: nn.Module,
    nvfp4_results: dict,
    fp8_results: dict,
    fp8_fraction: float = 0.2,
    assignment: str = "exact",
) -> dict:
    """Choose FP8 for the most NVFP4-sensitive layers, NVFP4 elsewhere.

    For every layer the squared reconstruction error of both formats is
    measured against the current model weight, and layers are upgraded
    to FP8 under the extra-memory budget. The default ``assignment``
    uses an exact sparse 0-1 knapsack DP; ``assignment="greedy"`` keeps
    the legacy error-reduction-per-byte heuristic for ablations. Layers
    that vLLM fuses (q/k/v, gate/up) are decided as one unit so each
    fused matrix keeps a single format.

    Args:
        model (nn.Module): Model carrying the original weights.
        nvfp4_results (dict): FloatQuant results with ``fmt="nvfp4"``.
        fp8_results (dict): FloatQuant results with ``fmt="fp8"`` for
            the same layers.
        fp8_fraction (float): Fraction of the *extra* FP8 memory budget
            to spend (0 = all NVFP4, 1 = all FP8).
        assignment (str): ``"exact"`` for sparse DP or ``"greedy"`` for
            the ratio heuristic.

    Returns:
        dict: Mapping of module name -> FloatQuantResult mixing both
        formats, ready for :func:`save_vllm_mixed_model`.
    """
    if set(nvfp4_results) != set(fp8_results):
        raise ValueError("select_mixed_formats: result key sets differ.")
    if assignment not in ("exact", "greedy"):
        raise ValueError("select_mixed_formats: assignment must be 'exact' or 'greedy'.")

    weights = {
        name.removesuffix(".weight"): param
        for name, param in model.named_parameters()
        if name.removesuffix(".weight") in nvfp4_results
    }
    missing = sorted(set(nvfp4_results) - set(weights))
    if missing:
        raise ValueError(f"select_mixed_formats: model is missing weights for {missing[:3]}.")

    # Decide per fusion unit (fused layers must share one format).
    units = _fused_groups(nvfp4_results.keys())
    grouped = {name for group in units for name in group}
    units += [[name] for name in nvfp4_results if name not in grouped]

    candidates = []
    total_extra_bytes = 0.0
    for unit in units:
        gain = 0.0
        extra_bytes = 0.0
        for name in unit:
            original = weights[name].detach().float()
            deq_nv = nvfp4_results[name].compute_dequantized_weight().float()
            deq_fp8 = fp8_results[name].compute_dequantized_weight().float()
            err_nv = (deq_nv - original).pow(2).sum()
            err_fp8 = (deq_fp8 - original).pow(2).sum()
            gain += float(err_nv - err_fp8)
            # nvfp4 = 4 bits + E4M3 scale per 16 (~4.5 b/elem); fp8 = 8 b/elem.
            extra_bytes += original.numel() * (8 - 4.5) / 8
        total_extra_bytes += extra_bytes
        candidates.append((gain / max(extra_bytes, 1.0), gain, extra_bytes, unit))

    budget = fp8_fraction * total_extra_bytes
    mixed = dict(nvfp4_results)
    if assignment == "exact":
        selected = _select_upgrade_units_exact(candidates, budget)
    else:
        selected = _select_upgrade_units_greedy(candidates, budget)
    for idx in selected:
        _, _, _, unit = candidates[idx]
        for name in unit:
            mixed[name] = fp8_results[name]
    upgraded = sum(1 for name in mixed if mixed[name].fmt == "fp8")
    logger.info(
        "select_mixed_formats: %d/%d layers upgraded to fp8 " "(budget %.0f%%, assignment=%s)",
        upgraded,
        len(mixed),
        100 * fp8_fraction,
        assignment,
    )
    return mixed


def save_vllm_mixed_model(  # pylint: disable=too-many-locals
    model: nn.Module,
    results: dict,
    save_directory: str,
    tokenizer=None,
) -> str:
    """Export mixed NVFP4 / FP8 FloatQuant results for vLLM.

    Uses the compressed-tensors ``mixed-precision`` format: one config
    group per format with explicit layer-name targets, so vLLM runs
    NVFP4 layers on the FP4 Marlin kernel (W4A16) and FP8 layers with
    W8A8 (dynamic per-token activations) in the same model.

    Layers fused by vLLM (q/k/v, gate/up) must use one format per fused
    group; :func:`select_mixed_formats` produces such assignments.

    Args:
        model (nn.Module): Model carrying the original weights.
        results (dict): Mapping of module name -> FloatQuantResult with
            ``fmt`` in ``{"nvfp4", "fp8"}`` (both may appear).
        save_directory (str): Destination directory.
        tokenizer (optional): Tokenizer saved alongside the model.

    Returns:
        str: The save directory.

    Raises:
        ValueError: If ``results`` is empty, contains formats other than
            nvfp4/fp8, or a fused group mixes formats.
    """
    from safetensors.torch import save_file

    if not results:
        raise ValueError("save_vllm_mixed_model: results is empty.")
    bad = {name for name, result in results.items() if result.fmt not in ("nvfp4", "fp8")}
    if bad:
        raise ValueError(f"save_vllm_mixed_model: unsupported formats for {sorted(bad)[:3]}.")

    for group in _fused_groups(results.keys()):
        fmts = {results[name].fmt for name in group}
        if len(fmts) != 1:
            raise ValueError(
                f"save_vllm_mixed_model: fused group {group} mixes formats {sorted(fmts)}; "
                "vLLM fuses these layers into one matrix (use select_mixed_formats)."
            )

    nvfp4_results = {name: result for name, result in results.items() if result.fmt == "nvfp4"}
    if any(result.block_size != 16 for result in nvfp4_results.values()):
        raise ValueError("save_vllm_mixed_model: vLLM requires block_size=16 for nvfp4.")
    nvfp4_tensors = _unify_nvfp4_global_scales(nvfp4_results) if nvfp4_results else {}

    os.makedirs(save_directory, exist_ok=True)
    state_dict = _dedupe_tied_state(model.state_dict())
    new_state: dict = {}
    for key, tensor in state_dict.items():
        module_name = key[: -len(".weight")] if key.endswith(".weight") else None
        if module_name not in results:
            new_state[key] = tensor.contiguous()
            continue
        _emit_quantized_tensors(
            new_state, module_name, results[module_name].fmt, results[module_name], nvfp4_tensors
        )

    save_file(new_state, os.path.join(save_directory, "model.safetensors"), {"format": "pt"})

    ignore = sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name not in results
    )
    if "lm_head" not in ignore and "lm_head" not in results:
        ignore.append("lm_head")

    config_groups = {}
    fp8_targets = sorted(name for name in results if results[name].fmt == "fp8")
    if nvfp4_results:
        config_groups["group_0"] = _format_group_dict("nvfp4", 16, sorted(nvfp4_results.keys()))
    if fp8_targets:
        config_groups[f"group_{len(config_groups)}"] = _format_group_dict("fp8", 0, fp8_targets)

    config_dict = model.config.to_dict()
    config_dict["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "quantization_status": "compressed",
        "config_groups": config_groups,
        "ignore": ignore,
        "producer": {"name": "onecomp", "quantizer": "FloatQuant", "fmt": "mixed"},
    }
    with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, sort_keys=True)

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(save_directory)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_directory)

    logger.info(
        "vLLM-native mixed checkpoint saved to %s (%d nvfp4 / %d fp8 layers)",
        save_directory,
        len(nvfp4_results),
        len(fp8_targets),
    )
    return save_directory


def _validate_results(results: dict) -> tuple:
    """Validate FloatQuant results for native export.

    Args:
        results (dict): Mapping of module name -> FloatQuantResult.

    Returns:
        tuple[str, int]: The common format and block size.

    Raises:
        ValueError: If ``results`` is empty, mixes formats, uses a block
            size vLLM does not support, or contains transposed layers.
    """
    if not results:
        raise ValueError("save_vllm_native_model: results is empty; run quantization first.")

    fmts = {result.fmt for result in results.values()}
    if len(fmts) != 1:
        raise ValueError(
            f"save_vllm_native_model: mixed formats {sorted(fmts)} are not supported."
        )
    fmt = fmts.pop()
    if fmt not in _CT_FORMATS:
        raise ValueError(f"save_vllm_native_model: unsupported format {fmt!r}.")

    if any(result.weight_transposed for result in results.values()):
        raise ValueError(
            "save_vllm_native_model: transposed (Conv1D) layers are not supported "
            "by the vLLM compressed-tensors layout."
        )

    block_sizes = {result.block_size for result in results.values()}
    required = _CT_REQUIRED_BLOCK_SIZE.get(fmt)
    if required is not None and block_sizes != {required}:
        raise ValueError(
            f"save_vllm_native_model: vLLM requires block_size={required} for "
            f"{fmt}, got {sorted(block_sizes)}."
        )
    return fmt, next(iter(block_sizes))


def _emit_quantized_tensors(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    new_state: dict,
    module_name: str,
    fmt: str,
    result,
    nvfp4_tensors,
    input_global_scales=None,
):
    """Write the packed tensors of one quantized Linear into ``new_state``."""
    if fmt == "nvfp4":
        codes, block_scales, tensor_scale = nvfp4_tensors[module_name]
        new_state[f"{module_name}.weight_packed"] = pack_fp4_codes(codes)
        new_state[f"{module_name}.weight_scale"] = block_scales.to(torch.float8_e4m3fn)
        new_state[f"{module_name}.weight_global_scale"] = (1.0 / tensor_scale.float()).reshape(1)
        if input_global_scales is not None:
            new_state[f"{module_name}.input_global_scale"] = (
                input_global_scales[module_name].float().reshape(1)
            )
    elif fmt == "mxfp4":
        new_state[f"{module_name}.weight_packed"] = pack_fp4_codes(result.codes)
        new_state[f"{module_name}.weight_scale"] = e8m0_scales_to_uint8(result.block_scales)
    else:  # fp8
        scales = result.block_scales.float()
        weight = result.compute_dequantized_weight().float()
        values = torch.clamp(weight / scales, -E4M3_MAX, E4M3_MAX)
        new_state[f"{module_name}.weight"] = values.to(torch.float8_e4m3fn)
        new_state[f"{module_name}.weight_scale"] = scales


def save_vllm_native_model(
    model: nn.Module,
    results: dict,
    save_directory: str,
    tokenizer=None,
    input_global_scales: Optional[dict] = None,
) -> str:
    """Export FloatQuant results as a vLLM-native compressed-tensors checkpoint.

    The saved directory loads directly in vLLM with real low-precision
    kernels (no plugin required)::

        LLM(model=save_directory)

    Per format the exported tensors per quantized Linear layer are:

    - ``nvfp4``: ``weight_packed`` (uint8, two FP4 codes per byte),
      ``weight_scale`` (FP8 E4M3 block scales, group 16),
      ``weight_global_scale`` (FP32, stored as ``1 / tensor_scale``),
      plus ``input_global_scale`` (FP32 divisor) when
      ``input_global_scales`` is provided (W4A4).
    - ``mxfp4``: ``weight_packed`` (uint8) and ``weight_scale``
      (uint8-encoded E8M0 exponents, group 32).
    - ``fp8``: ``weight`` (float8_e4m3fn) and ``weight_scale``
      (FP32 per-channel, shape ``(out_features, 1)``).

    NVFP4 layers fused by vLLM (q/k/v and gate/up projections) must
    share one global scale; shards whose per-tensor scale differs from
    the group maximum are re-quantized from their Hessian-compensated
    dequantized weights with the shared scale.

    Args:
        model (nn.Module): Model carrying the *original* (unquantized)
            weights; quantized layers are replaced from ``results``.
        results (dict): ``quantizer.results`` mapping module names to
            FloatQuantResult (a single format across all entries).
        save_directory (str): Destination directory.
        tokenizer (optional): Tokenizer saved alongside the model.
        input_global_scales (dict, optional): Per-layer activation global
            scales from :func:`collect_input_global_scales`. Only valid
            for ``nvfp4``; when provided the checkpoint quantizes
            activations as well (W4A4, Blackwell FP4 tensor cores)
            instead of weight-only (W4A16 Marlin).

    Returns:
        str: The save directory.

    Raises:
        ValueError: If ``results`` is empty, mixes formats, uses a block
            size vLLM does not support, or contains transposed
            (``transformers.Conv1D``) layers.

    Example:
        >>> runner.run()
        >>> from onecomp.quantizer.floatquant import save_vllm_native_model
        >>> model = runner.model_config.load_model(device_map="cpu")
        >>> save_vllm_native_model(
        ...     model, runner.quantizer.results, "./qwen_nvfp4_vllm"
        ... )
    """
    from safetensors.torch import save_file

    fmt, block_size = _validate_results(results)

    if input_global_scales is not None:
        if fmt != "nvfp4":
            raise ValueError(
                "save_vllm_native_model: input_global_scales (W4A4) is only "
                f"supported for nvfp4, got fmt={fmt!r}."
            )
        missing = sorted(set(results) - set(input_global_scales))
        if missing:
            raise ValueError(
                "save_vllm_native_model: input_global_scales is missing "
                f"entries for {missing[:3]}{'...' if len(missing) > 3 else ''}."
            )

    os.makedirs(save_directory, exist_ok=True)

    nvfp4_tensors = _unify_nvfp4_global_scales(results) if fmt == "nvfp4" else None

    # vLLM's MXFP4 kernel only supports bfloat16 activations (E8M0 scales
    # exceed the float16 exponent range), so mxfp4 checkpoints are saved
    # with bfloat16 as the runtime dtype.
    runtime_dtype = torch.bfloat16 if fmt == "mxfp4" else None
    if runtime_dtype is not None:
        logger.info("mxfp4: saving with torch_dtype=bfloat16 (vLLM kernel requirement)")

    state_dict = _dedupe_tied_state(model.state_dict())
    new_state: dict = {}
    for key, tensor in state_dict.items():
        module_name = key[: -len(".weight")] if key.endswith(".weight") else None
        if module_name not in results:
            if runtime_dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(runtime_dtype)
            new_state[key] = tensor.contiguous()
            continue
        _emit_quantized_tensors(
            new_state,
            module_name,
            fmt,
            results[module_name],
            nvfp4_tensors,
            input_global_scales,
        )

    save_file(new_state, os.path.join(save_directory, "model.safetensors"), {"format": "pt"})
    logger.info(
        "Exported %d Linear layers as %s (%s)",
        len(results),
        fmt,
        _CT_FORMATS[fmt],
    )

    # Unquantized Linear layers (e.g. lm_head) must be listed in `ignore`,
    # otherwise vLLM expects packed tensors for them as well.
    ignore = sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name not in results
    )
    if "lm_head" not in ignore and "lm_head" not in results:
        ignore.append("lm_head")

    config_dict = model.config.to_dict()
    if runtime_dtype is not None:
        # transformers >= 4.56 serializes the dtype under "dtype";
        # older versions (and vLLM fallbacks) read "torch_dtype".
        config_dict["dtype"] = "bfloat16"
        config_dict["torch_dtype"] = "bfloat16"
    config_dict["quantization_config"] = _quant_config_dict(
        fmt, block_size, ignore, quantize_activations=input_global_scales is not None
    )
    with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, sort_keys=True)

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(save_directory)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_directory)

    logger.info("vLLM-native %s checkpoint saved to %s", fmt, save_directory)
    return save_directory


def save_vllm_fp8_model(
    model: nn.Module,
    save_directory: str,
    tokenizer=None,
    ignored_layers: Optional[list] = None,
) -> str:
    """Quantize Linear weights to FP8 E4M3 and save a vLLM-native checkpoint.

    Every ``nn.Linear`` inside the transformer blocks (``*.layers.N.*``,
    excluding ``lm_head``) is quantized to ``torch.float8_e4m3fn`` with one
    per-tensor scale stored as ``<layer>.weight_scale``; all other tensors
    are kept as-is.  The saved directory loads directly in vLLM::

        LLM(model=save_directory)  # no plugin required

    Note:
        This legacy path re-quantizes with *per-tensor* scales.  Prefer
        :func:`save_vllm_native_model` with FloatQuant ``fmt="fp8"``
        results, which keeps the per-channel scales (and any Hessian
        compensation) intact.

    Args:
        model (nn.Module): Model to export.  Pass either the original model
            or a fake-quant model reloaded with ``load_quantized_model`` (the
            FP8 rounding is applied on top of whatever weights are present).
        save_directory (str): Destination directory.
        tokenizer (optional): Tokenizer saved alongside the model.
        ignored_layers (list, optional): Extra layer names recorded as
            unquantized in the config. ``lm_head`` is always included.

    Returns:
        str: The save directory.

    Example:
        >>> from onecomp.quantizer.floatquant import save_vllm_fp8_model
        >>> save_vllm_fp8_model(model, "./qwen_fp8_vllm", tokenizer=tokenizer)
    """
    from safetensors.torch import save_file

    os.makedirs(save_directory, exist_ok=True)

    targets = set(_collect_target_linears(model))
    if not targets:
        raise ValueError(
            "save_vllm_fp8_model: no transformer-block nn.Linear layers found. "
            "Expected modules matching '*.layers.N.*'."
        )

    state_dict = _dedupe_tied_state(model.state_dict())
    new_state: dict = {}
    for key, tensor in state_dict.items():
        module_name = key[: -len(".weight")] if key.endswith(".weight") else None
        if module_name in targets:
            weight = tensor.to(torch.float32)
            amax = weight.abs().amax()
            scale = torch.clamp(amax / E4M3_MAX, min=torch.finfo(torch.float32).tiny)
            quantized = torch.clamp(weight / scale, -E4M3_MAX, E4M3_MAX)
            new_state[key] = quantized.to(torch.float8_e4m3fn)
            new_state[f"{module_name}.weight_scale"] = scale.to(torch.float32)
        else:
            new_state[key] = tensor.contiguous()

    save_file(new_state, os.path.join(save_directory, "model.safetensors"), {"format": "pt"})
    logger.info("Quantized %d Linear layers to FP8 E4M3 (per-tensor scales)", len(targets))

    ignored = sorted(set(["lm_head"] + list(ignored_layers or [])))
    config_dict = model.config.to_dict()
    config_dict["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "ignored_layers": ignored,
    }
    with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, sort_keys=True)

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(save_directory)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_directory)

    logger.info("vLLM-native FP8 checkpoint saved to %s", save_directory)
    return save_directory
