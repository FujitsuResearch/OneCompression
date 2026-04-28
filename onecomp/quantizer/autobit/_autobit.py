"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional, Union

import torch

from onecomp.calibration import CalibrationConfig
from onecomp.utils import (
    effective_bits_per_param,
    effective_bits_for_quantizer,
)
from onecomp.quantizer._quantizer import Quantizer, QuantizationResult
from onecomp.quantizer.dbf import DBF
from onecomp.quantizer.gptq import GPTQ
from .ilp import assign_by_ilp, _find_candidates
from .manual import assign_manually
from .dbf_fallback import inject_dbf


class AssignmentStrategy(StrEnum):
    """Layer-to-quantizer assignment strategies."""

    MANUAL = "manual"
    ILP = "ilp"
    ACTIVATION_AWARE = "activation_aware"

    @property
    def fn(self):
        """Return the assignment function for this strategy."""
        return {
            "manual": assign_manually,
            "ilp": assign_by_ilp,
            "activation_aware": lambda q, m: assign_by_ilp(q, m, use_activation=True),
        }[self]


@dataclass
class AutoBitQuantizer(Quantizer):
    """Mixed-precision quantizer that assigns each layer to a child quantizer.

    Given a ``target_bit`` budget and a list of candidate ``quantizers``,
    this class solves the layer-to-quantizer assignment via ILP
    (optionally activation-aware) or manual rules, with optional DBF
    fallback for ultra-low-bit targets.

    ``quantizers`` must be provided.  Each candidate's ``groupsize`` is
    respected by both the RTN error evaluation and the effective-bpw
    budget, so mixing group sizes across candidates is fully supported.

    To estimate ``target_bit`` from available VRAM **before** creating
    this object, use :func:`onecomp.utils.estimate_wbits_from_vram`::

        from onecomp.utils import estimate_wbits_from_vram
        result = estimate_wbits_from_vram("meta-llama/Llama-2-7b-hf",
                                          total_vram_gb=24)
        autobit = AutoBitQuantizer(target_bit=result.target_bitwidth, ...)

    Following assignment strategies are supported:

    - ILP (``"ilp"``)
    - Activation-aware ILP (``"activation_aware"``)
    - Manual assignment (``"manual"``)

    DBF fallback for ultra-low-bit targets is supported when
    ``auto_dbf=True`` and ``target_bit`` falls below ``dbf_threshold``.

    When you specify ``save_path``, the assignment will be visualized
    as a heatmap.

    Examples:

        Activation-aware with explicit target::

            from onecomp.calibration import CalibrationConfig

            autobit = AutoBitQuantizer(
                assignment_strategy="activation_aware",
                target_bit=3.0,
                quantizers=[GPTQ(wbits=2), GPTQ(wbits=4)],
                calibration_config=CalibrationConfig(
                    num_calibration_samples=64,
                    max_length=256,
                ),
            )

        Mixed bit-width and group size::

            autobit = AutoBitQuantizer(
                target_bit=3.0,
                quantizers=[
                    GPTQ(wbits=2, groupsize=32),
                    GPTQ(wbits=4, groupsize=128),
                    GPTQ(wbits=4, groupsize=32),
                ],
            )

        Mixed bit-width and group size::

            autobit = AutoBitQuantizer(
                target_bit=3.0,
                quantizers=[
                    GPTQ(wbits=2, groupsize=32),
                    GPTQ(wbits=4, groupsize=128),
                    GPTQ(wbits=4, groupsize=32),
                ],
            )

        Ultra-low-bit with DBF fallback (target_bit <= dbf_threshold)::

            autobit = AutoBitQuantizer(
                target_bit=1.5,
                dbf_iters=10,       # fast testing
            )

        Manual assignment::

            autobit = AutoBitQuantizer(
                assignment_strategy="manual",
                quantizers=[
                    GPTQ(wbits=2, include_layer_keywords=["mlp"]),
                    GPTQ(wbits=4, include_layer_keywords=["self_attn"]),
                ],
            )
    """

    # --- core ---
    quantizers: list = field(default_factory=list)
    assignment_strategy: AssignmentStrategy = AssignmentStrategy.ACTIVATION_AWARE
    ratios: list = None
    target_bit: float = None
    target_bit_is_effective: bool = False

    # --- activation-aware parameters ---
    calibration_config: CalibrationConfig = None
    use_curvature_b: bool = True

    # --- visualisation ---
    save_path: str = None

    # --- DBF fallback for ultra-low bit ---
    auto_dbf: bool = True
    dbf_threshold: float = 2.0
    dbf_iters: int = None  # None → DBF default (600); set low (e.g. 10) for fast testing

    # --- vLLM fused-layer constraints ---
    # Groups of module suffixes that vLLM fuses into a single linear
    fused_groups: list = field(
        default_factory=lambda: [
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["mlp.gate_proj", "mlp.up_proj"],
        ]
    )
    enable_fused_groups: bool = True

    # internal variables
    _module_to_quantizer: dict = field(default_factory=dict, repr=False, init=False)
    _name_to_quantizer: dict = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.assignment_strategy, AssignmentStrategy):
            self.assignment_strategy = AssignmentStrategy(self.assignment_strategy)
        if not self.quantizers:
            raise ValueError("quantizers must be provided")
        if self.calibration_config is None:
            self.calibration_config = CalibrationConfig(
                max_length=256,
                num_calibration_samples=128,
            )
        self._sync_flags()

    def validate_params(self):
        """Validate AutoBitQuantizer parameters."""
        bad = []

        if self.target_bit is not None and (
            not isinstance(self.target_bit, (int, float)) or self.target_bit <= 0
        ):
            bad.append(
                f"Invalid parameter 'target_bit': {self.target_bit!r} "
                f"(expected positive number or None)"
            )

        if not isinstance(self.quantizers, list) or len(self.quantizers) == 0:
            bad.append(
                f"Invalid parameter 'quantizers': expected non-empty list, "
                f"got {type(self.quantizers).__name__} with {len(self.quantizers) if isinstance(self.quantizers, list) else 'N/A'} items"
            )

        if not isinstance(self.dbf_threshold, (int, float)) or self.dbf_threshold <= 0:
            bad.append(
                f"Invalid parameter 'dbf_threshold': {self.dbf_threshold!r} "
                f"(expected positive number)"
            )

        if self.dbf_iters is not None and (
            not isinstance(self.dbf_iters, int) or self.dbf_iters < 1
        ):
            bad.append(
                f"Invalid parameter 'dbf_iters': {self.dbf_iters!r} "
                f"(expected int >= 1 or None)"
            )

        calib_config = self.calibration_config
        if (
            not isinstance(calib_config.num_calibration_samples, int)
            or calib_config.num_calibration_samples < 1
        ):
            bad.append(
                f"Invalid parameter 'calibration_config.num_calibration_samples': "
                f"{calib_config.num_calibration_samples!r} (expected int >= 1)"
            )

        if not isinstance(calib_config.max_length, int) or calib_config.max_length < 1:
            bad.append(
                f"Invalid parameter 'calibration_config.max_length': "
                f"{calib_config.max_length!r} (expected int >= 1)"
            )

        for i, q in enumerate(self.quantizers):
            try:
                q.validate_params()
            except (ValueError, TypeError) as e:
                bad.append(f"quantizers[{i}] ({type(q).__name__}): {e}")

        _VLLM_SUPPORTED_BITS = {2, 3, 4, 8}
        if self.enable_fused_groups:
            for i, q in enumerate(self.quantizers):
                if not isinstance(q, GPTQ):
                    bad.append(
                        f"quantizers[{i}] ({type(q).__name__}): "
                        f"enable_fused_groups=True requires all quantizers to be GPTQ"
                    )
                elif q.wbits not in _VLLM_SUPPORTED_BITS:
                    bad.append(
                        f"quantizers[{i}] (GPTQ wbits={q.wbits}): "
                        f"vLLM only supports GPTQ bit-widths {sorted(_VLLM_SUPPORTED_BITS)}"
                    )

            if self.assignment_strategy == AssignmentStrategy.MANUAL:
                bad.extend(self._validate_manual_fused_consistency())

        if bad:
            raise ValueError("; ".join(bad))

    def setup(self, model):
        self.validate_params()
        assert len(self.module_to_name) == 0
        self._module_to_quantizer = {}
        self._name_to_quantizer = {}
        self.results = {}

        is_manual = self.assignment_strategy == AssignmentStrategy.MANUAL

        if not is_manual:
            if self.target_bit is not None and not self.target_bit_is_effective:
                self._convert_raw_to_effective_target(model)

            if self.target_bit is not None and self.target_bit < 1.0:
                raise ValueError(
                    f"target_bit={self.target_bit:.2f} is below 1.0 bpw. "
                    "Even 1-bit DBF cannot fit the model within this "
                    "budget. Use a higher target_bit or a smaller model."
                )

        if not is_manual and self.auto_dbf and self._needs_dbf_only():
            min_eff = min(effective_bits_for_quantizer(q) for q in self.quantizers)
            self.logger.warning(
                "target_bit=%.2f is at/below dbf_threshold=%.2f or "
                "minimum candidate (%.2f effective bpw). "
                "Switching to DBF for all layers.",
                self.target_bit,
                self.dbf_threshold,
                min_eff,
            )
            assignments = self._assign_all_dbf(model)
        else:
            assignments = self.assignment_strategy.fn(self, model)

        # Ensure layer order matches model structure
        model_order = {name: i for i, (name, _) in enumerate(model.named_modules())}
        assignments.sort(key=lambda x: model_order[x[0]])

        if not is_manual and self.auto_dbf:
            assignments = inject_dbf(
                assignments,
                self.quantizers,
                self.dbf_threshold,
                self.logger,
                dbf_iters=self.dbf_iters,
            )
            self._sync_flags()

        for name, module, child_q in assignments:
            self._assign_layer(name, module, child_q)

        for child_q in self.quantizers:
            count = sum(1 for q in self._module_to_quantizer.values() if q is child_q)
            if count > 0:
                self.logger.info("%s: %d layers assigned", child_q.name, count)

        if self.save_path is not None:
            self._visualize(assignments)

    def _needs_dbf_only(self):
        """True when target_bit is at/below the DBF threshold or every candidate's effective bpw."""
        if self.target_bit is None:
            return False
        if self.target_bit <= self.dbf_threshold:
            return True
        return self.target_bit < min(effective_bits_for_quantizer(q) for q in self.quantizers)

    def _assign_all_dbf(self, model):
        """Skip ILP and assign all quantisable layers to DBF.

        DBF supports arbitrary fractional ``target_bits`` via the middle
        dimension formula ``k = target_bits * n * m / (n + m)``,
        ``target_bit < 1.0`` is rejected earlier in
        ``setup()``, so the value here is always ≥ 1.0.
        """
        target_bits = self.target_bit
        dbf_kwargs = {"target_bits": target_bits}
        if self.dbf_iters is not None:
            dbf_kwargs["iters"] = self.dbf_iters
        dbf_q = DBF(**dbf_kwargs)
        self.quantizers.append(dbf_q)
        self._sync_flags()

        candidates = _find_candidates(self, model)
        self.logger.info(
            "target_bit=%.2f below all candidates — " "using %s for all %d layers (ILP skipped)",
            self.target_bit,
            dbf_q.name,
            len(candidates),
        )
        return [(name, module, dbf_q) for name, module in candidates]

    def _convert_raw_to_effective_target(self, model):
        """Convert raw-bpw target to effective bpw matching uniform GPTQ baseline.

        GPTQ packs scale (FP16) + zero (wbits) per group, so uniform N-bit
        costs ``N + (16+N)/group_size`` effective bpw.  Align the ILP budget
        to that so ``target_bit=3`` uses the same bits as uniform GPTQ 3-bit.

        When candidates use different group sizes the effective overhead is
        averaged across all quantizers (weighted by candidate count) so
        the budget reflects the actual distribution of group sizes.
        """
        candidates = _find_candidates(self, model)
        if not candidates:
            return

        group_sizes = [
            getattr(q, "groupsize", getattr(q, "group_size", -1)) or -1 for q in self.quantizers
        ]

        raw_target = self.target_bit
        total_params = sum(m.weight.numel() for _, m in candidates)
        total_eff_bits = 0.0
        for _, m in candidates:
            eff_per_q = [
                effective_bits_per_param(
                    wbits=raw_target,
                    group_size=gs,
                    in_features=m.weight.shape[1],
                )
                for gs in group_sizes
            ]
            total_eff_bits += sum(eff_per_q) / len(eff_per_q) * m.weight.numel()
        self.target_bit = total_eff_bits / total_params

        if self.target_bit != raw_target:
            self.logger.info(
                "target_bit adjusted: %.4f raw bpw → %.4f effective bpw "
                "(groupsizes=%s, equivalent to uniform %.3g-bit baseline)",
                raw_target,
                self.target_bit,
                group_sizes,
                raw_target,
            )

    def _sync_flags(self):
        if self.quantizers:
            self.flag_calibration = any(q.flag_calibration for q in self.quantizers)
            self.flag_hessian = any(q.flag_hessian for q in self.quantizers)
            self.flag_xtx = any(q.flag_xtx for q in self.quantizers)

    def _validate_manual_fused_consistency(self):
        """Check that manual keyword rules don't split fused groups."""
        bad = []
        for group in self.fused_groups:
            bits_for_suffix = {}
            for suffix in group:
                for child_q in self.quantizers:
                    if self._suffix_matches_quantizer(suffix, child_q):
                        bits_for_suffix[suffix] = getattr(
                            child_q, "wbits", getattr(child_q, "bits", None)
                        )
                        break
            unique_bits = set(bits_for_suffix.values())
            if len(unique_bits) > 1:
                detail = ", ".join(f"{s}={b}bit" for s, b in bits_for_suffix.items())
                bad.append(
                    f"Fused group {group}: mixed bit-widths ({detail}). "
                    f"vLLM requires identical bit-widths within each fused group"
                )
        return bad

    @staticmethod
    def _suffix_matches_quantizer(suffix, quantizer):
        """Return True if quantizer's keyword/name rules would match suffix."""
        include_names = getattr(quantizer, "include_layer_names", None)
        include_kw = getattr(quantizer, "include_layer_keywords", None)
        exclude_kw = getattr(quantizer, "exclude_layer_keywords", None)

        if include_names is not None:
            if not any(suffix in name for name in include_names):
                return False
        elif include_kw is not None:
            if not any(kw in suffix for kw in include_kw):
                return False

        if exclude_kw is not None:
            if any(kw in suffix for kw in exclude_kw):
                return False

        return True

    def _assign_layer(self, name, module, child_q):
        self.module_to_name[module] = name
        self._module_to_quantizer[module] = child_q
        self._name_to_quantizer[name] = child_q
        child_q.module_to_name[module] = name

    def _visualize(self, assignments):
        try:
            from .visualize import visualize_bit_assignment
        except ImportError:
            self.logger.warning(
                "matplotlib is not installed; skipping visualization. "
                "Install with: pip install onecomp[visualize]"
            )
            return

        layer_names = [name for name, _, _ in assignments]
        layer_bits = [
            float(effective_bits_for_quantizer(cq, in_features=mod.weight.shape[1]))
            for _, mod, cq in assignments
        ]
        layer_params = [module.weight.numel() for _, module, _ in assignments]
        layer_qnames = [cq.name for _, _, cq in assignments]

        total_p = sum(layer_params)
        weighted_avg = (
            sum(b * p for b, p in zip(layer_bits, layer_params)) / total_p if total_p > 0 else 0
        )

        title_parts = [
            f"strategy={self.assignment_strategy.value}",
            f"avg={weighted_avg:.2f} bpw",
        ]
        if self.target_bit is not None:
            title_parts.append(f"target={self.target_bit:.2f}")

        fig_path = visualize_bit_assignment(
            layer_names,
            layer_bits,
            layer_qnames,
            layer_params=layer_params,
            save_path=self.save_path,
            title=f"AutoBit Assignment ({', '.join(title_parts)})",
        )

        if fig_path:
            self.logger.info("Saved assignment heatmap: %s", fig_path)

    def quantize(
        self, module, input, output
    ):  # pylint: disable=redefined-builtin, unused-argument
        child_q = self._module_to_quantizer[module]
        child_q.quantize(module, input, output)
        name = self.module_to_name[module]
        self.results[name] = child_q.results[name]

    def quantize_with_qep(
        self,
        module,
        quant_input_activation,
        original_input_activation=None,
        percdamp=0.01,
        perccorr=0.5,
        hessian=None,
        delta_hatX=None,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        child_q = self._module_to_quantizer[module]
        child_q.quantize_with_qep(
            module,
            quant_input_activation,
            original_input_activation=original_input_activation,
            percdamp=percdamp,
            perccorr=perccorr,
            hessian=hessian,
            delta_hatX=delta_hatX,
        )
        name = self.module_to_name[module]
        self.results[name] = child_q.results[name]

    def quantize_layer(
        self,
        module,
        input=None,
        hessian=None,
    ) -> Union[torch.Tensor, QuantizationResult]:  # pylint: disable=redefined-builtin
        child_q = self._module_to_quantizer.get(module)
        if child_q is None:
            raise RuntimeError(
                "Module is not assigned to any child quantizer. " "Ensure setup() has been called."
            )
        return child_q.quantize_layer(module, input, hessian)

    def execute_post_processing(self):
        for child_q in self.quantizers:
            child_q.execute_post_processing()
        self.module_to_name = {}
        self._module_to_quantizer = {}

    def _all_children_gptq(self) -> bool:
        if not self._name_to_quantizer:
            return False
        return all(isinstance(q, GPTQ) for q in self._name_to_quantizer.values())

    def get_quant_config(self) -> dict:
        if self._all_children_gptq():
            return self._get_mixed_gptq_config()

        child_configs = []
        for child_q in self.quantizers:
            try:
                config = child_q.get_quant_config()
            except NotImplementedError:
                config = {"quant_method": child_q.name}
            config["layers"] = [n for n, q in self._name_to_quantizer.items() if q is child_q]
            child_configs.append(config)

        return {
            "quant_method": "autobit",
            "assignment_strategy": self.assignment_strategy.value,
            "target_bit": self.target_bit,
            "quantizers": child_configs,
        }

    def _get_mixed_gptq_config(self) -> dict:
        unique_quantizers = list({id(q): q for q in self._name_to_quantizer.values()}.values())
        dominant_q: GPTQ = max(
            unique_quantizers,
            key=lambda q: sum(1 for v in self._name_to_quantizer.values() if v is q),
        )
        result: dict = {
            "quant_method": "mixed_gptq",
            "bits": dominant_q.wbits,
            "groupsize": dominant_q.groupsize,
            "actorder": dominant_q.actorder,
            "group_size": dominant_q.groupsize,
            "desc_act": dominant_q.actorder,
            "sym": dominant_q.sym,
            "checkpoint_format": "gptq",
        }
        if dominant_q.mlp_wbits is not None:
            result["mlp_wbits"] = dominant_q.mlp_wbits
        if dominant_q.mlp_groupsize is not None:
            result["mlp_groupsize"] = dominant_q.mlp_groupsize
        if dominant_q.module_wbits:
            result["module_wbits"] = dict(dominant_q.module_wbits)
        return result

    def _build_quantization_bits(
        self,
        num_layers: int,
    ) -> list[dict[str, Any]]:
        _LAYER_RE = re.compile(r"\.layers\.(\d+)\.(.*)")

        skipped: list[str] = []
        layer_modules: dict[int, dict[str, Any]] = {}
        for name, child_q in self._name_to_quantizer.items():
            m = _LAYER_RE.search(name)
            if m is None:
                skipped.append(name)
                continue
            layer_idx = int(m.group(1))
            suffix = m.group(2)
            layer_modules.setdefault(layer_idx, {})[suffix] = {
                "bits": child_q.wbits,
                "method": "gptq",
                "params": {"group_size": child_q.groupsize},
            }

        if skipped:
            self.logger.debug(
                "Skipped %d module(s) not matching layer pattern: %s",
                len(skipped),
                skipped,
            )

        if not layer_modules:
            return []

        return [layer_modules.get(i, {}) for i in range(num_layers)]

    def finalize_quant_config_for_save(
        self,
        quant_config: dict[str, Any],
        quantized_layer_names: list[str],
        num_hidden_layers: Optional[int] = None,
    ) -> dict[str, Any]:
        if quant_config.get("quant_method") != "mixed_gptq":
            return quant_config

        if num_hidden_layers is None:
            raise ValueError(
                "num_hidden_layers is required for mixed_gptq quantization_bits "
                "(Runner passes model.config.num_hidden_layers)"
            )

        quant_config["quantization_bits"] = self._build_quantization_bits(
            num_layers=num_hidden_layers,
        )

        # TODO : DBF fallback is not supported yet

        return quant_config

    def create_inference_layer(self, result, linear_module, **kwargs):
        for name, stored_result in self.results.items():
            if stored_result is result:
                child_q = self._name_to_quantizer.get(name)
                if child_q is not None:
                    return child_q.create_inference_layer(
                        result,
                        linear_module,
                        **kwargs,
                    )
        raise RuntimeError(
            "Could not determine which child quantizer produced this result. "
            "Ensure setup() and quantization have been completed."
        )
