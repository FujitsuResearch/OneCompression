"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Union

import torch

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

    When ``quantizers`` is not provided, GPTQ candidates for 2, 3, 4, and 8 bit
    are generated automatically.

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

            autobit = AutoBitQuantizer(
                assignment_strategy="activation_aware",
                target_bit=3.0,
                quantizers=[GPTQ(wbits=2), GPTQ(wbits=4)],
                num_calib_samples=64,
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

    # --- activation-aware parameters ---
    num_calib_samples: int = 128
    calib_seqlen: int = 256
    use_curvature_b: bool = True

    # --- visualisation ---
    save_path: str = None

    # --- DBF fallback for ultra-low bit ---
    auto_dbf: bool = True
    dbf_threshold: float = 2.0
    dbf_iters: int = None  # None → DBF default (600); set low (e.g. 10) for fast testing

    # internal variables
    _module_to_quantizer: dict = field(default_factory=dict, repr=False, init=False)
    _name_to_quantizer: dict = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.assignment_strategy, AssignmentStrategy):
            self.assignment_strategy = AssignmentStrategy(self.assignment_strategy)
        if not self.quantizers:
            self.quantizers = self._default_quantizers()
        self._sync_flags()

    @staticmethod
    def _default_quantizers():
        """GPTQ candidates for 2–8 bit (auto-generated when ``quantizers=[]``)."""
        return [GPTQ(wbits=b) for b in (2, 3, 4, 5, 6, 7, 8)]

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

        if not isinstance(self.num_calib_samples, int) or self.num_calib_samples < 1:
            bad.append(
                f"Invalid parameter 'num_calib_samples': {self.num_calib_samples!r} "
                f"(expected int >= 1)"
            )

        if not isinstance(self.calib_seqlen, int) or self.calib_seqlen < 1:
            bad.append(
                f"Invalid parameter 'calib_seqlen': {self.calib_seqlen!r} " f"(expected int >= 1)"
            )

        for i, q in enumerate(self.quantizers):
            try:
                q.validate_params()
            except (ValueError, TypeError) as e:
                bad.append(f"quantizers[{i}] ({type(q).__name__}): {e}")

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
            if self.target_bit is not None:
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
        """
        gs = self._detect_group_size()
        candidates = _find_candidates(self, model)
        if not candidates:
            return

        raw_target = self.target_bit
        total_params = sum(m.weight.numel() for _, m in candidates)
        total_eff_bits = sum(
            effective_bits_per_param(
                wbits=raw_target,
                group_size=gs,
                in_features=m.weight.shape[1],
            )
            * m.weight.numel()
            for _, m in candidates
        )
        self.target_bit = total_eff_bits / total_params

        if self.target_bit != raw_target:
            self.logger.info(
                "target_bit adjusted: %.4f raw bpw → %.4f effective bpw "
                "(groupsize=%s, equivalent to uniform %.3g-bit baseline)",
                raw_target,
                self.target_bit,
                gs,
                raw_target,
            )

    def _detect_group_size(self) -> int:
        """Return the group_size from the first quantizer that uses grouping."""
        for q in self.quantizers:
            gs = getattr(q, "groupsize", getattr(q, "group_size", None))
            if gs is not None and gs > 0:
                return gs
        return -1

    def _sync_flags(self):
        if self.quantizers:
            self.flag_calibration = any(q.flag_calibration for q in self.quantizers)
            self.flag_hessian = any(q.flag_hessian for q in self.quantizers)
            self.flag_xtx = any(q.flag_xtx for q in self.quantizers)

    def _assign_layer(self, name, module, child_q):
        self.module_to_name[module] = name
        self._module_to_quantizer[module] = child_q
        self._name_to_quantizer[name] = child_q
        child_q.module_to_name[module] = name

    def _visualize(self, assignments):
        from .visualize import visualize_bit_assignment  # lazy: heavy matplotlib dep

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

    def get_quant_config(self) -> dict:
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
