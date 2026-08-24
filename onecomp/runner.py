"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

# pylint: disable=too-many-arguments, too-many-positional-arguments
import copy
import gc
import json
import math
import os
import time
from logging import getLogger
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .__version__ import __version__
from .calibration import CalibrationConfig, prepare_calibration_dataset
from .log import setup_logger
from .lpcd import LPCDConfig
from .model_config import ModelConfig
from .qep import QEPConfig
from .quantizer import GPTQ, Quantizer
from .quantizer.autobit import AssignmentStrategy, AutoBitQuantizer
from .quantizer.autobit.dbf_fallback import MPS_DBF_FALLBACK_ERROR
from .utils import calculate_accuracy as calc_accuracy
from .utils import calculate_perplexity as calc_perplexity
from .utils import empty_cache
from .utils.device import is_mps_device
from .utils.lora import LORA_ADAPTER_SUBDIR
from .utils.quant_config import get_quant_param, validate_quantized_model_config
from .utils.quantization_progress import QuantizationProgressTracker


def _get_num_experts(config):
    """Return the number of MoE experts declared in a model config.

    Different architectures use different attribute names:
    ``num_experts`` (Qwen-MoE, Gemma4) or ``num_local_experts``
    (GPT-OSS, Mixtral).  VLMs nest them under ``text_config``.
    Returns 0 for non-MoE models.
    """
    candidates = [config, getattr(config, "text_config", None)]
    for cfg in candidates:
        if cfg is None:
            continue
        for attr in ("num_experts", "num_local_experts"):
            value = getattr(cfg, attr, None)
            if value:
                return value
    return 0


class Runner:
    """Runner class for model quantization

    Runner class for executing quantization.
    Supports quantization using calibration data and parallel quantization on multiple GPUs.

    Examples:
        Single GPU quantization (default):

        >>> from onecomp import Runner, ModelConfig
        >>> from onecomp.quantizer.gptq import GPTQ
        >>> model_config = ModelConfig(model_id_or_path="meta-llama/Llama-2-7b-hf")
        >>> quantizer = GPTQ(wbits=4, groupsize=128)
        >>> runner = Runner(
        ...     model_config=model_config,
        ...     quantizer=quantizer,
        ... )
        >>> runner.run()

        Multi-GPU quantization (layer-wise parallel):

        >>> from onecomp.quantizer.jointq import JointQ
        >>> quantizer = JointQ(bits=4, group_size=128)
        >>> # Use all available GPUs
        >>> runner = Runner(
        ...     model_config=model_config,
        ...     quantizer=quantizer,
        ...     multi_gpu=True,
        ... )
        >>> runner.run()

        >>> # Use specific GPUs (e.g., GPU 0, 2, 3)
        >>> runner = Runner(
        ...     model_config=model_config,
        ...     quantizer=quantizer,
        ...     multi_gpu=True,
        ...     gpu_ids=[0, 2, 3],
        ... )
        >>> runner.run()

    """

    def __init__(
        self,
        model_config=None,
        quantizer=None,
        quantizers=None,
        calibration_config=None,
        qep=False,
        qep_config=None,
        lpcd=False,
        lpcd_config=None,
        multi_gpu=False,
        gpu_ids=None,
        post_processes=None,
        report_progress=True,
        moe_quant_experts=False,
    ):
        """__init__ method

        Args:
            model_config (ModelConfig):
                Model configuration.  Required.
            quantizer (Quantizer):
                The quantizer to use. Specify either ``quantizer`` or
                ``quantizers``, not both.  At least one must be given for
                ``run()``. ``None`` is only supported when assigning
                ``runner.quantized_model`` directly and calling
                ``run_post_processes()``.
            quantizers (list[Quantizer]):
                Specify multiple quantizers. When used with
                ``calibration_config.batch_size``, the X^T X accumulation
                is shared, reducing the forward pass to a single execution.
                Specify either ``quantizer`` or ``quantizers``, not both.
                Currently, this is only available when
                ``calibration_config.batch_size`` is set and ``qep=False``.
            calibration_config (CalibrationConfig or None):
                Calibration data configuration.  When ``None`` (default),
                a :class:`CalibrationConfig` with default values is
                created automatically.

                See :class:`CalibrationConfig` for available fields.
            qep (bool):
                Whether to use QEP.
            qep_config (QEPConfig or None):
                Configuration for QEP. If None and ``qep=True``,
                a default ``QEPConfig()`` is used.
            lpcd (bool):
                Whether to use LPCD.
            lpcd_config (LPCDConfig or None):
                Configuration for LPCD. If None and ``lpcd=True``,
                a default ``LPCDConfig()`` is used.
            multi_gpu (bool):
                Whether to use multi-GPU for layer-wise parallel quantization.
                Default is False.
            gpu_ids (list[int]):
                List of GPU IDs to use for multi-GPU quantization.
                If None and multi_gpu is True, all available GPUs will be used.
            post_processes (list[PostQuantizationProcess] or None):
                Optional list of post-quantization processes to execute
                after the main quantization step.  Each process receives
                a packed quantized model on CPU (built via
                ``create_quantized_model(pack_weights=True, use_gemlite=False)``)
                and may modify it in-place.  Processes preserve the
                incoming pack state, so the final ``self.quantized_model``
                remains packed in the production path.  Processes are
                executed in order.  Default is None.
            report_progress (bool):
                When ``True`` (default), emit ``[progress]`` log lines with
                completed steps, elapsed time, and a linear ETA estimate
                during long quantization (calibration, chunked, multi-GPU,
                QEP).  Set to ``False`` for quiet runs (e.g. CI).
            moe_quant_experts (bool):
                When ``True``, MoE experts are kept as per-expert GPTQ INT4
                tensors (``...experts.{i}.{gate,up,down}_proj.{qweight,...}``)
                and left in the quantization config so a 4-bit MoE vLLM path
                (e.g. the mixed_gptq plugin's gpt-oss WNA16 method) can serve
                them.  When ``False`` (default), experts are dequantized and
                fused into dense tensors for ``UnquantizedFusedMoEMethod``.
                Only enable this for architectures that have a quantized MoE
                serving path in vLLM (currently gpt-oss).

        Note:
            For zero-config quantization (VRAM auto-estimation +
            AutoBitQuantizer + QEP), use the class method
            :meth:`auto_run` instead.

        Examples:

            Chunked calibration with GPTQ (large-scale calibration data):

            >>> from onecomp import Runner, ModelConfig, CalibrationConfig
            >>> from onecomp.quantizer.gptq import GPTQ
            >>> model_config = ModelConfig(
            ...     model_id_or_path="meta-llama/Llama-2-7b-hf"
            ... )
            >>> quantizer = GPTQ(wbits=4, groupsize=128)
            >>> calib_config = CalibrationConfig(
            ...     max_length=2048,
            ...     num_calibration_samples=1024,
            ...     batch_size=128,
            ... )
            >>> runner = Runner(
            ...     model_config=model_config,
            ...     quantizer=quantizer,
            ...     calibration_config=calib_config,
            ... )
            >>> runner.run()

            With custom num_layers_per_group:

            >>> calib_config = CalibrationConfig(
            ...     max_length=2048,
            ...     num_calibration_samples=1024,
            ...     batch_size=128,
            ...     num_layers_per_group=14,
            ... )
            >>> runner = Runner(
            ...     model_config=model_config,
            ...     quantizer=quantizer,
            ...     calibration_config=calib_config,
            ... )
            >>> runner.run()

            Multiple quantizers (benchmark comparison):

            >>> from onecomp.quantizer.gptq import GPTQ
            >>> from onecomp.quantizer.jointq import JointQ
            >>> gptq = GPTQ(wbits=4, groupsize=128, calc_quant_error=True)
            >>> jointq = JointQ(bits=4, group_size=128, calc_quant_error=True,
            ...                 device=torch.device(0))
            >>> calib_config = CalibrationConfig(
            ...     max_length=2048,
            ...     num_calibration_samples=1024,
            ...     batch_size=128,
            ... )
            >>> runner = Runner(
            ...     model_config=model_config,
            ...     quantizers=[gptq, jointq],
            ...     calibration_config=calib_config,
            ... )
            >>> runner.run()
            >>> # Results are stored in gptq.results and jointq.results respectively
        """

        self.model_config = model_config
        self.logger = getLogger(__name__)

        self.quantizer = quantizer
        self.quantizers = quantizers

        if calibration_config is None:
            calibration_config = CalibrationConfig()
        self.calibration_config = calibration_config

        self.qep = qep
        self.multi_gpu = multi_gpu
        self.gpu_ids = gpu_ids
        self.post_processes = post_processes or []
        self.moe_quant_experts = moe_quant_experts
        self.quantized_model = None
        self.qep_config = None
        if qep:
            self.qep_config = qep_config if qep_config is not None else QEPConfig()
        self.lpcd_config = None
        if lpcd:
            self.lpcd_config = lpcd_config if lpcd_config is not None else LPCDConfig()
        self.report_progress = report_progress

    def check(self):
        """Check the settings

        Performs the following checks:

        1. ``model_config`` is a ``ModelConfig`` instance
        2. Mutual exclusion check for ``quantizer`` and ``quantizers`` (cannot specify both)
        3. Type check for ``quantizer`` / ``quantizers`` (must be ``Quantizer`` instances)
        4. At least one of them must be specified
        5. Parameter combination consistency check (see table below)
        6. When ``multi_gpu=True``, ``quantizer.flag_calibration=True`` must hold

        Valid parameter combinations:

        ===========  ====  ==========  ================================
        quantizers   qep   multi_gpu   calibration_config.batch_size
        ===========  ====  ==========  ================================
        Specified    False False       Specified
        None         True  False       None
        None         False True        None
        None         False False       Specified
        None         False False       None
        ===========  ====  ==========  ================================

        Note:
            ``multi_gpu=True`` requires a quantizer with ``flag_calibration=True``.

            This method is intended to be called from the ``run()`` flow only.
            It is *not* designed to be used in the
            ``load_quantized_model() -> Runner.run_post_processes()`` flow, and
            the checks here do not cover that use case.

        Raises:
            TypeError: Invalid type for ``model_config``, ``quantizer``, or ``quantizers``
            ValueError: Invalid parameter combination
        """

        if not isinstance(self.model_config, ModelConfig):
            raise TypeError("`model_config` is not a `ModelConfig` object")

        # Type check for quantizer / quantizers
        if self.quantizer is not None and self.quantizers is not None:
            raise ValueError(
                "Cannot specify both 'quantizer' and 'quantizers'. Use one or the other."
            )

        if self.quantizers is not None:
            for i, q in enumerate(self.quantizers):
                if not isinstance(q, Quantizer):
                    raise TypeError(f"`quantizers[{i}]` is not a `Quantizer` object")
        elif self.quantizer is not None:
            if not isinstance(self.quantizer, Quantizer):
                raise TypeError("`quantizer` is not a `Quantizer` object")
        else:
            raise ValueError("Either 'quantizer' or 'quantizers' must be specified.")

        # Parameter combination check
        batch_size = self.calibration_config.batch_size
        if self.quantizers is not None:
            # quantizers mode: qep=False, multi_gpu=False, batch_size required
            if self.qep:
                raise ValueError("'quantizers' cannot be used with qep=True.")
            if self.multi_gpu:
                raise ValueError("'quantizers' cannot be used with multi_gpu=True.")
            if batch_size is None:
                raise ValueError(
                    "'quantizers' requires 'calibration_config.batch_size' to be set."
                )
        else:
            # Single quantizer mode: combination check
            if self.qep and self.multi_gpu:
                raise ValueError("'qep' and 'multi_gpu' cannot be used together.")
            if self.qep and batch_size is not None:
                raise ValueError("'qep' cannot be used with 'calibration_config.batch_size'.")
            if self.multi_gpu and batch_size is not None:
                raise ValueError(
                    "'multi_gpu' cannot be used with 'calibration_config.batch_size'."
                )
            if self.multi_gpu and not self.quantizer.flag_calibration:
                raise ValueError("'multi_gpu' requires a quantizer with flag_calibration=True.")
            if self.qep and not self.quantizer.flag_qep_supported:
                raise ValueError(
                    f"Quantizer '{type(self.quantizer).__name__}' "
                    f"(or one of its candidate quantizers) does not support "
                    f"QEP (Quantization Error Propagation). "
                    f"Set qep=False, or use a QEP-compatible quantizer "
                    f"(e.g., GPTQ, DBF, AutoBitQuantizer with "
                    f"QEP-compatible candidates)."
                )

        # Cross-validate calibration_dataset when AutoBitQuantizer is used
        quantizer = self.quantizer or (self.quantizers[0] if self.quantizers else None)
        if isinstance(quantizer, AutoBitQuantizer) and quantizer.calibration_config is not None:
            runner_ds = self.calibration_config.calibration_dataset
            quantizer_ds = quantizer.calibration_config.calibration_dataset
            if runner_ds != quantizer_ds:
                raise ValueError(
                    f"Calibration dataset mismatch: Runner uses "
                    f"{runner_ds!r} but quantizer uses {quantizer_ds!r}. "
                    f"Set the same calibration_dataset in both "
                    f"CalibrationConfig objects."
                )

        # MPS device validation: only GPTQ (or AutoBitQuantizer whose
        # candidates are all GPTQ, without DBF fallback) is supported on MPS
        device = self.model_config.get_device()
        if is_mps_device(device):
            if self.multi_gpu:
                raise ValueError("multi_gpu is not supported on MPS device.")
            all_quantizers = self.quantizers if self.quantizers is not None else [self.quantizer]
            for i, q in enumerate(all_quantizers):
                label = f"quantizers[{i}]" if self.quantizers else "quantizer"
                if isinstance(q, AutoBitQuantizer):
                    for j, cand in enumerate(q.quantizers):
                        if not isinstance(cand, GPTQ):
                            raise ValueError(
                                f"{label}.quantizers[{j}] ({type(cand).__name__}) "
                                f"is not supported on MPS device. "
                                "AutoBitQuantizer on MPS requires all candidate "
                                "quantizers to be GPTQ."
                            )
                    if (
                        q.auto_dbf
                        and q.assignment_strategy != AssignmentStrategy.MANUAL
                        and q._needs_dbf_only()
                    ):
                        raise ValueError(MPS_DBF_FALLBACK_ERROR)
                elif not isinstance(q, GPTQ):
                    raise ValueError(
                        f"{label} ({type(q).__name__}) is not supported on MPS device. "
                        "Only GPTQ quantization supports MPS."
                    )

    def _exclude_moe_router_if_needed(self):
        """Exclude MoE router and shared-expert-gate layers from quantization.

        vLLM's GateLinear (used for MoE routing) hardcodes
        quant_config=None, so router weights must stay unquantized.
        Qwen3.6-A3B-style MoE models route through shared_expert_gate
        the same way, so it is excluded alongside the router.
        """
        config = self.model_config.load_config()
        num_experts = _get_num_experts(config)
        if num_experts == 0:
            return

        keywords = ["router", "shared_expert_gate"]
        target_quantizers = self.quantizers if self.quantizers is not None else [self.quantizer]
        for q in target_quantizers:
            existing = list(q.exclude_layer_keywords) if q.exclude_layer_keywords else []
            missing = [k for k in keywords if k not in existing]
            if missing:
                q.exclude_layer_keywords = existing + missing

        self.logger.info(
            "MoE model (num_experts=%d): excluding '%s' layers from "
            "quantization (vLLM GateLinear does not support quantization)",
            num_experts,
            ", ".join(keywords),
        )

    def run(self):
        """Execute quantization (and related) processing"""

        start_time = time.time()

        logger = self.logger
        logger.info("OneComp version: %s", __version__)
        logger.info("Model: %s", self.model_config.get_model_id_or_path())
        logger.info("Start the run method of Runner class")

        logger.info("Checking the settings...")
        self.check()
        self._exclude_moe_router_if_needed()

        if self.lpcd_config is not None:
            logger.info("Start quantization with LPCD")
            self.quantize_with_lpcd()
        elif self.qep:
            logger.info("Start quantization with error propagation (QEP)")
            self.quantize_with_qep()
        else:
            logger.info("Start quantization")
            self.quantize()

        if self.post_processes:
            self.run_post_processes()

        elapsed_time = time.time() - start_time
        logger.info(
            "Finished the run method of Runner class (elapsed time: %.2f seconds)",
            elapsed_time,
        )

        # Calculate total and average from per-layer quantization times and log them
        target_quantizers = self.quantizers if self.quantizers is not None else [self.quantizer]
        for q in target_quantizers:
            quant_times = [
                result.quantization_time
                for result in q.results.values()
                if result.quantization_time is not None
            ]
            if quant_times:
                total_quant_time = sum(quant_times)
                avg_quant_time = total_quant_time / len(quant_times)
                logger.info(
                    "[%s] Quantization time: total=%.2f seconds, "
                    "average=%.2f seconds/layer (%d layers)",
                    q.name,
                    total_quant_time,
                    avg_quant_time,
                    len(quant_times),
                )

    @classmethod
    def auto_run(
        cls,
        model_id: str,
        wbits: Optional[float] = None,
        total_vram_gb: Optional[float] = None,
        groupsize: int = 128,
        device: str = "cuda:0",
        qep: bool = True,
        evaluate: bool = True,
        eval_original_model: bool = False,
        save_dir: str = "auto",
        **kwargs,
    ):
        """One-liner quantization with sensible defaults.

        Sets up ModelConfig, AutoBitQuantizer (ILP-based mixed-precision),
        and QEP, then runs quantization.  When ``wbits`` is ``None``,
        the target bitwidth is estimated automatically from available VRAM.
        Optionally evaluates perplexity and accuracy, and saves the
        quantized model.

        Args:
            model_id (str): Hugging Face model ID or local path.
            wbits (float or None): Target quantization bitwidth.
                When ``None`` (default), estimated from VRAM via
                ``estimate_wbits_from_vram``.
            total_vram_gb (float or None): Total VRAM budget in GB for
                bitwidth estimation.  Only used when ``wbits`` is ``None``.
                When ``None``, the installed GPU VRAM is detected
                automatically.
            groupsize (int): GPTQ group size (default: 128).
                Use -1 to disable grouping.
            device (str): Device to place the model on (default: "cuda:0").
            qep (bool): Whether to use QEP (default: True).
            evaluate (bool): Whether to calculate perplexity and
                accuracy after quantization (default: True).
            eval_original_model (bool): Whether to also evaluate the
                original (unquantized) model (default: False).
            save_dir (str or None): Directory to save the quantized model.
                ``"auto"`` (default) derives the path from model_id
                (e.g., ``"TinyLlama-1.1B-...-autobit-3.5bit"``).
                Set to ``None`` to skip saving.
            **kwargs: Additional keyword arguments forwarded to the
                ``GPTQ`` constructor (e.g., ``actorder``, ``sym``).

        Returns:
            Runner: The configured Runner instance (with quantization
            results accessible via ``runner.quantizer.results``).

        Examples:
            Minimal usage (QEP + GPTQ 4-bit, groupsize=128, auto-save):

            >>> from onecomp import Runner
            >>> runner = Runner.auto_run(
            ...     model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
            ... )

            Custom save directory:

            >>> runner = Runner.auto_run(
            ...     model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            ...     save_dir="./my_quantized_model",
            ... )

            Skip saving:

            >>> runner = Runner.auto_run(
            ...     model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            ...     save_dir=None,
            ... )

            Evaluate both original and quantized models:

            >>> runner = Runner.auto_run(
            ...     model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            ...     eval_original_model=True,
            ... )
        """
        setup_logger()
        logger = getLogger(__name__)

        candidate_bits = (2, 3, 4, 8)

        if wbits is None:
            from .utils import estimate_wbits_from_vram

            result = estimate_wbits_from_vram(
                model_id,
                total_vram_gb=total_vram_gb,
                group_size=groupsize,
                logger=logger,
            )
            wbits = math.floor(result.target_bitwidth * 100) / 100
            logger.info(
                "VRAM estimation → target wbits=%.2f (%.2f GB total, ratio=80%%)",
                wbits,
                result.total_vram_gb,
            )

        _id_lower = model_id.lower()
        is_gemma4 = any(key in _id_lower for key in ("gemma-4", "gemma4", "gemma_4"))
        model_config = ModelConfig(model_id=model_id, device=device)

        if is_gemma4:
            valid_wbits = [b for b in candidate_bits if b <= wbits]
            if not valid_wbits:
                raise ValueError(
                    f"target wbits={wbits:.2f} is below all candidate "
                    f"bit-widths {candidate_bits}; cannot select a "
                    f"uniform GPTQ configuration for Gemma 4"
                )
            uniform_bit = max(valid_wbits)
            if save_dir == "auto":
                model_name = model_id.rstrip("/").split("/")[-1]
                save_dir = f"{model_name}-gptq-{uniform_bit}bit"
            logger.warning(
                "Gemma 4 detected → falling back to uniform GPTQ %d-bit " "(target wbits=%.2f)",
                uniform_bit,
                wbits,
            )
            quantizer = GPTQ(wbits=uniform_bit, groupsize=groupsize, **kwargs)
        else:
            if save_dir == "auto":
                model_name = model_id.rstrip("/").split("/")[-1]
                save_dir = f"{model_name}-autobit-{wbits}bit"

            from .quantizer.autobit import AutoBitQuantizer

            candidate_quantizers = [
                GPTQ(wbits=b, groupsize=groupsize, **kwargs) for b in candidate_bits
            ]
            quantizer = AutoBitQuantizer(
                assignment_strategy="activation_aware",
                quantizers=candidate_quantizers,
                target_bit=wbits,
                save_path=save_dir if save_dir is not None else None,
                enable_fused_groups=True,
            )
        qep_config = QEPConfig(device=device)
        runner = cls(
            model_config=model_config, quantizer=quantizer, qep=qep, qep_config=qep_config
        )
        runner.run()

        if evaluate:
            original_ppl, _, quantized_ppl = runner.calculate_perplexity(
                original_model=eval_original_model,
            )
            if eval_original_model:
                logger.info("Original model perplexity: %s", original_ppl)
            logger.info("Quantized model perplexity: %s", quantized_ppl)

            original_acc, _, quantized_acc = runner.calculate_accuracy(
                original_model=eval_original_model,
            )
            if eval_original_model:
                logger.info("Original model accuracy: %s", original_acc)
            logger.info("Quantized model accuracy: %s", quantized_acc)

        if save_dir is not None:
            runner.save_quantized_model(save_dir)

        return runner

    def quantize(self):
        """Quantize the model

        Assumes that parameter combinations have been validated by check().
        """

        if self.quantizers is not None:
            # Multiple quantizers mode (chunked quantization)
            self.quantize_with_calibration_chunked()
        elif self.multi_gpu:
            # Multi-GPU quantization (flag_calibration=True is guaranteed by check())
            self.quantize_with_calibration_on_multi_gpu()
        elif self.calibration_config.batch_size is not None:
            # Chunked quantization (single quantizer)
            self.quantize_with_calibration_chunked()
        elif self.quantizer.flag_calibration:
            # Standard calibration-based quantization
            self.quantize_with_calibration()
        else:
            # Quantization without calibration
            self.quantize_without_calibration()

    def quantize_with_calibration(self):
        """Quantize the model with calibration"""

        model = self.model_config.load_model()
        logger = self.logger
        input_device = next(model.parameters()).device
        inputs = self.prepare_calibration_dataset(input_device, model=model)

        # Setup the quantizer
        self.quantizer.setup(model)

        # Register hooks to all linear layers
        handles = []
        progress = None
        if self.report_progress:
            progress = QuantizationProgressTracker(
                logger,
                len(self.quantizer.module_to_name),
                "Quantization",
            )

        if progress:
            quantize_bound = self.quantizer.quantize

            def _quantize_hook(module, input, output):  # pylint: disable=redefined-builtin
                quantize_bound(module, input, output)
                progress.step_complete(self.quantizer.module_to_name[module])

            hook_fn = _quantize_hook
        else:
            hook_fn = self.quantizer.quantize

        for module in self.quantizer.module_to_name.keys():
            handle = module.register_forward_hook(hook_fn)
            handles.append(handle)

        logger.info("Quantizing the model using %s", self.quantizer.name)
        with torch.no_grad():
            model(**inputs)

        # Remove all hooks
        for handle in handles:
            handle.remove()

        self.quantizer.execute_post_processing()

    def quantize_with_calibration_chunked(self):
        """Quantize the model with calibration using chunked forward passes

        Designed for large-scale calibration data.
        Splits calibration data into chunks of calibration_batch_size and
        accumulates information needed for quantization across multiple forward passes.

        Processing flow:
        1. Prepare calibration data on CPU
        2. Load model and set up quantizer
        3. Divide layers into groups and for each group:
           a. Execute forward passes per chunk and accumulate X^T X in FP64
           b. Quantize each layer using X^T X

        Note:
            - X^T X is accumulated in FP64 (for reuse in error computation)
            - Cast to quantizer.hessian_dtype during quantization
            - CPU/GPU memory usage can be adjusted by controlling the number of layer groups
        """
        # Lazy import: load submodule only when needed
        # pylint: disable-next=import-outside-toplevel
        from .runner_methods.chunked_quantization import run_chunked_quantization

        run_chunked_quantization(
            model_config=self.model_config,
            quantizers=self.quantizers if self.quantizers is not None else [self.quantizer],
            calibration_config=self.calibration_config,
            report_progress=self.report_progress,
        )

    def quantize_with_calibration_on_multi_gpu(self):
        """Quantize the model with calibration using multiple GPUs

        Quantizes each linear layer in parallel across multiple GPUs.

        Processing flow:
        1. Load the model and prepare calibration data
        2. Capture input activations for all layers and save to CPU
        3. Distribute layers to each GPU and execute quantization in parallel
        4. Aggregate results

        Note:
            - Called from quantize() when multi_gpu=True
            - Uses all available GPUs when gpu_ids is None

        """
        # Lazy import: load submodule only when needed
        # pylint: disable-next=import-outside-toplevel
        from .runner_methods.multi_gpu_quantization import run_multi_gpu_quantization

        # Execute multi-GPU quantization
        result = run_multi_gpu_quantization(
            model_config=self.model_config,
            quantizer=self.quantizer,
            calibration_config=self.calibration_config,
            gpu_ids=self.gpu_ids,
            report_progress=self.report_progress,
        )

        # Store results in quantizer.results
        self.quantizer.results = result["results"]

        # Post-processing
        self.quantizer.execute_post_processing()

    def quantize_without_calibration(self):
        """Quantize the model without calibration

        Quantize each layer in the form ||W - hat_W||_F^2.

        """

        model = self.model_config.load_model()
        logger = self.logger

        # Setup the quantizer
        self.quantizer.setup(model)

        # Quantize each layer
        logger.info(
            "Quantizing the model without calibration using %s",
            self.quantizer.name,
        )
        progress = None
        if self.report_progress:
            progress = QuantizationProgressTracker(
                logger,
                len(self.quantizer.module_to_name),
                "Quantization without calibration (layers)",
            )
        for module in self.quantizer.module_to_name.keys():
            self.quantizer.quantize(module, None, None)
            if progress:
                progress.step_complete(self.quantizer.module_to_name[module])

        self.quantizer.execute_post_processing()

    def quantize_with_qep(self):
        """Quantize the model with QEP

        Dispatches to either the generic or architecture-aware
        implementation based on ``qep_config.general``.

        - ``general=True``: Generic implementation independent
          of model architecture. Captures input activations per layer.
        - ``general=False`` (default): Architecture-aware implementation that
          exploits shared activations (e.g., QKV in Llama).

        """
        kwargs = dict(
            model_config=self.model_config,
            quantizer=self.quantizer,
            qep_config=self.qep_config,
            calibration_config=self.calibration_config,
            report_progress=self.report_progress,
        )

        if self.qep_config.general:
            # Lazy import: load submodule only when needed
            # pylint: disable-next=import-outside-toplevel
            from .qep._quantize_with_qep import run_quantize_with_qep

            run_quantize_with_qep(**kwargs)
        else:
            # Lazy import: load submodule only when needed
            # pylint: disable-next=import-outside-toplevel
            from .qep._quantize_with_qep_arch import run_quantize_with_qep_arch

            run_quantize_with_qep_arch(**kwargs)

    def quantize_with_lpcd(self):
        """Quantize the model with LPCD"""
        # Lazy import: load submodule only when needed
        # pylint: disable-next=import-outside-toplevel
        from .lpcd._lpcd_runner import run_quantize_with_lpcd

        run_quantize_with_lpcd(
            model_config=self.model_config,
            quantizer=self.quantizer,
            qep_config=self.qep_config,
            lpcd_config=self.lpcd_config,
            calibration_config=self.calibration_config,
        )

    def quantize_with_jointq_error_propagation(
        self,
        max_layers=None,
        skip_threshold_increase=0.01,
        skip_threshold_error=0.01,
        skip_threshold_amplification=5.0,
        device=None,
        batch_size=None,
        variation_scale=0.1,
        variation_cap=0.05,
        degradation_threshold=0.1,
        max_iter=10,
        log_level=0,
        exclude_layer_keywords=None,
    ):
        """Quantize the model with JointQ error propagation

        A generic implementation independent of model architecture.
        Consumes extra CPU memory and incurs unnecessary forward passes.
        Could be faster by leveraging model structure to avoid redundant forward passes.

        Current procedure:
        1. Save input activations of the original model to CPU
        2. For each target layer l, perform the following sequentially:
        2-1. Save input activations of layer l in the quantized model to CPU
        2-2. Quantize the weights of layer l in the quantized model
        2-3. Update the weights of layer l in the quantized model

        TODO: Implement quantization that leverages model structure.

        Args:
            max_layers: Maximum number of layers to process (None for all layers; for testing)
            skip_threshold_increase: Skip threshold for error increase rate (default: 0.01)
            skip_threshold_error: Skip threshold for relative cumulative error (default: 0.01)
            skip_threshold_amplification (float): Skip threshold for error amplification rate.
                Re-quantize when amplification exceeds this value even if g_relative is small
                (default: 5.0)
            device: Device to use for computation (None uses each layer's device)
            batch_size (int): Batch size (default: None, solves the optimization problem all at once)
            variation_scale (float): Scaling coefficient from degradation rate to variation rate (default: 0.1)
            variation_cap (float): Upper limit for maximum variation rate (default: 0.05)
            degradation_threshold (float): Degradation rate threshold; variation rate is 0 below this (default: 0.1)
            max_iter (int): Maximum number of iterations for quantize_advanced (default: 10)
            log_level (int): Log level for quantize_advanced (default: 0)
            exclude_layer_keywords (list[str]): List of keywords for layers to exclude from Step 2.
                If a layer name contains any of these keywords, it is excluded from
                re-quantization in Step 2 (Step 1 results are used as-is).
                None targets all layers (default: None)
        """
        # Lazy import: load submodule only when needed
        # pylint: disable-next=import-outside-toplevel
        from .runner_methods.jointq_error_propagation import run_jointq_error_propagation

        model = self.model_config.load_model()
        logger = self.logger
        input_device = next(model.parameters()).device
        inputs = self.prepare_calibration_dataset(input_device, model=model)

        run_jointq_error_propagation(
            model=model,
            inputs=inputs,
            current_results=self.quantizer.results,
            logger=logger,
            max_layers=max_layers,
            skip_threshold_increase=skip_threshold_increase,
            skip_threshold_error=skip_threshold_error,
            skip_threshold_amplification=skip_threshold_amplification,
            device=device,
            batch_size=batch_size,
            variation_scale=variation_scale,
            variation_cap=variation_cap,
            degradation_threshold=degradation_threshold,
            max_iter=max_iter,
            log_level=log_level,
            exclude_layer_keywords=exclude_layer_keywords,
        )

    def run_post_processes(self):
        """Execute post-quantization processes.

        Uses ``self.quantized_model`` when one has already been assigned;
        otherwise builds a packed quantized model on CPU from
        ``quantizer.results`` with
        ``create_quantized_model(pack_weights=True, use_gemlite=False)`` by
        default.  The model is passed to each
        :class:`PostQuantizationProcess` in order, and each process preserves
        the incoming pack state (packed-in -> packed-out), so
        ``self.quantized_model`` stays packed for memory-efficient eval reuse.
        Processes that train (e.g. :class:`PostProcessLoraSFT`) unpack the base
        weights internally for the duration of training and re-pack on exit.

        ``use_gemlite=False`` is used because GemLite relies on fp16-only Triton
        kernels that break when LoRA SFT runs with bfloat16 autocast; plain
        buffers (qweight/scales) let training call ``base_layer.forward()``
        without dtype mismatch.

        If an unpacked layout is required, explicitly build the model before
        calling this method and assign it to ``runner.quantized_model``::

            runner.quantized_model, _ = runner.create_quantized_model(
                pack_weights=False, use_gemlite=False)

        Direct ``post_process.run(model, runner.model_config)`` execution can
        use the same explicitly built model.

        Each process records its own metadata in
        ``model.config.quantization_config["onecomp_post_processes"]`` when it
        finishes successfully, so direct process execution and Runner execution
        share the same history path.

        Raises:
            ValueError: If neither ``self.quantized_model`` nor
                ``self.quantizer`` is available (``quantizers`` mode is not
                yet supported for building post-process inputs).
        """
        logger = self.logger

        if self.quantized_model is not None:
            logger.info("Using existing quantized model for post-quantization processes")
            quantized_model = self.quantized_model
        elif self.quantizer is not None:
            logger.info("Building quantized model for post-quantization processes...")
            # use_gemlite=False: GemLite uses fp16-only Triton kernels that break when
            # LoRA SFT runs with bfloat16 autocast. Keep plain PyTorch inference
            # layers while preserving the default packed quantized-buffer layout.
            quantized_model, _ = self.create_quantized_model(
                pack_weights=True,
                use_gemlite=False,
            )
        else:
            raise ValueError(
                "post_processes requires either 'runner.quantized_model' or a single "
                "'quantizer'. 'quantizers' (multiple) is not yet supported with "
                "post_processes."
            )

        for process in self.post_processes:
            logger.info("Start post-quantization process: %s", process.name)
            process.run(quantized_model, self.model_config)
            logger.info("Finished post-quantization process: %s", process.name)

        self.quantized_model = quantized_model

    def prepare_calibration_dataset(self, device, model=None):
        """Prepare calibration data for quantization methods such as GPTQ.

        See calibration.calibration_data_loader.prepare_calibration_dataset for details.

        Args:
            device (torch.device): Device to place tensors on (CPU or GPU)
            model: Model instance (optional). Add model-specific fields
            (e.g. mm_token_type_ids for Gemma 4).

        Returns:
            dict: Input dictionary for the model
                - "input_ids": tensor of shape (num_chunks, max_length)
                - "attention_mask": tensor of shape (num_chunks, max_length)
        """
        tokenizer = self.model_config.load_tokenizer()

        return prepare_calibration_dataset(
            tokenizer=tokenizer,
            device=device,
            calibration_config=self.calibration_config,
            logger=self.logger,
            model=model,
        )

    def print_quantization_results(self, quantizer=None):
        """Log quantization results.

        Formats and logs the quantizer results.
        The following information is output for each layer:

        - Quantization time (seconds)
        - Output squared error (only if value exists)
        - Mean output squared error (only if value exists)
        - Weight squared error (only if value exists)
        - Mean weight squared error (only if value exists)

        Args:
            quantizer (Quantizer, optional):
                The quantizer. Uses self.quantizer if None.
                Specify explicitly when using quantizers mode.

        Examples:
            Single quantizer mode:

            >>> runner.print_quantization_results()

            Multiple quantizers mode:

            >>> runner.print_quantization_results(quantizer=gptq)
        """
        logger = self.logger

        if quantizer is None:
            quantizer = self.quantizer
        if quantizer is None:
            logger.warning(
                "print_quantization_results: 'quantizer' is None. "
                "Please specify a quantizer explicitly."
            )
            return

        logger.info("Quantization results for %s:", quantizer.name)

        for name, result in quantizer.results.items():
            logger.info("%s:", name)
            logger.info(
                "    Quantization time: %s seconds",
                f"{result.quantization_time:.2f}",
            )
            logger.info(
                "    Output squared error: %s",
                f"{result.output_squared_error:.2e}",
            )
            logger.info(
                "    Mean output squared error: %s",
                f"{result.mean_output_squared_error:.2e}",
            )
            logger.info(
                "    Weight squared error: %s",
                f"{result.weight_squared_error:.2e}",
            )
            logger.info(
                "    Mean weight squared error: %s",
                f"{result.mean_weight_squared_error:.2e}",
            )
            if result.relative_output_squared_error is not None:
                logger.info(
                    "    Relative output squared error: %s",
                    f"{result.relative_output_squared_error:.2e}",
                )
            if result.relative_weight_squared_error is not None:
                logger.info(
                    "    Relative weight squared error: %s",
                    f"{result.relative_weight_squared_error:.2e}",
                )

    def save_quantization_statistics(self, path: str, quantizer=None):
        """Save the quantization statistics

        Args:
            path (str): File path to save to
            quantizer (Quantizer, optional): Quantizer whose statistics to save.
                Uses self.quantizer if None.
                Specify explicitly when using quantizers mode.

        Examples:
            Single quantizer mode:

            >>> runner.save_quantization_statistics("stats.json")

            Multiple quantizers mode:

            >>> quantizers = [gptq, jointq]
            >>> runner.save_quantization_statistics("gptq_stats.json", quantizer=gptq)
            >>> runner.save_quantization_statistics("jointq_stats.json", quantizer=jointq)
        """

        logger = self.logger

        if quantizer is None:
            quantizer = self.quantizer
        if quantizer is None:
            logger.warning(
                "save_quantization_statistics: 'quantizer' is None. "
                "Please specify a quantizer explicitly."
            )
            return

        logger.info("Saving the quantization statistics to %s", path)

        statistics = {key: result.get_statistics() for key, result in quantizer.results.items()}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=4)

    def save_quantization_results(self, path: str, quantizer=None):
        """Save the quantization results to a file

        Save quantization results (QuantizationResult objects) to a file.
        The saved data includes dequantized weights, scales, zero points,
        integer assignments, and other quantization parameters.

        Args:
            path (str): The path to save the quantization results.
                The .pt extension is recommended.
            quantizer (Quantizer, optional): Quantizer whose results to save.
                Uses self.quantizer if None.
                Specify explicitly when using quantizers mode.

        Examples:
            Single quantizer mode:

            >>> runner.save_quantization_results("results.pt")

            Multiple quantizers mode:

            >>> quantizers = [gptq, jointq]
            >>> runner.save_quantization_results("gptq_results.pt", quantizer=gptq)
            >>> runner.save_quantization_results("jointq_results.pt", quantizer=jointq)
        """

        if quantizer is None:
            quantizer = self.quantizer
        if quantizer is None:
            self.logger.warning(
                "save_quantization_results: 'quantizer' is None. "
                "Please specify a quantizer explicitly."
            )
            return

        quantizer.save_results(path)

    def _calculate_evaluation(
        self,
        original_model: bool,
        dequantized_model: bool,
        quantized_model: bool,
        eval_name: str,
        eval_function,
        eval_args: dict,
        quantizer: Quantizer | None,
    ) -> tuple:
        """Calculate the evaluation metric (perplexity or accuracy).

        Each evaluation mode (original, dequantized, quantized) loads an
        independent model instance to prevent state contamination between
        evaluations.  This means multiple modes will trigger multiple
        ``load_model()`` calls, and calling both ``calculate_perplexity()``
        and ``calculate_accuracy()`` will load models independently as well.
        This trade-off prioritises correctness over load-time efficiency.
        """
        logger = self.logger

        if quantizer is None:
            quantizer = self.quantizer
        if quantizer is None:
            logger.warning(
                "calculate_%s: 'quantizer' is None. " "Please specify a quantizer explicitly.",
                eval_name,
            )
            return None, None, None

        original_result = None
        dequantized_result = None
        quantized_result = None

        if original_model:
            logger.info("Evaluating original model (%s)...", eval_name)
            model = self.model_config.load_model()
            tokenizer = self.model_config.load_tokenizer()
            original_result = eval_function(model=model, tokenizer=tokenizer, **eval_args)
            del model, tokenizer
            empty_cache(self.model_config.get_device())

        if quantized_model:
            try:
                logger.info("Evaluating quantized model (%s)...", eval_name)
                if self.quantized_model is not None:
                    model = self.quantized_model
                    model.to(self.model_config.get_device())
                    tokenizer = self.model_config.load_tokenizer()
                    quantized_result = eval_function(model=model, tokenizer=tokenizer, **eval_args)
                    model.to("cpu")
                    del tokenizer
                else:
                    model, tokenizer = self.create_quantized_model(quantizer=quantizer)
                    model.to(self.model_config.get_device())
                    quantized_result = eval_function(model=model, tokenizer=tokenizer, **eval_args)
                    del model, tokenizer
                empty_cache(self.model_config.get_device())
            except NotImplementedError:
                logger.warning(
                    "This quantization method does not support creating a quantized model; "
                    "evaluation will be performed using the dequantized model instead.",
                )
                dequantized_model = True

        if dequantized_model:
            logger.info("Evaluating dequantized model (%s)...", eval_name)
            model = self.model_config.load_model()
            tokenizer = self.model_config.load_tokenizer()
            # Unfuse MoE expert tensors so per-expert quantization results
            # (e.g. "...experts.4.gate_proj") can be matched by module name.
            from .utils.unfuse_moe import unfuse_moe_experts

            if unfuse_moe_experts(model, logger):
                logger.info("Unfused MoE expert tensors for dequantized evaluation")
            self.update_model_weights(model, quantizer=quantizer)
            dequantized_result = eval_function(model=model, tokenizer=tokenizer, **eval_args)
            del model, tokenizer
            empty_cache(self.model_config.get_device())

        return original_result, dequantized_result, quantized_result

    def calculate_perplexity(
        self,
        original_model=False,
        dequantized_model=False,
        quantized_model=True,
        dataset_name="Salesforce/wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="test",
        max_samples=None,
        max_length=2048,
        stride=2048,
        quantizer=None,
    ):
        """Calculate the perplexity of the model

        Args:
            original_model (bool):
                Whether to calculate the perplexity of the original model.
            dequantized_model (bool):
                Whether to calculate the perplexity of the dequantized model.
            quantized_model (bool):
                Whether to calculate the perplexity of the quantized model.
            dataset_name (str):
                The name of the dataset to use for calculating perplexity.
            dataset_config (str):
                The configuration of the dataset.
            split (str):
                The split of the dataset to use.
            max_samples (int):
                The maximum number of samples to use.
            max_length (int, optional):
                Maximum length of the sliding window.
                Uses model.config.max_position_embeddings if None.
                2048 is recommended to match standard paper values.
            stride (int, optional):
                Stride of the sliding window.
                Same as max_length (no overlap) if None.
            quantizer (Quantizer, optional):
                The quantizer. Uses self.quantizer if None.
                Specify explicitly when using quantizers mode.

        Returns:
            tuple: (original_ppl, dequantized_ppl, quantized_ppl)

        Note:
            Evaluating the original or dequantized model requires loading
            the full model on GPU.

            Quantized-model evaluation (``quantized_model=True``) is
            currently supported only for GPTQ and DBF quantizers.
            Support for other quantization methods is planned.

        Examples:
            Single quantizer mode:

            >>> original_ppl, dequantized_ppl, quantized_ppl = runner.calculate_perplexity()

            Multiple quantizers mode:

            >>> original_ppl, dequantized_ppl, quantized_ppl = runner.calculate_perplexity(
            ...     quantizer=gptq
            ... )
        """
        calculate_perplexity_args = {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
            "max_samples": max_samples,
            "max_length": max_length,
            "stride": stride,
        }

        return self._calculate_evaluation(
            original_model=original_model,
            dequantized_model=dequantized_model,
            quantized_model=quantized_model,
            eval_name="perplexity",
            eval_function=calc_perplexity,
            eval_args=calculate_perplexity_args,
            quantizer=quantizer,
        )

    def benchmark_perplexity(
        self,
        original_model=True,
        dequantized_model=False,
        quantized_model=True,
        dataset_name="Salesforce/wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="test",
        max_samples=None,
        max_length=2048,
        stride=2048,
        quantizers=None,
    ):
        """Calculate perplexity for all quantizers at once

        Internally calls calculate_perplexity for each quantizer.
        The original model PPL is calculated only once (on the first iteration).

        Args:
            original_model (bool):
                Whether to calculate the perplexity of the original model.
            dequantized_model (bool):
                Whether to calculate the perplexity of the dequantized model.
            quantized_model (bool):
                Whether to calculate the perplexity of the quantized model.
            dataset_name (str):
                The name of the dataset to use for calculating perplexity.
            dataset_config (str):
                The configuration of the dataset.
            split (str):
                The split of the dataset to use.
            max_samples (int):
                The maximum number of samples to use.
            max_length (int, optional):
                Maximum length of the sliding window.
                Uses model.config.max_position_embeddings if None.
            stride (int, optional):
                Stride of the sliding window.
                Same as max_length (no overlap) if None.
            quantizers (list[Quantizer], optional):
                List of quantizers. Uses self.quantizers or
                [self.quantizer] if None.

        Returns:
            dict: Dictionary of PPL values. Keys are as follows:

            - ``"original"``: PPL of the original model (not included if skipped)
            - ``quantizer.name``: PPL for each quantizer (quantized or
              dequantized, with quantized taking precedence)
            - ``quantizer.name + "_dequantized"``: PPL of the dequantized
              model (only included when ``dequantized_model=True``)

        Examples:
            >>> runner.run()
            >>> ppl_dict = runner.benchmark_perplexity()
            >>> print(ppl_dict)
            {'original': 5.47, 'GPTQ': 5.72, 'JointQ': 5.68}

            Specify quantizers explicitly:

            >>> ppl_dict = runner.benchmark_perplexity(quantizers=[gptq, jointq])

            Include dequantized model PPL:

            >>> ppl_dict = runner.benchmark_perplexity(dequantized_model=True)
            >>> print(ppl_dict)
            {'original': 5.47, 'GPTQ': 5.72, 'GPTQ_dequantized': 5.71}
        """

        logger = self.logger

        # Resolve quantizers
        if quantizers is None:
            if self.quantizers is not None:
                quantizers = self.quantizers
            elif self.quantizer is not None:
                quantizers = [self.quantizer]
            else:
                logger.warning("benchmark_perplexity: No quantizers available.")
                return {}

        ppl_dict = {}

        for i, q in enumerate(quantizers):
            logger.info("Calculating perplexity for %s ...", q.name)

            # Calculate original PPL only for the first quantizer
            calc_original = original_model and (i == 0)

            orig_ppl, dequant_ppl, quant_ppl = self.calculate_perplexity(
                original_model=calc_original,
                dequantized_model=dequantized_model,
                quantized_model=quantized_model,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                max_samples=max_samples,
                max_length=max_length,
                stride=stride,
                quantizer=q,
            )

            if calc_original:
                ppl_dict["original"] = orig_ppl
                logger.info("Original perplexity: %s", orig_ppl)

            if dequantized_model:
                ppl_dict[q.name + "_dequantized"] = dequant_ppl
                logger.info("%s dequantized perplexity: %s", q.name, dequant_ppl)

            # Fallback to dequantized PPL if quantized PPL is not available
            if quant_ppl is None:
                quant_ppl = dequant_ppl
            ppl_dict[q.name] = quant_ppl
            logger.info("%s perplexity: %s", q.name, quant_ppl)

        return ppl_dict

    def calculate_accuracy(
        self,
        original_model=False,
        dequantized_model=False,
        quantized_model=True,
        tasks=None,
        batch_size=8,
        num_fewshot=0,
        display_results=True,
        quantizer=None,
    ):
        """Calculate the zero-shot accuracy of the model

        Args:
            original_model (bool):
                Whether to calculate the accuracy of the original model.
            dequantized_model (bool):
                Whether to calculate the accuracy of the dequantized model.
            quantized_model (bool):
                Whether to calculate the accuracy of the quantized model.
            tasks (list):
                The list of tasks to evaluate.
                Default: ["arc_easy", "arc_challenge", "piqa", "winogrande"]
            batch_size (int):
                The batch size for evaluation.
            num_fewshot (int):
                The number of few-shot examples.
            display_results (bool):
                Whether to display the results.
            quantizer (Quantizer, optional):
                The quantizer. Uses self.quantizer if None.
                Specify explicitly when using quantizers mode.

        Returns:
            tuple: (original_acc, dequantized_acc, quantized_acc)

        Note:
            Evaluating the original or dequantized model requires loading
            the full model on GPU.

            Quantized-model evaluation (``quantized_model=True``) is
            currently supported only for GPTQ and DBF quantizers.
            Support for other quantization methods is planned.

        Examples:
            Single quantizer mode:

            >>> original_acc, dequantized_acc, quantized_acc = runner.calculate_accuracy()

            Multiple quantizers mode:

            >>> original_acc, dequantized_acc, quantized_acc = runner.calculate_accuracy(
            ...     quantizer=gptq
            ... )
        """
        calculate_accuracy_args = {
            "tasks": tasks,
            "batch_size": batch_size,
            "num_fewshot": num_fewshot,
            "display_results": display_results,
        }

        return self._calculate_evaluation(
            original_model=original_model,
            dequantized_model=dequantized_model,
            quantized_model=quantized_model,
            eval_name="accuracy",
            eval_function=calc_accuracy,
            eval_args=calculate_accuracy_args,
            quantizer=quantizer,
        )

    def benchmark_accuracy(
        self,
        original_model=True,
        dequantized_model=False,
        quantized_model=True,
        tasks=None,
        batch_size=8,
        num_fewshot=0,
        display_results=False,
        quantizers=None,
    ):
        """Calculate accuracy for all quantizers at once

        Internally calls calculate_accuracy for each quantizer.
        The original model accuracy is calculated only once (on the first iteration).

        Args:
            original_model (bool):
                Whether to calculate the accuracy of the original model.
            dequantized_model (bool):
                Whether to calculate the accuracy of the dequantized model.
            quantized_model (bool):
                Whether to calculate the accuracy of the quantized model.
            tasks (list):
                The list of tasks to evaluate.
                Default: ["arc_easy", "arc_challenge", "piqa", "winogrande"]
            batch_size (int):
                The batch size for evaluation.
            num_fewshot (int):
                The number of few-shot examples.
            display_results (bool):
                Whether to display the results.
            quantizers (list[Quantizer], optional):
                List of quantizers. Uses self.quantizers or
                [self.quantizer] if None.

        Returns:
            dict: Dictionary of accuracy values. Keys are as follows:

            - ``"original"``: Accuracy of the original model (not included if skipped)
            - ``quantizer.name``: Accuracy for each quantizer (quantized or
              dequantized, with quantized taking precedence)
            - ``quantizer.name + "_dequantized"``: Accuracy of the dequantized
              model (only included when ``dequantized_model=True``)

        Examples:
            >>> runner.run()
            >>> acc_dict = runner.benchmark_accuracy()
            >>> print(acc_dict)
            {'original': {...}, 'GPTQ': {...}, 'JointQ': {...}}

            Specify quantizers explicitly:

            >>> acc_dict = runner.benchmark_accuracy(quantizers=[gptq, jointq])

            Include dequantized model accuracy:

            >>> acc_dict = runner.benchmark_accuracy(dequantized_model=True)
        """

        logger = self.logger

        # Resolve quantizers
        if quantizers is None:
            if self.quantizers is not None:
                quantizers = self.quantizers
            elif self.quantizer is not None:
                quantizers = [self.quantizer]
            else:
                logger.warning("benchmark_accuracy: No quantizers available.")
                return {}

        acc_dict = {}

        for i, q in enumerate(quantizers):
            logger.info("Calculating accuracy for %s ...", q.name)

            # Calculate original accuracy only for the first quantizer
            calc_original = original_model and (i == 0)

            orig_acc, dequant_acc, quant_acc = self.calculate_accuracy(
                original_model=calc_original,
                dequantized_model=dequantized_model,
                quantized_model=quantized_model,
                tasks=tasks,
                batch_size=batch_size,
                num_fewshot=num_fewshot,
                display_results=display_results,
                quantizer=q,
            )

            if calc_original:
                acc_dict["original"] = orig_acc
                logger.info("Original accuracy: %s", orig_acc)

            if dequantized_model:
                acc_dict[q.name + "_dequantized"] = dequant_acc
                logger.info("%s dequantized accuracy: %s", q.name, dequant_acc)

            # Fallback to dequantized accuracy if quantized accuracy is not available
            if quant_acc is None:
                quant_acc = dequant_acc
            acc_dict[q.name] = quant_acc
            logger.info("%s accuracy: %s", q.name, quant_acc)

        return acc_dict

    def save_dequantized_model(self, path: str, quantizer=None):
        """Save the dequantized model to the specified path

        Args:
            path (str):
                The path to save the dequantized model.
            quantizer (Quantizer, optional):
                The quantizer. Uses self.quantizer if None.
                Specify explicitly when using quantizers mode.

        Examples:
            Single quantizer mode:

            >>> runner.save_dequantized_model("./dequantized_model")

            Multiple quantizers mode:

            >>> runner.save_dequantized_model("./gptq_model", quantizer=gptq)
            >>> runner.save_dequantized_model("./jointq_model", quantizer=jointq)
        """

        logger = self.logger

        if quantizer is None:
            quantizer = self.quantizer
        if quantizer is None:
            logger.warning(
                "save_dequantized_model: 'quantizer' is None. "
                "Please specify a quantizer explicitly."
            )
            return

        logger.info("Saving the dequantized model and tokenizer to %s", path)

        model = self.model_config.load_model(device_map="cpu")
        tokenizer = self.model_config.load_tokenizer()

        self.update_model_weights(model, quantizer=quantizer)

        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

        if self.model_config.has_additional_data():
            config_class = type(self.model_config).__name__
            logger.warning(
                "This model was loaded with '%s', which registers "
                "additional preprocessing (e.g., forward hooks). "
                "The saved model does NOT include these hooks. "
                "Please use '%s' (not ModelConfig) when "
                "loading the saved model from '%s'.",
                config_class,
                config_class,
                path,
            )

    def update_model_weights(self, model, quantizer=None):
        """Update the model weights"""

        logger = self.logger

        if quantizer is None:
            quantizer = self.quantizer
        if quantizer is None:
            logger.warning(
                "No quantizer specified. "
                "Use the 'quantizer' argument to specify which quantizer to use."
            )
            return

        logger.info("Updating the model weights with %s ...", quantizer.name)

        for name, module in model.named_modules():
            if name in quantizer.results:
                dtype = module.weight.data.dtype
                device = module.weight.data.device
                module.weight.data = (
                    quantizer.results[name].compute_dequantized_weight().to(device).to(dtype)
                )
                logger.debug("Updated the model weights for layer: %s", name)

    def create_quantized_model(self, pack_weights: bool = True, quantizer=None, use_gemlite=None):
        """Create a quantized model from quantization results.

        Loads the base model on CPU, replaces Linear layers with quantized
        inference layers (e.g. ``GPTQLinear``), and attaches quantization
        config to ``model.config``.

        Must be called after ``run()`` (i.e., ``quantizer.results`` must
        be populated).

        Args:
            pack_weights (bool):
                Whether to pack quantized weights for memory-efficient
                representation. Default is True.
            quantizer (Quantizer, optional):
                The quantizer to use. Uses self.quantizer if None.
                Specify explicitly when using quantizers mode.
            use_gemlite (bool or None):
                Whether to use GemLite for inference layers.
                Set to False when saving to avoid extra params in
                safetensors. Default is None (uses quantizer default).

        Returns:
            tuple[nn.Module, PreTrainedTokenizer]:
                (quantized_model, tokenizer)

        Examples:
            >>> runner.run()
            >>> model, tokenizer = runner.create_quantized_model()

            With post-process (manual single-process run; ``run_post_processes()``
            builds the model with ``pack_weights=True`` by default).  The same
            packed model can be passed directly to post-processes that preserve
            quantized layer structure, such as ``BlockWisePTQ`` or ``GlobalPTQ``:

            >>> model, tokenizer = runner.create_quantized_model()
            >>> post_process = BlockWisePTQ()
            >>> post_process.run(model, runner.model_config)
            >>> post_process = GlobalPTQ()
            >>> post_process.run(model, runner.model_config)

            LoRA SFT also accepts the model directly.  It introduces custom
            wrapper modules (``LoRAGPTQLinear``), which ``save_quantized_model``
            handles by writing the base weights as safetensors plus a
            PEFT-compatible adapter sidecar under ``lora_adapter/``:

            >>> model, tokenizer = runner.create_quantized_model(
            ...     pack_weights=True,
            ...     use_gemlite=False,
            ... )
            >>> post_process = PostProcessLoraSFT(data_files="train.jsonl")
            >>> post_process.run(model, runner.model_config)
            >>> runner.quantized_model = model  # so the LoRA model is the one saved
            >>> runner.save_quantized_model("./quantized_model_lora")

            Post-processes preserve the incoming pack state.  If a workflow
            requires unpacked quantized buffers (e.g. when intentionally
            debugging an unpacked-buffer path), build the model explicitly with
            ``pack_weights=False`` before direct execution:

            >>> model, tokenizer = runner.create_quantized_model(
            ...     pack_weights=False,
            ...     use_gemlite=False,
            ... )
            >>> post_process = BlockWisePTQ()
            >>> post_process.run(model, runner.model_config)
        """
        if quantizer is None:
            quantizer = self.quantizer

        # Delegate save config to quantizer (extensible via override)
        quant_config = quantizer.get_quant_config()

        # Load base model on CPU (GPU is not needed for saving)
        model = self.model_config.load_model(device_map="cpu")
        tokenizer = self.model_config.load_tokenizer()

        from .utils.unfuse_moe import (
            fuse_moe_experts,
            strip_moe_experts_from_quant_config,
            unfuse_moe_experts,
        )

        # Unfuse MoE experts so per-expert result keys can be resolved
        if unfuse_moe_experts(model, self.logger):
            self.logger.info("Unfused MoE expert tensors for quantized model save")

        # Replace Linear layers with quantized layers using quantizer.results
        self.logger.info("Replacing Linear layers with quantized inference layers...")
        quantizer.apply_results_to_model(model, pack_weights=pack_weights, use_gemlite=use_gemlite)

        # Re-register Hadamard hooks for rotation-preprocessed models.
        # apply_results replaces nn.Linear with quantized modules (e.g. GPTQLinear),
        # which discards hooks registered by RotatedModelConfig.load_model().
        fp32_had = getattr(self.model_config, "fp32_had", False)
        if self.model_config.has_additional_data():
            from .pre_process.rotation_utils import (
                collect_quantized_down_proj_types,
                register_online_hadamard_hooks,
            )

            quantized_down_proj_types = collect_quantized_down_proj_types(model)

            if quantized_down_proj_types:
                hooks = register_online_hadamard_hooks(
                    model,
                    layers_cls=quantized_down_proj_types,
                    fp32_had=fp32_had,
                )
                self.logger.info(
                    "Re-registered Hadamard pre-hooks on %d quantized layers (fp32_had=%s)",
                    len(hooks),
                    fp32_had,
                )

        # Build modules_in_block_to_quantize from actually-quantized layer names.
        quantized_names = sorted(quantizer.results.keys())
        modules_in_block = list(quantized_names)
        quant_config["modules_in_block_to_quantize"] = modules_in_block
        quant_config["quantized_layer_names"] = modules_in_block
        quant_config = quantizer.finalize_quant_config_for_save(
            quant_config=quant_config,
            quantized_layer_names=quantized_names,
            num_hidden_layers=(
                getattr(model.config, "num_hidden_layers", None)
                or getattr(getattr(model.config, "text_config", None), "num_hidden_layers", None)
            ),
        )
        quant_config["rotated"] = self.model_config.has_additional_data()
        quant_config["fp32_had"] = fp32_had

        # Rotated GPTQ models need the mixed_gptq plugin in vLLM so the
        # down_proj path can apply the online Hadamard transform.
        if quant_config.get("quant_method") == "gptq" and quant_config["rotated"]:
            quant_config["quant_method"] = "mixed_gptq"
            self.logger.info(
                "Rotated GPTQ model detected: switching quant_method to mixed_gptq "
                "for vLLM rotation compatibility"
            )

        # MoE expert layers are not nn.Linear but fused3d tensors and are skipped by the
        # quantizer.  vLLM's built-in "gptq" handler still assumes them
        # GPTQ-quantized.  "mixed_gptq" returns None
        # and passes the weights to UnquantizedFusedMoEMethod.
        # cf) https://docs.vllm.ai/en/stable/features/quantization/#implementing-a-quantized-moe-method
        num_experts = _get_num_experts(model.config)
        if quant_config.get("quant_method") == "gptq" and num_experts > 0:
            quant_config["quant_method"] = "mixed_gptq"
            self.logger.info(
                "MoE model detected (num_experts=%d): "
                "switching quant_method to mixed_gptq for vLLM compatibility",
                num_experts,
            )

        if num_experts > 0:
            if self.moe_quant_experts:
                # Keep experts as per-expert GPTQLinear tensors (4-bit) and leave
                # them in the quant config so the mixed_gptq vLLM plugin serves
                # them via GPTQMarlinMoEMethod.  No fuse/dequantize/strip.
                self.logger.info(
                    "MoE quant-experts mode: keeping %d-expert layers GPTQ-quantized "
                    "(no fuse/strip)",
                    num_experts,
                )
            else:
                strip_moe_experts_from_quant_config(quant_config)
                if fuse_moe_experts(model, self.logger):
                    self.logger.info("Fused MoE expert tensors for vLLM-compatible save")

        # Patch weights and quant config for architectures with shared
        # K/V projections (e.g. Gemma4 attention_k_eq_v) so that vLLM's
        # fused qkv_proj consistency check passes.
        self._patch_k_eq_v_for_vllm(model, quant_config)

        # Add quantization config to model config
        model.config.quantization_config = quant_config

        return model, tokenizer

    def _patch_k_eq_v_for_vllm(self, model, quant_config: dict) -> None:
        """Add synthetic v_proj weights and config for attention_k_eq_v layers.

        Gemma4 full-attention layers with attention_k_eq_v=True have no
        v_proj weight — the model reuses key states as value states.
        vLLM fuses q/k/v into a single qkv_proj and requires all shards
        to share the same quantization status.
        """
        text_cfg = getattr(model.config, "text_config", None)
        if text_cfg is None or not getattr(text_cfg, "attention_k_eq_v", False):
            return
        layer_types = getattr(text_cfg, "layer_types", [])
        k_eq_v_indices = {i for i, lt in enumerate(layer_types) if lt == "full_attention"}
        if not k_eq_v_indices:
            return

        # (1) Model weights: duplicate k_proj → v_proj
        layers = None
        for name, mod in model.named_modules():
            if name.endswith("language_model.layers"):
                layers = mod
                break

        if layers is not None:
            weight_count = 0
            for idx in sorted(k_eq_v_indices):
                if idx >= len(layers):
                    continue
                attn = getattr(layers[idx], "self_attn", None)
                if attn is None:
                    continue
                k_proj = getattr(attn, "k_proj", None)
                if k_proj is None or getattr(attn, "v_proj", None) is not None:
                    continue
                attn.v_proj = copy.deepcopy(k_proj)
                weight_count += 1
            if weight_count:
                self.logger.info(
                    "Added v_proj weights (copied from k_proj) to %d "
                    "attention_k_eq_v layers for vLLM compatibility",
                    weight_count,
                )

        # (2) Quant config: add v_proj entries cloned from k_proj
        for idx, layer_cfg in enumerate(quant_config.get("quantization_bits", [])):
            if (
                idx in k_eq_v_indices
                and "self_attn.k_proj" in layer_cfg
                and "self_attn.v_proj" not in layer_cfg
            ):
                layer_cfg["self_attn.v_proj"] = copy.deepcopy(layer_cfg["self_attn.k_proj"])

        for key in ("modules_in_block_to_quantize", "quantized_layer_names"):
            names = quant_config.get(key, [])
            added = [
                f"model.language_model.layers.{idx}.self_attn.v_proj"
                for idx in k_eq_v_indices
                if f"model.language_model.layers.{idx}.self_attn.k_proj" in names
                and f"model.language_model.layers.{idx}.self_attn.v_proj" not in names
            ]
            if added:
                quant_config[key] = sorted(names + added)

    # ========================================
    # Unified Save/Load Methods
    # ========================================

    @staticmethod
    def _packable_gptq_wbits(wbits: int) -> bool:
        """Return whether OneComp can export GPTQ tensors in packed format."""
        from .quantizer.gptq.gptq_layer import is_packable_wbits

        return is_packable_wbits(wbits)

    @staticmethod
    def _collect_lora_gptq_modules(model) -> list[tuple[str, torch.nn.Module]]:
        """Return ``LoRAGPTQLinear`` modules contained in *model*."""
        # Avoid importing post_process_lora_sft here; it pulls training deps.
        return [
            (name, mod)
            for name, mod in model.named_modules()
            if mod.__class__.__name__ == "LoRAGPTQLinear"
        ]

    @staticmethod
    def _iter_gptq_export_layers(
        model,
        lora_modules: list[tuple[str, torch.nn.Module]],
    ) -> list[tuple[str, torch.nn.Module]]:
        """Return GPTQLinear layers keyed by their base-model export prefix."""
        from .quantizer.gptq.gptq_layer import GPTQLinear

        layers: list[tuple[str, torch.nn.Module]] = []
        lora_base_names = set()
        for name, mod in lora_modules:
            base_layer = getattr(mod, "base_layer", None)
            if isinstance(base_layer, GPTQLinear):
                layers.append((name, base_layer))
                lora_base_names.add(f"{name}.base_layer" if name else "base_layer")

        for name, mod in model.named_modules():
            if name in lora_base_names:
                continue
            if isinstance(mod, GPTQLinear):
                layers.append((name, mod))
        return layers

    def _build_base_quantized_state_dict(
        self,
        model,
        lora_modules: list[tuple[str, torch.nn.Module]],
        pack_weights: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Build a base-model state_dict for HF/vLLM-compatible export.

        ``LoRAGPTQLinear`` wrappers are flattened back to the base GPTQLinear
        key layout, and LoRA tensors are omitted from the base weights. If a
        GPTQLinear is unpacked and the bit-width is packable, only the exported
        tensors are packed; the in-memory model is left unchanged.
        """
        export_state_dict: dict[str, torch.Tensor] = {}
        lora_modules = sorted(lora_modules, key=lambda item: len(item[0]), reverse=True)

        for key, tensor in model.state_dict().items():
            skip = False
            export_key = key
            for lora_name, _mod in lora_modules:
                prefix = f"{lora_name}." if lora_name else ""
                if key.startswith(f"{prefix}lora_A.") or key.startswith(f"{prefix}lora_B."):
                    skip = True
                    break
                base_prefix = f"{prefix}base_layer."
                if key.startswith(base_prefix):
                    export_key = f"{prefix}{key[len(base_prefix):]}"
                    break
            if not skip:
                export_state_dict[export_key] = tensor

        gptq_layers = self._iter_gptq_export_layers(model, lora_modules)
        if not gptq_layers or not pack_weights:
            return export_state_dict

        from .quantizer.gptq.gptq_layer import pack_int_weights, pack_zeros

        packed_layers = 0
        skipped_layers = []
        for layer_name, layer in gptq_layers:
            if getattr(layer, "_weight_is_packed", False):
                continue

            wbits = getattr(layer, "wbits", 0)
            if not self._packable_gptq_wbits(wbits):
                if wbits != 1:
                    skipped_layers.append((layer_name, wbits))
                continue
            wbits = int(wbits)

            prefix = f"{layer_name}." if layer_name else ""
            qweight_key = f"{prefix}qweight"
            qzeros_key = f"{prefix}qzeros"
            if qweight_key not in export_state_dict or qzeros_key not in export_state_dict:
                self.logger.warning(
                    "Skipping GPTQ export packing for %s because qweight/qzeros "
                    "were not found in state_dict",
                    layer_name,
                )
                continue

            export_state_dict[qweight_key] = pack_int_weights(
                layer.qweight.detach().to(torch.int32),
                wbits,
            ).contiguous()
            export_state_dict[qzeros_key] = pack_zeros(
                layer.qzeros.detach().to(torch.int32),
                wbits,
            ).contiguous()
            packed_layers += 1

        if packed_layers:
            self.logger.info(
                "Packed %d unpacked GPTQLinear layer(s) in export state_dict",
                packed_layers,
            )
        if skipped_layers:
            self.logger.warning(
                "Left %d unpacked GPTQLinear layer(s) unpacked in export state_dict "
                "because their wbits are not packable: %s",
                len(skipped_layers),
                skipped_layers,
            )
        return export_state_dict

    def _save_lora_adapter_sidecar(
        self, save_directory: str, model=None, save_format: str = "auto"
    ) -> bool:
        """Write a PEFT-compatible LoRA adapter sidecar if *model*
        contains ``LoRAGPTQLinear`` modules (typically produced by
        ``PostProcessLoraSFT``).

        The sidecar is placed in a ``lora_adapter/`` subdirectory rather than
        directly in ``save_directory``. Reason: vLLM's base-model safetensors
        loader globs ``*.safetensors`` at the top level of the model directory
        and would otherwise try to load ``adapter_model.safetensors`` as
        base-model weights, crashing with ``"no module or parameter named
        'base_model' in LlamaForCausalLM"``. Keeping the adapter under a
        subdirectory avoids that collision while still keeping the whole model
        self-contained under one directory tree.

        The subdirectory contains:
          - ``adapter_model.safetensors``
          - ``adapter_config.json``

        The format matches what vLLM's native PEFT LoRA loader expects, so::

            LLM(model=save_dir, enable_lora=True)
            LoRARequest(..., lora_path=os.path.join(save_dir, "lora_adapter"))

        will load and apply the adapter without any OneComp-specific changes
        to the vLLM plugin.

        Args:
            save_directory: Directory the model is being saved to.
            model: Model to collect LoRA modules from; defaults to
                ``self.quantized_model``.
            save_format: Must match the ``save_format`` used for the base
                weights. When ``"full_wrapper"``, adapter module paths are
                remapped from the text-only ``model.layers.*`` namespace to
                the composite ``model.language_model.*`` namespace so the
                sidecar keys line up with the remapped base layers.

        Returns:
            bool: True iff an adapter was written. False if there is no
            in-memory LoRA state to save (e.g. no post-process ran).
        """
        if model is None:
            model = self.quantized_model
        if model is None:
            return False

        # Inline imports keep runner.py import-time cheap and avoid any
        # circular-import risk with the post_process package.
        from safetensors.torch import save_file as _st_save_file

        lora_modules = self._collect_lora_gptq_modules(model)
        if not lora_modules:
            return False

        # Save in the base model's runtime dtype so the round-trip is a single
        # fp32(train) -> base-dtype rounding. Hardcoding float16 would add a
        # needless fp16 intermediate for bf16 models (fp32 -> fp16 -> bf16 in
        # vLLM). This save path expects model_config.dtype to be a concrete
        # "float16"/"bfloat16" that maps directly to a torch dtype; an
        # unexpected value (e.g. "auto") is out of scope and intentionally
        # raises via getattr rather than silently falling back.
        save_dtype = getattr(torch, self.model_config.dtype)

        # PEFT convention: keys are prefixed with "base_model.model." and the
        # module path matches what we will see on the loaded HF model. When
        # save_format='full_wrapper', the base weights and quantization_config
        # are remapped to the composite ``model.language_model.*`` namespace
        # (see _prepare_full_wrapper_quantized_save), so the adapter module
        # paths must be remapped the same way; otherwise the sidecar keeps the
        # text-only ``model.layers.*`` names and vLLM's full-wrapper loader
        # cannot match the adapter tensors to the base layers.
        remap = save_format == "full_wrapper"
        state_dict = {}
        for name, mod in lora_modules:
            adapter_name = (
                self._remap_text_only_module_name_to_full_wrapper(name) if remap else name
            )
            state_dict[f"base_model.model.{adapter_name}.lora_A.weight"] = (
                mod.lora_A.weight.detach().to("cpu", save_dtype).contiguous()
            )
            state_dict[f"base_model.model.{adapter_name}.lora_B.weight"] = (
                mod.lora_B.weight.detach().to("cpu", save_dtype).contiguous()
            )

        first = lora_modules[0][1]
        lora_r = int(first.lora_r)
        # scaling = alpha / r is stored as float; round-trip back to int alpha.
        lora_alpha = int(round(float(first.scaling) * float(first.lora_r)))
        lora_dropout = (
            float(first.dropout.p) if isinstance(first.dropout, torch.nn.Dropout) else 0.0
        )
        target_modules = sorted({name.rsplit(".", 1)[-1] for name, _ in lora_modules})

        adapter_config = {
            "peft_type": "LORA",
            "auto_mapping": None,
            "base_model_name_or_path": str(Path(save_directory).resolve()),
            "task_type": "CAUSAL_LM",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "modules_to_save": None,
            "init_lora_weights": True,
            "layers_to_transform": None,
            "layers_pattern": None,
            "revision": None,
        }

        adapter_dir = Path(save_directory) / LORA_ADAPTER_SUBDIR
        adapter_dir.mkdir(parents=True, exist_ok=True)
        _st_save_file(
            state_dict,
            str(adapter_dir / "adapter_model.safetensors"),
            metadata={"format": "pt"},
        )
        with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2, ensure_ascii=True)

        self.logger.info(
            "Saved LoRA adapter sidecar (%d layers) to %s",
            len(lora_modules),
            adapter_dir,
        )
        return True

    def _resolve_source_model_dir(self) -> Optional[str]:
        """Resolve the original model directory for auxiliary file copy.

        Returns the local directory of the source model used by this runner.
        If ``ModelConfig`` points at a Hugging Face Hub ID rather than a local
        directory, attempts to locate the snapshot via
        ``huggingface_hub.snapshot_download(local_files_only=True)``.

        Returns:
            The absolute path to the source model directory, or ``None`` if it
            could not be resolved (in which case auxiliary copying is skipped
            and a warning is logged by the caller).
        """
        src = self.model_config.get_model_id_or_path()
        if not src:
            return None
        if os.path.isdir(src):
            return src
        try:
            from huggingface_hub import snapshot_download

            return snapshot_download(src, local_files_only=True)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("Could not resolve source model dir for %s: %s", src, exc)
            return None

    # File patterns excluded from the auxiliary-config copy in
    # ``save_quantized_model``.  Weight tensors are written by HF
    # ``save_pretrained`` directly, so copying the originals would either
    # collide with the quantized weights or balloon the save directory.
    _AUX_COPY_EXCLUDE_FILES = frozenset(
        {
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        }
    )
    _AUX_COPY_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
    _AUX_COPY_INCLUDE_SUFFIXES = (".json", ".jinja")

    def _copy_auxiliary_files(self, src_dir: str, save_directory: str) -> int:
        """Copy auxiliary ``*.json`` / ``*.jinja`` files from ``src_dir``.

        Files already present in ``save_directory`` are left untouched so that
        the artifacts written by ``model.save_pretrained`` /
        ``tokenizer.save_pretrained`` (and the Gemma BOS post-processing) are
        never overwritten.  Weight tensors and weight index files are skipped.

        Args:
            src_dir: Resolved original model directory.
            save_directory: Destination directory.

        Returns:
            Number of files actually copied.
        """
        import shutil

        copied = 0
        try:
            entries = os.listdir(src_dir)
        except OSError as exc:
            self.logger.warning(
                "Failed to list source model dir %s for aux copy: %s", src_dir, exc
            )
            return 0

        for name in entries:
            src = os.path.join(src_dir, name)
            if not os.path.isfile(src):
                continue
            if name in self._AUX_COPY_EXCLUDE_FILES:
                continue
            lower = name.lower()
            if lower.endswith(self._AUX_COPY_WEIGHT_SUFFIXES):
                continue
            if not lower.endswith(self._AUX_COPY_INCLUDE_SUFFIXES):
                continue
            dst = os.path.join(save_directory, name)
            if os.path.exists(dst):
                # Don't clobber files already in the save directory.
                # Typically these were just written by
                # ``model.save_pretrained`` / ``tokenizer.save_pretrained``
                # (and possibly post-edited, e.g. the Gemma 4
                # ``add_bos_token`` patch on ``tokenizer_config.json``);
                # we deliberately keep that fresh copy instead of
                # overwriting it with the original-model file.  Logging
                # the skip simply makes the auxiliary-copy step easier
                # to follow alongside the ``Copied %s`` entries below.
                self.logger.info("Using existing %s in save directory", name)
                continue
            shutil.copy2(src, dst)
            copied += 1
            self.logger.info("Copied %s to save directory", name)
        return copied

    def save_quantized_model(
        self,
        save_directory: str,
        pack_weights: bool = True,
        save_format: str = "auto",
    ):
        """Save the quantized model to the specified directory

        If ``self.quantized_model`` is already set (e.g. after
        ``run_post_processes()``, or after loading a checkpoint and assigning it
        for a load -> post-process -> re-save flow), that in-place updated model
        is saved as-is so post-process results are preserved: its
        ``quantization_config`` is validated, any recorded
        ``onecomp_post_processes`` history is persisted to ``config.json``, and
        ``model_config`` is required (for the tokenizer).  Otherwise the base
        quantized model is built from ``quantizer.results`` via
        :meth:`create_quantized_model`.  The result is saved in
        HuggingFace-compatible safetensors format.

        If the selected model contains ``LoRAGPTQLinear`` wrappers, this method
        saves base weights with LoRA tensors excluded and additionally writes a
        PEFT-compatible LoRA adapter sidecar
        (``lora_adapter/adapter_model.safetensors`` +
        ``lora_adapter/adapter_config.json``).  The resulting directory can then
        be loaded back with
        :func:`onecomp.load_quantized_model` (which auto-detects the sidecar and
        re-wraps the layers) or served by vLLM via ``enable_lora=True``.

        Args:
            save_directory (str):
                The path to save the quantized model.
            pack_weights (bool):
                Whether to pack quantized weights for a more
                memory/storage-efficient representation.  When building from
                ``quantizer.results`` this controls the layout of the built
                model.  When saving an existing ``self.quantized_model``,
                packable unpacked GPTQ buffers are packed only in the export
                ``state_dict`` without mutating the in-memory model.
            save_format (str):
                One of ``"auto"``, ``"native"``, or ``"full_wrapper"``.
                ``"auto"``/``"native"`` save the model's own state_dict/config
                namespace as-is (after validating they agree); use this for
                all non-VLM models and it is the recommended default.
                ``"full_wrapper"`` is scoped to Qwen3.6 text-only checkpoints
                (``model_type`` ``qwen3_5_text`` / ``qwen3_5_moe_text``): it
                remaps them to the composite ``model.language_model.*``
                namespace that vLLM's full-wrapper VLM loader expects for
                ``Qwen3_5ForConditionalGeneration``-style serving. It is
                **not** a generic "save any VLM for vLLM" option — passing it
                for any other model (or any model whose original config isn't
                composite) raises ``RuntimeError``. Defaults to ``"auto"``.

        Examples:
            Single quantizer mode:

            >>> runner.save_quantized_model("./quantized_model")

            GPTQ + LoRA SFT:

            >>> runner = Runner(
            ...     model_config=model_config,
            ...     quantizer=GPTQ(wbits=4, groupsize=128),
            ...     post_processes=[PostProcessLoraSFT(data_files="train.jsonl")],
            ... )
            >>> runner.run()
            >>> runner.save_quantized_model("./quantized_model_lora")
        """
        logger = self.logger
        logger.info("Saving quantized model to %s", save_directory)

        if self.quantized_model is not None:
            logger.info("Using existing quantized model (post-process results preserved)")
            if self.model_config is None:
                raise RuntimeError(
                    "save_quantized_model with runner.quantized_model requires model_config."
                )
            model = self.quantized_model
            validate_quantized_model_config(
                model,
                "save_quantized_model",
            )
            tokenizer = self.model_config.load_tokenizer()
        else:
            # Disable GemLite when saving to avoid extra params in safetensors.
            model, tokenizer = self.create_quantized_model(
                pack_weights=pack_weights,
                use_gemlite=False,
            )

        # Save model and tokenizer
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        lora_modules = self._collect_lora_gptq_modules(model)
        gptq_layers = self._iter_gptq_export_layers(model, lora_modules)
        needs_export_state_dict = bool(lora_modules) or any(
            not getattr(layer, "_weight_is_packed", False) for _name, layer in gptq_layers
        )
        export_state_dict = None
        if needs_export_state_dict:
            export_state_dict = self._build_base_quantized_state_dict(
                model, lora_modules, pack_weights=pack_weights
            )

        # MoE models write fused expert tensors per-layer.  save_original_format=True
        # would run revert_weight_conversion (grouped_gemm) and collapse them into
        # shared gate_up_proj$/down_proj$ keys, so force it off for MoE checkpoints.
        # (fuse/strip of the experts themselves happens in
        # _prepare_model_for_quantized_save so save_format remapping stays consistent.)
        num_experts = _get_num_experts(model.config)
        extra_save_kwargs = {"save_original_format": False} if num_experts > 0 else {}

        orig_model_config_for_restore = model.config

        try:
            save_state_dict = self._prepare_model_for_quantized_save(
                model,
                save_format=save_format,
                state_dict=export_state_dict,
            )
            if save_state_dict is not None:
                model.save_pretrained(
                    save_directory, state_dict=save_state_dict, **extra_save_kwargs
                )
            else:
                model.save_pretrained(save_directory, **extra_save_kwargs)
        finally:
            model.config = orig_model_config_for_restore

        if num_experts > 0:
            if self.moe_quant_experts:
                from .utils.unfuse_moe import verify_saved_moe_quant_checkpoint

                n_experts = verify_saved_moe_quant_checkpoint(save_directory)
                logger.info(
                    "Saved checkpoint MoE (quant-experts): per-expert GPTQ layers=%d",
                    n_experts,
                )
            else:
                from .utils.unfuse_moe import verify_saved_moe_checkpoint

                n_layers = verify_saved_moe_checkpoint(save_directory)
                logger.info(
                    "Saved checkpoint MoE: per-layer gate_up_proj=%d, bad$ keys=[]",
                    n_layers,
                )

        tokenizer.save_pretrained(save_directory)
        if save_format == "full_wrapper":
            self._save_processor_files_if_available(save_directory)

        # Gemma 4 PT models require BOS token for coherent generation but the
        # upstream tokenizer_config.json omits add_bos_token.  Ensure it is
        # set so that vLLM (and other runtimes) prepend <bos> automatically.
        # See: https://github.com/vllm-project/vllm/issues/39827
        tc_path = Path(save_directory) / "tokenizer_config.json"
        if tc_path.exists():
            tc = json.loads(tc_path.read_text())
            if "add_bos_token" not in tc and tc.get("bos_token"):
                tc["add_bos_token"] = True
                tc_path.write_text(json.dumps(tc, indent=2, ensure_ascii=False) + "\n")
                logger.info("Set add_bos_token=true in tokenizer_config.json")

        # Copy auxiliary config / template files from the original model so the
        # save directory is self-contained for VLM inference and runtimes that
        # expect e.g. processor_config.json, preprocessor_config.json,
        # special_tokens_map.json, or chat_template.jinja next to weights.
        src_dir = self._resolve_source_model_dir()
        if src_dir and os.path.isdir(src_dir):
            self._copy_auxiliary_files(src_dir, save_directory)
        else:
            logger.warning("Source model dir not resolvable; skipping auxiliary file copy.")

        # LoRA sidecar: only written if selected model contains LoRAGPTQLinear.
        wrote_adapter = self._save_lora_adapter_sidecar(
            save_directory, model=model, save_format=save_format
        )
        if not wrote_adapter:
            # Remove any stale sidecar from a previous run so the directory is
            # self-consistent and load_quantized_model does not pick up an
            # adapter that no longer matches the saved base model.
            stale_adapter_dir = save_path / LORA_ADAPTER_SUBDIR
            if stale_adapter_dir.is_dir():
                for stale in ("adapter_model.safetensors", "adapter_config.json"):
                    stale_path = stale_adapter_dir / stale
                    if stale_path.exists():
                        stale_path.unlink()
                # Remove the (now-empty) subdirectory if nothing else lives there.
                try:
                    stale_adapter_dir.rmdir()
                except OSError:
                    pass
            # Also remove any top-level adapter files left by older versions of
            # this helper (previous layout put the sidecar directly in save_dir).
            for legacy in ("adapter_model.safetensors", "adapter_config.json"):
                legacy_path = save_path / legacy
                if legacy_path.exists():
                    legacy_path.unlink()

        logger.info(f"Quantized model saved to {save_directory}")
        return save_directory

    def save_quantized_model_pt(self, save_directory: str):
        """Save the quantized model as a PyTorch .pt file.

        This serializes the entire model object with ``torch.save``,
        preserving custom module types such as ``LoRAGPTQLinear``.  It is a
        legacy/alternative to :meth:`save_quantized_model`, which is preferred
        for all cases -- including LoRA post-processes, whose adapter is saved
        as a PEFT-compatible safetensors sidecar and is loadable by
        :func:`onecomp.load_quantized_model` and servable by vLLM.  Use this
        ``.pt`` method only when you specifically need a single serialized
        model object; note that loading it requires
        ``allow_unsafe_deserialization=True`` (see
        :func:`onecomp.load_quantized_model_pt`).

        The saved directory contains:
        - ``model.pt``: The model (``torch.save``)
        - Tokenizer files (via ``save_pretrained``)

        Args:
            save_directory (str):
                The path to save the model.

        See Also:
            :func:`onecomp.load_quantized_model_pt` to load models
            saved by this method.

        Examples:
            >>> runner.run()  # with post_processes=[PostProcessLoraSFT(...)]
            >>> runner.save_quantized_model_pt("./quantized_model_lora")
        """
        logger = self.logger

        if self.quantized_model is not None:
            model = self.quantized_model
        else:
            model, _ = self.create_quantized_model(pack_weights=False, use_gemlite=False)

        tokenizer = self.model_config.load_tokenizer()

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        model_path = save_path / "model.pt"
        logger.info("Saving quantized model (torch.save) to %s", model_path)
        torch.save(model, str(model_path))
        tokenizer.save_pretrained(save_directory)

        logger.info("Quantized model saved to %s", save_directory)
        return save_directory

    def analyze_cumulative_error(
        self,
        layer_keywords=None,
        plot_path=None,
        json_path=None,
        batch_keywords=False,
        quantizer=None,
    ):
        """Analyze cumulative quantization error for each linear layer.

        Cumulative error: ||W_orig X_orig - W_quant X_quant||^2_F

        Note:
            Must be used after calling the run() method.

        Args:
            layer_keywords: List of keywords to filter layers.
                Each keyword is analyzed and plotted separately.
                Default: ["mlp.down_proj"]
                Example: ["q_proj", "k_proj"]
            plot_path: Base path to save plots. Keyword is inserted before extension.
                Example: "error.png" -> "error_mlp.down_proj.png"
            json_path: Path to save results as JSON file.
                Example: "cumulative_error.json"
            batch_keywords: If True, process all keywords in a single forward pass.
                This is faster but uses more CPU memory because all target layers'
                outputs are stored simultaneously.
                If False (default), process each keyword separately with
                model reload per keyword. This uses less CPU memory but
                incurs overhead from repeated model loading and forward passes.
            quantizer (Quantizer, optional):
                The quantizer. Uses self.quantizer if None.
                Specify explicitly when using quantizers mode.

        Returns:
            dict: keyword -> {layer_name -> cumulative squared error}

        Examples:
            Single quantizer mode:

            >>> results = runner.analyze_cumulative_error()
            >>> results = runner.analyze_cumulative_error(plot_path="cumulative_error.png")

            Multiple quantizers mode:

            >>> results = runner.analyze_cumulative_error(quantizer=gptq)
        """
        # Lazy import: load submodule only when needed
        # pylint: disable-next=import-outside-toplevel
        # pylint: disable-next=import-outside-toplevel
        from .analyzer.cumulative_error import analyze_cumulative_error as _analyze
        from .analyzer.cumulative_error import plot_cumulative_error as _plot

        logger = self.logger

        # TODO: Support analyze_cumulative_error with self.quantized_model
        #       (use post-processed quantized model instead of quantizer.results)
        if self.quantized_model is not None:
            logger.error(
                "analyze_cumulative_error is not yet supported when "
                "post_processes have been applied (self.quantized_model is set). "
                "This will be implemented in a future version."
            )
            return {}

        if quantizer is None:
            quantizer = self.quantizer
        if quantizer is None:
            logger.warning(
                "analyze_cumulative_error: 'quantizer' is None. "
                "Please specify a quantizer explicitly."
            )
            return {}

        # Use default keywords if not specified
        if layer_keywords is None:
            layer_keywords = ["mlp.down_proj"]

        all_results = {}

        if batch_keywords:
            # All keywords in a single forward pass (faster, more CPU memory)
            logger.info(
                "Analyzing cumulative error in batch mode for keywords: %s",
                layer_keywords,
            )
            # Release fragmented GPU memory from previous operations (e.g., run())
            gc.collect()
            empty_cache(self.model_config.get_device())

            model = self.model_config.load_model()
            input_device = next(model.parameters()).device
            inputs = self.prepare_calibration_dataset(input_device, model=model)

            combined_results = _analyze(model, inputs, quantizer.results, layer_keywords)

            # Separate results by keyword
            for keyword in layer_keywords:
                all_results[keyword] = {
                    name: error_dict
                    for name, error_dict in combined_results.items()
                    if keyword in name
                }
        else:
            # Process each keyword separately (less CPU memory, more overhead)
            for keyword in layer_keywords:
                logger.info(
                    "Analyzing cumulative error for keyword: %s (reloading model)",
                    keyword,
                )
                # Release fragmented GPU memory from previous operations (e.g., run())
                gc.collect()
                empty_cache(self.model_config.get_device())

                model = self.model_config.load_model()
                input_device = next(model.parameters()).device
                inputs = self.prepare_calibration_dataset(input_device, model=model)

                keyword_results = _analyze(model, inputs, quantizer.results, [keyword])
                all_results[keyword] = {
                    name: error_dict
                    for name, error_dict in keyword_results.items()
                    if keyword in name
                }

                del model, inputs

        # Plot and save
        for keyword in layer_keywords:
            if plot_path:
                # Insert keyword into filename: "error.png" -> "error_keyword.png"
                base, ext = os.path.splitext(plot_path)
                keyword_safe = keyword.replace(".", "_")
                keyword_plot_path = f"{base}_{keyword_safe}{ext}"
                _plot(all_results[keyword], keyword_plot_path, [keyword])

        if json_path:
            # Exclude local_mean_squared_error from JSON output
            json_results = {}
            for keyword, layer_results in all_results.items():
                json_results[keyword] = {
                    layer_name: {
                        "squared_error": error_dict["squared_error"],
                        "mean_squared_error": error_dict["mean_squared_error"],
                    }
                    for layer_name, error_dict in layer_results.items()
                }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)

        return all_results

    def _save_processor_files_if_available(self, save_directory: str) -> None:
        """Save processor / image processor files required by full-wrapper VLM checkpoints.

        vLLM loads multimodal/full-wrapper checkpoints through Transformers
        processor utilities.  For VLM configs, tokenizer files alone are not
        enough: preprocessor_config.json is required for the image processor.
        """
        model_id_or_path = self.model_config.get_model_id_or_path()

        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                model_id_or_path,
                trust_remote_code=True,
            )
            processor.save_pretrained(save_directory)
            self.logger.info("Saved processor files from %s", model_id_or_path)
        except Exception as exc:
            self.logger.warning(
                "Could not save AutoProcessor files from %s: %s",
                model_id_or_path,
                exc,
            )

        try:
            from transformers import AutoImageProcessor

            image_processor = AutoImageProcessor.from_pretrained(
                model_id_or_path,
                trust_remote_code=True,
            )
            image_processor.save_pretrained(save_directory)
            self.logger.info("Saved image processor files from %s", model_id_or_path)
        except Exception as exc:
            self.logger.warning(
                "Could not save AutoImageProcessor files from %s: %s",
                model_id_or_path,
                exc,
            )

    # ------------------------------------------------------------------
    # Save namespace helpers
    # ------------------------------------------------------------------

    def _is_quantized_module(self, module) -> bool:
        """Return True for OneCompression quantized inference modules.

        Keep this name-based to avoid import cycles and to remain extensible
        for future quantizers.
        """
        return module.__class__.__name__ in {
            "GPTQLinear",
            "DoubleBinaryLinear",
        }

    def _collect_quantized_module_names(self, model) -> list[str]:
        """Collect actual quantized module names from the model to be saved.

        This is more reliable than quantizer.results.keys() because wrapper
        models may expose different names during quantization and save.
        """
        return sorted(
            name for name, module in model.named_modules() if self._is_quantized_module(module)
        )

    def _detect_weight_namespace(self, model) -> str:
        """Detect whether state_dict is text-only or full-wrapper style."""
        keys = list(model.state_dict().keys())

        if any(k.startswith("model.language_model.model.layers.") for k in keys):
            return "full_language_model"

        if any(k.startswith("model.language_model.layers.") for k in keys):
            return "full_language_model"

        if any(k.startswith("language_model.layers.") for k in keys):
            return "full_language_model"

        if any(".language_model.layers." in k for k in keys):
            return "full_language_model"

        if any(k.startswith("model.layers.") for k in keys):
            return "text_only"

        return "unknown"

    def _detect_config_namespace(self, model) -> str:
        """Detect whether config is text-only or full/composite style."""
        cfg = model.config
        model_type = getattr(cfg, "model_type", None)

        if model_type in {
            "qwen3_5_text",
            "qwen3_5_moe_text",
        }:
            return "text_only"

        if model_type in {
            "qwen3_5",
            "qwen3_5_moe",
        }:
            return "full_language_model"

        if getattr(cfg, "text_config", None) is not None:
            return "full_language_model"

        return "unknown"

    def _restore_original_composite_config_if_needed(self, model) -> None:
        """Restore outer/composite config if state_dict uses wrapper prefix.

        Example bad checkpoint:
          config:
            model_type = qwen3_5_text
            architectures = Qwen3_5ForCausalLM

          state_dict:
            model.language_model.layers.0....

        For such a model, save the original outer config instead:
          model_type = qwen3_5
          architectures = Qwen3_5ForConditionalGeneration
          text_config = {...}
          vision_config = {...}
        """
        cfg_ns = self._detect_config_namespace(model)
        sd_ns = self._detect_weight_namespace(model)

        if cfg_ns != "text_only" or sd_ns != "full_language_model":
            return

        orig_config = self.model_config.load_config()

        # Only restore when the original config is actually composite.
        if getattr(orig_config, "text_config", None) is None:
            return

        quant_config = getattr(model.config, "quantization_config", None)

        self.logger.warning(
            "Detected text-only config with full language_model state_dict. "
            "Restoring original composite config before save_pretrained()."
        )

        model.config = copy.deepcopy(orig_config)

        if quant_config is not None:
            model.config.quantization_config = quant_config

    def _assert_config_state_dict_namespace_consistent(self, model) -> None:
        """Fail before saving a checkpoint whose config and state_dict disagree."""
        cfg_ns = self._detect_config_namespace(model)
        sd_ns = self._detect_weight_namespace(model)

        if cfg_ns == "unknown" or sd_ns == "unknown":
            return

        if cfg_ns != sd_ns:
            raise RuntimeError(
                "config/state_dict namespace mismatch before save_pretrained().\n"
                f"  config namespace: {cfg_ns}\n"
                f"  state_dict namespace: {sd_ns}\n"
                f"  config model_type: {getattr(model.config, 'model_type', None)}\n"
                "This would create a checkpoint that may load with missing or "
                "zero-filled quantized buffers."
            )

    def _prepare_model_for_quantized_save(
        self,
        model,
        *,
        save_format: str,
        state_dict: dict | None = None,
    ) -> dict | None:
        """Prepare model.config and optional state_dict for save_pretrained.

        Returns:
            None:
                Use model.state_dict() as-is.

            dict:
                Pass this remapped state_dict to save_pretrained().
        """
        if save_format not in {"auto", "native", "full_wrapper"}:
            raise ValueError(
                f"Unknown save_format={save_format!r}. "
                "Expected one of: auto, native, full_wrapper."
            )

        # MoE models: fuse per-expert (dequantized) tensors back to the fused 3D
        # layout vLLM expects, and drop per-expert entries from quant_config so the
        # namespace assertions below (and the full_wrapper remap) do not flag the
        # now-absent per-expert modules.  Idempotent: a no-op when the model is
        # already fused/stripped (create_quantized_model handles this on the normal
        # path); this is the safety net for the self.quantized_model path.
        if _get_num_experts(model.config) > 0 and not self.moe_quant_experts:
            from .utils.unfuse_moe import (
                fuse_moe_experts,
                strip_moe_experts_from_quant_config,
            )

            if fuse_moe_experts(model, self.logger):
                self.logger.info("Fused MoE expert tensors before save")
            qcfg = getattr(model.config, "quantization_config", None)
            if isinstance(qcfg, dict):
                strip_moe_experts_from_quant_config(qcfg)

        if save_format in {"auto", "native"}:
            self._restore_original_composite_config_if_needed(model)
            self._assert_config_state_dict_namespace_consistent(model)
            self._assert_quant_config_matches_model_namespace(model)
            return state_dict

        # save_format == "full_wrapper"
        return self._prepare_full_wrapper_quantized_save(model, state_dict=state_dict)

    def _assert_quant_config_matches_model_namespace(self, model) -> None:
        quant_config = getattr(model.config, "quantization_config", None)
        if not quant_config:
            return

        names = quant_config.get("modules_in_block_to_quantize") or []
        names = [n for n in names if isinstance(n, str)]
        if not names:
            return

        model_module_names = set(dict(model.named_modules()).keys())

        missing = [n for n in names if n not in model_module_names]

        if missing:
            raise RuntimeError(
                "quantization_config contains module names that do not exist "
                "in the model being saved.\n"
                f"missing={missing[:50]}"
            )

    def _prepare_full_wrapper_quantized_save(self, model, state_dict: dict | None = None) -> dict:
        """Prepare full-wrapper checkpoint for vLLM-like runtimes.

        This does not mutate tensor objects; it only remaps state_dict keys and
        replaces model.config with the original composite config.
        """
        cfg_ns = self._detect_config_namespace(model)
        sd_ns = self._detect_weight_namespace(model)

        if cfg_ns == "full_language_model" and sd_ns == "full_language_model":
            self._assert_config_state_dict_namespace_consistent(model)
            self._assert_quant_config_matches_model_namespace(model)
            return state_dict

        orig_config = self.model_config.load_config()

        if getattr(orig_config, "text_config", None) is None:
            raise RuntimeError(
                "save_format='full_wrapper' was requested, but the original "
                "model config is not composite and has no text_config."
            )

        if cfg_ns != "text_only" or sd_ns != "text_only":
            raise RuntimeError(
                "save_format='full_wrapper' currently supports converting "
                "consistent text-only checkpoints only.\n"
                f"config namespace: {cfg_ns}\n"
                f"state_dict namespace: {sd_ns}"
            )

        old_quant_config = getattr(model.config, "quantization_config", None)
        if old_quant_config is None:
            raise RuntimeError("model.config has no quantization_config")

        full_quant_config = self._remap_text_only_quant_config_to_full_wrapper(old_quant_config)

        # Replace config with original composite config.
        model.config = copy.deepcopy(orig_config)
        model.config.quantization_config = full_quant_config

        # Remap tensors to match the full-wrapper config.
        source_state_dict = state_dict if state_dict is not None else model.state_dict()
        full_state_dict = self._remap_text_only_state_dict_to_full_wrapper(source_state_dict)
        full_state_dict = self._strip_moe_expert_g_idx_for_vllm(full_state_dict, full_quant_config)

        self.logger.info(
            "Prepared full-wrapper quantized save: model_type=%s, first_quantized=%s",
            getattr(model.config, "model_type", None),
            full_quant_config.get("modules_in_block_to_quantize", ["<none>"])[0],
        )

        return full_state_dict

    @staticmethod
    def _remap_text_only_quant_config_to_full_wrapper(quant_config: dict) -> dict:
        quant_config = copy.deepcopy(quant_config)

        def remap_name(name: str) -> str:
            if name.startswith("model.layers."):
                return "model.language_model.layers." + name[len("model.layers.") :]
            if name.startswith("model.embed_tokens."):
                return "model.language_model.embed_tokens." + name[len("model.embed_tokens.") :]
            if name.startswith("model.norm."):
                return "model.language_model.norm." + name[len("model.norm.") :]
            return name

        for key in ("modules_in_block_to_quantize", "quantized_layer_names"):
            names = quant_config.get(key)
            if isinstance(names, list):
                quant_config[key] = [remap_name(n) if isinstance(n, str) else n for n in names]

        return quant_config

    @staticmethod
    def _remap_text_only_module_name_to_full_wrapper(name: str) -> str:
        """Remap a single text-only module/tensor path to the composite
        ``model.language_model.*`` namespace used by the full-wrapper save.

        Example:
          model.layers.0...      -> model.language_model.layers.0...
          model.embed_tokens...  -> model.language_model.embed_tokens...
          model.norm...          -> model.language_model.norm...

        Top-level ``lm_head.*`` is left unchanged.
        """
        if name.startswith("lm_head."):
            return name
        if name.startswith("model.") and not name.startswith("model.language_model."):
            return "model.language_model." + name[len("model.") :]
        return name

    @staticmethod
    def _remap_text_only_state_dict_to_full_wrapper(state_dict: dict) -> dict:
        """Remap text-only CausalLM state_dict to composite language_model prefix.

        Example:
          model.layers.0...      -> model.language_model.layers.0...
          model.embed_tokens...  -> model.language_model.embed_tokens...
          model.norm...          -> model.language_model.norm...

        Keep top-level lm_head.* unchanged.
        """
        remapped = {}

        for key, tensor in state_dict.items():
            new_key = Runner._remap_text_only_module_name_to_full_wrapper(key)

            if new_key in remapped:
                raise RuntimeError(
                    f"State dict key collision during full-wrapper remap: " f"{key} -> {new_key}"
                )

            remapped[new_key] = tensor

        return remapped

    def _strip_moe_expert_g_idx_for_vllm(self, state_dict: dict, quant_config: dict) -> dict:
        """Drop per-expert GPTQ ``g_idx`` buffers from a vLLM-facing export.

        vLLM's GPTQ MoE kernel (MoeWNA16) has no ``g_idx`` parameter, so an
        unmapped ``g_idx`` key crashes weight loading. Safe to drop only
        when ``desc_act``/``actorder`` is disabled, since ``g_idx`` is then
        just the trivial ``arange(in_features) // group_size`` grouping the
        kernel already assumes; raises otherwise.
        """
        from vllm_plugins.utils.module import is_moe_expert_g_idx_key

        moe_g_idx_keys = [key for key in state_dict if is_moe_expert_g_idx_key(key)]
        if not moe_g_idx_keys:
            return state_dict

        desc_act = get_quant_param(quant_config, "desc_act", "actorder", default=False)
        if desc_act:
            raise RuntimeError(
                "Cannot export an actorder/desc_act GPTQ MoE checkpoint for "
                "vLLM: vLLM's GPTQ FusedMoE kernel (MoeWNA16Method) has no "
                "g_idx support, so activation-order MoE quantization is not "
                "servable by vLLM yet."
            )

        state_dict = dict(state_dict)
        for key in moe_g_idx_keys:
            del state_dict[key]
        self.logger.info(
            "Dropped %d trivial MoE expert g_idx buffer(s) unsupported by "
            "vLLM's GPTQ FusedMoE kernel",
            len(moe_g_idx_keys),
        )
        return state_dict
