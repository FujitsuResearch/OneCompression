"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

import gc
from logging import getLogger
import json
import os
import time

import torch

from .model_config import ModelConfig
from .qep import QEPConfig
from .quantizer import Quantizer
from .quantizer.dbf import DBF
from .quantizer.gptq import GPTQ
from .utils import calculate_perplexity
from .utils import calculate_accuracy as calc_accuracy
from .utils import prepare_calibration_dataset as prepare_calib_dataset
from .__version__ import __version__
from .log import setup_logger

from pathlib import Path


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
        calibration_dataset=None,
        max_length=512,
        num_calibration_samples=128,
        quantizer=None,
        quantizers=None,
        qep=False,
        qep_config=None,
        calibration_strategy="drop_rand",
        calibration_seed=0,
        multi_gpu=False,
        gpu_ids=None,
        calibration_batch_size=None,
        num_layers_per_group=7,
    ):
        """__init__ method

        Args:
            model_config (ModelConfig):
            calibration_dataset (datasets.Dataset):
            max_length (int):
                The maximum length of the input sequence.
            num_calibration_samples (int):
                The number of calibration samples to use when loading default dataset.
            quantizer (Quantizer):
                The quantizer to use. Specify either ``quantizer`` or
                ``quantizers``, not both.
            quantizers (list[Quantizer]):
                Specify multiple quantizers. When used with
                calibration_batch_size, the X^T X accumulation is shared,
                reducing the forward pass to a single execution.
                Specify either ``quantizer`` or ``quantizers``, not both.
                Currently, this is only available when
                ``calibration_batch_size`` is set and ``qep=False``.
            qep (bool):
                Whether to use QEP.
            qep_config (QEPConfig or None):
                Configuration for QEP. If None and ``qep=True``,
                a default ``QEPConfig()`` is used.
            calibration_strategy (str):
                Strategy for preparing calibration inputs.
                Default is "drop_rand".

                Available strategies:
                - "concat_chunk":
                    Concatenate all texts, tokenize once, and split into
                    fixed-length chunks (max_length).
                    Creates as many chunks as possible from the data.
                - "concat_chunk_align":
                    Same as concat_chunk, but adjusts the number of loaded
                    samples so that num_chunks == num_calibration_samples.
                    This ensures consistent token counts across experiments.
                - "drop_head":
                    No cross-document mixing. Tokenize each document
                    independently; drop samples with token length < max_length;
                    take the head window (first max_length tokens).
                - "drop_rand":
                    Same as above, but take a random window of length max_length
                    from each long document (reproducible with calibration_seed).
            calibration_seed (int):
                Random seed used by some calibration strategies
                (e.g., "drop_rand").
            multi_gpu (bool):
                Whether to use multi-GPU for layer-wise parallel quantization.
                Default is False.
            gpu_ids (list[int]):
                List of GPU IDs to use for multi-GPU quantization.
                If None and multi_gpu is True, all available GPUs will be used.
            calibration_batch_size (int or None):
                Batch size (number of sentences) for chunked calibration
                forward passes. Default is None (all calibration data in
                a single forward pass).
                When set to a positive integer (e.g., 128), calibration
                data is split into chunks of this size and forwarded in
                multiple passes to reduce GPU memory usage. The necessary
                statistics (e.g., X^T X for Hessian-based methods) are
                accumulated across chunks. This is mathematically exact,
                not an approximation.
            num_layers_per_group (int):
                Number of layers to process simultaneously in chunked
                calibration mode. Default is 7 (one Transformer block
                for Llama-like architectures: q,k,v,o,gate,up,down).
                Controls the trade-off between CPU memory usage for
                X^T X storage and the number of forward passes required.
                Only used when calibration_batch_size is set.

        Examples:
            Chunked calibration with GPTQ (large-scale calibration data):

            >>> from onecomp import Runner, ModelConfig
            >>> from onecomp.quantizer.gptq import GPTQ
            >>> model_config = ModelConfig(
            ...     model_id_or_path="meta-llama/Llama-2-7b-hf"
            ... )
            >>> quantizer = GPTQ(wbits=4, groupsize=128)
            >>> runner = Runner(
            ...     model_config=model_config,
            ...     quantizer=quantizer,
            ...     max_length=2048,
            ...     num_calibration_samples=1024,
            ...     calibration_batch_size=128,  # Forward 128 sentences at a time
            ... )
            >>> runner.run()

            With custom num_layers_per_group:

            >>> # When memory is sufficient: process 2 blocks (14 layers) simultaneously
            >>> runner = Runner(
            ...     model_config=model_config,
            ...     quantizer=quantizer,
            ...     max_length=2048,
            ...     num_calibration_samples=1024,
            ...     calibration_batch_size=128,
            ...     num_layers_per_group=14,
            ... )
            >>> runner.run()

            Multiple quantizers (benchmark comparison):

            >>> from onecomp.quantizer.gptq import GPTQ
            >>> from onecomp.quantizer.jointq import JointQ
            >>> gptq = GPTQ(wbits=4, groupsize=128, calc_quant_error=True)
            >>> jointq = JointQ(bits=4, group_size=128, calc_quant_error=True,
            ...                 device=torch.device(0))
            >>> runner = Runner(
            ...     model_config=model_config,
            ...     quantizers=[gptq, jointq],
            ...     max_length=2048,
            ...     num_calibration_samples=1024,
            ...     calibration_batch_size=128,
            ... )
            >>> runner.run()
            >>> # Results are stored in gptq.results and jointq.results respectively
        """

        self.model_config = model_config
        self.calibration_dataset = calibration_dataset
        self.max_length = max_length
        self.num_calibration_samples = num_calibration_samples
        self.logger = getLogger(__name__)
        self.quantizer = quantizer
        self.quantizers = quantizers
        self.qep = qep
        self.calibration_strategy = calibration_strategy
        self.calibration_seed = calibration_seed
        self.multi_gpu = multi_gpu
        self.gpu_ids = gpu_ids
        self.calibration_batch_size = calibration_batch_size
        self.num_layers_per_group = num_layers_per_group
        if qep:
            self.qep_config = qep_config if qep_config is not None else QEPConfig()

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

        ===========  ====  ==========  ======================
        quantizers   qep   multi_gpu   calibration_batch_size
        ===========  ====  ==========  ======================
        Specified    False False       Specified
        None         True  False       None
        None         False True        None
        None         False False       Specified
        None         False False       None
        ===========  ====  ==========  ======================

        Note:
            ``multi_gpu=True`` requires a quantizer with ``flag_calibration=True``.

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
        if self.quantizers is not None:
            # quantizers mode: qep=False, multi_gpu=False, calibration_batch_size required
            if self.qep:
                raise ValueError("'quantizers' cannot be used with qep=True.")
            if self.multi_gpu:
                raise ValueError("'quantizers' cannot be used with multi_gpu=True.")
            if self.calibration_batch_size is None:
                raise ValueError("'quantizers' requires 'calibration_batch_size' to be set.")
        else:
            # Single quantizer mode: combination check
            if self.qep and self.multi_gpu:
                raise ValueError("'qep' and 'multi_gpu' cannot be used together.")
            if self.qep and self.calibration_batch_size is not None:
                raise ValueError("'qep' cannot be used with 'calibration_batch_size'.")
            if self.multi_gpu and self.calibration_batch_size is not None:
                raise ValueError("'multi_gpu' cannot be used with 'calibration_batch_size'.")
            if self.multi_gpu and not self.quantizer.flag_calibration:
                raise ValueError("'multi_gpu' requires a quantizer with flag_calibration=True.")

    def run(self):
        """Execute quantization (and related) processing"""

        start_time = time.time()

        logger = self.logger
        logger.info("OneComp version: %s", __version__)
        logger.info("Model: %s", self.model_config.get_model_id_or_path())
        logger.info("Start the run method of Runner class")

        logger.info("Checking the settings...")
        self.check()

        if self.qep:
            logger.info("Start quantization with error propagation (QEP)")
            self.quantize_with_qep()
        else:
            logger.info("Start quantization")
            self.quantize()

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
        wbits: int = 4,
        groupsize: int = 128,
        device: str = "cuda:0",
        qep: bool = True,
        evaluate: bool = True,
        save_dir: str = "auto",
        **kwargs,
    ):
        """One-liner quantization with sensible defaults.

        Sets up ModelConfig, GPTQ quantizer, and QEP, then runs quantization.
        Optionally evaluates perplexity and accuracy, and saves the
        quantized model.

        Args:
            model_id (str): Hugging Face model ID or local path.
            wbits (int): Quantization bit width (default: 4).
            groupsize (int): GPTQ group size (default: 128).
                Use -1 to disable grouping.
            device (str): Device to place the model on (default: "cuda:0").
            qep (bool): Whether to use QEP (default: True).
            evaluate (bool): Whether to calculate perplexity and
                accuracy after quantization (default: True).
            save_dir (str or None): Directory to save the quantized model.
                ``"auto"`` (default) derives the path from model_id
                (e.g., ``"TinyLlama-1.1B-intermediate-step-1431k-3T-gptq-4bit"``).
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
        """
        setup_logger()
        logger = getLogger(__name__)

        if save_dir == "auto":
            model_name = model_id.rstrip("/").split("/")[-1]
            save_dir = f"{model_name}-gptq-{wbits}bit"

        model_config = ModelConfig(model_id=model_id, device=device)
        quantizer = GPTQ(wbits=wbits, groupsize=groupsize, **kwargs)
        runner = cls(model_config=model_config, quantizer=quantizer, qep=qep)
        runner.run()

        if evaluate:
            original_ppl, quantized_ppl = runner.calculate_perplexity()
            logger.info("Original model perplexity: %s", original_ppl)
            logger.info("Quantized model perplexity: %s", quantized_ppl)

            original_acc, quantized_acc = runner.calculate_accuracy()
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
        elif self.calibration_batch_size is not None:
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
        inputs = self.prepare_calibration_dataset(input_device)

        # Setup the quantizer
        self.quantizer.setup(model)

        # Register hooks to all linear layers
        handles = []
        for module in self.quantizer.module_to_name.keys():
            handle = module.register_forward_hook(self.quantizer.quantize)
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
            calibration_dataset=self.calibration_dataset,
            max_length=self.max_length,
            num_calibration_samples=self.num_calibration_samples,
            calibration_strategy=self.calibration_strategy,
            calibration_seed=self.calibration_seed,
            calibration_batch_size=self.calibration_batch_size,
            num_layers_per_group=self.num_layers_per_group,
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
            calibration_dataset=self.calibration_dataset,
            max_length=self.max_length,
            num_calibration_samples=self.num_calibration_samples,
            calibration_strategy=self.calibration_strategy,
            calibration_seed=self.calibration_seed,
            gpu_ids=self.gpu_ids,
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
        for module in self.quantizer.module_to_name.keys():
            self.quantizer.quantize(module, None, None)

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
            calibration_dataset=self.calibration_dataset,
            max_length=self.max_length,
            num_calibration_samples=self.num_calibration_samples,
            calibration_strategy=self.calibration_strategy,
            calibration_seed=self.calibration_seed,
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
        inputs = self.prepare_calibration_dataset(input_device)

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

    def prepare_calibration_dataset(self, device):
        """Prepare calibration data for quantization methods such as GPTQ.

        See utils.calibration.prepare_calibration_dataset for details.

        Args:
            device (torch.device): Device to place tensors on (CPU or GPU)

        Returns:
            dict: Input dictionary for the model
                - "input_ids": tensor of shape (num_chunks, max_length)
                - "attention_mask": tensor of shape (num_chunks, max_length)
        """
        tokenizer = self.model_config.load_tokenizer()

        return prepare_calib_dataset(
            tokenizer=tokenizer,
            device=device,
            calibration_dataset=self.calibration_dataset,
            max_length=self.max_length,
            num_calibration_samples=self.num_calibration_samples,
            strategy=self.calibration_strategy,
            seed=self.calibration_seed,
            logger=self.logger,
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

        statistics = {
            key: {
                "quantization_time": result.quantization_time,
                "output_squared_error": result.output_squared_error,
                "mean_output_squared_error": result.mean_output_squared_error,
                "weight_squared_error": result.weight_squared_error,
                "mean_weight_squared_error": result.mean_weight_squared_error,
                "relative_output_squared_error": result.relative_output_squared_error,
                "relative_weight_squared_error": result.relative_weight_squared_error,
            }
            for key, result in quantizer.results.items()
        }

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

    def calculate_perplexity(
        self,
        original_model=True,
        quantized_model=True,
        dataset_name="wikitext",
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
            tuple: (original_ppl, quantized_ppl)

        Examples:
            Single quantizer mode:

            >>> original_ppl, quantized_ppl = runner.calculate_perplexity()

            Multiple quantizers mode:

            >>> original_ppl, quantized_ppl = runner.calculate_perplexity(
            ...     quantizer=gptq
            ... )
        """

        logger = self.logger

        if quantizer is None:
            quantizer = self.quantizer
        if quantizer is None:
            logger.warning(
                "calculate_perplexity: 'quantizer' is None. "
                "Please specify a quantizer explicitly."
            )
            return None, None

        model = self.model_config.load_model()
        tokenizer = self.model_config.load_tokenizer()
        original_ppl = None
        quantized_ppl = None

        if original_model:
            original_ppl = calculate_perplexity(
                model=model,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                max_samples=max_samples,
                max_length=max_length,
                stride=stride,
            )

        if quantized_model:
            self.update_model_weights(model, quantizer=quantizer)
            quantized_ppl = calculate_perplexity(
                model=model,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                max_samples=max_samples,
                max_length=max_length,
                stride=stride,
            )

        return original_ppl, quantized_ppl

    def benchmark_perplexity(
        self,
        original_model=True,
        dataset_name="wikitext",
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
            - ``quantizer.name``: PPL for each quantizer

        Examples:
            >>> runner.run()
            >>> ppl_dict = runner.benchmark_perplexity()
            >>> print(ppl_dict)
            {'original': 5.47, 'GPTQ': 5.72, 'JointQ': 5.68}

            Specify quantizers explicitly:

            >>> ppl_dict = runner.benchmark_perplexity(quantizers=[gptq, jointq])
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

            orig_ppl, quant_ppl = self.calculate_perplexity(
                original_model=calc_original,
                quantized_model=True,
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

            ppl_dict[q.name] = quant_ppl
            logger.info("%s perplexity: %s", q.name, quant_ppl)

        return ppl_dict

    def calculate_accuracy(
        self,
        original_model=True,
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
            tuple: (original_acc, quantized_acc)

        Examples:
            Single quantizer mode:

            >>> original_acc, quantized_acc = runner.calculate_accuracy()

            Multiple quantizers mode:

            >>> original_acc, quantized_acc = runner.calculate_accuracy(
            ...     quantizer=gptq
            ... )
        """

        logger = self.logger

        if quantizer is None:
            quantizer = self.quantizer
        if quantizer is None:
            logger.warning(
                "calculate_accuracy: 'quantizer' is None. "
                "Please specify a quantizer explicitly."
            )
            return None, None

        model = self.model_config.load_model()
        tokenizer = self.model_config.load_tokenizer()
        original_acc = None
        quantized_acc = None

        if original_model:
            original_acc = calc_accuracy(
                model=model,
                tokenizer=tokenizer,
                tasks=tasks,
                batch_size=batch_size,
                num_fewshot=num_fewshot,
                display_results=display_results,
            )

        if quantized_model:
            self.update_model_weights(model, quantizer=quantizer)
            quantized_acc = calc_accuracy(
                model=model,
                tokenizer=tokenizer,
                tasks=tasks,
                batch_size=batch_size,
                num_fewshot=num_fewshot,
                display_results=display_results,
            )

        return original_acc, quantized_acc

    def benchmark_accuracy(
        self,
        original_model=True,
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
            - ``quantizer.name``: Accuracy for each quantizer

        Examples:
            >>> runner.run()
            >>> acc_dict = runner.benchmark_accuracy()
            >>> print(acc_dict)
            {'original': {...}, 'GPTQ': {...}, 'JointQ': {...}}

            Specify quantizers explicitly:

            >>> acc_dict = runner.benchmark_accuracy(quantizers=[gptq, jointq])
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

            orig_acc, quant_acc = self.calculate_accuracy(
                original_model=calc_original,
                quantized_model=True,
                tasks=tasks,
                batch_size=batch_size,
                num_fewshot=num_fewshot,
                display_results=display_results,
                quantizer=q,
            )

            if calc_original:
                acc_dict["original"] = orig_acc
                logger.info("Original accuracy: %s", orig_acc)

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

        model = self.model_config.load_model()
        tokenizer = self.model_config.load_tokenizer()

        self.update_model_weights(model, quantizer=quantizer)

        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

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
                    quantizer.results[name].dequantized_weight.to(device).to(dtype)
                )
                logger.debug("Updated the model weights for layer: %s", name)

    # ========================================
    # Unified Save/Load Methods (Using quantizer.results)
    # ========================================

    def save_quantized_model(self, save_directory: str, pack_weights: bool = True):
        logger = self.logger
        logger.info("Saving quantized model to %s", save_directory)

        # Delegate save config to quantizer (extensible via override)
        quant_config = self.quantizer.get_quant_config()

        # Load base model
        model = self.model_config.load_model()
        tokenizer = self.model_config.load_tokenizer()

        # Replace Linear layers with quantized layers using quantizer.results
        logger.info("Replacing Linear layers with quantized inference layers...")
        self.quantizer.apply_results_to_model(model, pack_weights=pack_weights)

        # Build modules_in_block_to_quantize from actually-quantized layer names.
        quantized_names = sorted(self.quantizer.results.keys())
        modules_in_block = list(quantized_names)
        quant_config["modules_in_block_to_quantize"] = modules_in_block
        quant_config["quantized_layer_names"] = modules_in_block
        quant_config = self.quantizer.finalize_quant_config_for_save(
            quant_config=quant_config,
            quantized_layer_names=quantized_names,
            num_hidden_layers=(
                getattr(model.config, "num_hidden_layers", None)
                or getattr(getattr(model.config, "text_config", None), "num_hidden_layers", None)
            ),
        )
        # Add quantization config to model config
        model.config.quantization_config = quant_config

        # Save model and tokenizer
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)

        logger.info(f"Quantized model saved to {save_directory}")
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
        from .analyzer.cumulative_error import analyze_cumulative_error as _analyze

        # pylint: disable-next=import-outside-toplevel
        from .analyzer.cumulative_error import plot_cumulative_error as _plot

        logger = self.logger

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
            torch.cuda.empty_cache()

            model = self.model_config.load_model()
            input_device = next(model.parameters()).device
            inputs = self.prepare_calibration_dataset(input_device)

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
                torch.cuda.empty_cache()

                model = self.model_config.load_model()
                input_device = next(model.parameters()).device
                inputs = self.prepare_calibration_dataset(input_device)

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
