"""
Base class for post-quantization processes.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from ..model_config import ModelConfig


@dataclass
class PostQuantizationProcess(metaclass=ABCMeta):
    """Abstract base class for post-quantization processes

    Post-quantization processes are executed after the main quantization
    step (e.g., GPTQ, DBF).  Each process receives a quantized model
    on CPU (with quantized inference layers such as ``GPTQLinear``)
    and may modify it in-place.

    Subclasses must implement ``run()`` method.
    ``name`` is automatically set to the class name if not provided.

    Args:
        name (str or None):
            Human-readable name used in log messages.
            If None, automatically set to the class name.

    Examples:
        Typical usage via ``Runner``:

        >>> from onecomp import Runner, ModelConfig, GPTQ, BlockWisePTQ
        >>> model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf")
        >>> quantizer = GPTQ(wbits=4, groupsize=128)
        >>> runner = Runner(
        ...     model_config=model_config,
        ...     quantizer=quantizer,
        ...     post_processes=[BlockWisePTQ()],
        ... )
        >>> runner.run()

    """

    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = type(self).__name__

    @abstractmethod
    def run(
        self,
        quantized_model: nn.Module,
        model_config: ModelConfig,
    ) -> None:
        """Execute the post-quantization process.

        The model is provided on CPU.  Implementations may move it to
        GPU for computation, but **must move it back to CPU before
        returning** so that subsequent processes and ``Runner`` methods
        (e.g. evaluation, saving) can work without device assumptions.

        Args:
            quantized_model (nn.Module):
                The quantized model on CPU.  Linear layers that were
                quantized have already been replaced with quantized
                inference layers (e.g. ``GPTQLinear``, ``DoubleBinaryLinear``).
                The process may modify the model in-place.
            model_config (ModelConfig):
                The model configuration (provides access to tokenizer,
                model id/path, device, etc.).
        """
