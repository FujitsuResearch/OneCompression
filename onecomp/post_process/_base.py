"""
Base class for post-quantization processes.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import json
from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional

import torch.nn as nn

from ..model_config import ModelConfig
from ..utils.quant_config import validate_quantized_model_config
from ._runtime import (
    append_post_process_metadata,
    prepare_quantized_model_for_post_process,
)


@dataclass
class PostQuantizationProcess(metaclass=ABCMeta):
    """Abstract base class for post-quantization processes

    Post-quantization processes are executed after the main quantization
    step (e.g., GPTQ, DBF).  Each process receives a quantized model
    on CPU (with quantized inference layers such as ``GPTQLinear``)
    and may modify it in-place.

    Subclasses must implement ``_run()`` method.
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

    def run(
        self,
        quantized_model: nn.Module,
        model_config: ModelConfig,
    ) -> None:
        """Execute the post-quantization process.

        The model is moved to CPU and validated before the subclass
        implementation runs. Implementations may move it to
        GPU for computation, but **must move it back to CPU before
        returning** so that subsequent processes and ``Runner`` methods
        (e.g. evaluation, saving) can work without device assumptions.
        Successful runs append audit metadata to
        ``model.config.quantization_config["onecomp_post_processes"]``.
        When a subclass ``_run`` method returns a result dict, its
        ``executed`` / ``global_executed`` and ``reason`` fields are mirrored
        into that metadata. Failed runs move the model back to CPU but do not
        append metadata.

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
        context = f"{type(self).__name__}.run"
        quantized_model = prepare_quantized_model_for_post_process(
            quantized_model,
            model_config,
            context,
        )

        run_result = None
        try:
            run_result = self._run(quantized_model, model_config)
        finally:
            if hasattr(quantized_model, "eval"):
                quantized_model.eval()
            if hasattr(quantized_model, "cpu"):
                quantized_model.cpu()

        quant_config = validate_quantized_model_config(quantized_model, context)
        metadata = self.build_metadata()
        metadata.update(self.build_execution_metadata(run_result))
        append_post_process_metadata(quant_config, [metadata])

    @abstractmethod
    def _run(
        self,
        quantized_model: nn.Module,
        model_config: ModelConfig,
    ) -> dict | None:
        """Run the post-process algorithm body (implemented by subclasses).

        Called by :meth:`run` after the input model has been moved to CPU and
        its ``quantization_config`` validated.  Implementations may move the
        model to GPU for computation and modify it in-place; :meth:`run`
        restores it to ``eval()`` on CPU afterwards (even if this method
        raises), so subclasses do not need to normalise the device themselves.

        Subclasses implement this method, not :meth:`run`. Returning ``None``
        is valid. Returning a dict lets the base class copy ``executed`` /
        ``global_executed`` and ``reason`` into the audit metadata.

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

    def build_metadata(self) -> dict:
        """Build JSON-serializable audit metadata for this post-process.

        The returned dict is appended to
        ``quantization_config["onecomp_post_processes"]`` and written to
        ``config.json`` by model save paths, so users can inspect which
        post-processes (and with which hyper-parameters) were applied to a
        saved checkpoint.

        ``asdict(self)`` already yields JSON-serializable primitives for every
        current post-process (all fields are ``str``/``int``/``float``/``bool``/
        ``None`` or nested dataclasses/lists/dicts thereof).  The ``json``
        round-trip with ``default=str`` is a cheap safety net: it normalises
        tuples to lists and coerces any future non-serializable field to a
        string here, instead of letting ``model.save_pretrained()`` raise after
        the (expensive) post-process has already run.
        """
        process_config = {
            key: value for key, value in asdict(self).items() if not key.startswith("_")
        }
        process_config.pop("name", None)
        return {
            "name": self.name or type(self).__name__,
            "class": type(self).__name__,
            "config": json.loads(json.dumps(process_config, default=str)),
        }

    def build_execution_metadata(self, run_result) -> dict:
        """Build audit metadata derived from an optional ``_run`` result.

        Post-process implementations historically returned ``None``. Newer
        implementations may return a small result dict such as
        ``{"global_executed": False, "reason": "not_quantized"}``. The audit
        history uses the generic ``executed`` name, while still accepting the
        GlobalPTQ-specific ``global_executed`` key for compatibility with the
        existing core helper result schema.
        """
        metadata = {"executed": True}
        if not isinstance(run_result, dict):
            return metadata

        executed = run_result.get("executed", run_result.get("global_executed"))
        if executed is not None:
            metadata["executed"] = bool(executed)

        if metadata["executed"] is False:
            metadata["reason"] = str(run_result.get("reason", "unknown"))
        elif run_result.get("reason") is not None:
            metadata["reason"] = str(run_result["reason"])

        return metadata
