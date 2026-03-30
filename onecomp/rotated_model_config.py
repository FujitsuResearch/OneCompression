"""Rotated Model Configuration for rotation/scaling preprocessing.

Provides a ``ModelConfig`` subclass for rotation-preprocessed models.
On ``load_model()``, deterministic Hadamard hooks are automatically
registered on ``down_proj`` layers so that the model can be quantized
with the existing ``Runner`` pipeline without modification.

Processing flow:
    rotated model load → Hadamard hooks on down_proj → quantization (existing code)

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura, Yusei Kawakami

"""

import json
import os

from .model_config import ModelConfig
from .pre_process.rotation_utils import register_online_hadamard_hooks


class RotatedModelConfig(ModelConfig):
    """ModelConfig subclass for rotation-preprocessed models.

    Inherits ``ModelConfig`` and automatically registers deterministic
    Hadamard ``forward_pre_hook`` on ``down_proj`` layers when
    ``load_model()`` is called.

    The saved model directory should contain:

    - ``config.json`` — HuggingFace model config (includes ``fp32_had`` field)
    - ``model.safetensors`` — rotation-applied weights
    - ``tokenizer.json``

    Args:
        path (str): Path to the saved rotated model (required).
        dtype (str): Data type. Defaults to "float16".
        device (str): Device. Defaults to "auto".
        fp32_had (bool or None): Use FP32 for online Hadamard transform.
            If None (default), auto-detect from ``rotation_config.json``
            in the model directory. Falls back to False if not found.

    Example:
        >>> from onecomp import Runner, RotatedModelConfig, GPTQ
        >>>
        >>> model_config = RotatedModelConfig(path="./rotated_model")
        >>> quantizer = GPTQ(wbits=4, groupsize=128)
        >>> runner = Runner(model_config=model_config, quantizer=quantizer)
        >>> runner.run()
    """

    def __init__(
        self,
        path: str = None,
        dtype: str = "float16",
        device: str = "auto",
        fp32_had: bool = None,
        **kwargs,
    ):
        if "model_id" in kwargs and kwargs["model_id"] is not None:
            raise ValueError(
                "RotatedModelConfig does not support model_id. "
                "Hub models are not rotation-preprocessed; Hadamard hooks on "
                "down_proj will produce incorrect results. "
                "Use 'path' to specify a locally saved rotated model instead."
            )
        if path is None:
            raise ValueError(
                "RotatedModelConfig requires 'path' to a locally saved "
                "rotated model. Use the pre_process module to create one."
            )
        super().__init__(path=path, dtype=dtype, device=device)

        if fp32_had is not None:
            self.fp32_had = fp32_had
        else:
            self.fp32_had = self._load_fp32_had()

    def _load_fp32_had(self) -> bool:
        """Read ``fp32_had`` from ``config.json`` if it exists."""
        config_path = os.path.join(self.path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as f:
                config_dict = json.load(f)
            return config_dict.get("fp32_had", False)
        return False

    def load_model(self, **kwargs):
        """Load the rotated model and register Hadamard hooks.

        Returns:
            nn.Module: Model with Hadamard pre-hooks registered on down_proj.
        """
        model = super().load_model(**kwargs)
        hooks = register_online_hadamard_hooks(model, fp32_had=self.fp32_had)
        self.logger.info(
            "Registered Hadamard pre-hooks on %d down_proj layers (fp32_had=%s)",
            len(hooks),
            self.fp32_had,
        )
        return model

    def has_additional_data(self):
        """Returns True (rotation metadata exists)."""
        return True
