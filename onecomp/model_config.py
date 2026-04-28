"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from logging import getLogger

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

from .utils.dtype import needs_bfloat16

try:
    from transformers import AutoModelForImageTextToText as _AutoVLM

    _HAS_VLM_AUTO = True
except ImportError:
    _HAS_VLM_AUTO = False


class ModelConfig:
    """Model and Tokenizer"""

    def __init__(
        self,
        model_id: str = None,
        path: str = None,
        dtype: str = "float16",
        device: str = "auto",
    ):
        """
        Args:
            model_id (str): Model ID (Hugging Face Hub ID).
            path (str): Path to the saved model and tokenizer.
            dtype (str, optional): Data type. Defaults to "float16".
            device (str, optional): Device to use ("cpu", "cuda", "auto"). Defaults to "auto".

        Example:
            >>> model_config = ModelConfig(model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
            >>> model = model_config.load_model()
            >>> tokenizer = model_config.load_tokenizer()

        """
        self.logger = getLogger(__name__)

        if model_id is None and path is None:
            raise ValueError("Either model_id or path must be provided")

        if needs_bfloat16(model_id or path):
            if dtype != "bfloat16":
                self.logger.warning(
                    "Overriding dtype to bfloat16 for %s "
                    "to prevent performance degradation.",
                    model_id or path,
                )
            dtype = "bfloat16"

        self.model_id = model_id
        self.path = path
        self.dtype = dtype
        self.device = device

        # If additional settings are needed, modify has_additional_data and load_model methods

    def get_model_id_or_path(self):
        """Get the model ID or path"""
        if self.model_id is not None:
            return self.model_id
        if self.path is not None:
            return self.path
        return None

    def load_config(self):
        """Load and cache the model config (no weights)."""
        if not hasattr(self, "_cached_config"):
            self._cached_config = AutoConfig.from_pretrained(
                self.get_model_id_or_path(), trust_remote_code=True
            )
        return self._cached_config

    def load_model(self, device_map=None):
        """Load the model

        Tries ``AutoModelForCausalLM`` first.  If the model is a
        Vision-Language Model (e.g. Qwen3-VL) that is not registered
        with ``AutoModelForCausalLM``, falls back to
        ``AutoModelForImageTextToText``.

        Args:
            device_map (str or None):
                Override the device placement for this load.
                If ``None`` (default), ``self.device`` is used.
        """
        effective_device = device_map if device_map is not None else self.device
        kwargs = dict(
            dtype=self.dtype if self.dtype == "auto" else getattr(torch, self.dtype),
            device_map=effective_device,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(self.get_model_id_or_path(), **kwargs)
        except ValueError as e:
            _vlm_hints = ("Unrecognized configuration class", "Unrecognized model", "is not supported")
            if not _HAS_VLM_AUTO or not any(h in str(e) for h in _vlm_hints):
                raise
            self.logger.info("AutoModelForCausalLM failed; trying AutoModelForImageTextToText.")
            model = _AutoVLM.from_pretrained(self.get_model_id_or_path(), **kwargs)
        model.eval()
        self.logger.info("Model loaded with dtype=%s", next(model.parameters()).dtype)
        return model

    def load_tokenizer(self):
        """Load the tokenizer"""

        tokenizer = AutoTokenizer.from_pretrained(self.get_model_id_or_path())

        # Handle models without pad_token (e.g., Llama2)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            self.logger.info("pad_token is not set. Using eos_token as pad_token.")

        return tokenizer

    def has_additional_data(self):
        """Check if the model has additional data

        Returns True if there are settings other than `model_id`, `path`, `dtype`, `device`.
        Currently always returns False. Should return True when additional settings are added.

        """

        return False
