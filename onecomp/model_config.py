"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from logging import getLogger

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch


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

    def load_model(self, device_map=None):
        """Load the model

        Automatically selects the appropriate AutoModel class based on
        the model's architecture (CausalLM, Vision2Seq, etc.).
        """

        if device_map is None:
            device_map = self.device

        pretrained = self.get_model_id_or_path()
        kwargs = dict(
            dtype=getattr(torch, self.dtype),
            device_map=device_map,
        )

        try:
            model = AutoModelForCausalLM.from_pretrained(pretrained, **kwargs)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText

            self.logger.info("AutoModelForCausalLM failed; trying AutoModelForImageTextToText.")
            model = AutoModelForImageTextToText.from_pretrained(pretrained, **kwargs)

        model.eval()
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
