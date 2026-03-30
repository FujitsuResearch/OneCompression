"""
Block-wise Post-Training Quantization process.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from dataclasses import dataclass

import torch.nn as nn

from ..model_config import ModelConfig
from ._base import PostQuantizationProcess


@dataclass
class BlockWisePTQ(PostQuantizationProcess):
    """Block-wise Post-Training Quantization

    After layer-wise PTQ (GPTQ / DBF) quantises each linear layer
    independently, block-wise PTQ minimises intermediate-representation
    MSE against an FP16 teacher model at the Transformer-block
    granularity.

    This is a placeholder class.  The actual optimisation logic will
    be implemented separately.

    Args:
        lr (float):
            Learning rate for block-wise optimisation.  Default is 1e-4.
        epochs (int):
            Number of optimisation epochs per block.  Default is 10.
        cbq_enable (bool):
            Whether to enable Cross-Block Quantisation (Phase 2) after
            greedy block-wise distillation.  Default is False.

    Examples:
        >>> from onecomp import Runner, ModelConfig, GPTQ, BlockWisePTQ
        >>> model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf")
        >>> quantizer = GPTQ(wbits=4, groupsize=128)
        >>> runner = Runner(
        ...     model_config=model_config,
        ...     quantizer=quantizer,
        ...     post_processes=[BlockWisePTQ(lr=1e-4, epochs=10, cbq_enable=True)],
        ... )
        >>> runner.run()

    """

    # TODO: パラメータは実装者が自由に変更・追加してください
    lr: float = 1e-4
    epochs: int = 10
    cbq_enable: bool = False

    def run(
        self,
        quantized_model: nn.Module,
        model_config: ModelConfig,
    ) -> None:
        """Execute block-wise PTQ on the quantized model.

        Not yet implemented.  This is a placeholder for the
        colleague's implementation.

        Args:
            quantized_model (nn.Module):
                Quantized model on CPU.
            model_config (ModelConfig):
                Model configuration.
        """
        raise NotImplementedError(
            "BlockWisePTQ.run() is not yet implemented. "
            "Please provide the block-wise optimisation logic."
        )
