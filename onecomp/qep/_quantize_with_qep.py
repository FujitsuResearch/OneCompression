"""
Quantization with QEP (Quantization Error Propagation) Module

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)

A generic implementation independent of model architecture.
Consumes extra CPU memory and incurs unnecessary forward passes.
Could be faster by leveraging model structure to avoid redundant forward passes.

Current procedure:
1. Save input activations of the original model to CPU
2. For each target layer l, perform the following sequentially:
   2-1. Save input activations of layer l in the quantized model to CPU
   2-2. Quantize the weights of layer l in the quantized model
   2-3. Update the weights of layer l in the quantized model

"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from logging import getLogger

import torch

from onecomp.model_config import ModelConfig
from onecomp.qep._qep_config import QEPConfig
from onecomp.quantizer._quantizer import Quantizer
from onecomp.utils import capture_input_activations
from onecomp.utils import prepare_calibration_dataset


logger = getLogger(__name__)


def run_quantize_with_qep(
    model_config: ModelConfig,
    quantizer: Quantizer,
    qep_config: QEPConfig,
    calibration_dataset,
    max_length: int,
    num_calibration_samples: int,
    calibration_strategy: str,
    calibration_seed: int,
):
    """Run quantization with Quantization Error Propagation (QEP).

    A generic implementation independent of model architecture.
    For each target layer, captures input activations from both the
    original and current (partially quantized) model, then quantizes
    using error propagation.

    Args:
        model_config (ModelConfig): Model configuration for loading the model
            and tokenizer.
        quantizer (Quantizer): The quantizer to use.
        qep_config (QEPConfig): Configuration for QEP
            (percdamp, perccorr, exclude_layer_keywords).
        calibration_dataset: Calibration dataset. If None, a default dataset
            is loaded.
        max_length (int): Maximum sequence length for calibration inputs.
        num_calibration_samples (int): Number of calibration samples.
        calibration_strategy (str): Strategy for preparing calibration inputs.
        calibration_seed (int): Random seed for calibration data preparation.

    """
    model = model_config.load_model()
    tokenizer = model_config.load_tokenizer()
    input_device = next(model.parameters()).device

    inputs = prepare_calibration_dataset(
        tokenizer=tokenizer,
        device=input_device,
        calibration_dataset=calibration_dataset,
        max_length=max_length,
        num_calibration_samples=num_calibration_samples,
        strategy=calibration_strategy,
        seed=calibration_seed,
        logger=logger,
    )

    # Setup the quantizer
    quantizer.setup(model)

    # 1. Save input activations of the original model to CPU
    original_input_activations = capture_input_activations(
        model=model,
        inputs=inputs,
        module_to_name=quantizer.module_to_name,
        exclude_layer_keywords=qep_config.exclude_layer_keywords,
        logger=logger,
    )
    torch.cuda.empty_cache()

    logger.info("Quantizing the model using %s", quantizer.name)

    # 2. For each target layer, perform the following sequentially
    for module, name in quantizer.module_to_name.items():

        logger.info(
            "Processing layer: %s =================================================",
            name,
        )

        # 2-1. Save input activations of only the target layer to CPU
        quant_input_activation = capture_input_activations(
            model=model,
            inputs=inputs,
            module_to_name={module: name},
            logger=logger,
        )[name]

        # 2-2. Quantize the weights of the target layer
        quantizer.quantize_with_qep(
            module,
            quant_input_activation,
            original_input_activation=original_input_activations.get(name, None),
            percdamp=qep_config.percdamp,
            perccorr=qep_config.perccorr,
        )

        # 2-3. Update the weights of the target layer
        dtype = module.weight.data.dtype
        device = module.weight.data.device
        module.weight.data = quantizer.results[name].dequantized_weight.to(device).to(dtype)

        # 2-4. Free memory
        del quant_input_activation

    del original_input_activations
    quantizer.execute_post_processing()
