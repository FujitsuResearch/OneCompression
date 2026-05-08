"""LPCD runner – top-level entry point for post-quantisation optimisation.

Iterates over decoder blocks and applies LPCD group optimisation where
enabled (QK, VO, MLP).  Designed to be called from ``Runner`` after
the base quantiser has already been set up.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yudai Fujimoto

"""

import copy
from logging import getLogger

import torch
from torch import nn

from onecomp.calibration import CalibrationConfig, prepare_calibration_dataset
from onecomp.lpcd import LPCDConfig
from onecomp.model_config import ModelConfig
from onecomp.qep import QEPConfig
from onecomp.quantizer._quantizer import Quantizer

from ._refiner import refiner, compute_mse
from ._metric import make_lpcd_metrics
from ..utils.blockwise import (
    get_blocks_and_inputs,
    move_kwargs_to_device,
    forward_input,
)
from ..qep._quantize_with_qep_arch import (
    make_grouped_module,
    compute_hessian_and_crossterm,
)

logger = getLogger(__name__)


@torch.no_grad()
def run_quantize_with_lpcd(
    model_config: ModelConfig,
    quantizer: Quantizer,
    qep_config: QEPConfig | None,
    lpcd_config: LPCDConfig,
    calibration_config: CalibrationConfig,
):
    """ Run quantization with LPCD.

    Args:
        model_config (ModelConfig): Model configuration
        quantizer (Quantizer): Quantizer
        qep_config (QEPConfig | None): QEP configuration
        lpcd_config (LPCDConfig): LPCD configuration
        calibration_config (CalibrationConfig): Calibration configuration
    """

    assert not (qep_config is not None and qep_config.general), \
        "qep_config.general must be False when qep is enabled."

    # TODO: Parameterize when necessary
    batch_size = 16

    model = model_config.load_model(
        device_map="cpu",
    )
    tokenizer = model_config.load_tokenizer()
    device = lpcd_config.device

    model_inputs = prepare_calibration_dataset(
        tokenizer=tokenizer,
        device=torch.device('cpu'),
        calibration_config=calibration_config,
        model=model,
        logger=logger,
    )

    # Setup the quantizer
    quantizer.setup(model)

    # 1. Prepare transformer blocks and their inputs
    blocks, inps, kwargs = get_blocks_and_inputs(model, model_inputs, batch_size)

    inps_q = inps
    inps_f = inps.clone()
    kwargs = move_kwargs_to_device(kwargs, device)

    logger.info("Quantizing the model using %s", quantizer.name)

    # 2. For each target transformer block, perform the following sequentially
    for block_idx, block in enumerate(blocks):

        logger.info(
            "Processing : %2d-th Transformer Block -------------------------------------------------",
            block_idx + 1,
        )

        block_q = block.to(device)
        block_f = copy.deepcopy(block_q)

        groups_q = make_grouped_module(block_q, inps_q, kwargs, device)
        
        # Build name→module map for block_f, then align groups_f to
        # groups_q by module name.  Using make_grouped_module on
        # block_f independently can produce a different group ordering
        # because tensor identity (used for grouping) is
        # non-deterministic across deepcopy + forward.
        name_to_module_f = {
            name: mod for name, mod in block_f.named_modules() if isinstance(mod, nn.Linear)
        }
        name_to_module_q = {
            name: mod for name, mod in block_q.named_modules() if isinstance(mod, nn.Linear)
        }
        groups_f = [
            [name_to_module_f[next(n for n, m2 in name_to_module_q.items() if m2 is m)] for m in gq]
            for gq in groups_q
        ]

        lpcd_metrics = make_lpcd_metrics(lpcd_config, block_q, block_f)
        lpcd_modules_q = [module for metric, _ in lpcd_metrics.metrics for _, module in metric.named_targets()]

        # 3. For each group of layers, perform the following sequentially
        for group_q, group_f in zip(groups_q, groups_f):

            logger.info(
                "Processing group of layers: %s",
                ", ".join([quantizer.module_to_name.get(m, "N/A") for m in group_q]),
            )

            # 3-1. compute hessian and cross-term matrix
            H, delta_hatX = compute_hessian_and_crossterm(
                block_q,
                block_f,
                group_q[0],
                group_f[0],
                inps_q,
                inps_f,
                kwargs,
                batch_size,
                device,
            )

            # Build reverse mapping module_q -> name for logging skipped layers
            module_q_to_name = {m: n for n, m in name_to_module_q.items()}

            # 3-2, For each module in the group, perform weight correction and quantization
            for module in group_q:

                # Skip layers not registered for quantization
                if module not in quantizer.module_to_name:
                    skipped_name = module_q_to_name.get(module, "<unknown>")
                    logger.info(
                        "Skipping layer (not in quantization targets): %s", skipped_name
                    )
                    continue
                name = quantizer.module_to_name[module]


                # Fall back to standard quantization if the module is not LPCD targets
                if not module in lpcd_modules_q:

                    logger.info(
                        "Processing layer: %s =================================================",
                        name,
                    )

                    if qep_config is None:
                        # Quatize without QEP
                        quantizer.quantize(
                            module=module,
                            input=None,
                            output=None,
                            hessian=H.clone(),
                        )
                    else:
                        # Quantize with QEP
                        exclude = any(kw in name for kw in qep_config.exclude_layer_keywords)
                        layer_delta = None if exclude else delta_hatX.clone()

                        quantizer.quantize_with_qep(
                            module,
                            quant_input_activation=None,
                            original_input_activation=None,
                            percdamp=qep_config.percdamp,
                            perccorr=qep_config.perccorr,
                            hessian=H.clone(),
                            delta_hatX=layer_delta,
                        )

                    # Update the weights of the target layer
                    dtype = module.weight.data.dtype
                    module.weight.data = (
                        quantizer.results[name]
                        .compute_dequantized_weight()
                        .to(device)
                        .to(dtype)
                    )

                lpcd_metrics.mark_as_ready(module)

                # 3-3. Perform LPCD optimisation
                for module_group in lpcd_metrics.get_refineable_metrics():

                    logger.info("perform LPCD optimization")
                    lpcd_names = [name for name, _ in module_group[0].named_targets()]
                    logger.info("LPCD targets: %s", lpcd_names)
                    refiner(
                        lpcd_config,
                        module_group,
                        inps_q,
                        inps_f,
                        quantizer,
                        kwargs,
                    )

        # forward input to the next block
        inps_q = forward_input(inps_q, block_q, kwargs, batch_size, device)
        inps_f = forward_input(inps_f, block_f, kwargs, batch_size, device)

        # DEBUG:Compute MSE between quantized and full-precision outputs
        mse = compute_mse(
            block_q,
            block_f,
            inps_q,
            inps_f,
            batch_size,
            device,
            kwargs
        )
        logger.info(f"[INFO] Layer {block_idx + 1} MSE: {mse:.6e}")

        # free memory
        block_q.cpu()
        torch.cuda.empty_cache()

    quantizer.execute_post_processing()
    