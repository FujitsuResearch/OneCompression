"""
Copyright 2025-2026 Fujitsu Ltd.

"""
import math
import torch
import torch.nn.functional as F

from onecomp.quantizer._quantizer import Quantizer
from ._metric import ClosedFormSolverArgument, LpcdMetric
from ._gradient_solver import gradient_solver
from ._lpcd_config import LPCDConfig
from ..qep._quantize_with_qep_arch import compute_hessian_and_crossterm

from logging import getLogger
logger = getLogger(__name__)


@torch.no_grad()
def compute_mse(
    metric_q: LpcdMetric,
    metric_f: LpcdMetric,
    inps_q: torch.Tensor,
    inps_f: torch.Tensor,
    batch_size: int,
    device: str,
    kwargs: dict,
) -> float:
    """ Compute the quantization error

    Args:
        metric_q (LpcdMetric): LPCD metric for the quantized block
        metric_f (LpcdMetric): LPCD metric for the full-precision block
        inps_q (torch.Tensor): Inputs for the quantized block
        inps_f (torch.Tensor): Inputs for the full-precision block
        batch_size (int): Batch size for computation
        device (str): Device for computation
        kwargs (dict): Additional arguments for the metric forward pass

    Returns:
        float: MSE between the quantized and full-precision metrics
    """
    mse = torch.zeros(1, device=device)
    chunked_inps = zip(inps_q.split(batch_size), inps_f.split(batch_size))
    for inp_q, inp_f in chunked_inps:
        out_q = metric_q(inp_q.to(device), **kwargs)
        out_f = metric_f(inp_f.to(device), **kwargs)
        mse += F.mse_loss(out_q.float(), out_f.float())
    n_iters = math.ceil(inps_q.size(0) / batch_size)
    return (mse / n_iters).item()


def refiner(
    lpcd_config: LPCDConfig,
    metric_group: tuple[LpcdMetric, LpcdMetric],
    inps_q: torch.Tensor,
    inps_f: torch.Tensor,
    quantizer: Quantizer,
    kwargs: dict,
) -> None:
    """ Perform the LPCD refinement for a single module group (e.g., qk-module)

    Args:
        lpcd_config (LPCDConfig): LPCD configuration
        metric_group (tuple[LpcdMetric, LpcdMetric]): LPCD metrics for the quantized and full-precision blocks
        inps_q (torch.Tensor): Inputs for the quantized block
        inps_f (torch.Tensor): Inputs for the full-precision block
        quantizer (Quantizer): Quantizer for the projection step
        kwargs (dict): Additional arguments for the metric forward pass
    """

    device = lpcd_config.device
    batch_size = 16

    assert lpcd_config.gd_batch_size % batch_size == 0, \
        f"gd_batch_size should be divisible by batch_size, " + \
        f"but got {lpcd_config.gd_batch_size} and {batch_size}"
    
    
    metric_q, metric_f = metric_group

    modules_q = [module for _, module in metric_q.named_targets()]
    modules_f = [module for _, module in metric_f.named_targets()]

    for alt_iter in range(lpcd_config.alt_steps):

        for module_idx, (module_q, module_f) in enumerate(zip(modules_q, modules_f)):

            # Skip layers not registered for quantization
            if module_q not in quantizer.module_to_name:
                logger.info("Skipping layer (not in quantization targets): %s", name)
                continue
            else:
                name = quantizer.module_to_name[module_q]

            logger.info(
                "Processing layer for refinement: %s =================================================",
                name,
            )
            
            default_mse = compute_mse(
                metric_q, metric_f, inps_q, inps_f, batch_size, device, kwargs
            )
            
            # Relaxation step
            cf_solver = metric_q.closed_form_solvers()[module_idx]

            if lpcd_config.use_closed_form and cf_solver is not None:
                logger.info("Relaxing with closed-form method.")
                cf_solver_arg = ClosedFormSolverArgument(
                    lpcd_config=lpcd_config,
                    block_q=metric_q.block,
                    block_f=metric_f.block,
                    inps_q=inps_q,
                    inps_f=inps_f,
                    kwargs=kwargs,
                    device=device,
                )
                cf_solver(cf_solver_arg)
            else:
                logger.info("Relaxing with gradient descent.")
                
                gradient_solver(
                    lpcd_config=lpcd_config,
                    target_modules=[module_q],
                    metric_q=metric_q,
                    metric_f=metric_f,
                    inps_q=inps_q,
                    inps_f=inps_f,
                    kwargs=kwargs,
                    penalty_fn_list=[],
                )

            
            # apply relaxed result with regularization
            with torch.no_grad():
                alpha  = lpcd_config.perccorr
                W      = module_f.weight.data
                W_corr = module_q.weight.data
                module_q.weight.data = (1 - alpha) * W + alpha * W_corr

            # Compute the relaxed MSE
            relaxed_mse = compute_mse(
                metric_q, metric_f, inps_q, inps_f, batch_size, device, kwargs
            )

            # Projection step
            H, _ = compute_hessian_and_crossterm(
                metric_q.block,
                metric_f.block,
                module_q,
                module_f,
                inps_q,
                inps_f,
                kwargs,
                batch_size,
                device,
            )
            quantizer.quantize(module_q, None, None, hessian=H.clone())

            # Update the weights of the target layer
            dtype = module_q.weight.data.dtype
            module_q.weight.data = (
                quantizer.results[name]
                .compute_dequantized_weight()
                .to(device)
                .to(dtype)
            )

            # Compute the projected MSE
            mse_after_projection = compute_mse(
                metric_q, metric_f, inps_q, inps_f, batch_size, device, kwargs
            )
            logger.debug(f"[LPCD] {default_mse:.5e} -> {relaxed_mse:.5e} -> {mse_after_projection:.5e}")

    metric_q.is_refined = True
