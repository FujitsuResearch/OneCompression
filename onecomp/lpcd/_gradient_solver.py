"""
Copyright 2025-2026 Fujitsu Ltd.

"""
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from transformers import get_cosine_schedule_with_warmup

from ._lpcd_config import LPCDConfig
from ._metric import LpcdMetric


class PenaltyFn(ABC):

    @abstractmethod
    def __call__(self, module_q: LpcdMetric, module_f: LpcdMetric) -> torch.Tensor:
        pass
 

def gradient_solver(
    lpcd_config: LPCDConfig,
    target_modules: list[nn.Module],
    metric_q: LpcdMetric,
    metric_f: LpcdMetric,
    inps_q: torch.Tensor,
    inps_f: torch.Tensor,
    kwargs: dict,
    penalty_fn_list: list[PenaltyFn] = [],
) -> None:
    """Gradient-based solver for LPCD relaxation."""
    device = lpcd_config.device
    batch_size = 16

    assert lpcd_config.gd_batch_size % batch_size == 0, \
        f"gd_batch_size should be divisible by batch_size, " + \
        f"but got {lpcd_config.gd_batch_size} and {batch_size}"
    
    assert target_modules[0].weight.data.dtype in [torch.float32, torch.bfloat16], \
        "The model must be loaded in float32 or bfloat16 " + \
        "for gradient-based LPCD refinement due to numerical stability."
    
    # backup and configure grad state
    grad_state_q = [p.requires_grad for p in metric_q.parameters()]
    grad_state_f = [p.requires_grad for p in metric_f.parameters()]

    # set target modules as trainablethe     
    metric_q.requires_grad_(False)
    metric_f.requires_grad_(False)
    for module in target_modules:
        module.requires_grad_(True)

    # Define gradient descent components
    epochs = lpcd_config.gd_steps
    accum_steps = lpcd_config.gd_batch_size // batch_size
    total_iters = epochs * inps_q.size(0) // accum_steps

    optimizer = optim.Adam(metric_q.parameters(), lr=lpcd_config.gd_base_lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_iters * 0.1),
        num_training_steps=total_iters
    )

    # Perform gradient descent to relax the weights
    for epoch in range(1, lpcd_config.gd_steps+1):

        chunked_inps = zip(
            inps_q.split(batch_size),
            inps_f.split(batch_size),
        )

        # TODO: shuffle the inputs for better convergence
        for iter, (inp_q, inp_f) in enumerate(chunked_inps):
        
            with torch.no_grad():
                out_f = metric_f(inp_f.to(device), **kwargs)

            with torch.enable_grad():
                out_q = metric_q(inp_q.to(device), **kwargs)
                task_loss = F.mse_loss(out_q.float(), out_f.float())
                reg_loss = sum([penalty(metric_q, metric_f) for penalty in penalty_fn_list])
                loss = (task_loss + reg_loss) / accum_steps
                loss.backward()

            if (iter + 1) % accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    # restore grad state
    for p, requires_grad in zip(metric_q.parameters(), grad_state_q):
        p.requires_grad_(requires_grad)
    for p, requires_grad in zip(metric_f.parameters(), grad_state_f):
        p.requires_grad_(requires_grad)