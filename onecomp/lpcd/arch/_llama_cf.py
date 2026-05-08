"""
Copyright 2025-2026 Fujitsu Ltd.

"""
import math

import torch
from torch import nn
from .._metric import ClosedFormSolverArgument

from logging import getLogger
logger = getLogger(__name__)


@torch.no_grad()
def _closed_form_solver(
    arg: ClosedFormSolverArgument, 
    module: nn.Linear,
    dest: dict
) -> None:
    """Common Closed-form solver for weight correction

    Args:
        arg (ClosedFormSolverArgument): The argument for the closed-form solver
        module (nn.Linear): The linear module to be corrected
        dest (dict): A dictionary to store activations, with keys:
            - "x_q": input to the quantized target layer
            - "x_f": input to the full-precision target layer
            - "h_q": residual input to the residual connection after the quantized target layer
            - "h_f": residual input to the residual connection after the full-precision target layer
    """
    batch_size = 16

    # Hessian, cross terms
    D_out, D_in = module.weight.shape
    H = torch.zeros((D_in, D_in), device=arg.device)
    deltax_x = torch.zeros((D_in, D_in), device=arg.device)
    deltah_x = torch.zeros((D_in, D_out), device=arg.device)

    chunked_inps = zip(
        arg.inps_q.split(batch_size),
        arg.inps_f.split(batch_size),
    )

    # compute Hessian and crossterms
    sample_cnt = 0

    # compute Hessian and crossterm in batches
    for inp_q, inp_f in chunked_inps:

        # capture activations
        _ = arg.block_q(inp_q.to(arg.device), **arg.kwargs)
        _ = arg.block_f(inp_f.to(arg.device), **arg.kwargs)
        
        x_q = dest['x_q'].view(-1, D_in).float()
        x_f = dest['x_f'].view(-1, D_in).float()

        h_q = dest['h_q'].view(-1, D_out).float()
        h_f = dest['h_f'].view(-1, D_out).float()

        n_tokens = x_q.size(0)
        
        # Compute Hessian and cross-term
        accum_scale = sample_cnt / (sample_cnt + n_tokens)
        H        *= accum_scale
        deltax_x *= accum_scale
        deltah_x *= accum_scale

        sample_cnt += n_tokens

        # Scale activations
        inp_scale = math.sqrt(2 / sample_cnt)
        x_q *= inp_scale
        x_f *= inp_scale
        h_q *= inp_scale
        h_f *= inp_scale

        # Compute residual cross-terms
        H        += torch.matmul(x_q.t(), x_q)
        deltax_x += torch.matmul(x_q.t(), (x_f - x_q))
        deltah_x += torch.matmul(x_q.t(), (h_f - h_q))

        dest.clear()

    # damping
    dtype = module.weight.dtype
    lam = arg.lpcd_config.percdamp * torch.trace(H) / D_in
    diag = lam * torch.eye(D_in, device=arg.device)
    H += diag

    try:
        # weight correction
        perccorr = arg.lpcd_config.perccorr
        W = module.weight.data
        L = torch.linalg.cholesky(H)
        corr_x = torch.cholesky_solve(deltax_x, L).to(dtype)
        corr_w = W + perccorr * torch.matmul(W, corr_x.t())
        
        corr_h = torch.cholesky_solve(deltah_x, L).to(dtype)
        corr_w += perccorr * corr_h.t()
        
        module.weight.data = corr_w
    except torch.linalg.LinAlgError as e:
        logger.warning(f"{e}")
        logger.warning(f"Cholesky decomposition failed with error. Skipping the relaxation.")


def _make_inp_hook(dest: dict, name: str):
    def hook(module, inp, out):
        dest[name] = (inp[0] if isinstance(inp, tuple) else inp).clone()
    return hook


def closed_form_solver_o_proj(arg: ClosedFormSolverArgument) -> None:
    """Closed-form solver for llama o_proj
    Args:
        arg (ClosedFormSolverArgument): closed-form solver argument
    """
    
    dest = {}
    
    handler = [
        arg.block_q.self_attn.o_proj.register_forward_hook(_make_inp_hook(dest, "x_q")),
        arg.block_f.self_attn.o_proj.register_forward_hook(_make_inp_hook(dest, "x_f")),
        arg.block_q.input_layernorm.register_forward_hook(_make_inp_hook(dest, "h_q")),
        arg.block_f.input_layernorm.register_forward_hook(_make_inp_hook(dest, "h_f")),
    ]

    _closed_form_solver(arg, arg.block_q.self_attn.o_proj, dest)

    for h in handler:
        h.remove()


def closed_form_solver_down_proj(arg: ClosedFormSolverArgument) -> None:
    """Closed-form solver for llama mlp down_proj
    Args:
        arg (ClosedFormSolverArgument): closed-form solver argument
    """

    dest = {}
    
    handler = [
        arg.block_q.mlp.down_proj.register_forward_hook(_make_inp_hook(dest, "x_q")),
        arg.block_f.mlp.down_proj.register_forward_hook(_make_inp_hook(dest, "x_f")),
        arg.block_q.post_attention_layernorm.register_forward_hook(_make_inp_hook(dest, "h_q")),
        arg.block_f.post_attention_layernorm.register_forward_hook(_make_inp_hook(dest, "h_f")),
    ]

    _closed_form_solver(arg, arg.block_q.mlp.down_proj, dest)

    for h in handler:
        h.remove()
