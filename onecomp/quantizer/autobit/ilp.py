"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

import torch
import torch.nn.functional as F
from ortools.linear_solver import pywraplp

from onecomp.utils import raw_bits_for_quantizer, effective_bits_for_quantizer
from onecomp.quantizer.autobit.activation_stats import collect_activation_stats_blockwise


def assign_by_ilp(quantizer, model, *, use_activation=False):
    """Assign layers by solving an ILP under an effective-bpw budget.

    ILP-based bit allocation for AutoBitQuantizer.

    Solves an Integer Linear Program that minimises quantisation error
    subject to an effective-bpw budget.  Two error metrics are supported:

        - RTN error  — ``||W - Ŵ||²_F``
        - Activation-aware error (``use_activation=True``) —
        ``Σ_{q,p} b_q * a_p * (ΔW_{qp})²``

        where ``a_p : (d_in,)`` (input Gram diagonal) and
        ``b_q : (d_out,)`` (output curvature diagonal) are the activation and
        curvature statistics collected via hooks.

    Args:
        quantizer: ``AutoBitQuantizer`` instance.
        model: The model to quantise.
        use_activation: When ``True``, collect activation / curvature
            statistics and use the activation-aware error metric.
    """
    logger = quantizer.logger

    if quantizer.target_bit is None:
        raise ValueError(
            "target_bit must be set when using the ILP strategy. "
            "Example: AutoBitQuantizer(..., target_bit=4.0)"
        )

    candidates = _find_candidates(quantizer, model)

    raw_bits = [raw_bits_for_quantizer(q) for q in quantizer.quantizers]
    if any(b is None for b in raw_bits):
        bad = [q.name for q, b in zip(quantizer.quantizers, raw_bits) if b is None]
        raise ValueError(
            f"Cannot infer bit-width from quantizer(s): {bad}. "
            "Each child quantizer must have a 'wbits', 'bits', "
            "or 'target_bits' attribute."
        )
    eff_matrix = _effective_bits_matrix(quantizer.quantizers, candidates)
    n_modules = len(candidates)

    tag = "Activation-Aware ILP" if use_activation else "ILP"
    logger.info(
        "%s: %d modules × %d candidates (raw %s), target=%.2f bpw",
        tag,
        n_modules,
        len(raw_bits),
        raw_bits,
        quantizer.target_bit,
    )

    # 1. activation / curvature collection (optional)
    a_diag = None
    b_diag = None
    if use_activation:
        a_diag, b_diag = collect_activation_stats_blockwise(
            model,
            candidates,
            num_samples=quantizer.num_calib_samples,
            seqlen=quantizer.calib_seqlen,
            use_curvature_b=quantizer.use_curvature_b,
            logger=logger,
        )

    # 2. error evaluation
    errors, params_per_module = _evaluate_errors(
        candidates,
        raw_bits,
        a_diag=a_diag,
        b_diag=b_diag,
        logger=logger,
    )

    # 3. solve ILP
    chosen = _solve_ilp(
        errors,
        params_per_module,
        eff_matrix,
        quantizer.target_bit,
        logger,
    )

    # 4. build assignments
    assignments = []
    for i, (name, module) in enumerate(candidates):
        bit_idx = chosen[i]
        child_q = quantizer.quantizers[bit_idx]
        assignments.append((name, module, child_q))
        logger.info(
            "Assigned '%s' -> %s (raw %s-bit, eff %.4f bpw, error=%.4e)",
            name,
            child_q.name,
            raw_bits[bit_idx],
            eff_matrix[i][bit_idx],
            errors[i][bit_idx],
        )

    total_params = sum(params_per_module)
    weighted_eff = sum(eff_matrix[i][chosen[i]] * params_per_module[i] for i in range(n_modules))
    logger.info(
        "%s result: %.4f effective bpw (target = %.2f)",
        tag,
        weighted_eff / total_params if total_params > 0 else 0,
        quantizer.target_bit,
    )

    return assignments


def _find_candidates(quantizer, model):
    """Collect quantisable layers in model order."""
    candidates = []
    for name, module in model.named_modules():
        if quantizer._should_quantize_layer(name, module):
            candidates.append((name, module))
        if quantizer.num_layers is not None and len(candidates) >= quantizer.num_layers:
            break

    if quantizer.num_layers is not None:
        assert len(candidates) == quantizer.num_layers, (
            f"Expected {quantizer.num_layers} layers, " f"but found {len(candidates)}"
        )
    return candidates


def _effective_bits_matrix(quantizers, candidates):
    """Build ``[n_modules × n_candidates]`` effective-bpw matrix.

    Each entry accounts for the actual ``in_features`` of that module,
    so per-channel (``groupsize=-1``) overhead is precisely reflected.
    """
    matrix = []
    for _name, module in candidates:
        in_f = module.weight.shape[1]
        matrix.append([effective_bits_for_quantizer(q, in_features=in_f) for q in quantizers])
    return matrix


def _evaluate_errors(candidates, raw_bits, *, a_diag=None, b_diag=None, logger):
    """Evaluate RTN errors for all (layer, candidate) pairs."""

    n_modules = len(candidates)
    errors = [None] * n_modules
    params_per_module = [0] * n_modules

    for i, (name, module) in enumerate(candidates):
        a = a_diag[name] if a_diag is not None else None
        b = b_diag[name] if b_diag is not None else None
        errors[i] = _rtn_errors(module.weight.data, raw_bits, a_diag=a, b_diag=b)
        params_per_module[i] = module.weight.numel()

    tag = "Activation-aware" if a_diag is not None else "RTN"
    logger.info(
        "%s evaluation: %d modules × %d candidates",
        tag,
        n_modules,
        len(raw_bits),
    )
    return errors, params_per_module


def _rtn_errors(weight, candidate_bits, *, a_diag=None, b_diag=None):
    """RTN quantisation error for each candidate bit-width.

    When both ``a_diag`` and ``b_diag`` are provided, returns the
    activation-weighted error (diagonal approximation of ΔL):

        Σ_{q,p}  b_q · a_p · (ΔW_{qp})²

    Otherwise returns the plain Frobenius norm squared ``||W - Ŵ||²_F``.
    """
    w = weight.float()
    if w.ndim > 2:
        w = w.flatten(1)

    use_activation = a_diag is not None and b_diag is not None
    if use_activation:
        a = a_diag.to(w.device).float()  # (d_in,)
        b = b_diag.to(w.device).float()  # (d_out,)

    w_min = w.min(dim=1, keepdim=True).values
    w_max = w.max(dim=1, keepdim=True).values

    errors = []
    for bits in candidate_bits:
        maxq = 2**bits - 1
        scale = ((w_max - w_min) / maxq).clamp(min=1e-10)
        zero = torch.round(-w_min / scale)
        w_q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
        w_deq = scale * (w_q - zero)
        dw = w - w_deq
        if use_activation:
            error = b @ (dw.pow(2) @ a)
        else:
            error = dw.pow(2).sum()
        errors.append(error.item())
    return errors


def _solve_ilp(errors, params_per_module, eff_matrix, target_bit, logger):
    """Solve the assignment ILP via OR-Tools SCIP solver.

    Args:
        errors: ``[n_modules][n_candidates]`` quantisation error matrix.
        params_per_module: parameter count per module.
        eff_matrix: ``[n_modules][n_candidates]`` effective bpw matrix
            (each entry includes the per-module scale/zero overhead).
        target_bit: target average effective bpw.
        logger: logger instance.
    """
    n_modules = len(errors)
    n_bits = len(eff_matrix[0])
    total_params = sum(params_per_module)
    bit_budget = target_bit * total_params

    max_err = max(max(mod_errs) for mod_errs in errors) if errors else 1.0
    if max_err <= 0:
        max_err = 1.0
    norm_errors = [[e / max_err for e in mod_errs] for mod_errs in errors]
    logger.info(
        "ILP error normalisation: max_err=%.4e, " "normalised range=[%.4e, %.4e]",
        max_err,
        min(min(me) for me in norm_errors),
        1.0,
    )

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError(
            "Failed to create SCIP solver. " "Ensure OR-Tools is installed: pip install ortools"
        )

    x = [[solver.IntVar(0, 1, f"x_{m}_{b}") for b in range(n_bits)] for m in range(n_modules)]

    logger.info(
        "SCIP ILP: %d modules × %d candidates = %d variables",
        n_modules,
        n_bits,
        solver.NumVariables(),
    )

    for m in range(n_modules):
        solver.Add(sum(x[m][b] for b in range(n_bits)) == 1)

    budget_expr = sum(
        eff_matrix[m][b] * params_per_module[m] * x[m][b]
        for m in range(n_modules)
        for b in range(n_bits)
    )
    solver.Add(budget_expr <= bit_budget)

    objective = sum(norm_errors[m][b] * x[m][b] for m in range(n_modules) for b in range(n_bits))
    solver.Minimize(objective)

    status = solver.Solve()

    if status == pywraplp.Solver.FEASIBLE:
        logger.warning("SCIP found a feasible but not proven-optimal solution")
    elif status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        min_eff = min(eff_matrix[m][b] for m in range(n_modules) for b in range(n_bits))
        raise RuntimeError(
            f"ILP solver failed (status={status}). "
            f"target_bit={target_bit} may be infeasible "
            f"(min candidate eff={min_eff:.4f})."
        )

    chosen = []
    for m in range(n_modules):
        for b in range(n_bits):
            if x[m][b].solution_value() > 0.5:
                chosen.append(b)
                break
        else:
            raise RuntimeError(f"No bit-width assigned to module {m}")

    norm_obj = solver.Objective().Value()
    raw_obj = norm_obj * max_err
    logger.info("ILP solved: total error = %.6e (normalised obj = %.6e)", raw_obj, norm_obj)
    return chosen
