"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

from onecomp.quantizer.dbf import DBF
from onecomp.utils import effective_bits_for_quantizer
from onecomp.utils.device import is_mps_device

MPS_DBF_FALLBACK_ERROR = (
    "AutoBitQuantizer DBF fallback is not supported on MPS device. "
    "DBF quantization is not implemented for MPS. "
    "Use a higher target_bit (above dbf_threshold), set auto_dbf=False, "
    "or run on CPU/CUDA."
)


def reject_mps_dbf_fallback(device) -> None:
    """Raise when AutoBit would run DBF fallback on MPS."""
    if is_mps_device(device):
        raise ValueError(MPS_DBF_FALLBACK_ERROR)


def inject_dbf(
    assignments,
    quantizers,
    threshold,
    logger,
    dbf_iters=None,
    device=None,
):
    """Inject DBF for ultra-low-bit assignments."""
    raw_of = _raw_bits_map(quantizers)

    total_params = 0
    weighted_eff = 0.0
    for _name, module, child_q in assignments:
        in_f = module.weight.shape[1]
        e = effective_bits_for_quantizer(child_q, in_features=in_f)
        p = module.weight.numel()
        weighted_eff += e * p
        total_params += p

    avg_eff = weighted_eff / total_params if total_params > 0 else 0
    if avg_eff > threshold:
        return assignments

    reject_mps_dbf_fallback(device)

    logger.info(
        "DBF fallback: avg effective bpw %.2f < threshold %.2f",
        avg_eff,
        threshold,
    )

    dbf_cache: dict = {}
    new_assignments = []

    for name, module, child_q in assignments:
        in_f = module.weight.shape[1]
        eff = effective_bits_for_quantizer(child_q, in_features=in_f)
        raw = raw_of[id(child_q)]

        if eff <= threshold and not isinstance(child_q, DBF):
            if raw not in dbf_cache:
                dbf_kwargs = {"target_bits": float(raw)}
                if dbf_iters is not None:
                    dbf_kwargs["iters"] = dbf_iters
                dbf_q = DBF(**dbf_kwargs)
                dbf_cache[raw] = dbf_q
                quantizers.append(dbf_q)
                logger.info(
                    "Created %s (target_bits=%.1f, no group overhead)",
                    dbf_q.name,
                    raw,
                )
            new_assignments.append((name, module, dbf_cache[raw]))
        else:
            new_assignments.append((name, module, child_q))

    return new_assignments


def _raw_bits_map(quantizers):
    """``id(q) → raw bits`` for every quantizer."""
    m: dict = {}
    for q in quantizers:
        for attr in ("wbits", "bits", "target_bits"):
            val = getattr(q, attr, None)
            if val is not None:
                m[id(q)] = val
                break
    return m
