"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

from onecomp.utils import effective_bits_for_quantizer
from onecomp.quantizer.dbf import DBF


def inject_dbf(assignments, quantizers, threshold, logger, dbf_iters=None):
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
