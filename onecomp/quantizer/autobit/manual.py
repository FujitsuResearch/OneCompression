"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""


def assign_manually(quantizer, model):
    """Assign layers using each child quantizer's selection criteria."""
    logger = quantizer.logger
    assignments = []

    for name, module in model.named_modules():
        for child_q in quantizer.quantizers:
            if child_q._should_quantize_layer(name, module):
                assignments.append((name, module, child_q))
                logger.info("Assigned layer '%s' -> %s", name, child_q.name)
                break

        if quantizer.num_layers is not None and len(assignments) >= quantizer.num_layers:
            break

    if quantizer.num_layers is not None:
        assert len(assignments) == quantizer.num_layers, (
            f"Expected {quantizer.num_layers} layers, " f"but found {len(assignments)}"
        )

    return assignments
