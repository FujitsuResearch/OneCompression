"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""


def assign_manually(quantizer, model):
    """Assign layers using each child quantizer's selection criteria.

    Bit allocation is performed on the text submodel for multi-modal models.
    """
    logger = quantizer.logger
    search_root, prefix = quantizer._get_text_search_root(model)

    assignments = []
    for name, module in search_root.named_modules():
        full_name = prefix + name if prefix else name
        for child_q in quantizer.quantizers:
            if child_q._should_quantize_layer(full_name, module):
                assignments.append((full_name, module, child_q))
                logger.info("Assigned layer '%s' -> %s", full_name, child_q.name)
                break

        if quantizer.num_layers is not None and len(assignments) >= quantizer.num_layers:
            break

    if quantizer.num_layers is not None:
        assert len(assignments) == quantizer.num_layers, (
            f"Expected {quantizer.num_layers} layers, " f"but found {len(assignments)}"
        )

    return assignments
