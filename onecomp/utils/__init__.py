"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from .perplexity import calculate_perplexity
from .accuracy import calculate_accuracy

from .activation_check import (
    check_activations,
)

from .activation_capture import (
    capture_input_activations,
)

from .vram_estimator import (
    estimate_target_bitwidth,
    estimate_wbits_from_vram,
    effective_bits_per_param,
    raw_bits_for_quantizer,
    effective_bits_for_quantizer,
    weight_memory_gb,
    VRAMBitwidthEstimation,
)

from .model_inputs import add_model_specific_inputs

from .blockwise import (
    get_blocks_and_inputs,
    forward_input,
    move_kwargs_to_device,
    expand_kwargs_batch,
)

from .dtype import needs_bfloat16
