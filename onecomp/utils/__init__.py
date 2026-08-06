"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from .accuracy import calculate_accuracy
from .activation_capture import capture_input_activations
from .activation_check import check_activations
from .blockwise import (
    expand_kwargs_batch,
    forward_input,
    get_blocks_and_inputs,
    move_kwargs_to_device,
)
from .device import (
    cleanup_memory,
    empty_cache,
    get_default_device,
)
from .dtype import needs_bfloat16
from .model_inputs import add_model_specific_inputs
from .perplexity import calculate_perplexity
from .vram_estimator import (
    VRAMBitwidthEstimation,
    effective_bits_for_quantizer,
    effective_bits_per_param,
    estimate_target_bitwidth,
    estimate_wbits_from_vram,
    raw_bits_for_quantizer,
    weight_memory_gb,
)
