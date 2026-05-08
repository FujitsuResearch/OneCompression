"""LPCD (Layer-Projected Coordinate Descent) module.

Implements the LPCD unified framework from arXiv:2512.01546.
Extends layer-wise PTQ by optimizing relaxed objectives across
submodule groups (QK, VO, MLP) and projecting solutions with
layer-wise quantizers.

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from ._lpcd_config import LPCDConfig

__all__ = [
    "LPCDConfig",
]
