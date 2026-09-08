"""
OneComp Global PTQ.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from .__version__ import __version__
from .global_ptq import GlobalPTQ, GlobalPTQDistributed

__all__ = [
    "GlobalPTQ",
    "GlobalPTQDistributed",
]
