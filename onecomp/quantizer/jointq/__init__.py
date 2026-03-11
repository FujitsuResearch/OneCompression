"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

try:
    from ._jointq import JointQ
except ImportError:  # Avoid error even if the jointq package is not installed
    JointQ = None
