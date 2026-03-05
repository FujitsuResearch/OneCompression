"""

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)

"""

try:
    from ._jointq import JointQ
except ImportError:  # Avoid error even if the jointq package is not installed
    JointQ = None
