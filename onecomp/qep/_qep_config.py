"""

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)

"""

from dataclasses import dataclass, field


@dataclass
class QEPConfig:
    """Configuration for Quantization Error Propagation (QEP).

    Attributes:
        general (bool): If True, use the generic (architecture-independent)
            implementation. If False, use the architecture-aware
            implementation that exploits shared activations (e.g., QKV
            layers in Llama sharing the same input activations).
            Default is False.
        percdamp (float): Damping percentage for Hessian regularization.
            Default is 0.01.
        perccorr (float): Correction percentage for error propagation.
            Default is 0.5.
        exclude_layer_keywords (list[str]): List of keywords to identify
            layers excluded from error propagation. Layers whose names
            contain any of these keywords will be excluded.
            Default is ``["mlp.down_proj"]``.

    Examples:
        >>> config = QEPConfig()
        >>> config.percdamp
        0.01

        >>> # Generic implementation
        >>> config = QEPConfig(general=True, percdamp=0.05, perccorr=0.3)

        >>> # Architecture-aware implementation (default)
        >>> config = QEPConfig(general=False)

        >>> config = QEPConfig(exclude_layer_keywords=["mlp.down_proj", "mlp.gate_proj"])

    Note:
        The default ``exclude_layer_keywords`` is designed for Llama-like
        architectures and may need to be adjusted for other model families.
    """

    general: bool = False
    percdamp: float = 0.01
    perccorr: float = 0.5
    exclude_layer_keywords: list[str] = field(default_factory=lambda: ["mlp.down_proj"])
    # TODO: exclude_layer_keywords depends on the architecture and needs to be fixed
