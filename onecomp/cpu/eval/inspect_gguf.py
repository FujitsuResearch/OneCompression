"""Inspect a GGUF file: per-tensor quant types, size and effective bit-width.

Especially useful for mixed-precision GGUFs produced by the llama.cpp plugin,
where each layer can carry a different quantization type. Reports the on-disk
type breakdown, the effective bits-per-weight, and a per-block-index view of how
each transformer block's tensors were quantized.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from logging import getLogger
from typing import Dict, List, Optional

logger = getLogger(__name__)

_BLOCK_RE = re.compile(r"\bblk\.(\d+)\.")


@dataclass
class TensorInfo:
    """One GGUF tensor's shape / type / size."""

    name: str
    ggml_type: str
    shape: tuple
    n_elements: int
    n_bytes: int

    @property
    def bits_per_weight(self) -> float:
        return (self.n_bytes * 8.0 / self.n_elements) if self.n_elements else 0.0


@dataclass
class GGUFReport:
    """Aggregate inspection of a GGUF file."""

    path: str
    architecture: str
    n_tensors: int
    total_bytes: int
    total_elements: int
    type_counts: Dict[str, int]
    type_bytes: Dict[str, int]
    tensors: List[TensorInfo] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def effective_bits_per_weight(self) -> float:
        return (self.total_bytes * 8.0 / self.total_elements) if self.total_elements else 0.0

    def per_block_types(self) -> Dict[int, Dict[str, str]]:
        """Return ``{block_index: {tensor_suffix: ggml_type}}`` for ``blk.N.*`` tensors."""
        out: Dict[int, Dict[str, str]] = {}
        for t in self.tensors:
            m = _BLOCK_RE.search(t.name)
            if not m:
                continue
            idx = int(m.group(1))
            suffix = t.name[m.end() :]
            out.setdefault(idx, {})[suffix] = t.ggml_type
        return dict(sorted(out.items()))

    def summary(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "architecture": self.architecture,
            "n_tensors": self.n_tensors,
            "total_MiB": round(self.total_bytes / 2**20, 2),
            "effective_bits_per_weight": round(self.effective_bits_per_weight, 4),
            "type_counts": self.type_counts,
            "type_MiB": {k: round(v / 2**20, 2) for k, v in self.type_bytes.items()},
        }


def inspect_gguf(path: str, keep_tensors: bool = True) -> GGUFReport:
    """Read a GGUF file and return a :class:`GGUFReport`.

    Args:
        path: Path to a ``.gguf`` file.
        keep_tensors: If False, drop the per-tensor list (only keep aggregates).
    """
    import gguf

    reader = gguf.GGUFReader(path)
    try:
        architecture = reader.get_field("general.architecture").contents()
    except Exception:  # pragma: no cover - defensive
        architecture = "?"

    type_counts: Dict[str, int] = {}
    type_bytes: Dict[str, int] = {}
    tensors: List[TensorInfo] = []
    total_bytes = 0
    total_elements = 0

    for t in reader.tensors:
        tname = t.tensor_type.name
        n_elements = int(t.n_elements)
        n_bytes = int(t.n_bytes)
        type_counts[tname] = type_counts.get(tname, 0) + 1
        type_bytes[tname] = type_bytes.get(tname, 0) + n_bytes
        total_bytes += n_bytes
        total_elements += n_elements
        if keep_tensors:
            tensors.append(
                TensorInfo(
                    name=t.name,
                    ggml_type=tname,
                    shape=tuple(int(x) for x in reversed(t.shape)),
                    n_elements=n_elements,
                    n_bytes=n_bytes,
                )
            )

    return GGUFReport(
        path=path,
        architecture=architecture,
        n_tensors=len(reader.tensors),
        total_bytes=total_bytes,
        total_elements=total_elements,
        type_counts=dict(sorted(type_counts.items())),
        type_bytes=dict(sorted(type_bytes.items())),
        tensors=tensors,
    )


def format_report(report: GGUFReport, max_block_rows: Optional[int] = 4) -> str:
    """Render a human-readable multi-line string for a :class:`GGUFReport`."""
    lines: List[str] = []
    lines.append(f"GGUF: {report.path}")
    lines.append(f"  architecture : {report.architecture}")
    lines.append(f"  tensors      : {report.n_tensors}")
    lines.append(f"  size         : {report.total_bytes / 2**20:.2f} MiB")
    lines.append(f"  eff. bits/w  : {report.effective_bits_per_weight:.4f}")
    lines.append("  type breakdown:")
    for tname in report.type_counts:
        cnt = report.type_counts[tname]
        mib = report.type_bytes[tname] / 2**20
        lines.append(f"    {tname:<8} x{cnt:<4} {mib:8.2f} MiB")

    blocks = report.per_block_types()
    if blocks:
        lines.append("  per-block tensor types:")
        shown = list(blocks.items())
        if max_block_rows is not None:
            shown = shown[:max_block_rows]
        for idx, suffixes in shown:
            joined = ", ".join(f"{k}={v}" for k, v in sorted(suffixes.items()))
            lines.append(f"    blk.{idx}: {joined}")
        if max_block_rows is not None and len(blocks) > max_block_rows:
            lines.append(f"    ... ({len(blocks) - max_block_rows} more blocks)")
    return "\n".join(lines)
