"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# ------------------------------------------------------------------
# Colour scheme
# ------------------------------------------------------------------

_BIT_CMAP_COLORS = [
    "#1a237e",
    "#283593",
    "#3949ab",
    "#5c6bc0",
    "#7986cb",
    "#9fa8da",
    "#c5cae9",
    "#ffffff",
]

_QUANTIZER_PALETTE = [
    "#2196F3",
    "#4CAF50",
    "#FF5722",
    "#9C27B0",
    "#FF9800",
    "#00BCD4",
    "#E91E63",
    "#795548",
    "#607D8B",
    "#8BC34A",
]


# ------------------------------------------------------------------
# Layer-name parser
# ------------------------------------------------------------------

_BLOCK_RE = re.compile(r"^(.*?)\.(\d+)\.(.*)")


def _parse_layer_structure(
    layer_names: Sequence[str],
) -> Optional[Tuple[List[int], List[str], List[str], int]]:
    """Extract (block_index, module_type) from transformer-style names.

    Returns ``None`` if any name fails to match the ``prefix.{idx}.rest``
    pattern.

    Returns:
        block_indices  – block index for each layer
        module_types   – module-type string for each layer
        unique_modules – ordered unique module types
        num_blocks     – total number of blocks
    """
    block_indices: List[int] = []
    module_types: List[str] = []

    for name in layer_names:
        m = _BLOCK_RE.match(name)
        if m is None:
            return None
        block_indices.append(int(m.group(2)))
        module_types.append(m.group(3))

    seen: dict = {}
    for mt in module_types:
        if mt not in seen:
            seen[mt] = len(seen)
    unique_modules = list(seen.keys())

    num_blocks = max(block_indices) + 1 if block_indices else 0
    return block_indices, module_types, unique_modules, num_blocks


# ------------------------------------------------------------------
# Short label for module types
# ------------------------------------------------------------------

_SHORT_LABELS: Dict[str, str] = {
    "self_attn.q_proj": "Q",
    "self_attn.k_proj": "K",
    "self_attn.v_proj": "V",
    "self_attn.o_proj": "O",
    "mlp.gate_proj": "Gate",
    "mlp.up_proj": "Up",
    "mlp.down_proj": "Down",
}


def _short_module_label(module_type: str) -> str:
    return _SHORT_LABELS.get(module_type, module_type.rsplit(".", 1)[-1])


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def visualize_bit_assignment(
    layer_names: Sequence[str],
    layer_bits: Sequence[float],
    layer_quantizer_names: Sequence[str],
    *,
    layer_params: Optional[Sequence[int]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
    dpi: int = 150,
) -> Optional[str]:
    """Visualise the bit-width assignment produced by AutoBitQuantizer.

    Generates a heatmap showing per-layer / per-module bit-width assignments.
    Transformer block structure is auto-detected from layer names
    (``model.layers.{i}.self_attn.q_proj`` etc.).  Falls back to a bar chart
    when the naming convention is not recognised.

    No external dependencies beyond numpy / matplotlib.

    Args:
        layer_names:           Layer names in model order.
        layer_bits:            Bit-width assigned to each layer.
        layer_quantizer_names: Name of the child quantizer for each layer.
        layer_params: Parameter count per layer (for weighted average).
            When ``None``, a simple average is shown instead.
        save_path:  Directory to save PNG (``None`` = don't save).
        title:      Plot title.
        figsize:    Figure size ``(width, height)``.
        dpi:        Resolution.

    Returns:
        Path to the saved figure, or ``None`` if not saved.
    """
    parsed = _parse_layer_structure(layer_names)

    if parsed is not None:
        fig_path = _plot_heatmap(
            parsed,
            layer_bits,
            layer_quantizer_names,
            layer_params=layer_params,
            save_path=save_path,
            title=title,
            figsize=figsize,
            dpi=dpi,
        )
    else:
        fig_path = _plot_bar(
            layer_names,
            layer_bits,
            layer_quantizer_names,
            layer_params=layer_params,
            save_path=save_path,
            title=title,
            figsize=figsize,
            dpi=dpi,
        )

    return fig_path


# ------------------------------------------------------------------
# Heatmap renderer
# ------------------------------------------------------------------


def _weighted_avg(bits, params):
    """Parameter-weighted average bit-width."""
    if params is not None:
        total = sum(params)
        if total > 0:
            return sum(b * p for b, p in zip(bits, params)) / total
    return float(np.nanmean(bits))


def _plot_heatmap(
    parsed,
    layer_bits,
    layer_quantizer_names,
    *,
    layer_params,
    save_path,
    title,
    figsize,
    dpi,
) -> Optional[str]:
    block_indices, module_types, unique_modules, num_blocks = parsed
    n_modules = len(unique_modules)

    mod_to_row = {m: i for i, m in enumerate(unique_modules)}

    matrix = np.full((n_modules, num_blocks), np.nan)
    annot = np.empty((n_modules, num_blocks), dtype=object)
    annot[:] = ""

    qname_matrix = np.empty((n_modules, num_blocks), dtype=object)
    qname_matrix[:] = ""

    for bidx, mtype, bits, qname in zip(
        block_indices,
        module_types,
        layer_bits,
        layer_quantizer_names,
    ):
        row = mod_to_row[mtype]
        matrix[row, bidx] = bits
        bits_str = f"{bits:.0f}" if bits == int(bits) else f"{bits:.1f}"
        annot[row, bidx] = bits_str
        qname_matrix[row, bidx] = qname

    quantized_vals = [b for b in layer_bits if b < 16]
    vmin = min(quantized_vals) if quantized_vals else 1.0
    vmax = max(quantized_vals) if quantized_vals else 8.0
    cmap = LinearSegmentedColormap.from_list("bits", _BIT_CMAP_COLORS, N=256)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    for row in range(n_modules):
        for col in range(num_blocks):
            txt = annot[row, col]
            if txt:
                color = "white" if matrix[row, col] < (vmin + vmax) / 2 else "black"
                ax.text(
                    col,
                    row,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color=color,
                )

    ax.set_xticks(range(0, num_blocks, max(1, num_blocks // 20)))
    ax.set_xticklabels(
        [f"L{i}" for i in range(0, num_blocks, max(1, num_blocks // 20))],
        fontsize=7,
    )
    ax.set_yticks(range(n_modules))
    ax.set_yticklabels(
        [_short_module_label(m) for m in unique_modules],
        fontsize=9,
    )

    ax.set_xticks(np.arange(-0.5, num_blocks, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_modules, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.3)
    ax.tick_params(which="minor", size=0)

    ax.set_xlabel("Layer Block", fontsize=11, fontweight="bold")
    ax.set_ylabel("Module", fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Bit-width", fontsize=10)

    unique_qnames = sorted(set(layer_quantizer_names))
    if len(unique_qnames) > 1:
        qn_to_bits: Dict[str, List[float]] = {}
        for qn, bits in zip(layer_quantizer_names, layer_bits):
            qn_to_bits.setdefault(qn, []).append(bits)
        norm = Normalize(vmin=vmin, vmax=vmax)
        handles = []
        for qn in unique_qnames:
            avg_bits = float(np.mean(qn_to_bits[qn]))
            color = cmap(norm(np.clip(avg_bits, vmin, vmax)))
            handles.append(
                mpatches.Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=qn)
            )
        ax.legend(
            handles=handles,
            title="Quantizer",
            title_fontsize=9,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.12, 1.0),
            frameon=True,
            fancybox=True,
        )

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    avg_bit = _weighted_avg(layer_bits, layer_params)
    avg_label = "Weighted avg" if layer_params is not None else "Simple avg"
    fig.text(
        0.5,
        -0.02,
        f"{avg_label} bit-width: {avg_bit:.3f} bpw  |  Layers: {len(layer_bits)}",
        ha="center",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout()
    return _save_fig(fig, save_path, "bit_assignment_heatmap", dpi)


# ------------------------------------------------------------------
# Bar-chart fallback
# ------------------------------------------------------------------


def _plot_bar(
    layer_names,
    layer_bits,
    layer_quantizer_names,
    *,
    layer_params,
    save_path,
    title,
    figsize,
    dpi,
) -> Optional[str]:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    unique_qnames = sorted(set(layer_quantizer_names))
    qn_to_color = {
        qn: _QUANTIZER_PALETTE[i % len(_QUANTIZER_PALETTE)] for i, qn in enumerate(unique_qnames)
    }

    colors = [qn_to_color[qn] for qn in layer_quantizer_names]
    x = np.arange(len(layer_names))
    ax.bar(x, layer_bits, color=colors, edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax.set_ylabel("Bit-width", fontsize=11, fontweight="bold")

    if len(layer_names) <= 40:
        ax.set_xticks(x)
        short = [n.rsplit(".", 1)[-1] if "." in n else n for n in layer_names]
        ax.set_xticklabels(short, rotation=90, fontsize=6)
    else:
        step = max(1, len(layer_names) // 20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f"L{i}" for i in x[::step]], fontsize=7)

    if len(unique_qnames) > 1:
        handles = [
            mpatches.Patch(facecolor=qn_to_color[qn], edgecolor="black", linewidth=0.5, label=qn)
            for qn in unique_qnames
        ]
        ax.legend(handles=handles, fontsize=8, frameon=True)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    return _save_fig(fig, save_path, "bit_assignment_bar", dpi)


# ------------------------------------------------------------------
# Save helper
# ------------------------------------------------------------------


def _save_fig(fig, save_path, prefix, dpi) -> Optional[str]:
    if save_path is None:
        plt.close(fig)
        return None

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = save_dir / f"{prefix}_{ts}.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)
