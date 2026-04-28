"""Quantization error visualization.

Plot relative_output_squared_error vs relative_weight_squared_error
from quantization statistics JSON files, grouped by layer type.

Typical Usage
-------------

>>> from onecomp.analyzer import plot_quantization_errors
>>>
>>> plot_quantization_errors(
...     files={
...         "GPTQ_4bit": "quantization_statistics_GPTQ_4bit.json",
...         "JointQ_4bit": "quantization_statistics_JointQ_4bit.json",
...     },
...     output_dir="./figures",
... )

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

_LAYER_PATTERN = re.compile(r"model\.layers\.(\d+)\.(.+)")


def _load_data(files: dict[str, Union[str, Path]]):
    """Load JSON files and organize by (label, layer_type, layer_idx).

    Returns:
        dict[str, dict[str, list[tuple]]]:
            ``{layer_type: {label: [(layer_idx, rw%, ro%), ...]}}``
    """
    data = defaultdict(lambda: defaultdict(list))

    for label, path in files.items():
        with open(path, encoding="utf-8") as f:
            stats = json.load(f)

        for key, val in stats.items():
            m = _LAYER_PATTERN.match(key)
            if not m:
                continue
            layer_idx = int(m.group(1))
            layer_type = m.group(2)

            rw = val["relative_weight_squared_error"] * 100
            ro = val["relative_output_squared_error"] * 100

            data[layer_type][label].append((layer_idx, rw, ro))

    for layer_type in data:
        for label in data[layer_type]:
            data[layer_type][label].sort(key=lambda t: t[0])

    return data


def _axis_limit(values, factor=1.5):
    """Compute an upper axis limit that excludes extreme outliers (IQR)."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    upper = q3 + factor * iqr
    return max(upper, np.percentile(values, 90))


def _build_colors(labels):
    """Assign a distinct color to each label."""
    cmap = plt.get_cmap("tab10")
    return {label: cmap(i % 10) for i, label in enumerate(labels)}


def _plot_layer_type(ax, layer_type, label_points, colors):
    """Plot a single layer type onto the given Axes."""
    all_xs, all_ys = [], []
    for points in label_points.values():
        all_xs.extend(rw for _, rw, _ in points)
        all_ys.extend(ro for _, _, ro in points)

    if not all_xs:
        return

    xlim = _axis_limit(all_xs) * 1.15
    ylim = _axis_limit(all_ys) * 1.15

    for label, points in label_points.items():
        xs = [rw for _, rw, _ in points]
        ys = [ro for _, _, ro in points]
        ks = [k for k, _, _ in points]
        color = colors[label]

        ax.scatter(xs, ys, c=[color], label=label, s=40, zorder=3)
        for k, x, y in zip(ks, xs, ys):
            if x > xlim or y > ylim:
                cx = min(x, xlim * 0.92)
                cy = min(y, ylim * 0.92)
                text = f"{k} ({x:.1f}, {y:.2f})"
                ax.annotate(
                    text, (cx, cy),
                    fontsize=6, color=color, fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8),
                )
            else:
                ax.annotate(
                    str(k), (x, y),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color=color,
                )

    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.set_xlabel("Relative Weight Squared Error (%)")
    ax.set_ylabel("Relative Output Squared Error (%)")
    ax.set_title(layer_type)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_quantization_errors(
    files: dict[str, Union[str, Path]],
    output_dir: Union[str, Path] = "./figures",
    dpi: int = 150,
):
    """Plot relative quantization errors from statistics JSON files.

    Produces one scatter plot per layer type (all labels overlaid)
    plus one combined PNG with all layer types in a grid.

    Args:
        files: ``{label: json_path}`` dict. Labels appear in legends.
        output_dir: Directory to save PNG files.
        dpi: Resolution for saved images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_data(files)
    labels = list(files.keys())
    colors = _build_colors(labels)
    layer_types = sorted(data.keys())

    # --- Individual PNGs ---
    for layer_type in layer_types:
        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_layer_type(ax, layer_type, data[layer_type], colors)
        fig.tight_layout()
        safe_name = layer_type.replace(".", "_")
        fig.savefig(output_dir / f"{safe_name}.png", dpi=dpi)
        plt.close(fig)
        print(f"Saved: {safe_name}.png")

    # --- Combined PNG ---
    n = len(layer_types)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows))
    axes_flat = np.asarray(axes).flatten()

    for i, layer_type in enumerate(layer_types):
        _plot_layer_type(axes_flat[i], layer_type, data[layer_type], colors)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "all_layer_types.png", dpi=dpi)
    plt.close(fig)
    print("Saved: all_layer_types.png")
