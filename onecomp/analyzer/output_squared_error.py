"""Output squared error sum-ratio visualization.

Typical Usage
-------------

>>> from onecomp.analyzer import plot_output_squared_error_sum_ratios
>>>
>>> plot_output_squared_error_sum_ratios(
...     files={
...         "GPTQ_4bit": "quantization_statistics_GPTQ_4bit.json",
...         "JointQ_4bit": "quantization_statistics_JointQ_4bit.json",
...     },
...     output_dir="./figures_output_error_sum_ratios",
...     markers={"JointQ_4bit": "s"},
...     title="Output squared error sum ratios",
...     ylabel="Output error sum ratio",
... )

Custom Category Usage
---------------------

``category_by_layer`` replaces the default mapping. Repeated category labels
are summed and their first appearance determines the x-axis order.

>>> plot_output_squared_error_sum_ratios(
...     files={
...         "GPTQ_4bit": "quantization_statistics_GPTQ_4bit.json",
...         "JointQ_4bit": "quantization_statistics_JointQ_4bit.json",
...     },
...     output_dir="./figures_output_error_group_ratios",
...     category_by_layer={
...         "q_proj": "attention",
...         "k_proj": "attention",
...         "v_proj": "attention",
...         "o_proj": "attention",
...         "gate_proj": "mlp",
...         "up_proj": "mlp",
...         "down_proj": "mlp",
...     },
...     markers={"JointQ_4bit": "s"},
...     title="Attention and MLP output error ratios",
... )

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import json
import math
import re
from pathlib import Path
from typing import Optional, TypedDict, Union, Unpack

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

_LAYER_PATTERN = re.compile(r"model\.layers\.(\d+)\.(.+)")
_DEFAULT_CATEGORY_BY_LAYER = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "o_proj": "o",
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
}


class _PlotTextOptions(TypedDict, total=False):
    title: Optional[str]
    ylabel: Optional[str]
    legend_location: Optional[Union[str, int]]
    ylim: Optional[tuple[Optional[float], Optional[float]]]
    show_values: bool
    figsize: Optional[tuple[float, float]]


def _resolve_categories(category_by_layer):
    """Validate the layer-to-category mapping and preserve category order."""
    mapping = (
        dict(_DEFAULT_CATEGORY_BY_LAYER) if category_by_layer is None else dict(category_by_layer)
    )
    if not mapping:
        raise ValueError("category_by_layer must contain at least one layer mapping.")
    if any(
        not isinstance(layer, str) or not layer or not isinstance(category, str) or not category
        for layer, category in mapping.items()
    ):
        raise ValueError("category_by_layer keys and values must be non-empty strings.")

    categories = tuple(dict.fromkeys(mapping.values()))
    if "total" in categories:
        raise ValueError("'total' is reserved for the sum of all layer categories.")
    return mapping, categories


def _validate_output_squared_error(values, label, layer_name):
    """Return a valid output squared error for one supported layer."""
    if "output_squared_error" not in values:
        raise ValueError(f"Missing output_squared_error for '{label}' layer '{layer_name}'.")

    error = values["output_squared_error"]
    if (
        isinstance(error, bool)
        or not isinstance(error, (int, float))
        or not math.isfinite(error)
        or error < 0
    ):
        raise ValueError(
            f"Invalid output_squared_error for '{label}' layer " f"'{layer_name}': {error!r}."
        )
    return error


def _sum_file_output_squared_errors(label, path, mapping, categories):
    """Sum one method's output squared errors by configured category."""
    category_sums = dict.fromkeys(categories, 0.0)
    supported_layer_count = 0
    with open(path, encoding="utf-8") as f:
        stats = json.load(f)

    for layer_name, values in stats.items():
        match = _LAYER_PATTERN.match(layer_name)
        if not match:
            continue

        layer_basename = re.split(r"[./]", match.group(2))[-1]
        category = mapping.get(layer_basename)
        if category is None:
            continue

        category_sums[category] += _validate_output_squared_error(values, label, layer_name)
        supported_layer_count += 1

    if supported_layer_count == 0:
        raise ValueError(f"No supported output_squared_error layer data found for '{label}'.")

    category_sums["total"] = sum(category_sums.values())
    return category_sums


def _load_output_squared_error_sums(files, category_by_layer):
    """Load output squared error sums for all methods."""
    if len(files) < 2:
        raise ValueError("files must contain at least two labels for comparison.")

    mapping, categories = _resolve_categories(category_by_layer)
    sums = {
        label: _sum_file_output_squared_errors(label, path, mapping, categories)
        for label, path in files.items()
    }
    return sums, categories


def _calculate_sum_ratios(sums, labels, categories):
    """Calculate category-sum ratios to the first method."""
    categories = [*categories, "total"]
    baseline_sums = sums[labels[0]]
    ratios = {}
    for label in labels[1:]:
        ratios[label] = [
            (
                np.nan
                if baseline_sums[category] == 0
                else sums[label][category] / baseline_sums[category]
            )
            for category in categories
        ]
    return categories, ratios


def _build_marker_styles(labels, markers):
    """Resolve marker and color styles for each method label."""
    markers = {} if markers is None else markers
    unknown_labels = set(markers) - set(labels)
    if unknown_labels:
        unknown = ", ".join(sorted(unknown_labels))
        raise ValueError(f"Unknown method labels in plot styles: {unknown}.")

    cmap = plt.get_cmap("tab10")
    return {
        label: {"color": cmap(index % 10), "marker": markers.get(label, "o")}
        for index, label in enumerate(labels)
    }


def _plot_ratio_series(ax, categories, ratios, styles, show_values):
    """Plot output squared error sum ratios as unconnected markers."""
    for series_index, (label, values) in enumerate(ratios.items()):
        style = styles[label]
        ax.scatter(
            categories,
            values,
            color=style["color"],
            marker=style["marker"],
            s=36,
            label=label,
            zorder=2,
        )
        if show_values:
            offset = 7 if series_index % 2 == 0 else -7
            vertical_alignment = "bottom" if offset > 0 else "top"
            for category_index, value in enumerate(values):
                if math.isfinite(value):
                    ax.annotate(
                        f"{value:.2f}",
                        (category_index, value),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha="center",
                        va=vertical_alignment,
                        fontsize=10,
                        zorder=3,
                    )


def _configure_ratio_axes(ax, categories, baseline_label, title, ylabel, legend_location, ylim):
    """Configure reference lines, labels, title, and the total highlight."""
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, zorder=1)
    total_index = len(categories) - 1
    ax.axvspan(
        total_index - 0.5,
        total_index + 0.5,
        color="gray",
        alpha=0.12,
        zorder=0,
    )
    if ylim is None:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.set_xlabel("Layer type", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    if ylabel is None:
        ylabel = f"Output Squared Error Sum Ratio to {baseline_label}"
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=18)
    legend_options = {"fontsize": 8}
    if legend_location is not None:
        legend_options["loc"] = legend_location
    ax.legend(**legend_options)
    ax.grid(True, alpha=0.3)


def _save_ratio_plot(fig, output_dir):
    """Save the output squared error sum-ratio figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "output_squared_error_sum_ratios.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {plot_path.name}")


def plot_output_squared_error_sum_ratios(
    files: dict[str, Union[str, Path]],
    output_dir: Union[str, Path] = "./figures_output_squared_error_sum_ratios",
    markers: Optional[dict[str, str]] = None,
    category_by_layer: Optional[dict[str, str]] = None,
    ax: Optional[Axes] = None,
    **text_options: Unpack[_PlotTextOptions],
):
    """Plot category-wise output squared error sums relative to the first method.

    By default, sums are calculated for ``q``, ``k``, ``v``, ``o``, ``gate``,
    ``up``, and ``down`` projection layers. ``total`` is the sum of all configured
    categories. Each method after the first is plotted as unconnected markers at
    ``method_sum / first_method_sum``. Ratios with a zero baseline sum are omitted.

    Args:
        files: ``{label: json_path}`` dict. The first label is the baseline.
        output_dir: Directory to save the PNG file.
        markers: Optional ``{label: matplotlib_marker}`` overrides.
        category_by_layer: Optional ``{layer_basename: category_label}`` mapping.
            Mapping values define category order and duplicate values are summed.
            The label ``"total"`` is reserved for the overall sum.
        ax: Optional axes on which to draw. When omitted, a new figure is created
            and saved to ``output_dir``. When provided, no file is saved.
        title: Optional graph title.
        ylabel: Optional y-axis label. Defaults to a label naming the baseline.
        legend_location: Optional Matplotlib legend location name or code.
        ylim: Optional ``(bottom, top)`` y-axis limits. Either limit may be
            ``None`` to let Matplotlib choose it automatically.
        show_values: Whether to annotate each finite plotted ratio. Defaults to
            ``False``.
        figsize: Optional figure size in inches as ``(width, height)``. When
            ``ax`` is provided, its entire figure is resized.

    Returns:
        The axes containing the plot.
    """
    sums, categories = _load_output_squared_error_sums(files, category_by_layer)
    labels = list(files.keys())
    baseline_label = labels[0]
    styles = _build_marker_styles(labels, markers)
    categories, ratios = _calculate_sum_ratios(sums, labels, categories)

    owns_figure = ax is None
    figsize = text_options.get("figsize")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6) if figsize is None else figsize)
    elif figsize is not None:
        ax.figure.set_size_inches(figsize)
    _plot_ratio_series(ax, categories, ratios, styles, text_options.get("show_values", False))
    _configure_ratio_axes(
        ax,
        categories,
        baseline_label,
        text_options.get("title"),
        text_options.get("ylabel"),
        text_options.get("legend_location"),
        text_options.get("ylim"),
    )
    if owns_figure:
        _save_ratio_plot(ax.figure, output_dir)
    return ax
