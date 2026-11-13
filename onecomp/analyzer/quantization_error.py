"""Quantization error visualization.

Plot quantization errors from statistics JSON files, grouped by layer type.

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

>>> from onecomp.analyzer import plot_relative_output_squared_errors
>>>
>>> plot_relative_output_squared_errors(
...     files={
...         "GPTQ_4bit": "quantization_statistics_GPTQ_4bit.json",
...         "JointQ_4bit": "quantization_statistics_JointQ_4bit.json",
...     },
...     output_dir="./figures_relative_output_error",
...     linestyles={"GPTQ_4bit": "--", "JointQ_4bit": "-"},
...     markers={"GPTQ_4bit": "s", "JointQ_4bit": "o"},
... )

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

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


def _load_relative_output_squared_errors(files: dict[str, Union[str, Path]]):
    """Load relative output squared errors by layer type and block index."""
    if not files:
        raise ValueError("files must contain at least one label and JSON path.")

    data = defaultdict(lambda: defaultdict(list))
    for label, path in files.items():
        with open(path, encoding="utf-8") as f:
            stats = json.load(f)

        for layer_name, values in stats.items():
            match = _LAYER_PATTERN.match(layer_name)
            if not match:
                continue

            if "relative_output_squared_error" not in values:
                raise ValueError(
                    f"Missing relative_output_squared_error for '{label}' layer "
                    f"'{layer_name}'."
                )

            error = values["relative_output_squared_error"]
            if (
                isinstance(error, bool)
                or not isinstance(error, (int, float))
                or not math.isfinite(error)
                or error < 0
            ):
                raise ValueError(
                    f"Invalid relative_output_squared_error for '{label}' layer "
                    f"'{layer_name}': {error!r}."
                )

            block_idx = int(match.group(1))
            layer_type = match.group(2)
            data[layer_type][label].append((block_idx, error * 100.0))

    if not data:
        raise ValueError("No relative_output_squared_error layer data found in files.")

    for layer_type in data:
        for label in data[layer_type]:
            data[layer_type][label].sort(key=lambda point: point[0])

    return data


def _build_relative_output_plot_options(labels, linestyles, markers, dpi):
    """Resolve plotting options for each method label."""
    linestyles = {} if linestyles is None else linestyles
    markers = {} if markers is None else markers
    known_labels = set(labels)
    unknown_labels = (set(linestyles) | set(markers)) - known_labels
    if unknown_labels:
        unknown = ", ".join(sorted(unknown_labels))
        raise ValueError(f"Unknown method labels in plot styles: {unknown}.")

    colors = _build_colors(labels)
    styles = {
        label: {
            "color": colors[label],
            "linestyle": linestyles.get(label, "-"),
            "marker": markers.get(label, "o"),
        }
        for label in labels
    }
    return {"baseline_label": labels[0], "dpi": dpi, "styles": styles}


def _add_missing_block_gaps(points):
    """Insert NaN points so lines do not span missing block indices."""
    block_indices = []
    values = []
    previous_block_idx = None

    for block_idx, value in points:
        if previous_block_idx is not None and block_idx > previous_block_idx + 1:
            block_indices.append(np.nan)
            values.append(np.nan)
        block_indices.append(block_idx)
        values.append(value)
        previous_block_idx = block_idx

    return block_indices, values


def _plot_block_series(ax, label_points, styles):
    """Plot all method series on a block-indexed axis."""
    for label, points in label_points.items():
        block_indices, values = _add_missing_block_gaps(points)
        ax.plot(
            block_indices,
            values,
            color=styles[label]["color"],
            linestyle=styles[label]["linestyle"],
            marker=styles[label]["marker"],
            markersize=4,
            linewidth=1.5,
            label=label,
            zorder=2,
        )


def _calculate_relative_output_error_ratios(label_points, baseline_label):
    """Calculate each method's error ratio to the baseline by block index."""
    baseline_by_block = dict(label_points.get(baseline_label, []))
    ratios = {}
    for label, points in label_points.items():
        ratios[label] = []
        for block_idx, error in points:
            baseline_error = baseline_by_block.get(block_idx)
            ratio = (
                np.nan if baseline_error is None or baseline_error == 0 else error / baseline_error
            )
            ratios[label].append((block_idx, ratio))
    return ratios


def _configure_block_axis(ax, all_block_indices):
    """Configure an axis to display integer block indices."""
    first_block = all_block_indices[0]
    last_block = all_block_indices[-1]
    ax.set_xlim(first_block - 0.5, last_block + 0.5)
    if len(all_block_indices) == 1:
        ax.set_xticks(all_block_indices)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _plot_relative_output_squared_error_layer_type(axes, layer_type, label_points, options):
    """Plot absolute and baseline-relative output errors for one layer type."""
    absolute_ax, ratio_ax = axes
    all_block_indices = sorted(
        {block_idx for points in label_points.values() for block_idx, _ in points}
    )
    styles = options["styles"]
    baseline_label = options["baseline_label"]

    _plot_block_series(absolute_ax, label_points, styles)
    ratios = _calculate_relative_output_error_ratios(label_points, baseline_label)
    _plot_block_series(ratio_ax, ratios, styles)

    for ax in axes:
        _configure_block_axis(ax, all_block_indices)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    absolute_ax.tick_params(labelbottom=False)
    absolute_ax.set_ylabel("Relative Output Squared Error (%)")
    absolute_ax.set_title(layer_type)
    absolute_ax.legend(fontsize=8)
    ratio_ax.axhline(1.0, color="black", linestyle="--", linewidth=1, zorder=1)
    ratio_ax.set_xlabel("Block index")
    ratio_ax.set_ylabel(f"Ratio to {baseline_label}")


def _save_relative_output_squared_error_layer_plots(output_dir, layer_types, data, options):
    """Save one relative output squared error plot per layer type."""
    for layer_type in layer_types:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(8, 8),
            gridspec_kw={"height_ratios": [2, 1]},
        )
        _plot_relative_output_squared_error_layer_type(axes, layer_type, data[layer_type], options)
        fig.tight_layout()
        safe_name = layer_type.replace(".", "_").replace("/", "_")
        plot_path = output_dir / f"relative_output_squared_error_{safe_name}.png"
        fig.savefig(plot_path, dpi=options["dpi"])
        plt.close(fig)
        print(f"Saved: {plot_path.name}")


def _save_relative_output_squared_error_combined_plot(output_dir, layer_types, data, options):
    """Save all relative output squared error plots in a single figure."""
    n = len(layer_types)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows * 2,
        ncols,
        figsize=(7 * ncols, 8 * nrows),
        squeeze=False,
        gridspec_kw={"height_ratios": [2, 1] * nrows},
    )

    for i, layer_type in enumerate(layer_types):
        grid_row = (i // ncols) * 2
        col = i % ncols
        _plot_relative_output_squared_error_layer_type(
            (axes[grid_row, col], axes[grid_row + 1, col]),
            layer_type,
            data[layer_type],
            options,
        )

    for i in range(n, nrows * ncols):
        grid_row = (i // ncols) * 2
        col = i % ncols
        axes[grid_row, col].set_visible(False)
        axes[grid_row + 1, col].set_visible(False)

    fig.tight_layout()
    plot_path = output_dir / "relative_output_squared_error_all_layer_types.png"
    fig.savefig(plot_path, dpi=options["dpi"])
    plt.close(fig)
    print(f"Saved: {plot_path.name}")


def plot_relative_output_squared_errors(
    files: dict[str, Union[str, Path]],
    output_dir: Union[str, Path] = "./figures_relative_output_error",
    dpi: int = 150,
    linestyles: Optional[dict[str, str]] = None,
    markers: Optional[dict[str, str]] = None,
):
    """Plot relative output squared error by block for multiple methods.

    The JSON ratio is converted to a percentage. One PNG is produced per layer
    type, plus a combined PNG containing all layer types. Each layer type has an
    absolute-error panel and a ratio panel relative to the first method in
    ``files``. Missing block indices and ratios with a zero or missing baseline
    are not interpolated; the corresponding line segment is left disconnected.

    Args:
        files: ``{label: json_path}`` dict. Labels appear in legends.
        output_dir: Directory to save PNG files.
        dpi: Resolution for saved images.
        linestyles: Optional ``{label: matplotlib_linestyle}`` overrides.
        markers: Optional ``{label: matplotlib_marker}`` overrides.
    """
    data = _load_relative_output_squared_errors(files)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(files.keys())
    options = _build_relative_output_plot_options(labels, linestyles, markers, dpi)
    layer_types = sorted(data.keys())

    _save_relative_output_squared_error_layer_plots(output_dir, layer_types, data, options)
    _save_relative_output_squared_error_combined_plot(output_dir, layer_types, data, options)


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
                    text,
                    (cx, cy),
                    fontsize=6,
                    color=color,
                    fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8),
                )
            else:
                ax.annotate(
                    str(k),
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 3),
                    fontsize=7,
                    color=color,
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


def _load_mean_output_squared_error(files: dict[str, Union[str, Path]]):
    """Load mean_output_squared_error by layer for each label.

    Returns:
        dict[str, dict[str, dict[str, Union[int, str, float]]]]:
            ``{label: {layer_name: {"layer_idx": int, "layer_type": str, "mse": float}}}``
    """
    data = {}

    for label, path in files.items():
        with open(path, encoding="utf-8") as f:
            stats = json.load(f)

        label_data = {}
        for layer_name, values in stats.items():
            m = _LAYER_PATTERN.match(layer_name)
            if not m:
                continue

            label_data[layer_name] = {
                "layer_idx": int(m.group(1)),
                "layer_type": m.group(2),
                "mse": values["mean_output_squared_error"],
            }
        data[label] = label_data

    return data


def plot_mean_output_squared_error_comparison(
    files: dict[str, Union[str, Path]],
    output_dir: Union[str, Path] = "./figures_mse",
    baseline_label: Optional[str] = None,
    compare_label: Optional[str] = None,
    title_suffix: Optional[str] = None,
    dpi: int = 150,
):
    """Plot mean_output_squared_error comparison for two quantization methods.

    Produces:
        1. Main figure:
            - small multiples by layer type
            - x-axis: layer index
            - y-axis: MSE improvement (%) = (1 - compare / baseline) * 100
        2. Supplementary figure:
            - heatmap over (layer type x layer index)
            - color: MSE improvement (%)
        3. Per-layer-type relative ratio bar plots (detail)

    Usage:
        >>> from onecomp.analyzer import plot_mean_output_squared_error_comparison
        >>> plot_mean_output_squared_error_comparison(
        ...     files={
        ...         "GPTQ": "quantization_statistics_GPTQ_4bit_gs128_sym_gs128.json",
        ...         "JointQ": "quantization_statistics_JointQ_4bit_gs128_sym.json",
        ...     },
        ...     output_dir="./figures_gptq_vs_jointq_mse_4bit",
        ...     baseline_label="GPTQ",
        ...     compare_label="JointQ",
        ...     title_suffix="4bit",
        ... )

    Args:
        files: ``{label: json_path}`` dict. Exactly two labels are required.
        output_dir: Directory to save PNG files.
        baseline_label: Baseline label in ``files`` (default: first key).
        compare_label: Compared label in ``files`` (default: second key).
        title_suffix: Optional suffix shown in each plot title (e.g., ``"4bit"``).
        dpi: Resolution for saved images.

    Returns:
        dict: Summary metrics and generated file paths.
    """
    if len(files) != 2:
        raise ValueError("files must contain exactly two labels for comparison.")

    labels = list(files.keys())
    if baseline_label is None:
        baseline_label = labels[0]
    if compare_label is None:
        compare_label = labels[1]

    if baseline_label not in files:
        raise ValueError(f"baseline_label '{baseline_label}' not found in files.")
    if compare_label not in files:
        raise ValueError(f"compare_label '{compare_label}' not found in files.")
    if baseline_label == compare_label:
        raise ValueError("baseline_label and compare_label must be different.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_mean_output_squared_error(files)
    baseline_data = data[baseline_label]
    compare_data = data[compare_label]

    common_layers = sorted(
        set(baseline_data).intersection(compare_data),
        key=lambda name: (
            baseline_data[name]["layer_idx"],
            baseline_data[name]["layer_type"],
        ),
    )
    if not common_layers:
        raise ValueError("No common layer keys found between the two files.")

    baseline_mses = [baseline_data[name]["mse"] for name in common_layers]
    compare_mses = [compare_data[name]["mse"] for name in common_layers]
    ratios = []
    for baseline_mse, compare_mse in zip(baseline_mses, compare_mses):
        if baseline_mse <= 0:
            raise ValueError("Encountered non-positive baseline MSE, ratio is undefined.")
        ratios.append(compare_mse / baseline_mse)

    baseline_mean = float(np.mean(baseline_mses))
    compare_mean = float(np.mean(compare_mses))
    baseline_median = float(np.median(baseline_mses))
    compare_median = float(np.median(compare_mses))

    layer_type_points = defaultdict(list)
    for name, baseline_mse, compare_mse in zip(common_layers, baseline_mses, compare_mses):
        layer_type = baseline_data[name]["layer_type"]
        layer_idx = baseline_data[name]["layer_idx"]
        relative_ratio = compare_mse / baseline_mse
        improvement_pct = (1.0 - relative_ratio) * 100.0
        layer_type_points[layer_type].append(
            (layer_idx, baseline_mse, compare_mse, relative_ratio, improvement_pct)
        )

    layer_types = sorted(layer_type_points.keys())
    layer_type_plots = {}
    layer_type_mean_ratios = {}
    layer_type_mean_improvements = {}
    better_layer_type_count = 0

    # --- One relative-ratio bar chart per layer type ---
    for layer_type in layer_types:
        points = sorted(layer_type_points[layer_type], key=lambda x: x[0])
        layer_indices = [layer_idx for layer_idx, _, _, _, _ in points]
        baseline_vals = [baseline_mse for _, baseline_mse, _, _, _ in points]
        compare_vals = [compare_mse for _, _, compare_mse, _, _ in points]
        relative_vals = [relative_ratio for _, _, _, relative_ratio, _ in points]
        improvement_vals = [improvement_pct for _, _, _, _, improvement_pct in points]

        baseline_type_mean = float(np.mean(baseline_vals))
        compare_type_mean = float(np.mean(compare_vals))
        if baseline_type_mean <= 0:
            raise ValueError(
                "Encountered non-positive layer-type baseline mean MSE, ratio is undefined."
            )
        ratio = compare_type_mean / baseline_type_mean
        improvement = (1.0 - ratio) * 100.0
        layer_type_mean_ratios[layer_type] = ratio
        layer_type_mean_improvements[layer_type] = improvement
        if ratio < 1.0:
            better_layer_type_count += 1

        x = np.arange(len(layer_indices))
        fig_width = max(10.0, 0.45 * len(layer_indices))
        fig, ax = plt.subplots(figsize=(fig_width, 5.8))
        bar_colors = ["tab:green" if rel < 1.0 else "tab:red" for rel in relative_vals]
        bars = ax.bar(x, relative_vals, color=bar_colors, width=0.78)
        ax.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Same as {baseline_label} (ratio=1.0)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(layer_indices)
        ax.set_xlabel("Layer index")
        ax.set_ylabel(f"{compare_label} / {baseline_label} MSE")
        if title_suffix:
            ax.set_title(
                f"{layer_type} ({title_suffix}) - relative mean squared error (MSE) by layer index"
            )
        else:
            ax.set_title(f"{layer_type} (relative mean squared error (MSE) by layer index)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right")
        ax.text(
            0.01,
            0.98,
            f"mean improvement: {improvement:.1f}% (positive favors {compare_label})",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        )
        y_max = max(max(relative_vals), 1.0)
        ax.set_ylim(0, y_max * 1.2)
        for bar, rel in zip(bars, relative_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                rel + y_max * 0.02,
                f"{rel:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        fig.tight_layout()
        safe_name = layer_type.replace(".", "_").replace("/", "_")
        plot_path = (
            output_dir / f"mean_output_squared_error_relative_{safe_name}_by_layer_index.png"
        )
        fig.savefig(plot_path, dpi=dpi)
        plt.close(fig)
        print(f"Saved: {plot_path.name}")
        layer_type_plots[layer_type] = plot_path

    # --- Main figure: small multiples of improvement (%) ---
    n = len(layer_types)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 * ncols, 3.8 * nrows),
        sharey=True,
    )
    axes_flat = np.asarray(axes).flatten()
    all_improvements = [
        improvement_pct
        for layer_type in layer_types
        for _, _, _, _, improvement_pct in layer_type_points[layer_type]
    ]
    y_limit = max(
        1.0, max(abs(float(np.min(all_improvements))), abs(float(np.max(all_improvements))))
    )
    y_limit *= 1.15

    for i, layer_type in enumerate(layer_types):
        ax = axes_flat[i]
        points = sorted(layer_type_points[layer_type], key=lambda x: x[0])
        layer_indices = [layer_idx for layer_idx, _, _, _, _ in points]
        improvement_vals = [improvement_pct for _, _, _, _, improvement_pct in points]
        colors = ["tab:green" if val >= 0 else "tab:red" for val in improvement_vals]

        ax.bar(layer_indices, improvement_vals, color=colors, width=0.75)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        if title_suffix:
            ax.set_title(f"{layer_type} ({title_suffix})", fontsize=10)
        else:
            ax.set_title(layer_type, fontsize=10)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("MSE improvement (%)")
        ax.set_ylim(-y_limit, y_limit)
        ax.grid(True, axis="y", alpha=0.25)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title_suffix:
        fig.suptitle(
            (
                f"MSE improvement vs {baseline_label} ({title_suffix})\n"
                f"improvement (%) = (1 - {compare_label}/{baseline_label}) * 100"
            ),
            fontsize=13,
        )
    else:
        fig.suptitle(
            (
                f"MSE improvement vs {baseline_label}\n"
                f"improvement (%) = (1 - {compare_label}/{baseline_label}) * 100"
            ),
            fontsize=13,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    main_plot = output_dir / "mean_output_squared_error_improvement_main.png"
    fig.savefig(main_plot, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {main_plot.name}")

    # --- Supplementary figure: improvement heatmap ---
    all_layer_indices = sorted(
        {
            layer_idx
            for layer_type in layer_types
            for layer_idx, _, _, _, _ in layer_type_points[layer_type]
        }
    )
    index_to_col = {layer_idx: i for i, layer_idx in enumerate(all_layer_indices)}
    heatmap = np.full((len(layer_types), len(all_layer_indices)), np.nan, dtype=float)

    for row, layer_type in enumerate(layer_types):
        for layer_idx, _, _, _, improvement_pct in layer_type_points[layer_type]:
            heatmap[row, index_to_col[layer_idx]] = improvement_pct

    min_val = float(np.nanmin(heatmap))
    max_val = float(np.nanmax(heatmap))
    # TwoSlopeNorm requires vmin < vcenter < vmax.
    # Keep 0.0 as center even when values are all positive/negative.
    min_bound = min(min_val, 0.0)
    max_bound = max(max_val, 0.0)
    if min_bound >= 0.0:
        min_bound = -max(max_bound * 0.05, 1e-6)
    if max_bound <= 0.0:
        max_bound = max(abs(min_bound) * 0.05, 1e-6)
    if max_bound == min_bound:
        min_bound -= 1.0
        max_bound += 1.0

    fig, ax = plt.subplots(
        figsize=(max(10.0, 0.35 * len(all_layer_indices)), max(4.5, 0.9 * len(layer_types)))
    )
    im = ax.imshow(
        heatmap,
        aspect="auto",
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=min_bound, vcenter=0.0, vmax=max_bound),
        interpolation="nearest",
    )

    step = max(1, len(all_layer_indices) // 16)
    xticks = list(range(0, len(all_layer_indices), step))
    if xticks[-1] != len(all_layer_indices) - 1:
        xticks.append(len(all_layer_indices) - 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([all_layer_indices[i] for i in xticks])
    ax.set_yticks(np.arange(len(layer_types)))
    ax.set_yticklabels(layer_types)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Layer type")
    if title_suffix:
        ax.set_title(f"MSE improvement heatmap ({title_suffix})")
    else:
        ax.set_title("MSE improvement heatmap")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(
        f"MSE improvement (%) = (1 - {compare_label}/{baseline_label}) * 100",
        rotation=90,
    )

    fig.tight_layout()
    supplementary_plot = output_dir / "mean_output_squared_error_improvement_heatmap.png"
    fig.savefig(supplementary_plot, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {supplementary_plot.name}")

    better_count = int(np.sum(np.array(compare_mses) < np.array(baseline_mses)))
    total_count = len(common_layers)

    return {
        "baseline_label": baseline_label,
        "compare_label": compare_label,
        "num_common_layers": total_count,
        "baseline_mean_mse": baseline_mean,
        "compare_mean_mse": compare_mean,
        "baseline_median_mse": baseline_median,
        "compare_median_mse": compare_median,
        "compare_better_count": better_count,
        "compare_better_ratio": better_count / total_count,
        "better_layer_type_count": better_layer_type_count,
        "num_layer_types": len(layer_types),
        "layer_type_mean_ratios": layer_type_mean_ratios,
        "layer_type_mean_improvements": layer_type_mean_improvements,
        "geometric_mean_ratio": float(np.exp(np.mean(np.log(ratios)))),
        "main_plot": main_plot,
        "supplementary_plot": supplementary_plot,
        "layer_type_plots": layer_type_plots,
    }
