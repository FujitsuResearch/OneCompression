"""MT-Bench radar chart visualization.

Reads summary_<model>.json files written by show_result and
renders an 8-category radar chart.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
from logging import getLogger
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAGothic",
        "DejaVu Sans",
        "Arial",
        "Helvetica",
    ]

logger = getLogger(__name__)

CATEGORIES = [
    "writing",
    "roleplay",
    "extraction",
    "stem",
    "humanities",
    "reasoning",
    "math",
    "coding",
]

CATEGORY_DISPLAY = {
    "writing": "Writing",
    "roleplay": "Roleplay",
    "extraction": "Extraction",
    "stem": "STEM",
    "humanities": "Humanities",
    "reasoning": "Reasoning",
    "math": "Math",
    "coding": "Coding",
}

CATEGORY_GROUP = {
    "writing": "language",
    "roleplay": "language",
    "extraction": "knowledge",
    "stem": "knowledge",
    "humanities": "knowledge",
    "reasoning": "logic",
    "math": "logic",
    "coding": "logic",
}

GROUP_COLORS = {
    "language": "#E8F4FD",
    "knowledge": "#E8F5E9",
    "logic": "#FCE4EC",
}

GROUP_LABELS = {
    "language": "Language",
    "knowledge": "Knowledge",
    "logic": "Logic & Reasoning",
}

MODEL_COLORS = [
    "#2196F3",
    "#FF5722",
    "#4CAF50",
    "#9C27B0",
    "#FF9800",
    "#00BCD4",
    "#E91E63",
    "#607D8B",
]


def load_mt_bench_results(results_dir: str | Path) -> dict[str, dict[str, float]]:
    """Load every summary_*.json under <results_dir>/mt_bench/."""
    models_data: dict[str, dict[str, float]] = {}
    mt_dir = Path(results_dir) / "mt_bench"
    if not mt_dir.exists():
        return models_data

    for f in sorted(mt_dir.glob("summary_*.json")):
        try:
            with open(f) as fp:
                data = json.load(fp)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", f, e)
            continue
        model = data.get("model", f.stem.replace("summary_", ""))
        cats = data.get("categories", {})
        if cats:
            models_data[model] = cats
    return models_data


def plot_radar(
    models_data: dict[str, dict[str, float]],
    categories: list[str],
    output_path: str | Path,
    title: str = "MT-Bench",
) -> Path:
    """Render the radar chart and save it as a PNG."""
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(18, 18), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    drawn_groups: set[str] = set()
    for i, cat in enumerate(categories):
        grp = CATEGORY_GROUP.get(cat, "other")
        color = GROUP_COLORS.get(grp, "#F5F5F5")
        theta1 = angles[i] - np.pi / n
        theta2 = angles[i] + np.pi / n
        theta_range = np.linspace(theta1, theta2, 50)
        ax.fill_between(theta_range, 0, 10.5, color=color, alpha=0.3, zorder=0)
        drawn_groups.add(grp)

    ax.set_ylim(0, 10.5)
    grid_values = [2, 4, 6, 8, 10]
    ax.set_yticks(grid_values)
    ax.set_yticklabels(
        [str(v) for v in grid_values],
        fontsize=14,
        color="#888888",
        fontweight="medium",
    )
    ax.yaxis.set_tick_params(pad=8)
    ax.grid(True, color="#CCCCCC", linewidth=0.6, alpha=0.7)
    ax.spines["polar"].set_visible(False)

    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 10], color="#DDDDDD", linewidth=0.8, zorder=1)

    for idx, (model_name, results) in enumerate(models_data.items()):
        values = [results.get(cat, 0.0) for cat in categories]
        values += values[:1]

        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2.8,
            markersize=7,
            color=color,
            label=model_name,
            zorder=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
        ax.fill(angles, values, alpha=0.12, color=color, zorder=5)

        for a, v in zip(angles[:-1], values[:-1]):
            if v > 0:
                ax.annotate(
                    f"{v:.1f}",
                    xy=(a, v),
                    fontsize=12,
                    fontweight="bold",
                    color=color,
                    ha="center",
                    va="bottom",
                    xytext=(0, 8),
                    textcoords="offset points",
                    zorder=15,
                )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    for angle, cat in zip(angles[:-1], categories):
        display = CATEGORY_DISPLAY.get(cat, cat)
        angle_deg = np.degrees(angle) % 360
        ha = "center"
        if 10 < angle_deg < 170:
            ha = "left"
        elif 190 < angle_deg < 350:
            ha = "right"
        va = "center"
        if 80 < angle_deg < 100:
            va = "top"
        elif 260 < angle_deg < 280:
            va = "bottom"
        ax.text(
            angle,
            10 + 1.5,
            display,
            fontsize=17,
            fontweight="bold",
            ha=ha,
            va=va,
            color="#333333",
            transform=ax.transData,
        )

    ax.set_title(title, fontsize=28, fontweight="bold", color="#222222", pad=60, y=1.05)

    if models_data:
        handles, labels = ax.get_legend_handles_labels()
        cat_handles = []
        for grp in ["language", "knowledge", "logic"]:
            if grp in drawn_groups:
                cat_handles.append(
                    mpatches.Patch(color=GROUP_COLORS[grp], alpha=0.5, label=GROUP_LABELS[grp])
                )
        all_h = handles + cat_handles
        all_l = labels + [h.get_label() for h in cat_handles]
        legend = ax.legend(
            all_h,
            all_l,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            fontsize=15,
            framealpha=0.95,
            edgecolor="#CCCCCC",
            fancybox=True,
            shadow=True,
            ncol=min(len(all_h), 4),
            title="Models & Categories",
            title_fontsize=16,
        )
        legend.get_title().set_fontweight("bold")

    plt.subplots_adjust(bottom=0.12)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info("Saved MT-Bench radar chart to %s", output)
    plt.close(fig)
    return output


def generate_radar_chart(
    results_dir: str | Path,
    output_path: str | Path,
    *,
    models: Iterable[str] | None = None,
    labels: Iterable[str] | None = None,
    title: str = "MT-Bench",
) -> Path | None:
    """Programmatic entry point for chart generation."""
    all_data = load_mt_bench_results(results_dir)

    if models:
        wanted = list(models)
        models_data = {k: all_data[k] for k in wanted if k in all_data}
        missing = [k for k in wanted if k not in all_data]
        if missing:
            logger.warning("No summary found for models: %s", missing)
    else:
        models_data = all_data

    if labels and models:
        labels_list = list(labels)
        keys = list(models_data.keys())
        if len(labels_list) == len(keys):
            models_data = {lbl: models_data[k] for lbl, k in zip(labels_list, keys)}
        else:
            logger.warning(
                "labels length (%d) != models length (%d); ignoring labels",
                len(labels_list),
                len(keys),
            )

    if not models_data:
        logger.warning("No MT-Bench results available for chart generation")
        return None

    logger.info("Plotting radar chart for models: %s", list(models_data.keys()))
    return plot_radar(models_data, CATEGORIES, output_path, title=title)
