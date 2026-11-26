"""
Professional light single-run dashboard for formal compaction telemetry.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from itertools import cycle, islice
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...compaction import CompactionRunner
from .report import build_run_report, load_run_report, validate_run_report

_BG = "#FFFFFF"
_PANEL = "#FFFFFF"
_CARD = "#F7F9FC"
_CARD_EDGE = "#DDE5EF"
_GRID = "#E4EAF1"
_TRACK = "#E9EEF5"
_TEXT = "#172033"
_MUTED = "#66758A"
_CYAN = "#0D9488"
_BLUE = "#2563EB"
_AMBER = "#E8871E"
_MAGENTA = "#7C3AED"
_RED = "#D9485F"


def visualize_run(
    source: CompactionRunner | Mapping[str, Any] | str | Path,
    output: str | Path,
    *,
    run_name: str = "compaction-run",
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 180,
) -> list[Path]:
    """Render one run to PNG/SVG with collision-resistant constrained layout."""
    if isinstance(source, CompactionRunner):
        report = build_run_report(source, run_name=run_name)
    elif isinstance(source, (str, Path)):
        report = load_run_report(source)
    elif isinstance(source, Mapping):
        report = validate_run_report(source)
    else:
        raise TypeError("source must be a runner, report mapping, or report path")
    selected = tuple(dict.fromkeys(str(item).lower() for item in formats))
    if not selected or any(item not in {"png", "svg"} for item in selected):
        raise ValueError("formats must contain png and/or svg")
    if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi < 72:
        raise ValueError("dpi must be an integer >= 72")

    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from matplotlib.patches import FancyBboxPatch  # pylint: disable=import-outside-toplevel
    from matplotlib.ticker import (  # pylint: disable=import-outside-toplevel
        FuncFormatter,
        MaxNLocator,
    )

    metrics = report["metrics"]
    events = report["events"]
    observations = report["observations"]
    with plt.rc_context(
        {
            "figure.facecolor": _BG,
            "axes.facecolor": _PANEL,
            "axes.edgecolor": _GRID,
            "axes.labelcolor": _MUTED,
            "xtick.color": _MUTED,
            "ytick.color": _MUTED,
            "text.color": _TEXT,
            "font.size": 9,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.titlepad": 11,
            "grid.color": _GRID,
            "grid.alpha": 0.9,
            "axes.axisbelow": True,
            "font.family": "DejaVu Sans",
        }
    ):
        figure = plt.figure(figsize=(16, 9))
        grid = figure.add_gridspec(
            3,
            12,
            height_ratios=(1.0, 3.2, 2.7),
            left=0.055,
            right=0.975,
            bottom=0.07,
            top=0.86,
            hspace=0.72,
            wspace=0.78,
        )
        _title(figure, report)
        _kpis(figure, grid[0, :], metrics, FancyBboxPatch)
        # Keep one GridSpec column empty between the two upper panels. Long
        # observation IDs live on the left of the right panel and must never
        # collide with the context legend or axis annotations.
        number_formatter = FuncFormatter(lambda value, _position: _compact_axis_number(value))
        _context_panel(
            figure.add_subplot(grid[1, :6]),
            events,
            MaxNLocator,
            number_formatter,
        )
        _observation_panel(
            figure.add_subplot(grid[1, 7:]),
            observations,
            number_formatter,
        )
        _timeline_panel(figure.add_subplot(grid[2, :6]), events)
        _operations_panel(figure.add_subplot(grid[2, 6:9]), events)
        _integrity_panel(figure.add_subplot(grid[2, 9:]), metrics, FancyBboxPatch)
        _footer(figure)

        base = Path(output).expanduser().resolve()
        if base.suffix.lower() in {".png", ".svg"}:
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for extension in selected:
            path = base.with_suffix(f".{extension}")
            figure.savefig(
                path,
                dpi=dpi if extension == "png" else None,
                facecolor=figure.get_facecolor(),
                bbox_inches="tight",
                pad_inches=0.18,
            )
            paths.append(path)
        plt.close(figure)
    return paths


def _title(figure, report: Mapping[str, Any]) -> None:
    title = str(report.get("run_name", "Compaction run"))
    if len(title) > 80:
        title = title[:79] + "…"
    figure.suptitle(
        title,
        x=0.025,
        y=0.975,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=_TEXT,
    )
    config = report.get("config", {})
    figure.text(
        0.025,
        0.925,
        f"FORMAL RUN  /  {report.get('profile', 'unknown')}  /  "
        f"summary ≤ {_format_limit(config.get('summary_max_chars'))} chars  /  "
        f"recall ≤ {_format_limit(config.get('max_recall_units'))} units",
        ha="left",
        va="top",
        fontsize=9.5,
        color=_MUTED,
    )


def _kpis(figure, slot, metrics: Mapping[str, Any], patch_class) -> None:
    axis = figure.add_subplot(slot)
    axis.set_axis_off()
    values = [
        ("ARCHIVED", _compact_number(metrics["archived_chars"]), "exact chars", _CYAN),
        ("COMPACTIONS", str(metrics["compactions"]), "bounded rewrites", _BLUE),
        ("SAVED", _compact_number(metrics["cumulative_saved_chars"]), "cumulative chars", _AMBER),
        ("RECALL", str(metrics["recall_actions"]), "bounded actions", _MAGENTA),
    ]
    gap = 0.018
    width = (1 - gap * 3) / 4
    for index, (label, value, detail, color) in enumerate(values):
        x = index * (width + gap)
        axis.add_patch(
            patch_class(
                (x, 0.08),
                width,
                0.76,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                transform=axis.transAxes,
                facecolor=_CARD,
                edgecolor=_CARD_EDGE,
                linewidth=1.0,
            )
        )
        axis.add_patch(
            patch_class(
                (x + 0.014, 0.19),
                0.004,
                0.48,
                boxstyle="round,pad=0,rounding_size=0.003",
                transform=axis.transAxes,
                facecolor=color,
                edgecolor="none",
            )
        )
        axis.text(x + 0.025, 0.69, label, transform=axis.transAxes, color=_MUTED, fontsize=8)
        axis.text(
            x + 0.025,
            0.38,
            value,
            transform=axis.transAxes,
            color=color,
            fontsize=20,
            fontweight="bold",
            va="center",
        )
        axis.text(x + 0.025, 0.17, detail, transform=axis.transAxes, color=_MUTED, fontsize=8)


def _context_panel(
    axis,
    events: list[Mapping[str, Any]],
    locator_class: Any,
    number_formatter: Any,
) -> None:
    rows = [event for event in events if event.get("kind") == "compaction"]
    _panel(axis, "Context trajectory", "Model-visible text before and after each rewrite")
    if not rows:
        _empty(axis, "No compaction event in this run")
        return
    x = list(range(1, len(rows) + 1))
    before = [int(row.get("before_chars", 0)) for row in rows]
    after = [int(row.get("after_chars", 0)) for row in rows]
    axis.fill_between(x, after, before, color=_BLUE, alpha=0.14, label="Removed")
    axis.plot(x, before, color=_AMBER, marker="o", linewidth=2.0, label="Before")
    axis.plot(x, after, color=_CYAN, marker="o", linewidth=2.4, label="After")
    axis.set_xlabel("Compaction event")
    axis.set_ylabel("Model-visible characters")
    axis.yaxis.set_major_formatter(number_formatter)
    if len(x) == 1:
        axis.set_xlim(0.5, 1.5)
        axis.set_xticks([1])
    else:
        axis.xaxis.set_major_locator(locator_class(integer=True, nbins=8))
    axis.grid(axis="y")
    axis.legend(
        loc="upper right",
        ncols=3,
        frameon=False,
        labelcolor=_TEXT,
        columnspacing=1.0,
        handlelength=1.8,
        fontsize=8,
    )


def _observation_panel(
    axis,
    rows: list[Mapping[str, Any]],
    number_formatter: Any,
) -> None:
    _panel(axis, "Archived observations", "Largest exact bodies; labels contain metadata only")
    if not rows:
        _empty(axis, "No archived observation")
        return
    selected = sorted(rows, key=lambda row: int(row["char_count"]), reverse=True)[:10]
    selected.reverse()
    labels = [f"#{row['sequence']}  {str(row['obs_id'])[-8:]}" for row in selected]
    values = [int(row["char_count"]) for row in selected]
    colors = [_CYAN if int(row["returncode"]) == 0 else _RED for row in selected]
    bars = axis.barh(labels, values, color=colors, alpha=0.84, height=0.58)
    axis.bar_label(
        bars, labels=[_compact_number(value) for value in values], padding=5, color=_MUTED
    )
    axis.set_xlabel("Exact archived characters")
    axis.xaxis.set_major_formatter(number_formatter)
    axis.grid(axis="x")
    axis.margins(x=0.18)


def _timeline_panel(axis, events: list[Mapping[str, Any]]) -> None:
    _panel(axis, "Operation timeline", "Step-aligned lifecycle; marker area scales with payload")
    if not events:
        _empty(axis, "No telemetry event")
        return
    kinds = ("observation", "compaction", "action")
    y_map = {kind: index for index, kind in enumerate(kinds)}
    colors = {"observation": _CYAN, "compaction": _BLUE, "action": _MAGENTA}
    for kind in kinds:
        selected = [event for event in events if event.get("kind") == kind]
        axis.scatter(
            [int(event.get("step", 0)) for event in selected],
            [y_map[kind]] * len(selected),
            s=[20 + min(180, int(event.get("chars", 0)) ** 0.5 * 4) for event in selected],
            color=colors[kind],
            alpha=0.85,
            edgecolors=_PANEL,
            linewidths=0.7,
        )
    axis.set_yticks(range(len(kinds)), [kind.title() for kind in kinds])
    axis.set_xlabel("Agent step")
    axis.set_ylabel("Operation")
    axis.grid(axis="x")


def _operations_panel(axis, events: list[Mapping[str, Any]]) -> None:
    _panel(axis, "Recovery mix", "Model-facing ARC actions in this run")
    actions = [event for event in events if event.get("kind") == "action"]
    counts: dict[str, int] = {}
    for event in actions:
        label = str(event.get("action_kind", "unknown")).replace("recall_", "")
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        _empty(axis, "No recall action")
        return
    labels = list(counts)
    values = [counts[label] for label in labels]
    palette = tuple(islice(cycle((_MAGENTA, _CYAN, _BLUE, _AMBER)), len(labels)))
    axis.bar(labels, values, color=palette)
    axis.set_xlabel("Recovery operation")
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y")


def _integrity_panel(axis, metrics: Mapping[str, Any], patch_class) -> None:
    _panel(axis, "Archive health", "Coverage computed without reading bodies")
    axis.set_axis_off()
    total = max(1, int(metrics["observations"]))
    checksums = int(metrics["checksum_coverage"])
    recalled = int(metrics["recalled_observations"])
    success = int(metrics["successful_observations"])
    rows = [
        ("Checksum", checksums / total, _CYAN),
        ("Successful", success / total, _BLUE),
        ("Recalled", recalled / total, _MAGENTA),
    ]
    for index, (label, ratio, color) in enumerate(rows):
        y = 0.72 - index * 0.25
        axis.text(0.03, y + 0.09, label, transform=axis.transAxes, color=_MUTED, fontsize=8)
        axis.add_patch(
            patch_class(
                (0.03, y),
                0.86,
                0.055,
                boxstyle="round,pad=0,rounding_size=0.02",
                transform=axis.transAxes,
                facecolor=_TRACK,
                edgecolor="none",
            )
        )
        axis.add_patch(
            patch_class(
                (0.03, y),
                0.86 * min(1.0, ratio),
                0.055,
                boxstyle="round,pad=0,rounding_size=0.02",
                transform=axis.transAxes,
                facecolor=color,
                edgecolor="none",
            )
        )
        axis.text(
            0.94,
            y + 0.027,
            f"{ratio:.0%}",
            transform=axis.transAxes,
            ha="right",
            va="center",
            color=_TEXT,
            fontweight="bold",
        )


def _panel(axis, title: str, subtitle: str) -> None:
    # Draw panel headings explicitly so Matplotlib's scientific-offset text
    # cannot occupy the same top-left slot as the subtitle.
    axis.set_title("")
    axis.text(
        0,
        1.14,
        title,
        transform=axis.transAxes,
        va="bottom",
        color=_TEXT,
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )
    axis.text(
        0,
        1.07,
        subtitle,
        transform=axis.transAxes,
        va="bottom",
        color=_MUTED,
        fontsize=8,
        clip_on=False,
    )
    for spine in axis.spines.values():
        spine.set_color(_CARD_EDGE)
        spine.set_linewidth(0.9)


def _footer(figure) -> None:
    figure.text(
        0.975,
        0.018,
        "ONECOMP  /  BODY-FREE RUN TELEMETRY",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=_MUTED,
    )


def _empty(axis, label: str) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(0.5, 0.47, label, transform=axis.transAxes, ha="center", color=_MUTED)


def _compact_number(value: int) -> str:
    number = float(value)
    for suffix in ("", "K", "M", "B"):
        if abs(number) < 1000 or suffix == "B":
            return f"{number:.0f}" if not suffix else f"{number:.1f}{suffix}"
        number /= 1000
    return str(value)


def _compact_axis_number(value: float) -> str:
    """Render ticks without a detached scientific-offset annotation."""
    magnitude = abs(value)
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= threshold:
            scaled = value / threshold
            precision = 0 if abs(scaled - round(scaled)) < 1e-9 else 1
            return f"{scaled:.{precision}f}{suffix}"
    return f"{value:g}"


def _format_limit(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return "—"


__all__ = ["visualize_run"]
