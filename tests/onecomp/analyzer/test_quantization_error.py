"""Tests for quantization error visualization."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from onecomp.analyzer import (
    plot_output_squared_error_sum_ratios,
    plot_relative_output_squared_errors,
)


def _write_statistics(path, statistics):
    path.write_text(json.dumps(statistics), encoding="utf-8")
    return path


def _assert_png(path):
    assert path.stat().st_size > 8
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def _capture_plot_calls(monkeypatch):
    """Capture line-series and horizontal-line calls while rendering normally."""
    plot_calls = []
    horizontal_lines = []
    original_plot = Axes.plot
    original_axhline = Axes.axhline

    def capture_plot(axis, x_values, y_values, *args, **kwargs):
        """Capture a plotted series."""
        plot_calls.append(
            {
                "label": kwargs["label"],
                "linestyle": kwargs["linestyle"],
                "marker": kwargs["marker"],
                "x_values": list(x_values),
                "values": np.asarray(y_values, dtype=float),
            }
        )
        return original_plot(axis, x_values, y_values, *args, **kwargs)

    def capture_axhline(axis, *args, **kwargs):
        """Capture a horizontal reference line."""
        y_value = args[0] if args else kwargs.get("y", 0)
        horizontal_lines.append(y_value)
        return original_axhline(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture_plot)
    monkeypatch.setattr(Axes, "axhline", capture_axhline)
    return plot_calls, horizontal_lines


def _capture_scatter_plot_details(monkeypatch):
    """Capture scatter series and axes configuration calls."""
    scatter_calls = []
    horizontal_lines = []
    highlighted_spans = []
    titles = []
    ylabels = []
    legend_calls = []
    originals = {
        "scatter": Axes.scatter,
        "axhline": Axes.axhline,
        "axvspan": Axes.axvspan,
        "set_title": Axes.set_title,
        "set_ylabel": Axes.set_ylabel,
        "legend": Axes.legend,
    }

    def capture_scatter(axis, x_values, y_values, *args, **kwargs):
        """Capture a scatter series."""
        scatter_calls.append(
            {
                "label": kwargs["label"],
                "marker": kwargs["marker"],
                "x_values": list(x_values),
                "values": np.asarray(y_values, dtype=float),
            }
        )
        return originals["scatter"](axis, x_values, y_values, *args, **kwargs)

    def capture_axhline(axis, *args, **kwargs):
        """Capture a horizontal reference line."""
        y_value = args[0] if args else kwargs.get("y", 0)
        horizontal_lines.append(y_value)
        return originals["axhline"](axis, *args, **kwargs)

    def capture_axvspan(axis, xmin, xmax, *args, **kwargs):
        """Capture a highlighted vertical span."""
        highlighted_spans.append((xmin, xmax, kwargs.get("alpha")))
        return originals["axvspan"](axis, xmin, xmax, *args, **kwargs)

    def capture_set_title(axis, title, *args, **kwargs):
        """Capture the configured plot title and font size."""
        titles.append((title, kwargs.get("fontsize")))
        return originals["set_title"](axis, title, *args, **kwargs)

    def capture_set_ylabel(axis, ylabel, *args, **kwargs):
        """Capture the configured y-axis label."""
        ylabels.append(ylabel)
        return originals["set_ylabel"](axis, ylabel, *args, **kwargs)

    def capture_legend(axis, *args, **kwargs):
        """Capture the configured legend options."""
        legend_calls.append(kwargs)
        return originals["legend"](axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    monkeypatch.setattr(Axes, "axhline", capture_axhline)
    monkeypatch.setattr(Axes, "axvspan", capture_axvspan)
    monkeypatch.setattr(Axes, "set_title", capture_set_title)
    monkeypatch.setattr(Axes, "set_ylabel", capture_set_ylabel)
    monkeypatch.setattr(Axes, "legend", capture_legend)
    return scatter_calls, horizontal_lines, highlighted_spans, titles, ylabels, legend_calls


def test_plot_relative_output_squared_errors_for_multiple_methods(tmp_path):
    """Generate layer-type and combined plots for multiple methods."""
    gptq_path = _write_statistics(
        tmp_path / "gptq.json",
        {
            "model.layers.0.self_attn.q_proj": {"relative_output_squared_error": 0.01},
            "model.layers.2.self_attn.q_proj": {"relative_output_squared_error": 0.03},
            "model.layers.0.mlp/down_proj": {"relative_output_squared_error": 0.02},
            "lm_head": {"relative_output_squared_error": 0.99},
        },
    )
    jointq_path = _write_statistics(
        tmp_path / "jointq.json",
        {
            "model.layers.0.self_attn.q_proj": {"relative_output_squared_error": 0.008},
            "model.layers.1.self_attn.q_proj": {"relative_output_squared_error": 0.015},
            "model.layers.2.self_attn.q_proj": {"relative_output_squared_error": 0.025},
            "model.layers.0.mlp/down_proj": {"relative_output_squared_error": 0.018},
        },
    )
    output_dir = tmp_path / "plots"

    plot_relative_output_squared_errors(
        files={"GPTQ": gptq_path, "JointQ": jointq_path},
        output_dir=output_dir,
        dpi=50,
    )

    expected_paths = [
        output_dir / "relative_output_squared_error_self_attn_q_proj.png",
        output_dir / "relative_output_squared_error_mlp_down_proj.png",
        output_dir / "relative_output_squared_error_all_layer_types.png",
    ]
    for path in expected_paths:
        _assert_png(path)


def test_plot_relative_output_squared_errors_adds_ratios_and_method_styles(tmp_path, monkeypatch):
    """Plot ratios against the first method and apply styles to both panels."""
    baseline_path = _write_statistics(
        tmp_path / "baseline.json",
        {
            "model.layers.0.self_attn.q_proj": {"relative_output_squared_error": 0.0},
            "model.layers.1.self_attn.q_proj": {"relative_output_squared_error": 0.02},
        },
    )
    compare_path = _write_statistics(
        tmp_path / "compare.json",
        {
            "model.layers.0.self_attn.q_proj": {"relative_output_squared_error": 0.005},
            "model.layers.1.self_attn.q_proj": {"relative_output_squared_error": 0.03},
        },
    )
    plot_calls, horizontal_lines = _capture_plot_calls(monkeypatch)

    plot_relative_output_squared_errors(
        files={"Baseline": baseline_path, "Compare": compare_path},
        output_dir=tmp_path / "plots",
        linestyles={"Baseline": "--", "Compare": ":"},
        markers={"Baseline": "s", "Compare": "^"},
        dpi=50,
    )

    expected_series = {
        "Baseline": ([0.0, 2.0], [np.nan, 1.0], "--", "s"),
        "Compare": ([0.5, 3.0], [np.nan, 1.5], ":", "^"),
    }
    for label, (absolute, ratio, linestyle, marker) in expected_series.items():
        method_calls = [call for call in plot_calls if call["label"] == label]
        assert len(method_calls) == 4
        assert all(call["linestyle"] == linestyle for call in method_calls)
        assert all(call["marker"] == marker for call in method_calls)
        assert (
            sum(np.allclose(call["values"], absolute, equal_nan=True) for call in method_calls)
            == 2
        )
        assert (
            sum(np.allclose(call["values"], ratio, equal_nan=True) for call in method_calls) == 2
        )

    assert horizontal_lines == [1.0, 1.0]


@pytest.mark.parametrize(
    ("option_name", "option_value"),
    [
        ("linestyles", {"Unknown": "--"}),
        ("markers", {"Unknown": "s"}),
    ],
)
def test_plot_relative_output_squared_errors_rejects_unknown_style_labels(
    tmp_path, option_name, option_value
):
    """Reject style mappings whose labels are absent from files."""
    path = _write_statistics(
        tmp_path / "method.json",
        {
            "model.layers.0.self_attn.q_proj": {
                "relative_output_squared_error": 0.01,
            }
        },
    )

    with pytest.raises(ValueError, match="Unknown method labels"):
        plot_relative_output_squared_errors(
            {"Method": path}, tmp_path / "plots", **{option_name: option_value}
        )


@pytest.mark.parametrize(
    "value, message",
    [
        ({}, "Missing relative_output_squared_error"),
        ({"relative_output_squared_error": None}, "Invalid relative_output_squared_error"),
        ({"relative_output_squared_error": float("nan")}, "Invalid relative_output_squared_error"),
        ({"relative_output_squared_error": -0.1}, "Invalid relative_output_squared_error"),
    ],
)
def test_plot_relative_output_squared_errors_rejects_invalid_values(tmp_path, value, message):
    """Reject missing, non-finite, non-numeric, and negative error values."""
    path = _write_statistics(tmp_path / "invalid.json", {"model.layers.0.self_attn.q_proj": value})

    with pytest.raises(ValueError, match=message):
        plot_relative_output_squared_errors({"Method": path}, tmp_path / "plots")


def test_plot_relative_output_squared_errors_rejects_empty_files(tmp_path):
    """Reject an empty mapping of labels to statistics files."""
    with pytest.raises(ValueError, match="at least one label"):
        plot_relative_output_squared_errors({}, tmp_path / "plots")


def test_plot_relative_output_squared_errors_rejects_no_layer_data(tmp_path):
    """Reject statistics files without model layer entries."""
    path = _write_statistics(
        tmp_path / "empty.json",
        {"lm_head": {"relative_output_squared_error": 0.01}},
    )

    with pytest.raises(ValueError, match="No relative_output_squared_error layer data"):
        plot_relative_output_squared_errors({"Method": path}, tmp_path / "plots")


def test_plot_output_squared_error_sum_ratios(tmp_path, monkeypatch):
    """Sum layer categories and plot each comparison ratio to the first method."""
    baseline_path = _write_statistics(
        tmp_path / "baseline.json",
        {
            "model.layers.0.self_attn.q_proj": {"output_squared_error": 2.0},
            "model.layers.1.self_attn.q_proj": {"output_squared_error": 3.0},
            "model.layers.0.self_attn.k_proj": {"output_squared_error": 4.0},
            "model.layers.0.self_attn.v_proj": {"output_squared_error": 0.0},
            "model.layers.0.self_attn.o_proj": {"output_squared_error": 10.0},
            "model.layers.0.mlp.gate_proj": {"output_squared_error": 20.0},
            "model.layers.0.mlp.up_proj": {"output_squared_error": 25.0},
            "model.layers.0.mlp.down_proj": {"output_squared_error": 50.0},
            "model.layers.0.input_layernorm": {"output_squared_error": 999.0},
            "lm_head": {"output_squared_error": 999.0},
        },
    )
    compare_path = _write_statistics(
        tmp_path / "compare.json",
        {
            "model.layers.0.self_attn.q_proj": {"output_squared_error": 10.0},
            "model.layers.0.self_attn.k_proj": {"output_squared_error": 2.0},
            "model.layers.0.self_attn.v_proj": {"output_squared_error": 3.0},
            "model.layers.0.self_attn.o_proj": {"output_squared_error": 20.0},
            "model.layers.0.mlp.gate_proj": {"output_squared_error": 10.0},
            "model.layers.0.mlp.up_proj": {"output_squared_error": 50.0},
            "model.layers.0.mlp.down_proj": {"output_squared_error": 25.0},
        },
    )
    scatter_calls, horizontal_lines, highlighted_spans, titles, ylabels, legend_calls = (
        _capture_scatter_plot_details(monkeypatch)
    )
    output_dir = tmp_path / "plots"

    plotted_ax = plot_output_squared_error_sum_ratios(
        files={"Baseline": baseline_path, "Compare": compare_path},
        output_dir=output_dir,
        markers={"Compare": "s"},
        title="Output error comparison",
        ylabel="Normalized output error",
        legend_location="lower left",
        figsize=(7, 3),
    )

    assert len(scatter_calls) == 1
    call = scatter_calls[0]
    assert call["label"] == "Compare"
    assert call["x_values"] == ["q", "k", "v", "o", "gate", "up", "down", "total"]
    assert np.allclose(
        call["values"],
        [2.0, 0.5, np.nan, 2.0, 0.5, 2.0, 0.5, 120.0 / 114.0],
        equal_nan=True,
    )
    assert call["marker"] == "s"
    assert horizontal_lines == [1.0]
    assert highlighted_spans == [(6.5, 7.5, 0.12)]
    assert titles == [("Output error comparison", 18)]
    assert ylabels == ["Normalized output error"]
    assert legend_calls == [{"fontsize": 8, "loc": "lower left"}]
    assert plotted_ax.figure.get_size_inches() == pytest.approx((7, 3))
    _assert_png(output_dir / "output_squared_error_sum_ratios.png")


def test_plot_output_squared_error_sum_ratios_draws_on_provided_axes(tmp_path):
    """Draw repeated calls on one figure without saving individual plots."""
    paths = {
        label: _write_statistics(
            tmp_path / f"{label}.json",
            {"model.layers.0.self_attn.q_proj": {"output_squared_error": error}},
        )
        for label, error in (("Baseline", 2.0), ("Compare", 1.0))
    }
    output_dir = tmp_path / "plots"
    fig, axes = plt.subplots(1, 2)

    returned_axes = [
        plot_output_squared_error_sum_ratios(paths, output_dir, ax=ax, title=f"Plot {index}")
        for index, ax in enumerate(axes)
    ]

    assert returned_axes == list(axes)
    assert all(ax.figure is fig and len(ax.collections) == 1 for ax in axes)
    assert not output_dir.exists()
    plt.close(fig)


def test_plot_output_squared_error_sum_ratios_supports_limits_and_value_labels(tmp_path):
    """Apply y-axis limits, larger labels, and finite-value annotations."""
    paths = {
        label: _write_statistics(
            tmp_path / f"{label}.json",
            {"model.layers.0.self_attn.q_proj": {"output_squared_error": error}},
        )
        for label, error in (("Baseline", 2.0), ("Compare", 1.0))
    }
    fig, ax = plt.subplots()

    plot_output_squared_error_sum_ratios(
        paths,
        ax=ax,
        title="Ratio plot",
        ylim=(0.25, 0.75),
        show_values=True,
    )

    assert ax.get_ylim() == pytest.approx((0.25, 0.75))
    assert [text.get_text() for text in ax.texts] == ["0.50", "0.50"]
    assert {text.get_fontsize() for text in ax.texts} == {10}
    assert ax.title.get_fontsize() == 18
    assert ax.xaxis.label.get_fontsize() == 12
    assert {label.get_fontsize() for label in ax.get_xticklabels()} == {11}
    plt.close(fig)


def test_plot_output_squared_error_sum_ratios_supports_custom_categories(tmp_path, monkeypatch):
    """Aggregate custom layer basenames using mapping order for the x-axis."""
    baseline_path = _write_statistics(
        tmp_path / "baseline.json",
        {
            "model.layers.0.self_attn.q_proj": {"output_squared_error": 2.0},
            "model.layers.0.mlp.custom_proj": {"output_squared_error": 4.0},
            "model.layers.0.mlp.down_proj": {"output_squared_error": 999.0},
        },
    )
    compare_path = _write_statistics(
        tmp_path / "compare.json",
        {
            "model.layers.0.self_attn.q_proj": {"output_squared_error": 4.0},
            "model.layers.0.mlp.custom_proj": {"output_squared_error": 2.0},
            "model.layers.0.mlp.down_proj": {"output_squared_error": 999.0},
        },
    )
    scatter_calls, _, highlighted_spans, _, ylabels, legend_calls = _capture_scatter_plot_details(
        monkeypatch
    )

    plot_output_squared_error_sum_ratios(
        files={"Baseline": baseline_path, "Compare": compare_path},
        output_dir=tmp_path / "plots",
        category_by_layer={"q_proj": "query", "custom_proj": "custom"},
    )

    assert len(scatter_calls) == 1
    assert scatter_calls[0]["x_values"] == ["query", "custom", "total"]
    assert np.allclose(scatter_calls[0]["values"], [2.0, 0.5, 1.0])
    assert highlighted_spans == [(1.5, 2.5, 0.12)]
    assert ylabels == ["Output Squared Error Sum Ratio to Baseline"]
    assert legend_calls == [{"fontsize": 8}]


@pytest.mark.parametrize(
    "value, message",
    [
        ({}, "Missing output_squared_error"),
        ({"output_squared_error": None}, "Invalid output_squared_error"),
        ({"output_squared_error": float("nan")}, "Invalid output_squared_error"),
        ({"output_squared_error": -0.1}, "Invalid output_squared_error"),
    ],
)
def test_plot_output_squared_error_sum_ratios_rejects_invalid_values(tmp_path, value, message):
    """Reject invalid output squared errors in supported layer categories."""
    baseline_path = _write_statistics(
        tmp_path / "baseline.json", {"model.layers.0.self_attn.q_proj": value}
    )
    compare_path = _write_statistics(
        tmp_path / "compare.json",
        {"model.layers.0.self_attn.q_proj": {"output_squared_error": 1.0}},
    )

    with pytest.raises(ValueError, match=message):
        plot_output_squared_error_sum_ratios(
            {"Baseline": baseline_path, "Compare": compare_path},
            tmp_path / "plots",
        )


def test_plot_output_squared_error_sum_ratios_requires_two_methods(tmp_path):
    """Require a baseline and at least one comparison method."""
    path = _write_statistics(
        tmp_path / "method.json",
        {"model.layers.0.self_attn.q_proj": {"output_squared_error": 1.0}},
    )

    with pytest.raises(ValueError, match="at least two labels"):
        plot_output_squared_error_sum_ratios({"Method": path}, tmp_path / "plots")


def test_plot_output_squared_error_sum_ratios_rejects_no_supported_layers(tmp_path):
    """Reject files without any of the seven supported layer categories."""
    paths = {
        label: _write_statistics(
            tmp_path / f"{label}.json",
            {"model.layers.0.input_layernorm": {"output_squared_error": 1.0}},
        )
        for label in ("Baseline", "Compare")
    }

    with pytest.raises(ValueError, match="No supported output_squared_error layer data"):
        plot_output_squared_error_sum_ratios(paths, tmp_path / "plots")
