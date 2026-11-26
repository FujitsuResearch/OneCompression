"""Import-boundary tests for optional analyzer visualization dependencies."""

import importlib
import sys
from types import ModuleType

import pytest


def test_analyzer_package_does_not_eagerly_import_plotting_module():
    sys.modules.pop("onecomp.analyzer.output_squared_error", None)
    sys.modules.pop("onecomp.analyzer.quantization_error", None)
    sys.modules.pop("onecomp.analyzer.weight_outlier", None)
    sys.modules.pop("onecomp.analyzer", None)

    analyzer = importlib.import_module("onecomp.analyzer")

    assert callable(analyzer.plot_mean_output_squared_error_comparison)
    assert callable(analyzer.plot_output_squared_error_sum_ratios)
    assert callable(analyzer.plot_quantization_errors)
    assert callable(analyzer.plot_relative_output_squared_errors)
    assert "onecomp.analyzer.output_squared_error" not in sys.modules
    assert "onecomp.analyzer.quantization_error" not in sys.modules
    assert "onecomp.analyzer.weight_outlier" not in sys.modules


@pytest.mark.parametrize(
    ("public_name", "module_name"),
    [
        ("plot_mean_output_squared_error_comparison", "onecomp.analyzer.quantization_error"),
        ("plot_output_squared_error_sum_ratios", "onecomp.analyzer.output_squared_error"),
        ("plot_quantization_errors", "onecomp.analyzer.quantization_error"),
        ("plot_relative_output_squared_errors", "onecomp.analyzer.quantization_error"),
    ],
)
def test_analyzer_plot_wrappers_forward_arguments(monkeypatch, public_name, module_name):
    analyzer = importlib.import_module("onecomp.analyzer")
    plotting_module = ModuleType(module_name)
    expected = object()
    received = []

    def fake_plot(*args, **kwargs):
        received.append((args, kwargs))
        return expected

    setattr(plotting_module, public_name, fake_plot)
    monkeypatch.setitem(sys.modules, module_name, plotting_module)

    result = getattr(analyzer, public_name)("measurement", title="comparison")

    assert result is expected
    assert received == [(("measurement",), {"title": "comparison"})]
