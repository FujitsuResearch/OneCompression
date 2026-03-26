"""Pytest hooks for shared parametrization.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import pytest


def pytest_collection_modifyitems(items):
    """Reorder so that tests marked slow run last."""
    slow = [t for t in items if t.get_closest_marker("slow")]
    fast = [t for t in items if not t.get_closest_marker("slow")]
    items[:] = fast + slow


def pytest_generate_tests(metafunc):
    """Apply parameters defined on derived test classes to shared parametrized tests."""
    if "params" not in metafunc.fixturenames:
        return

    cls = metafunc.cls
    if cls is None:
        return

    if metafunc.function.__name__ == "test_parameters_boundary":
        metafunc.parametrize("params", cls.boundary_parameters)
        return

    if metafunc.function.__name__ == "test_parameters_abnormal_values_raise":
        metafunc.parametrize("params", cls.abnormal_parameters)
