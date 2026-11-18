"""Body-safe, single-run analysis for context compaction."""

from .report import build_run_report, load_run_report, save_run_report, validate_run_report


def visualize_run(*args, **kwargs):
    """Render a run dashboard without importing matplotlib at package import."""
    from .visualization import (
        visualize_run as _visualize_run,  # pylint: disable=import-outside-toplevel
    )

    return _visualize_run(*args, **kwargs)


__all__ = [
    "build_run_report",
    "load_run_report",
    "save_run_report",
    "validate_run_report",
    "visualize_run",
]
