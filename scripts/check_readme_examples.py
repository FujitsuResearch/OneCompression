#!/usr/bin/env python3
"""Validate that the README Examples table lists every runnable example.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
EXAMPLES_HEADING = "## 🚀 Examples"
EXAMPLE_LINK_PATTERN = re.compile(r"\[[^]]+\]\((\./(?:example|notebook)/[^)]+)\)")


def get_examples_section(readme: str) -> str:
    """Return the README section headed by ``## 🚀 Examples``."""
    try:
        _, section = readme.split(EXAMPLES_HEADING, maxsplit=1)
    except ValueError as error:
        raise ValueError(f"Missing {EXAMPLES_HEADING!r} heading in {README_PATH}") from error

    return re.split(r"^## ", section, maxsplit=1, flags=re.MULTILINE)[0]


def get_expected_paths() -> set[str]:
    """Return all runnable example paths that must appear in the README."""
    paths = {
        path.relative_to(REPO_ROOT).as_posix() for path in (REPO_ROOT / "example").rglob("*.py")
    }
    # The README lists the tutorial alongside scripts although it is outside example/.
    paths.add("notebook/01_tutorial.ipynb")
    return paths


def main() -> int:
    section = get_examples_section(README_PATH.read_text(encoding="utf-8"))
    # Compare repository-relative paths, not the Markdown link labels.
    linked_paths = [match.removeprefix("./") for match in EXAMPLE_LINK_PATTERN.findall(section)]
    linked_path_set = set(linked_paths)
    expected_paths = get_expected_paths()

    missing_paths = sorted(expected_paths - linked_path_set)
    unexpected_paths = sorted(linked_path_set - expected_paths)
    duplicate_paths = sorted(path for path, count in Counter(linked_paths).items() if count > 1)

    if not (missing_paths or unexpected_paths or duplicate_paths):
        print(f"README Examples table is synchronized ({len(expected_paths)} entries).")
        return 0

    print("README Examples table is out of sync.", file=sys.stderr)
    for label, paths in (
        ("Missing README entries", missing_paths),
        ("README entries without a matching runnable example", unexpected_paths),
        ("Duplicate README entries", duplicate_paths),
    ):
        if paths:
            print(f"{label}:", file=sys.stderr)
            print(*(f"  - {path}" for path in paths), sep="\n", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
