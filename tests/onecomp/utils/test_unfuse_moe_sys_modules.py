"""

Copyright 2025-2026 Fujitsu Ltd.

Checks that running test_unfuse_moe.py leaves imported onecomp modules intact.

"""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UNFUSE_MOE_TESTS = Path(__file__).with_name("test_unfuse_moe.py")

# Run in a child process: replaced modules would otherwise leak into this pytest session.
_CHILD_SCRIPT = """
import sys

import pytest

import onecomp  # noqa: F401
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

before = {
    name: module
    for name, module in sys.modules.copy().items()
    if name == "onecomp" or name.startswith("onecomp.")
}
exit_code = pytest.main([sys.argv[1], "-q", "-p", "no:cacheprovider"])
replaced = sorted(name for name, module in before.items() if sys.modules.get(name) is not module)
if sys.modules["onecomp.quantizer.gptq.gptq_layer"].GPTQLinear is not GPTQLinear:
    replaced.append("GPTQLinear")
print(f"replaced: {replaced}")
sys.exit(int(exit_code) or int(bool(replaced)))
"""


def test_unfuse_moe_tests_keep_imported_onecomp_modules() -> None:
    """Running test_unfuse_moe.py replaces no imported onecomp module or GPTQLinear."""
    env = os.environ.copy()
    # Options meant for this session (-k, --junitxml) would deselect the child's tests
    # or write to this session's report.
    env.pop("PYTEST_ADDOPTS", None)
    # The check needs no GPU; hiding it keeps the child from creating a CUDA context.
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT, str(_UNFUSE_MOE_TESTS)],
        capture_output=True,
        text=True,
        timeout=300,
        # From the repository root, tests/onecomp cannot shadow the onecomp package.
        cwd=_REPO_ROOT,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
