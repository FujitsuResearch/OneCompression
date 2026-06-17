"""Pre-commit hook: verify that Python files contain the Fujitsu copyright header.

Copyright 2025-2026 Fujitsu Ltd.
"""

import sys

EXPECTED = "Copyright 2025-2026 Fujitsu Ltd."
HEADER_LINES = 100


def check_file(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            head = "".join(f.readline() for _ in range(HEADER_LINES))
    except (OSError, UnicodeDecodeError):
        return True
    if EXPECTED not in head:
        print(f"{path}: missing '{EXPECTED}' in the first {HEADER_LINES} lines")
        return False
    return True


def main() -> int:
    ok = True
    for path in sys.argv[1:]:
        if not check_file(path):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
