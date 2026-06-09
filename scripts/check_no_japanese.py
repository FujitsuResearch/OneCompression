"""
Copyright 2025-2026 Fujitsu Ltd.

"""

import re
import sys

_JAPANESE_RE = re.compile(
    "[\u3000-\u303f"  # CJK symbols & punctuation
    "\u3040-\u309f"  # Hiragana
    "\u30a0-\u30ff"  # Katakana
    "\u4e00-\u9fff"  # CJK unified ideographs
    "\uff00-\uffef]"  # Full-width forms
)


def check_file(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return True
    ok = True
    for i, line in enumerate(lines, 1):
        if _JAPANESE_RE.search(line):
            print(f"{path}:{i}: {line.rstrip()}")
            ok = False
    return ok


def main() -> int:
    ok = True
    for path in sys.argv[1:]:
        if not check_file(path):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
