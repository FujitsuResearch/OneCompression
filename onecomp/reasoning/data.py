# Copyright 2025-2026 Fujitsu Ltd.

"""Deliberately conservative answer grading for boxed final answers."""

import re
from fractions import Fraction

PROMPT_TEMPLATE = (
    "Solve the following problem. Put only your final answer inside \\boxed{{}}.\n\n{question}"
)


def extract_answer(text):
    """Last balanced boxed answer, then explicit answer line or one-line output.

    Never extract an arbitrary last number from unfinished reasoning. This is an
    exact/numeric grader, not a full symbolic equivalence engine for MATH-500.
    """
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(starts):
        depth = 1
        for end in range(match.end(), len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            if depth == 0:
                return text[match.end() : end].strip()
    match = re.search(r"(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)", text, re.I)
    if match:
        return match.group(1).strip().rstrip(".")
    stripped = text.strip()
    return stripped.rstrip(".") if "\n" not in stripped and len(stripped) <= 120 else ""


def normalize_answer(answer):
    answer = str(answer).strip().replace("$", "").replace(r"\left", "").replace(r"\right", "")
    answer = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\\(?:dfrac|tfrac|frac)\{(-?\d+)\}\{(-?\d+)\}", r"\1/\2", answer)
    answer = re.sub(r"\s+", "", answer)
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", answer):
        answer = answer.replace(",", "")
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", answer):
        try:
            return str(Fraction(answer))
        except (ValueError, ZeroDivisionError):
            pass
    return answer


def grade_answer(text, expected):
    actual = extract_answer(text)
    return bool(actual) and normalize_answer(actual) == normalize_answer(expected)
