# Copyright 2025-2026 Fujitsu Ltd.

"""Shared text and result handling for the reasoning backends."""

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from onecomp.reasoning import RequestState

DEER_PROBE_CUE = "\n**Final Answer**\n\\boxed"
ASAG_PROBE_CUE = "\n\n Final Answer\n\n \\boxed"
ASAG_JUMP = (
    "Wait, my previous reasoning is not correct. I should adopt a more concise "
    "and different approach to reexamine this problem.\n\n"
)


@dataclass
class GenerationResult:
    state: RequestState
    text: str
    final_answer: str
    metrics: dict[str, Any]


class TextStream:
    """Incremental detokenization that holds back incomplete characters."""

    def __init__(self, decode: Callable[[Sequence[int]], str]):
        self._decode = decode
        self._ids: list[int] = []
        self._prefix = 0
        self._read = 0

    def push(self, token_id: int, final: bool = False) -> str:
        self._ids.append(token_id)
        prefix_text = self._decode(self._ids[self._prefix : self._read])
        new_text = self._decode(self._ids[self._prefix :])
        if not final and (len(new_text) <= len(prefix_text) or new_text.endswith("�")):
            return ""
        self._prefix, self._read = self._read, len(self._ids)
        return new_text[len(prefix_text) :]


def boxed_content(token_ids, decode):
    r"""Return the first complete boxed answer and its content token indices.

    The cue already ends in \boxed, so a leading braced group is its argument.
    A group elsewhere must follow an explicit \boxed command.

    A character split across tokens decodes to U+FFFD until its last token, so
    character offsets are measured only at complete-character prefixes and
    every token of a split character shares that character's span.
    """
    text = decode(token_ids)
    for match in re.finditer(r"^\s*\{|\\boxed\s*\{", text):
        start = match.end() - 1
        depth = 0
        for end in range(start, len(text)):
            depth += (text[end] == "{") - (text[end] == "}")
            if depth == 0:
                break
        answer = text[start + 1 : end].strip() if depth == 0 else ""
        if answer:
            break
    else:
        return "", [], False
    selected = []
    pending = []
    left = 0
    for index in range(len(token_ids)):
        pending.append(index)
        prefix = decode(token_ids[: index + 1])
        if prefix.endswith("�") and index + 1 < len(token_ids):
            continue
        right = len(prefix)
        if text[max(left, start + 1) : min(right, end)].strip():
            selected.extend(pending)
        if right > end:
            break
        pending = []
        left = right
    return answer, selected, bool(selected)
