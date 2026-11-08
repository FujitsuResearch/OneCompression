# Copyright 2025-2026 Fujitsu Ltd.

"""Request-local state and tokenizer-specific reasoning formats."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReasoningFormat:
    start_token_ids: tuple[int, ...]
    end_token_ids: tuple[int, ...]
    starts_in_reasoning: bool = True
    name: str = "custom"

    def __post_init__(self):
        object.__setattr__(self, "start_token_ids", tuple(self.start_token_ids))
        object.__setattr__(self, "end_token_ids", tuple(self.end_token_ids))
        if not self.end_token_ids:
            raise ValueError("An explicit nonempty reasoning-end sequence is required")
        if not self.starts_in_reasoning and not self.start_token_ids:
            raise ValueError("A start marker is required when the prompt is outside reasoning")
        if self.start_token_ids == self.end_token_ids:
            raise ValueError("Start and end markers must differ")
        if any(type(x) is not int or x < 0 for x in self.start_token_ids + self.end_token_ids):
            raise ValueError("Marker token IDs must be nonnegative integers")

    @classmethod
    def from_tokenizer(
        cls, tokenizer, *, start="<think>", end="</think>", starts_in_reasoning=True, name="qwen3"
    ):
        return cls(
            tuple(tokenizer.encode(start, add_special_tokens=False)),
            tuple(tokenizer.encode(end, add_special_tokens=False)),
            starts_in_reasoning,
            name,
        )


@dataclass
class RequestState:
    request_id: str
    format: ReasoningFormat
    generated_token_ids: list[int] = field(default_factory=list)
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    marker_tokens: int = 0
    phase: str = field(init=False)
    boundaries: list[int] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    interventions: list[dict] = field(default_factory=list)
    final_answer: str = ""
    timing: dict = field(default_factory=dict)
    last_text: str = ""
    stop_boundary: int | None = None
    token_kinds: list[str] = field(default_factory=list)
    last_decision_boundary: int | None = None

    def __post_init__(self):
        self.phase = "reasoning" if self.format.starts_in_reasoning else "preamble"

    @property
    def total_generated_tokens(self):
        return len(self.generated_token_ids)

    @property
    def pending_end_marker(self):
        ids = self.generated_token_ids
        end = self.format.end_token_ids
        return any(tuple(ids[-i:]) == end[:i] for i in range(1, min(len(end), len(ids) + 1)))

    def append(self, token_id: int, text: str = ""):
        self.generated_token_ids.append(int(token_id))
        self.last_text = text
        kind = self.phase
        self.token_kinds.append(kind)
        if kind == "reasoning":
            self.reasoning_tokens += 1
        elif kind == "answer":
            self.answer_tokens += 1
        marker = None
        # Some templates include the opening marker, others generate it.
        if self.phase in {"preamble", "reasoning"} and self.format.start_token_ids:
            if (
                tuple(self.generated_token_ids[-len(self.format.start_token_ids) :])
                == self.format.start_token_ids
            ):
                marker = self.format.start_token_ids
                self.phase = "reasoning"
        if (
            self.phase == "reasoning"
            and tuple(self.generated_token_ids[-len(self.format.end_token_ids) :])
            == self.format.end_token_ids
        ):
            marker = self.format.end_token_ids
            self.phase = "answer"
        if marker:
            for i in range(len(self.token_kinds) - len(marker), len(self.token_kinds)):
                old = self.token_kinds[i]
                if old == "reasoning":
                    self.reasoning_tokens -= 1
                elif old == "answer":
                    self.answer_tokens -= 1
                if old != "marker":
                    self.marker_tokens += 1
                self.token_kinds[i] = "marker"
