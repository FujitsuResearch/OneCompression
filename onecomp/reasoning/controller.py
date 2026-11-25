# Copyright 2025-2026 Fujitsu Ltd.

"""Coordinate request-local observations, policies and intervention records."""

import math
from time import perf_counter

from .actions import Decision
from .boundaries import make_boundary
from .config import ReasoningConfig
from .policies import make_policy
from .signals import relative_entropy_change


class ReasoningCompressionController:
    def __init__(self, config=None):
        self.config = config or ReasoningConfig()
        self.boundary = make_boundary(self.config)
        self.policy = make_policy(self.config)

    def is_boundary(self, state, next_token_text="", *, last_token_forced=False):
        c = self.config
        if state.phase != "reasoning":
            return False
        if state.last_decision_boundary == state.total_generated_tokens:
            return False
        # Budgets are hard limits; their stop completes a partially emitted end marker.
        forced = self.policy.forced_boundary(state)
        if forced is not None:
            return forced
        # Adaptive checks wait while the model may be writing the end marker itself.
        if state.pending_end_marker or state.reasoning_tokens < c.min_reasoning_tokens:
            return False
        # A sentence ending in an inserted jump is not a new model-generated boundary.
        if c.boundary == "sentence" and last_token_forced:
            return False
        return self.boundary.detect(state, next_token_text)

    def needs_probe(self, state, observation):
        return self.policy.needs_probe(state, observation)

    def decide(self, state, observation):
        started = perf_counter()
        if self.config.policy == "none":
            return Decision()
        if state.phase != "reasoning":
            return Decision()
        values = dict(observation)
        # Labels belong to offline analysis only; reject accidental leakage.
        if "safe_stop" in values or "correct" in values:
            raise ValueError("Oracle correctness labels cannot be policy observations")
        boundary = state.total_generated_tokens
        if state.last_decision_boundary == boundary:
            # Repeated engine callbacks for the same prefix are not fresh probes
            # and must not accumulate evidence or apply an intervention twice.
            return Decision()
        complete = values.get("probe_complete", True)
        if type(complete) is not bool:
            raise ValueError("probe_complete must be boolean")
        if not complete:
            values["confidence"] = None
            values["answer"] = ""
        confidence = values.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("confidence must be a finite probability or None")
        if confidence is not None:
            state.probe_confidences.append(values["confidence"])
        if "answer" in values:
            state.probe_answers.append(values["answer"])
            w = self.config.stability_window
            answers = state.probe_answers[-w:]
            values["answer_stability"] = float(
                len(answers) == w and bool(answers[0]) and len(set(answers)) == 1
            )
        entropy = values.get("attention_entropy")
        if entropy is not None:
            first = state.first_entropy is None
            values["first_attention_boundary"] = first
            if first:
                state.first_entropy = entropy
            # Derive the ratio from request-local history; caller-provided values
            # must not silently use a different denominator or previous boundary.
            values["relative_entropy"] = (
                None if first else relative_entropy_change(entropy, state.first_entropy)
            )
        decision = self.policy.decide(state, values)
        state.last_decision_boundary = boundary
        if not state.boundaries or state.boundaries[-1] != boundary:
            state.boundaries.append(boundary)
        state.signals.append(
            {
                "boundary": boundary,
                "reasoning_tokens": state.reasoning_tokens,
                **values,
                "decision": decision.kind,
            }
        )
        if decision.kind != "continue":
            state.interventions.append(
                {
                    "boundary": boundary,
                    "reasoning_tokens": state.reasoning_tokens,
                    "kind": decision.kind,
                    "reason": decision.reason,
                }
            )
            if decision.kind == "stop":
                state.stop_boundary = boundary
            if decision.kind == "jump":
                state.jump_count += 1
        state.timing["policy_s"] = state.timing.get("policy_s", 0) + perf_counter() - started
        return decision
