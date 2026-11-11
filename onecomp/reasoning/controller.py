# Copyright 2025-2026 Fujitsu Ltd.

"""Coordinate request-local observations, policies and intervention records."""

from time import perf_counter

from .actions import Decision
from .boundaries import make_boundary
from .config import ReasoningConfig
from .policies import make_policy


class ReasoningCompressionController:
    def __init__(self, config=None):
        self.config = config or ReasoningConfig()
        self.boundary = make_boundary(self.config)
        self.policy = make_policy(self.config)

    def is_boundary(self, state, next_token_text=""):
        c = self.config
        if state.phase != "reasoning" or state.pending_end_marker:
            return False
        if state.last_decision_boundary == state.total_generated_tokens:
            return False
        forced = self.policy.forced_boundary(state)
        if forced is not None:
            return forced
        if state.reasoning_tokens < c.min_reasoning_tokens:
            return False
        return self.boundary.detect(state, next_token_text)

    def needs_probe(self, state, observation):
        return self.policy.needs_probe(state, observation)

    def decide(self, state, observation):
        started = perf_counter()
        if self.config.policy == "none":
            return Decision()
        if state.phase != "reasoning" or state.pending_end_marker:
            return Decision()
        values = dict(observation)
        # Labels belong to offline analysis only; reject accidental leakage.
        if "safe_stop" in values or "correct" in values:
            raise ValueError("Oracle correctness labels cannot be policy observations")
        boundary = state.total_generated_tokens
        if state.last_decision_boundary == boundary:
            # Repeated engine callbacks for the same prefix must not apply an
            # intervention twice.
            return Decision()
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
        state.timing["policy_s"] = state.timing.get("policy_s", 0) + perf_counter() - started
        return decision
