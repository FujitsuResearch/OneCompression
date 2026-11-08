"""CPU-only controller regression tests; also runnable with unittest."""

import unittest
from dataclasses import replace

from onecomp.reasoning import (
    ReasoningCompressionController,
    ReasoningConfig,
    ReasoningFormat,
    RequestState,
    estimate_prompt_budget,
)
from onecomp.reasoning.actions import StopAction
from onecomp.reasoning.boundaries import (
    SentenceBoundaryDetector,
    TokenBoundaryDetector,
    TransitionBoundaryDetector,
)


class CoreTests(unittest.TestCase):
    def state(self, name="test"):
        return RequestState(name, ReasoningFormat((10, 11), (20, 21), False))

    def test_marker_accounting_and_phase(self):
        s = self.state()
        for t in (10, 11, 1, 2, 20, 21, 3):
            s.append(t)
        self.assertEqual((s.reasoning_tokens, s.answer_tokens, s.marker_tokens), (2, 1, 4))
        self.assertEqual(s.total_generated_tokens, 7)
        self.assertEqual(s.phase, "answer")

    def test_partial_marker_is_counted_until_complete(self):
        s = RequestState("a", ReasoningFormat((10,), (20, 21)))
        s.append(20)
        self.assertTrue(s.pending_end_marker)
        s.append(2)
        self.assertFalse(s.pending_end_marker)
        self.assertEqual(s.reasoning_tokens, 2)

    def test_budget_stop_completes_a_persistent_partial_end_marker(self):
        s = RequestState("a", ReasoningFormat((10,), (20, 21, 22)))
        c = ReasoningCompressionController(ReasoningConfig(policy="fixed", budget=1))
        for token in (20, 20, 21):
            s.append(token)
        self.assertEqual(s.pending_end_prefix, 2)
        self.assertTrue(c.is_boundary(s))
        self.assertEqual(c.decide(s, {}).kind, "stop")
        pending = []
        StopAction(s.format, s.pending_end_prefix).apply(pending)
        self.assertEqual(pending, [22])
        s.append(22)
        self.assertEqual(s.phase, "answer")
        self.assertEqual((s.reasoning_tokens, s.marker_tokens), (1, 3))

    def test_independent_interleaved_requests(self):
        fmt = ReasoningFormat((10,), (20,))
        a, b = RequestState("a", fmt), RequestState("b", fmt)
        c = ReasoningCompressionController(ReasoningConfig(policy="fixed", budget=1))
        a.append(5)
        self.assertEqual(c.decide(a, {}).kind, "stop")
        self.assertEqual(c.decide(b, {}).kind, "continue")
        self.assertEqual(b.interventions, [])
        self.assertEqual(b.boundaries, [0])

    def test_disabled_identity(self):
        s = self.state()
        c = ReasoningCompressionController()
        self.assertEqual(c.decide(s, {"logit_margin": -100}).kind, "continue")
        self.assertEqual(s.signals, [])
        self.assertEqual(s.interventions, [])

    def test_fixed_budget_zero_and_no_sentence_delay(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(ReasoningConfig(policy="fixed", budget=0))
        self.assertTrue(c.is_boundary(s))
        self.assertEqual(c.decide(s, {}).kind, "stop")
        pending = []
        StopAction(s.format).apply(pending)
        self.assertEqual(pending, [20])

    def test_prompt_budget_uses_substituted_per_prompt_budget(self):
        self.assertEqual(estimate_prompt_budget("What is 17 * 24?"), 512)
        self.assertEqual(estimate_prompt_budget(" ".join(["word"] * 100)), 2048)
        self.assertEqual(estimate_prompt_budget(" ".join(["word"] * 200)), 8192)
        config = replace(ReasoningConfig(policy="prompt_budget"), budget=2)
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(config)
        s.append(1)
        self.assertFalse(c.is_boundary(s))
        s.append(1)
        self.assertTrue(c.is_boundary(s))
        self.assertEqual(c.decide(s, {}).reason, "reasoning_budget")

    def test_thinkbrake_gap_direction(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(ReasoningConfig(policy="thinkbrake", threshold=0.1))
        self.assertEqual(c.decide(s, {"logit_margin": 0.2}).kind, "continue")
        s.append(1)
        self.assertEqual(c.decide(s, {"logit_margin": 0.05}).kind, "stop")
        self.assertEqual(s.interventions[0]["reason"], "end_logit_gap")

    def test_thinkbrake_without_margin_continues(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(ReasoningConfig(policy="thinkbrake", threshold=10))
        self.assertEqual(c.decide(s, {"logit_margin": None}).kind, "continue")

    def test_deer_stops_only_above_confidence_threshold(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(
            ReasoningConfig(policy="deer", confidence_threshold=0.9)
        )
        s.append(1)
        self.assertEqual(c.decide(s, {"confidence": 0.9, "answer": "42"}).kind, "continue")
        s.append(1)
        self.assertEqual(c.decide(s, {"confidence": 0.95, "answer": "42"}).kind, "stop")
        self.assertEqual(s.probe_confidences, [0.9, 0.95])
        self.assertEqual(s.interventions[0]["reason"], "probe_confidence")
        s = RequestState("b", ReasoningFormat((10,), (20,)))
        with self.assertRaises(ValueError):
            c.decide(s, {"confidence": 1.5})

    def test_deer_relative_confidence_variant(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(
            ReasoningConfig(policy="deer", confidence_threshold=0.99, relative_confidence=True)
        )
        for confidence, kind in ((0.6, "continue"), (0.5, "continue"), (0.7, "stop")):
            s.append(1)
            self.assertEqual(c.decide(s, {"confidence": confidence}).kind, kind)

    def test_stability_uses_answer_and_requires_window(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(ReasoningConfig(policy="stability", stability_window=3))
        for answer in ("42", "42"):
            s.append(1)
            self.assertEqual(c.decide(s, {"answer": answer}).kind, "continue")
        s.append(1)
        self.assertEqual(c.decide(s, {"answer": "42"}).kind, "stop")

    def test_repeated_boundary_does_not_accumulate_evidence(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(ReasoningConfig(policy="stability", stability_window=2))
        s.append(1, "Done.")
        self.assertEqual(c.decide(s, {"answer": "42"}).kind, "continue")
        self.assertFalse(c.is_boundary(s))
        self.assertEqual(c.decide(s, {"answer": "42"}).kind, "continue")
        self.assertEqual(len(s.probe_answers), 1)
        self.assertEqual(len(s.signals), 1)

    def test_incomplete_probes_never_imply_confidence_or_stability(self):
        for policy in ("deer", "stability"):
            c = ReasoningCompressionController(
                ReasoningConfig(
                    policy=policy,
                    confidence_threshold=0,
                    stability_window=1,
                    relative_confidence=True,
                )
            )
            s = RequestState("a", ReasoningFormat((10,), (20,)))
            for _ in range(2):
                s.append(1)
                decision = c.decide(s, {"answer": "42", "confidence": 1, "probe_complete": False})
                self.assertEqual(decision.kind, "continue")
            self.assertEqual(s.probe_confidences, [])

    def test_boundaries(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        s.append(1, "done.\n")
        self.assertTrue(SentenceBoundaryDetector().detect(s))
        self.assertTrue(TransitionBoundaryDetector().detect(s, " Wait"))
        self.assertFalse(TokenBoundaryDetector(2).detect(s))
        s.append(2)
        self.assertTrue(TokenBoundaryDetector(2).detect(s))

    def test_preamble_tokens_are_neither_reasoning_nor_answer(self):
        s = RequestState("a", ReasoningFormat((10,), (20,), starts_in_reasoning=False))
        for token in (1, 1, 10, 2, 20, 3):
            s.append(token)
        self.assertEqual(
            s.token_kinds, ["preamble", "preamble", "marker", "reasoning", "marker", "answer"]
        )
        self.assertEqual((s.reasoning_tokens, s.answer_tokens, s.marker_tokens), (1, 1, 2))

    def test_repeated_boundary_is_decided_once(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(ReasoningConfig(policy="fixed", budget=1))
        s.append(1)
        self.assertEqual(c.decide(s, {}).kind, "stop")
        self.assertFalse(c.is_boundary(s))
        self.assertEqual(c.decide(s, {}).kind, "continue")
        self.assertEqual(len(s.signals), 1)
        self.assertEqual(len(s.interventions), 1)

    def test_no_oracle_label_leakage(self):
        c = ReasoningCompressionController(ReasoningConfig(policy="fixed"))
        with self.assertRaises(ValueError):
            c.decide(RequestState("a", ReasoningFormat((10,), (20,))), {"safe_stop": True})

    def test_config_validation(self):
        self.assertEqual(ReasoningConfig.from_dict({}).policy, "none")
        for values in (
            {"bad": 1},
            {"policy": "ASAG"},
            {"budget": -1},
            {"interval": 0},
            {"boundary": "paragraph"},
            {"budget": True},
            {"threshold": float("nan")},
            {"threshold": True},
            {"temperature": 0},
            {"temperature": float("inf")},
            {"confidence_threshold": 1.1},
            {"probe_max_tokens": 0},
            {"stability_window": 0},
            {"relative_confidence": 1},
        ):
            with self.assertRaises((ValueError, TypeError)):
                ReasoningConfig.from_dict(values)


if __name__ == "__main__":
    unittest.main()
