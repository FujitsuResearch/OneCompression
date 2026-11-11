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
        ):
            with self.assertRaises((ValueError, TypeError)):
                ReasoningConfig.from_dict(values)


if __name__ == "__main__":
    unittest.main()
