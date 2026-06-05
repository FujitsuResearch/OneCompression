"""Specification tests for ``_strip_thinking_tokens`` in gen_answer.

Each test docstring documents concrete Before → After behaviour
(see onecomp/eval/README.md).
"""

from __future__ import annotations

from onecomp.eval.evals.mt_bench.gen_answer import _strip_thinking_tokens as strip_thinking_tokens


class TestPassthrough:
    def test_plain_text_unchanged(self):
        """Before: Hello world  →  After: Hello world"""
        assert strip_thinking_tokens("Hello world") == "Hello world"

    def test_empty_string(self):
        """Before: (empty)  →  After: (empty)"""
        assert strip_thinking_tokens("") == ""


class TestHarmony:
    def test_final_channel_message_body_only(self):
        """Harmony: take text after the last <|message|> in the final channel.

        Before:
          <|channel|>analysis<|message|>reasoning...
          <|channel|>final<|message|>Answer text
        After:
          Answer text
        """
        text = (
            "<|channel|>analysis<|message|>reasoning..." "<|channel|>final<|message|>Answer text"
        )
        assert strip_thinking_tokens(text) == "Answer text"

    def test_single_final_channel_with_message_prefix(self):
        """Before: <|channel|>final<|message|>Answer  →  After: Answer"""
        assert strip_thinking_tokens("<|channel|>final<|message|>Answer") == "Answer"

    def test_last_channel_without_message_keeps_segment_tail(self):
        """Before: <|channel|>a<|channel|>final segment  →  After: final segment"""
        text = "<|channel|>analysis stuff<|channel|>final segment"
        assert strip_thinking_tokens(text) == "final segment"

    def test_residual_control_tokens_removed(self):
        """Before: Answer <|message|> x  →  After: Answer  x"""
        assert strip_thinking_tokens("Answer <|message|> x") == "Answer  x"

    def test_start_end_return_tokens_removed(self):
        """Before: <|start|>Hi<|end|>  →  After: Hi"""
        assert strip_thinking_tokens("<|start|>Hi<|end|>") == "Hi"


class TestRedactedThinking:
    def test_paired_block_before_answer(self):
        """Before: <think>…</think>Answer  →  After: Answer"""
        text = "<think>reasoning</think>Answer"
        assert strip_thinking_tokens(text) == "Answer"

    def test_paired_block_multiline(self):
        """Before: <think>line1\\nline2</think>Answer
        After: Answer"""
        text = "<think>line1\nline2</think>Answer"
        assert strip_thinking_tokens(text) == "Answer"

    def test_paired_block_between_answer_parts(self):
        """Before: Part1 <think>t</think> Part2
        After: Part1  Part2"""
        text = "Part1 <think>t</think> Part2"
        assert strip_thinking_tokens(text) == "Part1  Part2"


class TestCohereThinking:
    def test_pipe_delimited_pair(self):
        """Before: <|START_THINKING|>t<|END_THINKING|>Answer  →  After: Answer"""
        text = "<|START_THINKING|>thought<|END_THINKING|>Answer"
        assert strip_thinking_tokens(text) == "Answer"

    def test_angle_bracket_pair(self):
        """Before: <START_THINKING>t<END_THINKING>Answer  →  After: Answer"""
        text = "<START_THINKING>thought<END_THINKING>Answer"
        assert strip_thinking_tokens(text) == "Answer"


class TestCombined:
    def test_harmony_then_redacted_thinking(self):
        """Harmony final channel wins; thinking tags inside earlier segments are dropped.

        Before:
          <|channel|>analysis<|message|><think>x</think>
          <|channel|>final<|message|>Visible answer
        After:
          Visible answer
        """
        text = (
            "<|channel|>analysis<|message|>"
            "<think>x</think>"
            "<|channel|>final<|message|>Visible answer"
        )
        assert strip_thinking_tokens(text) == "Visible answer"
