"""Tests for onecomp.compaction.protocol."""

from onecomp.compaction.protocol import (
    format_citation,
    format_meta_response,
    format_recall_response,
    looks_like_recall_intent,
    normalise_cmd_signature,
    parse_any_recall,
    parse_recall,
    parse_recall_meta,
    suggest_closest_obs_id,
)


def test_parse_recall_positive_cases():
    assert parse_recall("_recall \u00a7a3f2b9c1") == "a3f2b9c1"
    assert parse_recall("  _recall \u00a7a3f2b9c1  ") == "a3f2b9c1"
    assert parse_recall("_recall \u00a7A3F2B9C1") == "a3f2b9c1"  # lowercased
    assert parse_recall("_recall \u00a7a3f2b9c1  # need full grep output") == "a3f2b9c1"


def test_parse_recall_negative_cases():
    assert parse_recall("_recall a3f2b9c1") is None  # missing sigil
    assert parse_recall("_recall \u00a7a3f2b9c1 && ls") is None  # extra shell
    assert parse_recall("echo _recall \u00a7a3f2b9c1") is None
    assert parse_recall("") is None
    assert parse_recall(None) is None


def test_parse_recall_meta_and_dispatcher():
    assert parse_recall_meta("_recall_meta \u00a7deadbeef") == "deadbeef"
    assert parse_recall_meta("_recall \u00a7deadbeef") is None
    assert parse_any_recall("_recall \u00a7deadbeef") == ("recall", "deadbeef")
    assert parse_any_recall("_recall_meta \u00a7deadbeef") == ("recall_meta", "deadbeef")
    assert parse_any_recall("ls -la") is None


def test_looks_like_recall_intent():
    assert looks_like_recall_intent("_recall deadbeef")  # forgot the sigil
    assert looks_like_recall_intent("_recall \u00a7deadbeef | tee out.txt")
    assert not looks_like_recall_intent("_recall \u00a7deadbeef")  # valid, no intent-error
    assert not looks_like_recall_intent("ls -la")
    assert not looks_like_recall_intent("")


def test_suggest_closest_obs_id():
    known = ["a3f2b9c1", "a3f29999", "deadbeef"]
    # Prefix match wins.
    assert suggest_closest_obs_id("a3f", known)[0] in ("a3f2b9c1", "a3f29999")
    # Fuzzy match on a near-miss.
    assert "deadbeef" in suggest_closest_obs_id("deadbeaf", known)
    assert suggest_closest_obs_id("", known) == []
    assert suggest_closest_obs_id("abc", []) == []


def test_format_citation_is_bounded():
    citation = format_citation(
        obs_id="a3f2b9c1",
        cmd_signature="grep -rn foo /testbed",
        returncode=0,
        line_count=10_000,
        char_count=1_000_000,
        head="H" * 500,
        tail="T" * 500,
        head_chars=200,
        tail_chars=200,
    )
    assert "\u00a7a3f2b9c1" in citation
    assert "_recall \u00a7a3f2b9c1" in citation
    # Bounded regardless of the underlying observation size.
    assert len(citation) < 200 + 200 + 400


def test_format_meta_response_shape():
    meta = format_meta_response(
        obs_id="deadbeef",
        cmd_signature="cat big.log",
        returncode=1,
        line_count=42,
        char_count=9000,
        head="start",
        tail="end",
    )
    assert meta.startswith("<recalled_meta")
    assert meta.endswith("</recalled_meta>")
    assert 'rc="1"' in meta


def test_format_recall_response_marks_truncation():
    full = format_recall_response(
        obs_id="deadbeef",
        cmd_signature="cat big.log",
        returncode=0,
        char_count=100,
        line_count=1,
        body="BODY",
    )
    assert "truncated" not in full
    windowed = format_recall_response(
        obs_id="deadbeef",
        cmd_signature="cat big.log",
        returncode=0,
        char_count=100,
        line_count=1,
        body="BODY",
        elided_chars=50,
    )
    assert 'truncated="50"' in windowed


def test_normalise_cmd_signature():
    assert normalise_cmd_signature("  ls   -la\n /tmp ") == "ls -la /tmp"
    long = normalise_cmd_signature("x" * 500)
    assert len(long) == 200
