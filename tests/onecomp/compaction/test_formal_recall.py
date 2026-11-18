"""Exact formal recall and model-facing protocol tests."""

import hashlib
import random

import pytest

from onecomp.compaction.formal import (
    ArcCoreService,
    CatalogAction,
    ExactRecoveryError,
    InvalidAction,
    RecallChunkAction,
    RecallFullAction,
    RecallMetadataAction,
    RecallRangeAction,
    RecallRequiresChunking,
    SQLiteObservationArchive,
    exact_text_chunks,
    exact_token_id_chunks,
    looks_like_action_intent,
    parse_action,
    reconstruct_token_chunks,
    verify_text_chunks,
)


class CharacterCodec:
    def encode(self, text):
        return [ord(character) for character in text]

    def decode(self, token_ids):
        return "".join(chr(token) for token in token_ids)


class NormalizingCodec(CharacterCodec):
    def decode(self, token_ids):
        return super().decode(token_ids).lower()


def test_huge_recall_never_silently_elides_the_middle(tmp_path):
    body = "HEAD-" + "x" * 4000 + "MIDDLE_NEEDLE" + "y" * 4000 + "-TAIL"
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        service = ArcCoreService(archive, max_recall_units=128)
        record = service.record("cat huge.log", body)

        with pytest.raises(RecallRequiresChunking):
            service.recall_full(record.obs_id)
        chunks = service.text_chunks(record.obs_id)
        assert verify_text_chunks(chunks, body)
        assert "".join(chunk.text for chunk in chunks) == body
        assert any("MIDDLE_NEEDLE" in chunk.text for chunk in chunks)
        assert all(chunk.measured_units <= 128 for chunk in chunks)
        assert {chunk.content_sha256 for chunk in chunks} == {
            hashlib.sha256(body.encode()).hexdigest()
        }


def test_text_chunk_property_over_diverse_unicode_and_limits():
    randomizer = random.Random(20260829)
    alphabet = "abcXYZ012\n αβ🙂漢字"  # jp-check: ignore
    for _ in range(100):
        text = "".join(randomizer.choice(alphabet) for _ in range(randomizer.randrange(0, 400)))
        limit = randomizer.randrange(1, 35)
        chunks = exact_text_chunks("abcd1234", text, max_units=limit, measure=len)
        assert verify_text_chunks(chunks, text)
        assert all(chunk.measured_units <= limit for chunk in chunks)


def test_text_chunker_stays_safe_for_non_monotonic_counter():
    def non_monotonic(value):
        return 1 if len(value) % 2 else 10

    chunks = exact_text_chunks("abcd1234", "abcdef", max_units=1, measure=non_monotonic)
    assert "".join(chunk.text for chunk in chunks) == "abcdef"
    assert all(chunk.measured_units <= 1 for chunk in chunks)


def test_token_chunks_preserve_the_original_token_sequence():
    codec = CharacterCodec()
    text = "boundary-sensitive🙂text"
    chunks = exact_token_id_chunks(
        "abcd1234",
        text,
        max_tokens=4,
        codec=codec,
    )
    assert all(len(chunk.token_ids) <= 4 for chunk in chunks)
    assert reconstruct_token_chunks(reversed(chunks), codec) == text

    with pytest.raises(ExactRecoveryError):
        exact_token_id_chunks(
            "abcd1234",
            "UPPER",
            max_tokens=3,
            codec=NormalizingCodec(),
        )


def test_strict_action_grammar_includes_discovery_metadata_chunk_and_range():
    obs_id = "0123abcd"
    assert parse_action("_catalog") == CatalogAction()
    assert parse_action("_catalog find shard 17") == CatalogAction(query="shard 17")
    assert parse_action("_catalog start 10") == CatalogAction(limit=10)
    assert parse_action("_catalog 19 5") == CatalogAction(19, 5)
    assert parse_action(f"_recall §{obs_id}") == RecallFullAction(obs_id)
    assert parse_action(f"_recall_meta §{obs_id}") == RecallMetadataAction(obs_id)
    assert parse_action(f"_recall_chunk §{obs_id} 3") == RecallChunkAction(obs_id, 3)
    assert parse_action(f"_recall_range §{obs_id} 10:20") == RecallRangeAction(obs_id, 10, 20)
    for invalid in (
        f"please _recall §{obs_id}",
        f"_recall §{obs_id} # comment",
        f"_recall_range §{obs_id} 20:10",
        "_catalog start 501",
        "_catalog find trailing ",
        "_recall §NOTHEX",
    ):
        with pytest.raises(InvalidAction):
            parse_action(invalid)
    assert looks_like_action_intent(f"  _RECALL §{obs_id}")
    with pytest.raises(InvalidAction):
        parse_action(f"  _RECALL §{obs_id}")


def test_execute_action_keeps_exact_body_separate_and_catalog_bounded(tmp_path):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        service = ArcCoreService(archive, max_recall_units=600)
        records = [
            service.record(
                "command " + "x" * 500 + str(index),
                f"body-{index}",
                source_key=f"call-{index}",
            )
            for index in range(20)
        ]
        recalled = service.execute_action(f"_recall §{records[0].obs_id}")
        assert recalled.exact
        assert recalled.body == "body-0"
        assert "<recalled" not in recalled.body

        catalog = service.execute_action("_catalog start 20")
        assert len(catalog.body) <= 600
        assert catalog.metadata["has_more"]
        assert catalog.metadata["next_cursor"] is not None

        recovered = []
        cursor = None
        while True:
            action = "_catalog" if cursor is None else f"_catalog {cursor} 20"
            response = service.execute_action(action)
            page_ids = [
                item.obs_id
                for item in service.catalog(
                    after_sequence=cursor,
                    limit=response.metadata["count"],
                ).items
            ]
            recovered.extend(page_ids)
            if not response.metadata["has_more"]:
                break
            cursor = response.metadata["next_cursor"]
        assert recovered == [record.obs_id for record in records]


def test_range_and_chunk_operations_are_individually_bounded(tmp_path):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        service = ArcCoreService(archive, max_recall_units=8)
        record = service.record("cmd", "0123456789abcdef")
        assert service.recall_range(record.obs_id, 4, 12) == "456789ab"
        with pytest.raises(RecallRequiresChunking):
            service.recall_range(record.obs_id, 0, 9)
        assert service.recall_text_chunk(record.obs_id, 1).text == "89abcdef"
        with pytest.raises(InvalidAction):
            service.recall_text_chunk(record.obs_id, 2)


def test_catalog_find_filters_model_facing_discovery(tmp_path):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        service = ArcCoreService(archive, max_recall_units=4096)
        wanted = service.record("inspect target.py", "body")
        service.record("inspect unrelated.py", "body")
        response = service.execute_action("_catalog find target.py")
        assert wanted.obs_id in response.body
        assert "unrelated.py" not in response.body
