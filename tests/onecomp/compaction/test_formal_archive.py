"""Formal archive invariants and failure-injection tests."""

import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from onecomp.compaction.formal import (
    ArchiveCapacityExceeded,
    ArchiveConfigurationMismatch,
    ArchiveIntegrityError,
    OccurrenceIdConfig,
    SourceOccurrenceConflict,
    SQLiteObservationArchive,
)


def test_identical_results_remain_distinct_occurrences(tmp_path):
    path = tmp_path / "formal.db"
    with SQLiteObservationArchive(path) as archive:
        first = archive.append("same command", "same result", source_key="call-1")
        second = archive.append("same command", "same result", source_key="call-2")

        assert first.obs_id != second.obs_id
        assert first.sequence == 0
        assert second.sequence == 1
        assert archive.count == 2
        assert archive.verify_integrity(first.obs_id)

    assert path.stat().st_mode & 0o077 == 0


def test_source_key_retry_is_exactly_once_and_conflicts_fail_closed(tmp_path):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        first = archive.append(
            "cmd",
            "body",
            source_key="event",
            extra={"attempt": 1},
            tags=("test", "stdout"),
        )
        retry = archive.append(
            "cmd",
            "body",
            source_key="event",
            created_step=99,
            extra={"attempt": 1},
            tags=("test", "stdout"),
        )
        assert retry.obs_id == first.obs_id
        with pytest.raises(SourceOccurrenceConflict):
            archive.append(
                "cmd",
                "body",
                source_key="event",
                extra={"attempt": 1},
                tags=("different",),
            )
        assert archive.count == 1


def test_source_key_retry_compares_canonical_json_metadata(tmp_path):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        first = archive.append(
            "cmd",
            "body",
            source_key="event",
            extra={"nested": (1, 2)},
        )
        retry = archive.append(
            "cmd",
            "body",
            source_key="event",
            extra={"nested": [1, 2]},
        )
        assert retry.obs_id == first.obs_id
        with pytest.raises(ValueError, match="JSON-serializable"):
            archive.append("cmd", "body", extra={"invalid": float("nan")})


def test_returned_record_metadata_is_deeply_immutable(tmp_path):
    source = {"nested": {"values": [1, 2]}}
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        record = archive.append("cmd", "body", extra=source)
        source["nested"]["values"].append(3)

        assert record.extra["nested"]["values"] == (1, 2)
        with pytest.raises(TypeError, match="immutable"):
            record.extra["new"] = "value"
        with pytest.raises(TypeError, match="immutable"):
            record.extra["nested"]["new"] = "value"


def test_transaction_rolls_back_body_and_sequence_on_metadata_failure(tmp_path, monkeypatch):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        original = archive._meta_set

        def fail_sequence(key, value):
            if key == "next_sequence":
                raise OSError("injected durable metadata failure")
            return original(key, value)

        monkeypatch.setattr(archive, "_meta_set", fail_sequence)
        with pytest.raises(OSError, match="injected"):
            archive.append("cmd", "body")
        monkeypatch.setattr(archive, "_meta_set", original)

        record = archive.append("cmd", "body")
        assert record.sequence == 0
        assert archive.count == 1


def test_archive_quota_and_identifier_configuration_are_persistent(tmp_path):
    path = tmp_path / "formal.db"
    config = OccurrenceIdConfig(max_occurrences=4, fingerprint_bits=32)
    with SQLiteObservationArchive(
        path,
        id_config=config,
        max_archive_bytes=5,
    ) as archive:
        archive.append("cmd", "12345")
        with pytest.raises(ArchiveCapacityExceeded):
            archive.append("cmd", "6")

    with SQLiteObservationArchive(path) as reopened:
        assert reopened.id_config == config
        assert reopened.max_archive_bytes == 5

    with pytest.raises(ArchiveConfigurationMismatch):
        SQLiteObservationArchive(
            path,
            id_config=OccurrenceIdConfig(max_occurrences=8, fingerprint_bits=32),
        )


def test_archive_owner_binding_is_persistent_and_body_free(tmp_path):
    path = tmp_path / "formal.db"
    with SQLiteObservationArchive(path) as archive:
        archive.bind_owner("tenant-a")
        archive.bind_owner("tenant-a")
    raw = sqlite3.connect(path)
    owner_digest = raw.execute("SELECT value FROM arc_meta WHERE key = 'owner_sha256'").fetchone()[
        0
    ]
    raw.close()
    assert owner_digest != "tenant-a"
    assert len(owner_digest) == 64

    with SQLiteObservationArchive(path) as archive:
        with pytest.raises(ArchiveConfigurationMismatch, match="different owner"):
            archive.bind_owner("tenant-b")


def test_legacy_aliases_match_the_model_recall_grammar(tmp_path):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        record = archive.append("cmd", "body")
        archive.register_alias("DEADBEEF", record.obs_id)
        assert archive.get("deadbeef").obs_id == record.obs_id
        for invalid in ("name", "123", "g123", "x" * 129):
            with pytest.raises(ValueError, match="hexadecimal"):
                archive.register_alias(invalid, record.obs_id)


def test_exact_line_verification_rejects_partial_matches(tmp_path):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        archive.append("cmd", "prefix token suffix\r\nexact-token\n")
        assert archive.contains_value("token")
        assert not archive.contains_exact_line("token")
        assert archive.contains_exact_line("exact-token")


def test_catalog_is_complete_body_free_filtered_and_cursor_stable(tmp_path):
    secret = "SECRET_MIDDLE_NEVER_RENDERED"
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        records = [
            archive.append(
                f"command {index}",
                f"body-{index}-{secret}",
                returncode=index % 2,
                created_step=index,
                tags=("even" if index % 2 == 0 else "odd",),
            )
            for index in range(7)
        ]
        pages = []
        cursor = None
        while True:
            page = archive.catalog_page(after_sequence=cursor, limit=3)
            pages.append(page)
            if not page.has_more:
                break
            cursor = page.next_cursor
        assert [item.obs_id for page in pages for item in page.items] == [
            record.obs_id for record in records
        ]
        assert secret not in repr(pages)
        assert [item.obs_id for item in archive.catalog_page(limit=10, tag="even").items] == [
            records[index].obs_id for index in (0, 2, 4, 6)
        ]
        assert [item.obs_id for item in archive.catalog_page(limit=10, returncode=1).items] == [
            records[index].obs_id for index in (1, 3, 5)
        ]
        assert archive.catalog_count() == 7
        assert archive.catalog_count(query="command 3") == 1
        assert archive.recent_ids(2) == [records[-2].obs_id, records[-1].obs_id]

        first_recent = archive.recent_catalog_page(limit=3)
        second_recent = archive.recent_catalog_page(
            before_sequence=first_recent.next_cursor,
            limit=3,
        )
        assert [item.obs_id for item in first_recent.items] == [
            records[index].obs_id for index in (6, 5, 4)
        ]
        assert [item.obs_id for item in second_recent.items] == [
            records[index].obs_id for index in (3, 2, 1)
        ]


def test_body_search_is_literal_exact_and_tracks_new_appends(tmp_path):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        archive.append("first", 'prefix Exact-Needle "quoted" suffix')
        assert archive.contains_value('Exact-Needle "quoted"')
        assert not archive.contains_value("exact-needle")

        latest = archive.append("second", "late-search-marker")

        assert [row.obs_id for row in archive.find_records_containing("search-marker")] == [
            latest.obs_id
        ]


def test_catalog_query_uses_unicode_casefolding(tmp_path):
    with SQLiteObservationArchive(tmp_path / "unicode-search.db") as archive:
        record = archive.append("Prüfe Straße", "body", tags=("ÜBERBLICK",))

        assert archive.catalog_page(query="STRASSE").items[0].obs_id == record.obs_id
        assert archive.catalog_count(query="überblick") == 1


def test_serializable_iterator_excludes_appends_past_its_high_watermark(tmp_path):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        first = archive.append("first", "first body")
        second = archive.append("second", "second body")
        snapshot = archive.iter_serializable(include_bodies=False, batch_size=1)
        assert next(snapshot)["obs_id"] == first.obs_id

        archive.append("late", "late body")

        assert [row["obs_id"] for row in snapshot] == [second.obs_id]


def test_multiple_connections_allocate_each_occurrence_once(tmp_path):
    path = tmp_path / "concurrent.db"
    SQLiteObservationArchive(path).close()

    def write(worker):
        with SQLiteObservationArchive(path) as archive:
            return [
                archive.append(
                    "same",
                    "same",
                    source_key=f"worker-{worker}-call-{index}",
                ).obs_id
                for index in range(12)
            ]

    with ThreadPoolExecutor(max_workers=6) as pool:
        batches = list(pool.map(write, range(6)))
    obs_ids = [obs_id for batch in batches for obs_id in batch]
    assert len(obs_ids) == len(set(obs_ids)) == 72
    with SQLiteObservationArchive(path) as archive:
        assert archive.count == 72
        assert [
            archive.id_factory.sequence_from_id(obs_id) for obs_id in archive.known_ids()
        ] == list(range(72))


def test_nested_batch_rollback_is_local_when_caller_recovers(tmp_path):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        with archive.batch():
            with pytest.raises(RuntimeError, match="reject row"):
                with archive.batch():
                    archive.append("rejected", "must roll back")
                    raise RuntimeError("reject row")
            accepted = archive.append("accepted", "committed")

        assert accepted.sequence == 0
        assert archive.known_ids() == [accepted.obs_id]


def test_read_only_archive_recalls_without_chmod_or_writes(tmp_path, monkeypatch):
    path = tmp_path / "formal.db"
    body = "prefix\n" + "日本語" * 10_000 + "\nsuffix"  # jp-check: ignore
    with SQLiteObservationArchive(path) as archive:
        record = archive.append("cat large.log", body)
        archive.checkpoint(truncate=True)

    original_connect = sqlite3.connect
    connection_targets = []

    def capture_connect(target, *args, **kwargs):
        connection_targets.append(str(target))
        return original_connect(target, *args, **kwargs)

    def reject_chmod(*_args, **_kwargs):
        raise OSError(30, "read-only file system")

    monkeypatch.setattr("onecomp.compaction.formal.archive.sqlite3.connect", capture_connect)
    monkeypatch.setattr("onecomp.compaction.formal.archive.os.chmod", reject_chmod)
    with SQLiteObservationArchive(path, read_only=True) as archive:
        assert archive.get(record.obs_id).full_content == body
        assert archive.read_range(record.obs_id, 16_000, 17_000) == body[16_000:17_000]
        archive.record_recall(record.obs_id, 3)
        assert archive.get(record.obs_id).n_recalls == 0
        with pytest.raises(RuntimeError, match="read-only"):
            archive.append("rejected", "body")
    assert connection_targets[-1].endswith("?mode=ro&immutable=1")


def test_read_verification_detects_body_tampering(tmp_path):
    path = tmp_path / "formal.db"
    with SQLiteObservationArchive(path) as archive:
        record = archive.append("cmd", "trusted")
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE observations SET full_content = ? WHERE obs_id = ?",
        ("tampered", record.obs_id),
    )
    connection.commit()
    connection.close()

    with SQLiteObservationArchive(path) as archive:
        with pytest.raises(ArchiveIntegrityError):
            archive.get(record.obs_id)


def test_metadata_reads_detect_record_metadata_tampering(tmp_path):
    path = tmp_path / "metadata-tamper.db"
    with SQLiteObservationArchive(path) as archive:
        record = archive.append("trusted command", "trusted body", tags=("trusted",))

    raw = sqlite3.connect(path)
    raw.execute(
        "UPDATE observations SET cmd_signature = ? WHERE obs_id = ?",
        ("tampered command", record.obs_id),
    )
    raw.commit()
    raw.close()

    with SQLiteObservationArchive(path) as archive:
        with pytest.raises(ArchiveIntegrityError, match="metadata integrity"):
            archive.get_metadata(record.obs_id)
        with pytest.raises(ArchiveIntegrityError, match="metadata integrity"):
            archive.catalog_page()


@pytest.mark.parametrize("key,value", [("next_sequence", "99"), ("body_bytes", "0")])
def test_reopen_rejects_mismatched_archive_counters(tmp_path, key, value):
    path = tmp_path / f"counter-{key}.db"
    with SQLiteObservationArchive(path) as archive:
        archive.append("cmd", "non-empty body")

    raw = sqlite3.connect(path)
    raw.execute("UPDATE arc_meta SET value = ? WHERE key = ?", (value, key))
    raw.commit()
    raw.close()

    with pytest.raises(ArchiveIntegrityError, match="does not match records"):
        SQLiteObservationArchive(path)


def test_v1_archive_is_verified_before_integrity_metadata_upgrade(tmp_path):
    path = tmp_path / "legacy-v1.db"
    with SQLiteObservationArchive(path) as archive:
        record = archive.append("cmd", "trusted body")

    raw = sqlite3.connect(path)
    raw.execute("UPDATE arc_meta SET value = '1' WHERE key = 'schema_version'")
    raw.execute("ALTER TABLE observations DROP COLUMN record_sha256")
    raw.commit()
    raw.close()

    with SQLiteObservationArchive(path) as archive:
        assert archive.get(record.obs_id).full_content == "trusted body"

    raw = sqlite3.connect(path)
    version = raw.execute("SELECT value FROM arc_meta WHERE key = 'schema_version'").fetchone()[0]
    columns = {row[1] for row in raw.execute("PRAGMA table_info(observations)")}
    raw.close()
    assert version == "2"
    assert "record_sha256" in columns


def test_v1_upgrade_does_not_launder_existing_body_corruption(tmp_path):
    path = tmp_path / "corrupt-v1.db"
    with SQLiteObservationArchive(path) as archive:
        archive.append("cmd", "trusted body")

    raw = sqlite3.connect(path)
    raw.execute("UPDATE arc_meta SET value = '1' WHERE key = 'schema_version'")
    raw.execute("ALTER TABLE observations DROP COLUMN record_sha256")
    raw.execute("UPDATE observations SET full_content = 'tampered body'")
    raw.commit()
    raw.close()

    with pytest.raises(ArchiveIntegrityError, match="v1 integrity"):
        SQLiteObservationArchive(path)


def test_every_body_returning_read_detects_tampering(tmp_path):
    path = tmp_path / "formal.db"
    with SQLiteObservationArchive(path) as archive:
        record = archive.append("cmd", "trusted needle")
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE observations SET full_content = ? WHERE obs_id = ?",
        ("tampered needle", record.obs_id),
    )
    connection.commit()
    connection.close()

    with SQLiteObservationArchive(path) as archive:
        with pytest.raises(ArchiveIntegrityError):
            archive.find_by_signature("cmd")
        with pytest.raises(ArchiveIntegrityError):
            archive.find_records_containing("needle")
        with pytest.raises(ArchiveIntegrityError):
            archive.contains_value("needle")
        with pytest.raises(ArchiveIntegrityError):
            archive.to_serializable(include_bodies=True)
        assert "full_content" not in archive.to_serializable(include_bodies=False)[0]


def test_range_reads_use_verified_substrings_and_invalidate_on_external_commit(
    tmp_path, monkeypatch
):
    path = tmp_path / "formal.db"
    with SQLiteObservationArchive(path) as archive:
        record = archive.append("cmd", "0123456789" * 1000)
        verified_reads = 0
        original = archive._verified_record_from_row

        def count_verified(row):
            nonlocal verified_reads
            verified_reads += 1
            return original(row)

        monkeypatch.setattr(archive, "_verified_record_from_row", count_verified)
        assert archive.read_range(record.obs_id, 100, 110) == "0123456789"
        assert archive.read_range(record.obs_id, 200, 210) == "0123456789"
        assert verified_reads == 0, "a freshly committed body is already verified"

        connection = sqlite3.connect(path)
        connection.execute(
            "UPDATE observations SET full_content = ? WHERE obs_id = ?",
            ("tampered" * 1000, record.obs_id),
        )
        connection.commit()
        connection.close()
        with pytest.raises(ArchiveIntegrityError):
            archive.read_range(record.obs_id, 0, 8)


def test_range_reads_are_exact_across_utf8_block_boundaries(tmp_path):
    text = "a" * 16_380 + "é😊漢字" + "z" * 20_000  # jp-check: ignore
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        record = archive.append("unicode", text)

        assert archive.read_range(record.obs_id, 16_378, 16_390) == text[16_378:16_390]
        assert archive.read_range(record.obs_id, 20_000, 20_010) == text[20_000:20_010]


def test_existing_archive_range_index_is_backfilled_after_full_verification(tmp_path):
    path = tmp_path / "formal.db"
    with SQLiteObservationArchive(path) as archive:
        record = archive.append("legacy-layout", "0123456789" * 10_000)
    connection = sqlite3.connect(path)
    connection.execute("DELETE FROM observation_body_blocks")
    connection.execute("DELETE FROM observation_body_block_meta")
    connection.commit()
    connection.close()

    with SQLiteObservationArchive(path) as archive:
        assert archive.read_range(record.obs_id, 50_000, 50_010) == "0123456789"
        count = archive._conn.execute(
            "SELECT COUNT(*) FROM observation_body_blocks WHERE sequence = ?",
            (record.sequence,),
        ).fetchone()[0]
        assert count > 1


def test_append_returns_committed_record_without_reselecting_body(tmp_path, monkeypatch):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        monkeypatch.setattr(
            archive,
            "get",
            lambda _obs_id: pytest.fail("append must not reload the committed body"),
        )
        record = archive.append("cmd", "large" * 1000)
        assert record.sequence == 0
        assert record.char_count == 5000


def test_recall_metrics_are_buffered_but_visible_and_flushed(tmp_path):
    path = tmp_path / "formal.db"
    with SQLiteObservationArchive(path) as archive:
        record = archive.append("cmd", "body")
        archive.record_recall(record.obs_id, 3)
        archive.record_recall(record.obs_id, 5)
        visible = archive.get(record.obs_id)
        assert visible.n_recalls == 2
        assert visible.last_recalled == 5
        assert archive.stats().recall_count == 2
    with SQLiteObservationArchive(path) as reopened:
        persisted = reopened.get(record.obs_id)
        assert persisted.n_recalls == 2
        assert persisted.last_recalled == 5


def test_archive_permissions_do_not_depend_on_process_umask(tmp_path):
    path = tmp_path / "private" / "formal.db"
    previous = os.umask(0)
    try:
        SQLiteObservationArchive(path).close()
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o077 == 0
    assert path.parent.stat().st_mode & 0o077 == 0


def test_archive_is_private_before_sqlite_opens_it(tmp_path, monkeypatch):
    path = tmp_path / "formal.db"
    original = sqlite3.connect
    observed_modes = []

    def connect(candidate, *args, **kwargs):
        observed_modes.append(Path(candidate).stat().st_mode & 0o777)
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr("onecomp.compaction.formal.archive.sqlite3.connect", connect)
    SQLiteObservationArchive(path).close()
    assert observed_modes == [0o600]


def test_archive_preserves_existing_parent_mode_and_protects_sidecars(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    path = shared / "formal.db"

    with SQLiteObservationArchive(path) as archive:
        archive.append("cmd", "body")
        assert shared.stat().st_mode & 0o777 == 0o755
        artifacts = [
            candidate
            for candidate in (
                path,
                Path(str(path) + "-wal"),
                Path(str(path) + "-shm"),
                Path(str(path) + "-journal"),
            )
            if candidate.exists()
        ]
        assert path in artifacts
        assert all(candidate.stat().st_mode & 0o077 == 0 for candidate in artifacts)


def test_committed_occurrence_survives_abrupt_process_exit(tmp_path):
    path = tmp_path / "crash.db"
    script = (
        "import os\n"
        "from onecomp.compaction.formal import SQLiteObservationArchive\n"
        f"a = SQLiteObservationArchive({str(path)!r})\n"
        "a.append('crash-command', 'committed-body', source_key='crash-call')\n"
        "os._exit(17)\n"
    )
    process = subprocess.run([sys.executable, "-c", script], check=False)
    assert process.returncode == 17
    with SQLiteObservationArchive(path) as archive:
        assert archive.n_records == 1
        committed = archive.get(archive.known_ids()[0])
        assert committed.full_content == "committed-body"
        following = archive.append("next", "body", source_key="next-call")
        assert following.sequence == 1


def test_occurrence_metadata_has_explicit_denial_of_service_bounds():
    with SQLiteObservationArchive() as archive:
        with pytest.raises(ValueError, match="cmd_signature"):
            archive.append("c" * 20_000, "body")
        with pytest.raises(ValueError, match="source_key"):
            archive.append("cmd", "body", source_key="s" * 2_000)
        with pytest.raises(ValueError, match="at most 64"):
            archive.append("cmd", "body", tags=[f"tag-{index}" for index in range(65)])
        with pytest.raises(ValueError, match="65536"):
            archive.append("cmd", "body", extra={"payload": "x" * 70_000})
        with pytest.raises(ValueError, match="valid Unicode"):
            archive.append("cmd", "bad\ud800")
        with pytest.raises(ValueError, match="JSON-serializable"):
            archive.append("cmd", "body", extra={"invalid": "bad\ud800"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("returncode", 1 << 63),
        ("returncode", -(1 << 63) - 1),
        ("created_step", 1 << 63),
        ("created_ns", 1 << 63),
    ],
)
def test_occurrence_integers_fail_with_domain_error_before_sqlite(field, value):
    with SQLiteObservationArchive() as archive:
        with pytest.raises(ValueError, match="signed 64-bit"):
            archive.append("cmd", "body", **{field: value})
