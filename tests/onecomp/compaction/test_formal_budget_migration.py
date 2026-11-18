"""Provider guard and experimental WAL migration tests."""

import asyncio
import json
from pathlib import Path

import pytest

from onecomp.compaction.formal import (
    BudgetExceeded,
    BudgetSnapshot,
    InvocationBudgetGuard,
    LegacyMigrationError,
    SQLiteObservationArchive,
    migrate_experimental_wal,
    validate_budget,
)


def test_component_budget_rejects_hidden_or_overlapping_components():
    valid = BudgetSnapshot(
        context_window=100,
        completion_reserve=20,
        baseline=10,
        memory=10,
        recall=20,
        protected_history=10,
        query=10,
        serialization_margin=5,
        labels={"tool_schemas": 5},
    )
    assert validate_budget(valid).remaining == 10
    with pytest.raises(BudgetExceeded):
        validate_budget(
            BudgetSnapshot(
                context_window=100,
                completion_reserve=20,
                baseline=81,
            )
        )
    with pytest.raises(ValueError, match="conflicts"):
        validate_budget(
            BudgetSnapshot(
                context_window=100,
                completion_reserve=20,
                labels={"baseline": 0},
            )
        )
    with pytest.raises(TypeError, match="immutable"):
        valid.labels["new"] = 1


def test_guard_rejects_before_sync_dispatch_and_exposes_coverage():
    invoked = []
    guard = InvocationBudgetGuard(
        context_window=20,
        completion_reserve=5,
        serialization_margin=2,
        count_request=lambda request: len(request["serialized"]),
    )
    assert (
        guard.dispatch(
            {"serialized": "x" * 13},
            lambda request: invoked.append(request) or "ok",
        )
        == "ok"
    )
    with pytest.raises(BudgetExceeded):
        guard.dispatch(
            {"serialized": "x" * 14},
            lambda request: invoked.append(request),
        )
    assert len(invoked) == 1
    assert guard.audit().checked_requests == 2
    assert guard.audit().dispatched_requests == 1
    assert guard.audit().rejected_requests == 1
    assert guard.audit().dispatch_successes == 1
    assert guard.audit().dispatch_failures == 0


def test_guard_distinguishes_dispatch_attempts_from_provider_failures():
    guard = InvocationBudgetGuard(
        context_window=20,
        completion_reserve=5,
        count_request=lambda _request: 1,
    )

    def fail(_request):
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        guard.dispatch({}, fail)

    audit = guard.audit()
    assert audit.dispatched_requests == 1
    assert audit.dispatch_successes == 0
    assert audit.dispatch_failures == 1


def test_guard_rejects_before_async_dispatch():
    invoked = []
    guard = InvocationBudgetGuard(
        context_window=10,
        completion_reserve=2,
        count_request=len,
    )

    async def invoke(request):
        invoked.append(request)
        return "ok"

    async def scenario():
        assert await guard.dispatch_async("12345678", invoke) == "ok"
        with pytest.raises(BudgetExceeded):
            await guard.dispatch_async("123456789", invoke)

    asyncio.run(scenario())
    assert invoked == ["12345678"]


def test_legacy_migration_preserves_old_citations_and_is_resumable(tmp_path, monkeypatch):
    wal = tmp_path / "experimental.jsonl"
    entries = [
        {
            "obs_id": "deadbeef",
            "cmd_signature": "same",
            "full_content": "same",
            "returncode": 0,
            "created_step": 1,
            "extra": {"call": 1},
        },
        {
            "obs_id": "cafebabe",
            "cmd_signature": "same",
            "full_content": "same",
            "returncode": 0,
            "created_step": 2,
            "extra": {"call": 2},
        },
    ]
    wal.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _self: pytest.fail("migration must stream instead of read_bytes"),
    )
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        report = migrate_experimental_wal(wal, archive)
        assert report.loaded == 2
        assert report.aliases_registered == 2
        assert archive.count == 2
        assert archive.get("deadbeef").created_step == 1
        assert archive.get("cafebabe").created_step == 2
        assert archive.get("deadbeef").obs_id != archive.get("cafebabe").obs_id

        repeated = migrate_experimental_wal(wal, archive)
        assert repeated.loaded == 2
        assert archive.count == 2


def test_migration_identity_survives_an_appended_source_file(tmp_path):
    wal = tmp_path / "growing.jsonl"
    first = {"cmd_signature": "first", "full_content": "one"}
    second = {"cmd_signature": "second", "full_content": "two"}
    wal.write_text(json.dumps(first) + "\n")

    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        migrate_experimental_wal(wal, archive, migration_id="stable-run")
        with wal.open("a") as stream:
            stream.write(json.dumps(second) + "\n")
        migrate_experimental_wal(wal, archive, migration_id="stable-run")

        assert archive.count == 2
        assert [record.cmd_signature for record in archive.catalog_page().items] == [
            "first",
            "second",
        ]


def test_migration_rejects_non_standard_json_numbers(tmp_path):
    wal = tmp_path / "nan.jsonl"
    wal.write_text('{"cmd_signature":"cmd","full_content":"body","extra":{"value":NaN}}\n')
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        with pytest.raises(LegacyMigrationError, match="JSON-serializable"):
            migrate_experimental_wal(wal, archive)
        assert archive.count == 0


def test_migration_rolls_back_if_source_changes_between_validation_and_import(
    tmp_path, monkeypatch
):
    from onecomp.compaction.formal import migration  # pylint: disable=import-outside-toplevel

    wal = tmp_path / "changing.jsonl"
    wal.write_text(json.dumps({"cmd_signature": "cmd", "full_content": "body"}) + "\n")
    monkeypatch.setattr(migration, "_file_sha256", lambda _path: "0" * 64)

    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        with pytest.raises(LegacyMigrationError, match="changed while migration"):
            migrate_experimental_wal(wal, archive)
        assert archive.n_records == 0


def test_migration_strict_mode_validates_before_writing(tmp_path):
    wal = tmp_path / "broken.jsonl"
    wal.write_text(
        json.dumps(
            {
                "obs_id": "deadbeef",
                "cmd_signature": "cmd",
                "full_content": "body",
            }
        )
        + "\n{broken\n"
    )
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        with pytest.raises(LegacyMigrationError, match="line 2"):
            migrate_experimental_wal(wal, archive, strict=True)
        assert archive.count == 0

        report = migrate_experimental_wal(wal, archive, strict=False)
        assert report.loaded == 1
        assert report.skipped == 1
        assert archive.get("deadbeef").full_content == "body"


def test_lenient_migration_rolls_back_an_alias_conflict_row(tmp_path):
    wal = tmp_path / "alias-conflict.jsonl"
    wal.write_text(
        json.dumps(
            {
                "obs_id": "deadbeef",
                "cmd_signature": "rejected",
                "full_content": "must not survive",
            }
        )
        + "\n"
    )
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        existing = archive.append("existing", "trusted")
        archive.register_alias("deadbeef", existing.obs_id)

        report = migrate_experimental_wal(wal, archive, strict=False)

        assert report.loaded == 0
        assert report.skipped == 1
        assert archive.known_ids() == [existing.obs_id]
        accepted = archive.append("after", "next")
        assert accepted.sequence == 1
