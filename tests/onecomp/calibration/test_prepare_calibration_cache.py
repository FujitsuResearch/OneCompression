"""Test fixed-cache integrity and replacement without network or shared storage."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import datasets
import pytest

_SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "prepare_calibration_cache.py"
_SPEC = importlib.util.spec_from_file_location("prepare_calibration_cache", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
_CACHE_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CACHE_SCRIPT)


def _dataset(values):
    """Create a minimal C4-shaped dataset for fast, self-contained tests."""

    return datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "text": values,
                    "url": [f"https://example.com/{index}" for index in range(len(values))],
                }
            )
        }
    )


def _write_cache(cache_root, values):
    """Write a valid synthetic cache and the manifest expected by verification."""

    destination = cache_root / "c4"
    _dataset(values).save_to_disk(destination)
    persisted = datasets.load_from_disk(destination)
    manifest = {
        "dataset_id": _CACHE_SCRIPT.DATASET_ID,
        "dataset_revision": _CACHE_SCRIPT.DATASET_REVISION,
        "data_files": {"train": _CACHE_SCRIPT.DATA_FILE},
        "source_sha256": _CACHE_SCRIPT.SOURCE_SHA256,
        "datasets_version": datasets.__version__,
        "files": _CACHE_SCRIPT._file_records(destination),
        "splits": {
            "train": {
                "num_rows": len(persisted["train"]),
                "fingerprint": persisted["train"]._fingerprint,
                "features": persisted["train"].features.to_dict(),
            }
        },
    }
    (destination / _CACHE_SCRIPT.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    (cache_root / ".c4.lock").touch()
    return destination


@pytest.fixture(autouse=True)
def _small_expected_dataset(monkeypatch):
    """Match production row-count validation to the two-row test datasets."""

    monkeypatch.setattr(_CACHE_SCRIPT, "EXPECTED_TRAIN_ROWS", 2)


def test_verify_accepts_matching_file_checksums(tmp_path):
    """An untouched cache whose files match the manifest must verify successfully."""

    _write_cache(tmp_path, ["first", "second"])

    _CACHE_SCRIPT.verify(tmp_path)


def test_verify_rejects_replaced_arrow_file(tmp_path):
    """File checksums must detect Arrow replacement missed by saved metadata."""

    destination = _write_cache(tmp_path, ["first", "second"])
    replacement = tmp_path / "replacement"
    _dataset(["other-a", "other-b"]).save_to_disk(replacement)
    source_arrow = next(replacement.rglob("*.arrow"))
    destination_arrow = next(destination.rglob("*.arrow"))
    shutil.copyfile(source_arrow, destination_arrow)

    with pytest.raises(ValueError, match="files do not match"):
        _CACHE_SCRIPT.verify(tmp_path)


def test_force_restores_existing_cache_when_install_fails(tmp_path, monkeypatch):
    """A failed forced install must restore the previous cache and remove its backup."""

    destination = tmp_path / "c4"
    destination.mkdir()
    marker = destination / "existing"
    marker.write_text("keep", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("source", encoding="utf-8")
    monkeypatch.setattr(_CACHE_SCRIPT, "SOURCE_SHA256", hashlib.sha256(b"source").hexdigest())
    monkeypatch.setattr(_CACHE_SCRIPT, "hf_hub_download", lambda **_kwargs: str(source))
    monkeypatch.setattr(
        _CACHE_SCRIPT.datasets, "load_dataset", lambda *_args, **_kwargs: _dataset(["a", "b"])
    )

    original_rename = Path.rename

    def fail_install(path, target):
        # Let the old cache move to backup, then fail only the replacement move.
        if path.name.startswith(".c4.tmp-"):
            raise OSError("simulated install failure")
        return original_rename(path, target)

    with patch.object(Path, "rename", fail_install):
        with pytest.raises(OSError, match="simulated install failure"):
            _CACHE_SCRIPT.prepare(tmp_path, force=True)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".c4.backup-*"))
