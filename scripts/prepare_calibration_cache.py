#!/usr/bin/env python3
"""Prepare and verify the fixed C4 calibration cache used by cluster CI.

Copyright 2025-2026 Fujitsu Ltd.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import datasets
from huggingface_hub import hf_hub_download

DATASET_ID = "allenai/c4"
DATASET_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
DATA_FILE = "en/c4-train.00001-of-01024.json.gz"
SOURCE_SHA256 = "b945059cd1a343cabe311881b7840a6f0363f570e745a0eff0e687e266f6b55d"
EXPECTED_TRAIN_ROWS = 356318
MANIFEST_NAME = "onecomp-calibration-manifest.json"


def _sha256(path):
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(cache_root):
    """Return the save-to-disk directory consumed by the C4 loader."""

    return Path(cache_root).expanduser().resolve() / "c4"


def _file_records(root):
    """Describe every persisted cache file so content changes are detectable."""

    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == MANIFEST_NAME:
            # The manifest cannot include its own digest without becoming self-referential.
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


@contextmanager
def _cache_lock(cache_root, *, exclusive):
    """Coordinate cache readers with the short final installation step.

    Verification and CI use a shared lock for the entire period in which cache
    files may be read. Regeneration uses an exclusive lock only while replacing
    the verified temporary cache, so building and hashing it does not block CI.
    """

    cache_root = Path(cache_root).expanduser().resolve()
    lock_path = cache_root / ".c4.lock"
    if exclusive:
        cache_root.mkdir(parents=True, exist_ok=True)
        lock_path.touch(exist_ok=True)
        mode = "r+"
        operation = fcntl.LOCK_EX
    else:
        mode = "r"
        operation = fcntl.LOCK_SH

    try:
        lock_file = lock_path.open(mode, encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Fixed calibration cache lock is missing: {lock_path}. "
            "Regenerate the cache with this script."
        ) from exc
    with lock_file:
        fcntl.flock(lock_file, operation)
        yield


def prepare(cache_root, *, local_files_only=False, force=False):
    """Build, verify, and transactionally install the pinned C4 cache."""

    destination = _cache_path(cache_root)
    if destination.exists() and not force:
        raise FileExistsError(
            f"Calibration cache already exists: {destination}. "
            "Use --force only for an intentional regeneration."
        )

    source = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=DATA_FILE,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_files_only=local_files_only,
        )
    )
    source_sha256 = _sha256(source)
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"Unexpected SHA-256 for {DATA_FILE}: {source_sha256}; " f"expected {SOURCE_SHA256}"
        )

    dataset = datasets.load_dataset("json", data_files={"train": str(source)})
    if len(dataset["train"]) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"Unexpected C4 train row count: {len(dataset['train'])}; "
            f"expected {EXPECTED_TRAIN_ROWS}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)

    try:
        dataset.save_to_disk(temporary)
        # Fingerprints can change during save_to_disk(), so record the value that
        # consumers will observe after loading the persisted cache.
        persisted_dataset = datasets.load_from_disk(temporary)
        manifest = {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "data_files": {"train": DATA_FILE},
            "source_sha256": source_sha256,
            "datasets_version": datasets.__version__,
            "files": _file_records(temporary),
            "splits": {
                "train": {
                    "num_rows": len(persisted_dataset["train"]),
                    "fingerprint": persisted_dataset["train"]._fingerprint,
                    "features": persisted_dataset["train"].features.to_dict(),
                }
            },
        }
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Reject an incomplete or internally inconsistent cache before it can
        # become visible at the shared destination.
        _verify_cache(temporary)

        backup = destination.with_name(f".{destination.name}.backup-{os.getpid()}")
        with _cache_lock(cache_root, exclusive=True):
            if destination.exists() and not force:
                raise FileExistsError(
                    f"Calibration cache already exists: {destination}. "
                    "Use --force only for an intentional regeneration."
                )
            if backup.exists():
                raise FileExistsError(f"Calibration cache backup already exists: {backup}")

            moved_existing = False
            try:
                # Keep the previous cache available for rollback until the
                # already-verified replacement has been installed.
                if destination.exists():
                    destination.rename(backup)
                    moved_existing = True
                temporary.rename(destination)
            except Exception:
                if moved_existing and not destination.exists():
                    backup.rename(destination)
                raise

            if moved_existing:
                try:
                    shutil.rmtree(backup)
                except OSError as exc:
                    print(
                        f"WARNING: failed to remove calibration cache backup: {exc}",
                        file=sys.stderr,
                    )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Prepared fixed C4 calibration cache: {destination}")


def _verify_cache(destination):
    """Validate the pinned source identity and every persisted cache artifact."""

    manifest_path = destination / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Fixed calibration cache manifest is missing: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "data_files": {"train": DATA_FILE},
        "source_sha256": SOURCE_SHA256,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"Invalid fixed calibration cache manifest field {key!r}: "
                f"{manifest.get(key)!r}; expected {expected_value!r}"
            )

    files = _file_records(destination)
    if manifest.get("files") != files:
        raise ValueError("Fixed calibration cache files do not match its manifest")

    dataset = datasets.load_from_disk(destination)
    if "train" not in dataset:
        raise ValueError("Fixed calibration cache has no train split")
    train_manifest = manifest.get("splits", {}).get("train", {})
    if len(dataset["train"]) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"Invalid fixed calibration cache row count: {len(dataset['train'])}; "
            f"expected {EXPECTED_TRAIN_ROWS}"
        )
    if train_manifest.get("num_rows") != EXPECTED_TRAIN_ROWS:
        raise ValueError("Fixed calibration cache manifest has an invalid row count")
    if dataset["train"]._fingerprint != train_manifest.get("fingerprint"):
        raise ValueError("Fixed calibration cache fingerprint does not match its manifest")
    if dataset["train"].features.to_dict() != train_manifest.get("features"):
        raise ValueError("Fixed calibration cache schema does not match its manifest")

    print(
        "Verified fixed C4 calibration cache: "
        f"{destination} ({len(dataset['train'])} train rows)"
    )


def verify(cache_root):
    """Verify a cache while preventing concurrent regeneration from replacing it."""

    with _cache_lock(cache_root, exclusive=False):
        _verify_cache(_cache_path(cache_root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("ONECOMP_CALIB_CACHE"),
        help="Cache root containing the c4 directory (default: ONECOMP_CALIB_CACHE)",
    )
    parser.add_argument("--verify", action="store_true", help="Verify without writing")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use an already downloaded Hugging Face source shard",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing cache intentionally",
    )
    args = parser.parse_args()
    if not args.cache_root:
        parser.error("--cache-root or ONECOMP_CALIB_CACHE is required")

    if args.verify:
        verify(args.cache_root)
    else:
        prepare(
            args.cache_root,
            local_files_only=args.local_files_only,
            force=args.force,
        )


if __name__ == "__main__":
    main()
