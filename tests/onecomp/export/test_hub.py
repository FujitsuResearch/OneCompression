"""Tests for the Hugging Face Hub upload wrapper (network mocked).

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import sys
import types

import pytest

from onecomp.export import push_to_hub


class _FakeHfApi:
    """Records calls made by ``push_to_hub`` without touching the network."""

    instances = []

    def __init__(self, token=None):
        self.token = token
        self.created = []
        self.uploaded = []
        _FakeHfApi.instances.append(self)

    def create_repo(self, repo_id, private=None, exist_ok=None, repo_type=None):
        self.created.append((repo_id, private, exist_ok, repo_type))
        return f"https://huggingface.co/{repo_id}"

    def upload_folder(self, repo_id=None, folder_path=None, commit_message=None):
        self.uploaded.append((repo_id, folder_path, commit_message))


@pytest.fixture(name="fake_hub")
def fixture_fake_hub(monkeypatch):
    """Install a fake ``huggingface_hub`` module exposing ``HfApi``."""
    _FakeHfApi.instances = []
    module = types.ModuleType("huggingface_hub")
    module.HfApi = _FakeHfApi
    module.get_token = lambda: "cached-token"
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return _FakeHfApi


def test_push_to_hub_uploads_folder(tmp_path, fake_hub):
    save_dir = tmp_path / "model"
    save_dir.mkdir()
    (save_dir / "config.json").write_text("{}")

    url = push_to_hub(str(save_dir), "user/repo", model_card="# card", token="tok")

    assert url == "https://huggingface.co/user/repo"
    api = fake_hub.instances[-1]
    assert api.token == "tok"
    assert api.created == [("user/repo", True, True, "model")]
    assert len(api.uploaded) == 1
    assert api.uploaded[0][0] == "user/repo"
    assert api.uploaded[0][1] == str(save_dir)


def test_push_to_hub_writes_model_card(tmp_path, fake_hub):
    save_dir = tmp_path / "model"
    save_dir.mkdir()

    push_to_hub(str(save_dir), "user/repo", model_card="# my card")

    assert (save_dir / "README.md").read_text(encoding="utf-8") == "# my card"


def test_push_to_hub_keeps_existing_readme(tmp_path, fake_hub):
    save_dir = tmp_path / "model"
    save_dir.mkdir()
    (save_dir / "README.md").write_text("original", encoding="utf-8")

    push_to_hub(str(save_dir), "user/repo", model_card="# new card")

    assert (save_dir / "README.md").read_text(encoding="utf-8") == "original"


def test_push_to_hub_public_repo(tmp_path, fake_hub):
    save_dir = tmp_path / "model"
    save_dir.mkdir()

    push_to_hub(str(save_dir), "user/repo", private=False)

    assert fake_hub.instances[-1].created == [("user/repo", False, True, "model")]


def test_push_to_hub_rejects_missing_dir(tmp_path, fake_hub):
    with pytest.raises(ValueError, match="not a directory"):
        push_to_hub(str(tmp_path / "missing"), "user/repo")


def test_push_to_hub_rejects_missing_token(tmp_path, fake_hub, monkeypatch):
    save_dir = tmp_path / "model"
    save_dir.mkdir()
    monkeypatch.setattr(sys.modules["huggingface_hub"], "get_token", lambda: None)

    with pytest.raises(ValueError, match="no Hugging Face token"):
        push_to_hub(str(save_dir), "user/repo")

    # An explicit token bypasses the cached-login check.
    push_to_hub(str(save_dir), "user/repo", token="tok")
    assert fake_hub.instances[-1].token == "tok"
