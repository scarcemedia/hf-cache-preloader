import os

import pytest

from hf_cache_preloader.config import ModelConfig
from hf_cache_preloader.downloader import download_model


def test_download_model_omits_revision_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return "/cache/snapshots/abc"

    monkeypatch.setattr("hf_cache_preloader.downloader.snapshot_download", fake_snapshot_download)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    path = download_model(
        ModelConfig(
            repo_id="org/repo",
            repo_type="model",
            revision=None,
            cache_dir="/cache",
            max_workers=8,
            allow_patterns=None,
            ignore_patterns=None,
        )
    )

    assert path == "/cache/snapshots/abc"
    assert captured["repo_id"] == "org/repo"
    assert captured["repo_type"] == "model"
    assert captured["cache_dir"] == "/cache"
    assert captured["max_workers"] == 8
    assert captured["allow_patterns"] is None
    assert captured["ignore_patterns"] is None
    assert "revision" not in captured
    assert "token" not in captured


def test_download_model_passes_revision_and_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return "/cache/snapshots/def"

    monkeypatch.setattr("hf_cache_preloader.downloader.snapshot_download", fake_snapshot_download)
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    download_model(
        ModelConfig(
            repo_id="org/repo",
            repo_type="model",
            revision="main",
            cache_dir="/cache",
            max_workers=4,
            allow_patterns=["*.json"],
            ignore_patterns=["*.onnx"],
        )
    )

    assert captured["revision"] == "main"
    assert captured["token"] == os.environ["HF_TOKEN"]
