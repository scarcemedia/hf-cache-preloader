from pathlib import Path
from typing import cast

from hf_cache_preloader.app import PreloaderApp, Settings
from hf_cache_preloader.cleanup import CleanupResult
from hf_cache_preloader.config import AppConfig, ModelConfig
from hf_cache_preloader.state import AppState


def test_sync_once_downloads_models_and_marks_state_ready(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
models:
  - repo_id: org/repo
""",
        encoding="utf-8",
    )
    downloaded: list[str] = []

    def fake_download_model(model: ModelConfig) -> str:
        downloaded.append(model.repo_id)
        return "/cache/models--org--repo/snapshots/abc"

    def fake_cleanup(config: AppConfig) -> CleanupResult:
        return CleanupResult(deleted=[], errors=[])

    state = AppState(loop_interval_seconds=10, cache_dir="/models/huggingface/hub")
    app = PreloaderApp(
        settings=Settings(config_path=str(config_path)),
        state=state,
        download_model=fake_download_model,
        cleanup=fake_cleanup,
    )

    assert app.sync_once() is True
    assert downloaded == ["org/repo"]
    assert state.is_ready() is True


def test_sync_once_continues_after_model_failure_and_marks_state_not_ready(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
models:
  - repo_id: org/fails
  - repo_id: org/succeeds
""",
        encoding="utf-8",
    )
    attempted: list[str] = []

    def fake_download_model(model: ModelConfig) -> str:
        attempted.append(model.repo_id)
        if model.repo_id == "org/fails":
            raise RuntimeError("download failed")
        return "/cache/models--org--succeeds/snapshots/abc"

    def fake_cleanup(config: AppConfig) -> CleanupResult:
        return CleanupResult(deleted=[], errors=[])

    state = AppState(loop_interval_seconds=10, cache_dir="/models/huggingface/hub")
    app = PreloaderApp(
        settings=Settings(config_path=str(config_path)),
        state=state,
        download_model=fake_download_model,
        cleanup=fake_cleanup,
    )

    assert app.sync_once() is False
    assert attempted == ["org/fails", "org/succeeds"]
    assert state.is_ready() is False
    models = cast(list[dict[str, object]], state.to_status()["models"])
    statuses = {model["repo_id"]: model["last_status"] for model in models}
    assert statuses == {"org/fails": "failed", "org/succeeds": "downloaded"}
