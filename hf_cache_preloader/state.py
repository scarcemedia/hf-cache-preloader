from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

from . import __version__
from .config import APPLICATION_NAME, AppConfig, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelStatus:
    repo_id: str
    repo_type: str
    revision: str | None
    last_status: str
    last_path: str | None
    last_error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
            "last_status": self.last_status,
            "last_path": self.last_path,
            "last_error": self.last_error,
        }


class AppState:
    def __init__(
        self,
        loop_interval_seconds: float,
        cache_dir: str,
        application_name: str = APPLICATION_NAME,
        application_version: str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._application_name = application_name
        self._version = (
            application_version if application_version is not None else _application_version()
        )
        self._started_at = _now_iso()
        self._last_sync_started_at: str | None = None
        self._last_sync_completed_at: str | None = None
        self._last_success_at: str | None = None
        self._last_error_at: str | None = None
        self._last_error: str | None = None
        self._configured_model_count = 0
        self._models: dict[str, ModelStatus] = {}
        self._cleanup_enabled = False
        self._loop_interval_seconds = loop_interval_seconds
        self._cache_dir = cache_dir
        self._config_loaded_once = False
        self._last_loop_success = False

    def start_sync(self) -> None:
        with self._lock:
            self._last_sync_started_at = _now_iso()

    def config_loaded(self, config: AppConfig, loop_interval_seconds: float) -> None:
        with self._lock:
            self._config_loaded_once = True
            self._configured_model_count = len(config.models)
            self._cleanup_enabled = config.defaults.remove_unlisted_models
            self._loop_interval_seconds = loop_interval_seconds
            self._cache_dir = config.defaults.cache_dir
            current = self._models
            next_models: dict[str, ModelStatus] = {}
            for model in config.models:
                key = _model_key(model)
                existing = current.get(key)
                next_models[key] = existing or ModelStatus(
                    repo_id=model.repo_id,
                    repo_type=model.repo_type,
                    revision=model.revision,
                    last_status="pending",
                    last_path=None,
                    last_error=None,
                )
            self._models = next_models

    def model_started(self, model: ModelConfig) -> None:
        with self._lock:
            status = self._models.get(_model_key(model))
            if status is None:
                status = ModelStatus(
                    repo_id=model.repo_id,
                    repo_type=model.repo_type,
                    revision=model.revision,
                    last_status="downloading",
                    last_path=None,
                    last_error=None,
                )
                self._models[_model_key(model)] = status
            status.last_status = "downloading"
            status.last_error = None

    def model_succeeded(self, model: ModelConfig, path: str) -> None:
        with self._lock:
            status = self._ensure_model_status(model)
            status.last_status = "downloaded"
            status.last_path = path
            status.last_error = None

    def model_failed(self, model: ModelConfig, error: str) -> None:
        with self._lock:
            status = self._ensure_model_status(model)
            status.last_status = "failed"
            status.last_error = error

    def complete_sync(self, success: bool, error: str | None) -> None:
        with self._lock:
            self._last_sync_completed_at = _now_iso()
            self._last_loop_success = success
            if success:
                self._last_success_at = self._last_sync_completed_at
                self._last_error = None
            else:
                self._last_error_at = self._last_sync_completed_at
                self._last_error = error

    def is_ready(self) -> bool:
        with self._lock:
            return self._config_loaded_once and self._last_loop_success

    def to_status(self) -> dict[str, object]:
        with self._lock:
            status: dict[str, object] = {
                "application": self._application_name,
                "version": self._version,
                "started_at": self._started_at,
                "last_sync_started_at": self._last_sync_started_at,
                "last_sync_completed_at": self._last_sync_completed_at,
                "last_success_at": self._last_success_at,
                "last_error_at": self._last_error_at,
                "last_error": self._last_error,
                "configured_model_count": self._configured_model_count,
                "models": [model.to_dict() for model in self._models.values()],
                "cleanup_enabled": self._cleanup_enabled,
                "loop_interval_seconds": self._loop_interval_seconds,
                "cache_dir": self._cache_dir,
            }
            return status

    def _ensure_model_status(self, model: ModelConfig) -> ModelStatus:
        key = _model_key(model)
        status = self._models.get(key)
        if status is None:
            status = ModelStatus(
                repo_id=model.repo_id,
                repo_type=model.repo_type,
                revision=model.revision,
                last_status="pending",
                last_path=None,
                last_error=None,
            )
            self._models[key] = status
        return status


def _model_key(model: ModelConfig) -> str:
    return f"{model.cache_dir}\0{model.repo_type}\0{model.repo_id}\0{model.revision or ''}"


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _application_version() -> str | None:
    try:
        return version("hf-cache-preloader")
    except PackageNotFoundError:
        return __version__
