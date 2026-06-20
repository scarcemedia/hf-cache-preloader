from __future__ import annotations

import logging
import os
import signal
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import FrameType

from .cleanup import CleanupResult, cleanup_unlisted_repositories
from .config import DEFAULT_CACHE_DIR, AppConfig, ConfigError, ModelConfig, load_config
from .downloader import download_model as default_download_model
from .health import HealthServer
from .logging_config import configure_logging
from .state import AppState

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "/config/models.yaml"
DEFAULT_LOOP_INTERVAL_SECONDS = 10.0
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_HEALTH_HOST = "0.0.0.0"
DEFAULT_HEALTH_PORT = 8080

DownloadModel = Callable[[ModelConfig], str]
CleanupRepositories = Callable[[AppConfig], CleanupResult]


@dataclass(frozen=True)
class Settings:
    config_path: str = DEFAULT_CONFIG_PATH
    loop_interval_seconds: float = DEFAULT_LOOP_INTERVAL_SECONDS
    log_level: str = DEFAULT_LOG_LEVEL
    health_host: str = DEFAULT_HEALTH_HOST
    health_port: int = DEFAULT_HEALTH_PORT

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = env if env is not None else os.environ
        return cls(
            config_path=source.get("CONFIG_PATH", DEFAULT_CONFIG_PATH),
            loop_interval_seconds=_parse_positive_float(
                source.get("LOOP_INTERVAL_SECONDS", str(DEFAULT_LOOP_INTERVAL_SECONDS)),
                "LOOP_INTERVAL_SECONDS",
            ),
            log_level=source.get("LOG_LEVEL", DEFAULT_LOG_LEVEL),
            health_host=source.get("HEALTH_HOST", DEFAULT_HEALTH_HOST),
            health_port=_parse_port(source.get("HEALTH_PORT", str(DEFAULT_HEALTH_PORT))),
        )


class PreloaderApp:
    def __init__(
        self,
        settings: Settings,
        state: AppState,
        download_model: DownloadModel = default_download_model,
        cleanup: CleanupRepositories = cleanup_unlisted_repositories,
    ) -> None:
        self.settings = settings
        self.state = state
        self._download_model = download_model
        self._cleanup = cleanup

    def sync_once(self) -> bool:
        self.state.start_sync()
        logger.info("sync loop started", extra={"config_path": self.settings.config_path})
        try:
            config = load_config(self.settings.config_path)
        except ConfigError as exc:
            logger.exception(
                "config parse failed", extra={"config_path": self.settings.config_path}
            )
            self.state.complete_sync(success=False, error=str(exc))
            return False
        except Exception as exc:
            logger.exception("unexpected sync loop failure")
            self.state.complete_sync(success=False, error=str(exc))
            return False

        try:
            self.state.config_loaded(
                config=config, loop_interval_seconds=self.settings.loop_interval_seconds
            )
            errors: list[str] = []
            for model in config.models:
                self.state.model_started(model)
                try:
                    path = self._download_model(model)
                except Exception as exc:
                    logger.exception(
                        "snapshot_download failed",
                        extra={
                            "repo_id": model.repo_id,
                            "repo_type": model.repo_type,
                            "revision": model.revision,
                        },
                    )
                    errors.append(f"{model.repo_id}: {exc}")
                    self.state.model_failed(model=model, error=str(exc))
                    continue
                self.state.model_succeeded(model=model, path=path)

            if config.defaults.remove_unlisted_models:
                try:
                    cleanup_result = self._cleanup(config)
                except Exception as exc:
                    logger.exception("unexpected sync loop failure")
                    errors.append(f"cleanup: {exc}")
                else:
                    errors.extend(cleanup_result.errors)

            if errors:
                error = "; ".join(errors)
                self.state.complete_sync(success=False, error=error)
                logger.error("sync loop failed", extra={"error_count": len(errors)})
                return False

            self.state.complete_sync(success=True, error=None)
            logger.info("sync loop completed", extra={"configured_model_count": len(config.models)})
            return True
        except Exception as exc:
            logger.exception("unexpected sync loop failure")
            self.state.complete_sync(success=False, error=str(exc))
            return False

    def run_forever(self, shutdown_event: threading.Event) -> None:
        while not shutdown_event.is_set():
            self.sync_once()
            shutdown_event.wait(self.settings.loop_interval_seconds)


def main() -> int:
    try:
        settings = Settings.from_env()
        configure_logging(settings.log_level)
    except Exception:
        logging.basicConfig(level=logging.ERROR)
        logger.exception("startup settings failed")
        return 1

    state = AppState(
        loop_interval_seconds=settings.loop_interval_seconds,
        cache_dir=DEFAULT_CACHE_DIR,
    )
    logger.info(
        "startup settings",
        extra={
            "config_path": settings.config_path,
            "loop_interval_seconds": settings.loop_interval_seconds,
            "log_level": settings.log_level,
            "health_host": settings.health_host,
            "health_port": settings.health_port,
            "cache_dir": DEFAULT_CACHE_DIR,
        },
    )

    shutdown_event = threading.Event()
    _install_signal_handlers(shutdown_event)
    health_server = HealthServer(
        host=settings.health_host,
        port=settings.health_port,
        state=state,
    )
    app = PreloaderApp(settings=settings, state=state)

    try:
        health_server.start()
        app.run_forever(shutdown_event)
        return 0
    except Exception:
        logger.exception("unexpected application failure")
        return 1
    finally:
        health_server.stop()


def _install_signal_handlers(shutdown_event: threading.Event) -> None:
    def request_shutdown(signum: int, frame: FrameType | None) -> None:
        logger.info("shutdown requested", extra={"signal": signum})
        shutdown_event.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)


def _parse_positive_float(value: str, env_name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a positive number") from exc
    if parsed <= 0:
        raise ValueError(f"{env_name} must be a positive number")
    return parsed


def _parse_port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("HEALTH_PORT must be an integer") from exc
    if parsed < 1 or parsed > 65535:
        raise ValueError("HEALTH_PORT must be between 1 and 65535")
    return parsed
