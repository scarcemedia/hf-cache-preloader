from __future__ import annotations

import logging
import os
from importlib import import_module
from operator import attrgetter
from pathlib import Path
from typing import Protocol, cast

from .config import ModelConfig

logger = logging.getLogger(__name__)


class SnapshotDownload(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        repo_type: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        max_workers: int = 8,
        token: bool | str | None = None,
    ) -> str: ...


_snapshot_download_object: object = attrgetter("snapshot_download")(
    import_module("huggingface_hub")
)
snapshot_download = cast(SnapshotDownload, _snapshot_download_object)


def download_model(model: ModelConfig) -> str:
    logger.info(
        "model download started",
        extra={
            "repo_id": model.repo_id,
            "repo_type": model.repo_type,
            "revision": model.revision,
            "cache_dir": model.cache_dir,
        },
    )
    token = os.getenv("HF_TOKEN")
    if model.revision is not None and token:
        path = snapshot_download(
            repo_id=model.repo_id,
            repo_type=model.repo_type,
            revision=model.revision,
            cache_dir=model.cache_dir,
            allow_patterns=model.allow_patterns,
            ignore_patterns=model.ignore_patterns,
            max_workers=model.max_workers,
            token=token,
        )
    elif model.revision is not None:
        path = snapshot_download(
            repo_id=model.repo_id,
            repo_type=model.repo_type,
            revision=model.revision,
            cache_dir=model.cache_dir,
            allow_patterns=model.allow_patterns,
            ignore_patterns=model.ignore_patterns,
            max_workers=model.max_workers,
        )
    elif token:
        path = snapshot_download(
            repo_id=model.repo_id,
            repo_type=model.repo_type,
            cache_dir=model.cache_dir,
            allow_patterns=model.allow_patterns,
            ignore_patterns=model.ignore_patterns,
            max_workers=model.max_workers,
            token=token,
        )
    else:
        path = snapshot_download(
            repo_id=model.repo_id,
            repo_type=model.repo_type,
            cache_dir=model.cache_dir,
            allow_patterns=model.allow_patterns,
            ignore_patterns=model.ignore_patterns,
            max_workers=model.max_workers,
        )
    logger.info(
        "model download completed",
        extra={
            "repo_id": model.repo_id,
            "repo_type": model.repo_type,
            "revision": model.revision,
            "path": path,
        },
    )
    return path
