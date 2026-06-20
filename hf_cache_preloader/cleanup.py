from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig

logger = logging.getLogger(__name__)

REPO_TYPE_PREFIXES = {
    "model": "models",
    "dataset": "datasets",
    "space": "spaces",
}
KNOWN_CACHE_PREFIXES = frozenset(REPO_TYPE_PREFIXES.values())


@dataclass(frozen=True)
class CleanupResult:
    deleted: list[str]
    errors: list[str]


def cache_directory_name(repo_type: str, repo_id: str) -> str:
    try:
        prefix = REPO_TYPE_PREFIXES[repo_type]
    except KeyError as exc:
        raise ValueError(f"unsupported repo_type for cleanup: {repo_type}") from exc
    return f"{prefix}--{repo_id.replace('/', '--')}"


def cleanup_unlisted_repositories(config: AppConfig) -> CleanupResult:
    if not config.defaults.remove_unlisted_models:
        logger.debug("cleanup disabled", extra={"cache_dir": config.defaults.cache_dir})
        return CleanupResult(deleted=[], errors=[])

    logger.warning("cleanup enabled", extra={"cache_dir": config.defaults.cache_dir})
    keep_by_cache_dir = _build_keep_sets(config)
    deleted: list[str] = []
    errors: list[str] = []

    for cache_dir, keep_names in keep_by_cache_dir.items():
        logger.debug(
            "cache scan details",
            extra={"cache_dir": str(cache_dir), "listed_repo_count": len(keep_names)},
        )
        if not cache_dir.exists():
            logger.debug("skip cleanup missing cache dir", extra={"cache_dir": str(cache_dir)})
            continue
        if not cache_dir.is_dir():
            logger.warning(
                "skip cleanup non-directory cache path", extra={"cache_dir": str(cache_dir)}
            )
            continue

        for child in sorted(cache_dir.iterdir()):
            if not _is_known_complete_repo_dir(child):
                logger.debug("skip non-repo cache entry", extra={"path": str(child)})
                continue
            if child.name in keep_names:
                logger.debug("skip listed repo cache directory", extra={"path": str(child)})
                continue

            logger.warning("unlisted model scheduled for deletion", extra={"path": str(child)})
            try:
                shutil.rmtree(child)
            except Exception as exc:
                logger.exception("cleanup delete failed", extra={"path": str(child)})
                errors.append(f"{child}: {exc}")
                continue
            logger.info("cleanup deleted repo cache directory", extra={"path": str(child)})
            deleted.append(str(child))

    logger.info(
        "cleanup completed", extra={"deleted_count": len(deleted), "error_count": len(errors)}
    )
    return CleanupResult(deleted=deleted, errors=errors)


def _build_keep_sets(config: AppConfig) -> dict[Path, set[str]]:
    keep_by_cache_dir: dict[Path, set[str]] = {}
    for model in config.models:
        keep_by_cache_dir.setdefault(Path(model.cache_dir), set()).add(
            cache_directory_name(model.repo_type, model.repo_id)
        )
    if not keep_by_cache_dir:
        keep_by_cache_dir[Path(config.defaults.cache_dir)] = set()
    return keep_by_cache_dir


def _is_known_complete_repo_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not any(path.name.startswith(f"{prefix}--") for prefix in KNOWN_CACHE_PREFIXES):
        return False
    return (path / "blobs").is_dir() and (path / "snapshots").is_dir()
