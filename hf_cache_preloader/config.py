from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

logger = logging.getLogger(__name__)

APPLICATION_NAME = "hf-cache-preloader"
DEFAULT_CACHE_DIR = "/models/huggingface/hub"
DEFAULT_REPO_TYPE = "model"
DEFAULT_MAX_WORKERS = 8
SUPPORTED_REPO_TYPES = frozenset({"model", "dataset", "space"})

_DEFAULT_FIELDS = frozenset(
    {
        "cache_dir",
        "repo_type",
        "revision",
        "max_workers",
        "remove_unlisted_models",
        "allow_patterns",
        "ignore_patterns",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "repo_id",
        "cache_dir",
        "repo_type",
        "revision",
        "max_workers",
        "allow_patterns",
        "ignore_patterns",
    }
)
_ROOT_FIELDS = frozenset({"defaults", "models"})


class ConfigError(ValueError):
    """Raised when models.yaml cannot be loaded or validated."""


@dataclass(frozen=True)
class DefaultsConfig:
    cache_dir: str = DEFAULT_CACHE_DIR
    repo_type: str = DEFAULT_REPO_TYPE
    revision: str | None = None
    max_workers: int = DEFAULT_MAX_WORKERS
    remove_unlisted_models: bool = False
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None


@dataclass(frozen=True)
class ModelConfig:
    repo_id: str
    repo_type: str
    revision: str | None
    cache_dir: str
    max_workers: int
    allow_patterns: list[str] | None
    ignore_patterns: list[str] | None


@dataclass(frozen=True)
class AppConfig:
    defaults: DefaultsConfig
    models: list[ModelConfig]


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        logger.warning("config file missing", extra={"config_path": str(path)})
        raise ConfigError(f"config file missing: {path}")

    try:
        raw_data: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config parse failed: {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"config read failed: {path}: {exc}") from exc

    if raw_data is None:
        raw_data = {}
    root = _as_mapping(raw_data, "root")
    logger.debug(
        "parsed raw config",
        extra={"config_path": str(path), "root_keys": sorted(root.keys())},
    )
    _warn_unknown_fields(root, _ROOT_FIELDS, "root")

    defaults = _parse_defaults(root.get("defaults", {}))
    models = _parse_models(root, defaults)
    return AppConfig(defaults=defaults, models=models)


def _parse_defaults(value: object) -> DefaultsConfig:
    if value is None:
        value = {}
    raw_defaults = _as_mapping(value, "defaults")
    _warn_unknown_fields(raw_defaults, _DEFAULT_FIELDS, "defaults")

    cache_dir = _optional_string(raw_defaults, "cache_dir", "defaults") or DEFAULT_CACHE_DIR
    repo_type = _optional_string(raw_defaults, "repo_type", "defaults") or DEFAULT_REPO_TYPE
    _validate_repo_type(repo_type, "defaults.repo_type")
    revision = _optional_string(raw_defaults, "revision", "defaults")
    max_workers = (
        _optional_positive_int(raw_defaults, "max_workers", "defaults") or DEFAULT_MAX_WORKERS
    )
    remove_unlisted_models = _optional_bool(raw_defaults, "remove_unlisted_models", "defaults")
    if remove_unlisted_models is None:
        remove_unlisted_models = False

    defaults = DefaultsConfig(
        cache_dir=cache_dir,
        repo_type=repo_type,
        revision=revision,
        max_workers=max_workers,
        remove_unlisted_models=remove_unlisted_models,
        allow_patterns=_optional_patterns(raw_defaults, "allow_patterns", "defaults"),
        ignore_patterns=_optional_patterns(raw_defaults, "ignore_patterns", "defaults"),
    )
    if defaults.remove_unlisted_models:
        logger.warning("cleanup enabled", extra={"cache_dir": defaults.cache_dir})
    return defaults


def _parse_models(root: dict[str, object], defaults: DefaultsConfig) -> list[ModelConfig]:
    if "models" not in root:
        raise ConfigError("models is required and must be a list")
    raw_models = _as_list(root["models"], "models")

    models: list[ModelConfig] = []
    for index, raw_model in enumerate(raw_models):
        context = f"models[{index}]"
        model_data = _as_mapping(raw_model, context)
        _warn_unknown_fields(model_data, _MODEL_FIELDS, context)
        repo_id = _required_string(model_data, "repo_id", context)
        cache_dir = _optional_string(model_data, "cache_dir", context) or defaults.cache_dir
        repo_type = _optional_string(model_data, "repo_type", context) or defaults.repo_type
        _validate_repo_type(repo_type, f"{context}.repo_type")
        revision = (
            _optional_string(model_data, "revision", context)
            if "revision" in model_data
            else defaults.revision
        )
        max_workers = (
            _optional_positive_int(model_data, "max_workers", context) or defaults.max_workers
        )
        allow_patterns = (
            _optional_patterns(model_data, "allow_patterns", context)
            if "allow_patterns" in model_data
            else _copy_patterns(defaults.allow_patterns)
        )
        ignore_patterns = (
            _optional_patterns(model_data, "ignore_patterns", context)
            if "ignore_patterns" in model_data
            else _copy_patterns(defaults.ignore_patterns)
        )
        model = ModelConfig(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            max_workers=max_workers,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        logger.debug(
            "effective per-model config",
            extra={
                "repo_id": model.repo_id,
                "repo_type": model.repo_type,
                "revision": model.revision,
                "cache_dir": model.cache_dir,
                "max_workers": model.max_workers,
                "has_allow_patterns": model.allow_patterns is not None,
                "has_ignore_patterns": model.ignore_patterns is not None,
            },
        )
        models.append(model)
    return models


def _as_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    raw_mapping = cast(dict[object, object], value)
    for key, item in raw_mapping.items():
        if not isinstance(key, str):
            raise ConfigError(f"{context} keys must be strings")
        result[key] = item
    return result


def _as_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ConfigError(f"{context} is required and must be a list")
    return list(cast(list[object], value))


def _warn_unknown_fields(
    mapping: dict[str, object], allowed_fields: frozenset[str], context: str
) -> None:
    unknown = sorted(set(mapping) - allowed_fields)
    if unknown:
        logger.warning("unknown YAML fields", extra={"context": context, "fields": unknown})


def _required_string(mapping: dict[str, object], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context}.{key} is required and must be a non-empty string")
    return value


def _optional_string(mapping: dict[str, object], key: str, context: str) -> str | None:
    if key not in mapping or mapping[key] is None:
        return None
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context}.{key} must be a non-empty string when set")
    return value


def _optional_positive_int(mapping: dict[str, object], key: str, context: str) -> int | None:
    if key not in mapping or mapping[key] is None:
        return None
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{context}.{key} must be a positive integer")
    return value


def _optional_bool(mapping: dict[str, object], key: str, context: str) -> bool | None:
    if key not in mapping or mapping[key] is None:
        return None
    value = mapping[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{context}.{key} must be a boolean")
    return value


def _optional_patterns(mapping: dict[str, object], key: str, context: str) -> list[str] | None:
    if key not in mapping or mapping[key] is None:
        return None
    value = mapping[key]
    if not isinstance(value, list):
        raise ConfigError(f"{context}.{key} must be a list of strings or null")
    patterns: list[str] = []
    raw_patterns = cast(list[object], value)
    for index, item in enumerate(raw_patterns):
        if not isinstance(item, str) or not item:
            raise ConfigError(f"{context}.{key}[{index}] must be a non-empty string")
        patterns.append(item)
    return patterns


def _copy_patterns(patterns: list[str] | None) -> list[str] | None:
    if patterns is None:
        return None
    return list(patterns)


def _validate_repo_type(repo_type: str, context: str) -> None:
    if repo_type not in SUPPORTED_REPO_TYPES:
        raise ConfigError(f"{context} must be one of: {', '.join(sorted(SUPPORTED_REPO_TYPES))}")
