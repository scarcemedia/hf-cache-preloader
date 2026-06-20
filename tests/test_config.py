from pathlib import Path

import pytest

from hf_cache_preloader.config import DEFAULT_CACHE_DIR, ConfigError, load_config


def write_config(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config_applies_defaults(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "models.yaml",
        """
models:
  - repo_id: Qwen/Qwen2.5-VL-7B-Instruct
""",
    )

    config = load_config(config_path)

    assert config.defaults.cache_dir == DEFAULT_CACHE_DIR
    assert config.defaults.repo_type == "model"
    assert config.defaults.max_workers == 8
    assert config.defaults.remove_unlisted_models is False
    assert len(config.models) == 1
    model = config.models[0]
    assert model.repo_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert model.repo_type == "model"
    assert model.revision is None
    assert model.cache_dir == DEFAULT_CACHE_DIR
    assert model.max_workers == 8
    assert model.allow_patterns is None
    assert model.ignore_patterns is None


def test_model_settings_override_defaults(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "models.yaml",
        """
defaults:
  cache_dir: /cache/default
  repo_type: model
  revision: stable
  max_workers: 8
  allow_patterns:
    - "*.json"
  ignore_patterns:
    - "*.onnx"
models:
  - repo_id: org/repo
    cache_dir: /cache/override
    repo_type: dataset
    revision: main
    max_workers: 4
    allow_patterns:
      - "*.txt"
    ignore_patterns: null
""",
    )

    config = load_config(config_path)

    model = config.models[0]
    assert model.cache_dir == "/cache/override"
    assert model.repo_type == "dataset"
    assert model.revision == "main"
    assert model.max_workers == 4
    assert model.allow_patterns == ["*.txt"]
    assert model.ignore_patterns is None


def test_default_patterns_apply_unless_model_overrides_them(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "models.yaml",
        """
defaults:
  allow_patterns:
    - "*.json"
  ignore_patterns:
    - "*.onnx"
models:
  - repo_id: org/uses-defaults
  - repo_id: org/overrides
    allow_patterns: null
    ignore_patterns:
      - "*.bin"
""",
    )

    config = load_config(config_path)

    assert config.models[0].allow_patterns == ["*.json"]
    assert config.models[0].ignore_patterns == ["*.onnx"]
    assert config.models[1].allow_patterns is None
    assert config.models[1].ignore_patterns == ["*.bin"]


def test_invalid_config_without_models_fails(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "models.yaml",
        """
defaults:
  repo_type: model
""",
    )

    with pytest.raises(ConfigError, match="models"):
        load_config(config_path)


def test_invalid_model_without_repo_id_fails(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "models.yaml",
        """
models:
  - revision: main
""",
    )

    with pytest.raises(ConfigError, match="repo_id"):
        load_config(config_path)
