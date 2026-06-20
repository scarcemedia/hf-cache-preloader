from pathlib import Path

from hf_cache_preloader.cleanup import cache_directory_name, cleanup_unlisted_repositories
from hf_cache_preloader.config import AppConfig, DefaultsConfig, ModelConfig


def make_config(cache_dir: Path, remove_unlisted_models: bool) -> AppConfig:
    model = ModelConfig(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        repo_type="model",
        revision=None,
        cache_dir=str(cache_dir),
        max_workers=8,
        allow_patterns=None,
        ignore_patterns=None,
    )
    defaults = DefaultsConfig(
        cache_dir=str(cache_dir), remove_unlisted_models=remove_unlisted_models
    )
    return AppConfig(defaults=defaults, models=[model])


def make_complete_repo(cache_dir: Path, name: str) -> Path:
    repo_dir = cache_dir / name
    (repo_dir / "blobs").mkdir(parents=True)
    (repo_dir / "snapshots").mkdir()
    return repo_dir


def test_cache_directory_name_converts_repo_id_to_hugging_face_layout() -> None:
    assert (
        cache_directory_name("model", "Qwen/Qwen2.5-VL-7B-Instruct")
        == "models--Qwen--Qwen2.5-VL-7B-Instruct"
    )
    assert cache_directory_name("dataset", "org/data") == "datasets--org--data"
    assert cache_directory_name("space", "org/demo") == "spaces--org--demo"


def test_cleanup_keeps_listed_model_repo_directories(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hub"
    listed = make_complete_repo(cache_dir, "models--Qwen--Qwen2.5-VL-7B-Instruct")

    result = cleanup_unlisted_repositories(make_config(cache_dir, remove_unlisted_models=True))

    assert listed.exists()
    assert result.deleted == []
    assert result.errors == []


def test_cleanup_deletes_unlisted_model_repo_directories_when_enabled(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hub"
    listed = make_complete_repo(cache_dir, "models--Qwen--Qwen2.5-VL-7B-Instruct")
    unlisted = make_complete_repo(cache_dir, "models--other-org--other-model")

    result = cleanup_unlisted_repositories(make_config(cache_dir, remove_unlisted_models=True))

    assert listed.exists()
    assert not unlisted.exists()
    assert result.deleted == [str(unlisted)]
    assert result.errors == []


def test_cleanup_does_nothing_when_disabled(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hub"
    unlisted = make_complete_repo(cache_dir, "models--other-org--other-model")

    result = cleanup_unlisted_repositories(make_config(cache_dir, remove_unlisted_models=False))

    assert unlisted.exists()
    assert result.deleted == []
    assert result.errors == []


def test_cleanup_does_not_delete_unknown_or_incomplete_directories(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hub"
    unknown = cache_dir / "not-a-hf-repo"
    incomplete = cache_dir / "models--other-org--incomplete"
    unknown.mkdir(parents=True)
    incomplete.mkdir()

    result = cleanup_unlisted_repositories(make_config(cache_dir, remove_unlisted_models=True))

    assert unknown.exists()
    assert incomplete.exists()
    assert result.deleted == []
    assert result.errors == []
