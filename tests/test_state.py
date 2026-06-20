from hf_cache_preloader.config import AppConfig, DefaultsConfig, ModelConfig
from hf_cache_preloader.state import AppState


def make_model() -> ModelConfig:
    return ModelConfig(
        repo_id="org/repo",
        repo_type="model",
        revision=None,
        cache_dir="/cache",
        max_workers=8,
        allow_patterns=None,
        ignore_patterns=None,
    )


def test_state_readiness_tracks_successful_and_failed_syncs() -> None:
    model = make_model()
    config = AppConfig(
        defaults=DefaultsConfig(cache_dir="/cache", remove_unlisted_models=False),
        models=[model],
    )
    state = AppState(loop_interval_seconds=10, cache_dir="/cache")

    assert state.is_ready() is False

    state.start_sync()
    state.config_loaded(config=config, loop_interval_seconds=10)
    state.model_succeeded(model=model, path="/cache/models--org--repo/snapshots/abc")
    state.complete_sync(success=True, error=None)

    assert state.is_ready() is True
    status = state.to_status()
    assert status["application"] == "hf-cache-preloader"
    assert status["configured_model_count"] == 1
    assert status["cleanup_enabled"] is False
    assert status["loop_interval_seconds"] == 10
    assert status["cache_dir"] == "/cache"
    assert status["models"] == [
        {
            "repo_id": "org/repo",
            "repo_type": "model",
            "revision": None,
            "last_status": "downloaded",
            "last_path": "/cache/models--org--repo/snapshots/abc",
            "last_error": None,
        }
    ]

    state.start_sync()
    state.complete_sync(success=False, error="boom")

    assert state.is_ready() is False
    assert state.to_status()["last_error"] == "boom"
