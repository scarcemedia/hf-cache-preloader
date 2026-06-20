import json
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from hf_cache_preloader.config import AppConfig, DefaultsConfig, ModelConfig
from hf_cache_preloader.health import HealthServer
from hf_cache_preloader.state import AppState


def test_health_server_exposes_health_ready_and_status() -> None:
    state = AppState(loop_interval_seconds=10, cache_dir="/cache")
    server = HealthServer(host="127.0.0.1", port=0, state=state)
    server.start()
    try:
        base_url = f"http://127.0.0.1:{server.port}"

        with urlopen(f"{base_url}/healthz", timeout=2) as response:
            assert response.status == 200

        with pytest.raises(HTTPError) as error:
            urlopen(f"{base_url}/readyz", timeout=2)
        assert error.value.code == 503

        model = ModelConfig(
            repo_id="org/repo",
            repo_type="model",
            revision=None,
            cache_dir="/cache",
            max_workers=8,
            allow_patterns=None,
            ignore_patterns=None,
        )
        config = AppConfig(
            defaults=DefaultsConfig(cache_dir="/cache", remove_unlisted_models=False),
            models=[model],
        )
        state.start_sync()
        state.config_loaded(config=config, loop_interval_seconds=10)
        state.model_succeeded(model=model, path="/cache/models--org--repo/snapshots/abc")
        state.complete_sync(success=True, error=None)

        with urlopen(f"{base_url}/readyz", timeout=2) as response:
            assert response.status == 200

        with urlopen(f"{base_url}/status", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["application"] == "hf-cache-preloader"
        assert "HF_TOKEN" not in json.dumps(payload)
    finally:
        server.stop()
