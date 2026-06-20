# hf-cache-preloader

`hf-cache-preloader` is a small Python 3.12 Kubernetes utility that warms a node-local Hugging Face cache. It is intended to run as a DaemonSet on k3s GPU nodes so inference workloads such as vLLM can start from a local cache instead of downloading model repositories during pod startup.

The container image is published to:

```text
ghcr.io/scarcemedia/hf-cache-preloader
```

## How It Works

The process runs forever. Every sync loop it reads `models.yaml`, resolves defaults and model-level overrides, downloads each configured repository with `huggingface_hub.snapshot_download()`, optionally removes unlisted repository cache directories, then waits for the next loop interval.

This provides the same practical cache-warming behavior as:

```sh
hf download <model-name>
```

It does not shell out to the `hf` CLI. It calls `snapshot_download()` directly and uses the standard Hugging Face cache layout. It does not use `local_dir`.

Default paths:

```text
Host cache path:      /var/lib/huggingface
Container HF_HOME:    /models/huggingface
Container hub cache:  /models/huggingface/hub
```

## Configuration

The app reads `/config/models.yaml` by default. Override with `CONFIG_PATH`.

Example:

```yaml
defaults:
  cache_dir: /models/huggingface/hub
  repo_type: model
  max_workers: 8
  remove_unlisted_models: false
  allow_patterns:
    - "*.json"
    - "*.safetensors"
    - "*.model"
    - "*.txt"
    - "*.py"
    - "*.tiktoken"
    - "tokenizer*"
    - "generation_config.json"
    - "preprocessor_config.json"
    - "processor_config.json"
    - "chat_template*"
  ignore_patterns:
    - "*.onnx"
    - "*.msgpack"
    - "*.h5"
    - "*.ot"
    - "*.pb"
models:
  - repo_id: Qwen/Qwen2.5-VL-7B-Instruct
  - repo_id: OpenGVLab/InternVL3_5-8B
    max_workers: 4
  - repo_id: some-org/some-model
    revision: main
    allow_patterns:
      - "*.json"
      - "*.safetensors"
      - "*.txt"
```

Rules:

- `models` is required and must be a list.
- Each model requires `repo_id`.
- `revision` is optional. If unset, it is not passed to `snapshot_download()`.
- `repo_type` defaults to `model`; `model`, `dataset`, and `space` are supported.
- `cache_dir` defaults to `/models/huggingface/hub`.
- `max_workers` defaults to `8`.
- `allow_patterns` and `ignore_patterns` default to `null` unless set in `defaults`.
- Model-level values override defaults. Set a model pattern field to `null` to pass `None`.
- `remove_unlisted_models` is only read from `defaults`.

## Environment

Runtime settings:

```text
CONFIG_PATH=/config/models.yaml
LOOP_INTERVAL_SECONDS=10
LOG_LEVEL=INFO
HEALTH_HOST=0.0.0.0
HEALTH_PORT=8080
HF_HOME=/models/huggingface
HF_HUB_CACHE=/models/huggingface/hub
```

Proxy behavior is controlled only by environment variables. If `HF_ENDPOINT` is set, `huggingface_hub` uses it. If it is not set, the app does not set one. The app also does not hard-code `HTTP_PROXY` or `HTTPS_PROXY`.

For a Hugging Face compatible proxy:

```yaml
- name: HF_ENDPOINT
  value: https://hf-proxy.example.internal
```

For private models, create an optional token secret:

```sh
kubectl -n model-serving create secret generic hf-token --from-literal=token='<hf-token>'
```

The DaemonSet references this secret with `optional: true`, so the secret is not required for public models.

## Cleanup

Cleanup is disabled by default:

```yaml
defaults:
  remove_unlisted_models: false
```

When enabled, the app deletes only complete Hugging Face repository cache directories under the configured cache directory that are not listed in `models`.

Examples of directories it understands:

```text
models--Qwen--Qwen2.5-VL-7B-Instruct
datasets--org--repo
spaces--org--repo
```

It does not delete blobs directly, does not delete snapshots for listed repositories, and does not delete unknown or incomplete directories. If a listed repository has old revisions in its cache directory, the whole listed repo cache directory is kept.

## Kubernetes Deployment

Create the namespace if needed:

```sh
kubectl create namespace model-serving --dry-run=client -o yaml | kubectl apply -f -
```

Apply the manifests:

```sh
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/daemonset.yaml
```

The DaemonSet targets only GPU nodes with this selector:

```yaml
nvidia.com/gpu.present: "true"
```

Label a GPU node if needed:

```sh
kubectl label node <node-name> nvidia.com/gpu.present=true
```

The DaemonSet mounts `/var/lib/huggingface` from the host read/write at `/models/huggingface` in the container. It mounts the ConfigMap at `/config` as a directory, not with `subPath`, so ConfigMap updates are eventually reflected in the running pod. Kubernetes ConfigMap volume updates are eventually consistent; the app reloads the file every loop.

Do not mount the ConfigMap with `subPath` if you expect live updates. `subPath` mounts do not receive ConfigMap updates.

The container currently runs as root because v1 needs reliable write access to the hostPath cache directory. A future hardening pass can use a fixed UID/GID and host directory ownership management.

## Consumer Mount Example

Consumers such as vLLM should mount the same host cache read-only:

```yaml
volumeMounts:
  - name: hf-cache
    mountPath: /models/huggingface
    readOnly: true
env:
  - name: HF_HOME
    value: /models/huggingface
  - name: HF_HUB_CACHE
    value: /models/huggingface/hub
volumes:
  - name: hf-cache
    hostPath:
      path: /var/lib/huggingface
      type: Directory
```

## Health And Status

The health server uses Python stdlib `ThreadingHTTPServer`.

Endpoints:

- `GET /healthz` returns `200` while the process is alive.
- `GET /readyz` returns `200` only after a successful config load and sync loop; otherwise `503`.
- `GET /status` returns JSON with sync timestamps, configured models, cleanup status, loop interval, cache directory, and the last error.

The `/status` response does not include `HF_TOKEN` or proxy credentials.

Port-forward a pod to inspect it:

```sh
kubectl -n model-serving get pods -l app.kubernetes.io/name=hf-cache-preloader
kubectl -n model-serving port-forward pod/<pod-name> 8080:8080
curl http://127.0.0.1:8080/status
```

## Local Development

Install and test with uv:

```sh
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
```

Run locally:

```sh
CONFIG_PATH=./models.yaml uv run python -m hf_cache_preloader
```

## Releases

GitHub Actions builds the Docker image on pull requests and publishes only the Docker image to GitHub Container Registry on `main` and release tags.

Release tags use date-based versions with a leading `v`:

```text
v2026.06.01
```

The Docker tag matches the Git tag:

```text
ghcr.io/scarcemedia/hf-cache-preloader:v2026.06.01
```
