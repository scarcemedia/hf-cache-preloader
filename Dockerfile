FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/models/huggingface \
    HF_HUB_CACHE=/models/huggingface/hub

COPY pyproject.toml uv.lock README.md ./
COPY hf_cache_preloader ./hf_cache_preloader

RUN uv sync --frozen --no-dev

ENTRYPOINT ["python", "-m", "hf_cache_preloader"]
