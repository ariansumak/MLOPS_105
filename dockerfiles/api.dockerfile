FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/app/src
ENV UV_LINK_MODE=copy
ENV DVC_NO_SCM=1

COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY configs/ configs/
COPY uv.lock uv.lock

RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev

EXPOSE $PORT

CMD ["sh", "-c", "uv run uvicorn pneumoniaclassifier.api:app --host 0.0.0.0 --port ${PORT}"]
