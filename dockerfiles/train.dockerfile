FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt update && \
        apt install --no-install-recommends -y build-essential gcc && \
        apt clean && rm -rf /var/lib/apt/lists/*

# Use a project directory inside the image
WORKDIR /app
ENV PYTHONPATH=/app/src
ENV UV_LINK_MODE=copy

# Copy project metadata and sources
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY README.md README.md
COPY src/ src/
COPY configs/ configs/
COPY .dvc/config .dvc/config
COPY .dvcignore .dvcignore
COPY data.dvc data.dvc

RUN --mount=type=cache,target=/root/.cache/uv uv sync
RUN uv run dvc pull --no-run-cache

ENTRYPOINT ["uv","run","python","-m","pneumoniaclassifier.train"]

