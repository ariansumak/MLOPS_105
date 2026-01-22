FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt update && \
        apt install --no-install-recommends -y build-essential gcc git && \
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
COPY .dvc/ .dvc/
COPY .dvcignore .dvcignore
COPY data.dvc data.dvc

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv uv sync

# Install DVC via uv (ensures compatibility with project environment)
RUN uv pip install dvc[gs]

# Pull data at runtime, not build time
# DVC will use configuration from .dvc/config
# GCP credentials are available via service account
ENTRYPOINT ["sh", "-c", "dvc pull && uv run python -m pneumoniaclassifier.train"]

