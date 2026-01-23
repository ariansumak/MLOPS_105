FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt update && \
        apt install --no-install-recommends -y build-essential gcc git && \
        apt clean && rm -rf /var/lib/apt/lists/*

# Use a project directory inside the image
WORKDIR /app
ENV PYTHONPATH=/app/src
ENV UV_LINK_MODE=copy
ENV DVC_NO_SCM=1

# Copy project metadata and sources
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY README.md README.md
COPY src/ src/
COPY configs/ configs/
COPY frontend_django/ frontend_django/

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv uv sync

RUN uv run dvc init --no-scm
COPY .dvc/config .dvc/config
COPY data.dvc data.dvc
RUN uv run dvc config core.no_scm true

# Pull data at runtime, not build time
# DVC will use configuration from .dvc/config
# GCP credentials are available via service account
ENTRYPOINT ["sh", "-c", "uv run dvc pull && ls -l /app && ls -l /app/data/ && uv run python -m pneumoniaclassifier.train"]


