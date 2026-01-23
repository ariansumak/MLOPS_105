FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt update && \
        apt install --no-install-recommends -y build-essential gcc git && \
        apt clean && rm -rf /var/lib/apt/lists/*

# Use a project directory inside the image
WORKDIR /app
ENV PYTHONPATH=/app/src
ENV UV_LINK_MODE=copy
ENV DVC_NO_SCM=1

# Copy project files
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY configs/ configs/
COPY uv.lock uv.lock
COPY frontend_django/ frontend_django/
# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv uv sync

EXPOSE $PORT


# Default: FastAPI
CMD ["sh", "-c", "uv run uvicorn pneumoniaclassifier.api:app --host 0.0.0.0 --port ${PORT}"]
