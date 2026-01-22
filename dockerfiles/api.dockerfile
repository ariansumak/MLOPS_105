FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app/src
ENV UV_LINK_MODE=copy

# Copy metadata and source code
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY configs/ configs/

# IMPORTANT: Copy your trained model weights so the API can load them
# Based on your train.py, they are likely in the models/ folder
COPY models/ models/

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv uv sync

# The API needs to listen on the port provided by Cloud Run ($PORT)
# We use 8000 as a default for local testing
ENTRYPOINT ["uv", "run", "uvicorn", "pneumoniaclassifier.api:app", "--host", "0.0.0.0", "--port", "8000"]