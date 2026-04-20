# ── Verdictra LAW.ai — Backend Dockerfile ─────────────────────────────────────
# Python 3.13 slim — matches local dev environment
FROM python:3.13-slim

# Install system dependencies
# tesseract-ocr — for scanned PDF pages
# poppler-utils — for pdf2image
# libgl1 — for OpenCV (GLiNER dependency)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry==1.8.3

# Install PyTorch CPU-only BEFORE poetry install.
# This prevents poetry from pulling in massive NVIDIA GPU packages
# (nvidia-cuda, nvidia-cublas etc.) which are useless on a CPU server
# and cause download timeouts during docker build.
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Copy dependency files first (layer caching)
COPY pyproject.toml poetry.lock ./

# Install all other dependencies via poetry
# torch is already installed above so poetry skips it
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev

# Ensure CPU torch is used (poetry may have reinstalled GPU version)
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu --force-reinstall

# Copy application code
COPY . .

# Create cases directory (mounted as volume in production)
RUN mkdir -p cases data/embeddings

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI with uvicorn
CMD ["uvicorn", "resolver_ui.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]