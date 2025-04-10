FROM python:3.10-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface \
    HF_HOME=/home/appuser/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    ca-certificates \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --upgrade pip

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Create app directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory for huggingface
RUN mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser/.cache

# Copy application code
COPY app.py .
COPY scripts ./scripts/

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
# We'll use the number of workers based on environment variable with default of 1
CMD gunicorn app:app \
    --bind 0.0.0.0:8000 \
    --workers ${WORKERS:-1} \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 300 \
    --log-level info