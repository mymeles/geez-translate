FROM nvcr.io/nvidia/pytorch:23.07-py3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    TRANSFORMERS_CACHE=/app/models/cache \
    HF_HOME=/app/models/cache \
    LOG_LEVEL=INFO \
    # Optimized Torch environment settings
    TORCH_CUDNN_V8_API_ENABLED=1 \
    CUDA_MODULE_LOADING=LAZY \
    # Enable PyTorch optimizations
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    # Force CUDA to use spawn for multiprocessing
    PYTORCH_MULTIPROCESSING_START_METHOD=spawn \
    # Don't force offline mode by default - will be set in docker-compose
    # TRANSFORMERS_OFFLINE=1
    PATH="/home/appuser/.local/bin:${PATH}"

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    wget \
    git \
    curl \
    ca-certificates \
    build-essential \
    libsndfile1 \
    ffmpeg \
    # Update certificates
    && update-ca-certificates \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify ffmpeg installation
RUN ffmpeg -version

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Create app directory and set permissions
WORKDIR /app
RUN chown -R appuser:appuser /app

# Create model cache directory and model directory structure
RUN mkdir -p /app/models/cache && \
    mkdir -p /app/models/seamless-m4t-v2-large && \
    mkdir -p /app/models/seamless-m4t-v2-medium && \
    chown -R appuser:appuser /app/models

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN chown appuser:appuser /app/requirements.txt

# Switch to non-root user before pip operations
USER appuser

# Update pip as non-root user
RUN pip install --user --upgrade pip

# Install dependencies from requirements.txt with optimizations
RUN pip install --user -r requirements.txt && \
    pip install --user soundfile && \
    pip install --user numba && \
    pip install --user psutil

# Copy application code and download script
USER root
COPY app.py .
COPY download_model.py .
RUN chmod +x download_model.py

# Create data directory (useful for potential future use)
RUN mkdir -p /app/data

# Copy test.mp3 into the container
# Use RUN commands to handle potential non-existence gracefully
RUN if [ -f ./test.mp3 ]; then cp ./test.mp3 /app/test.mp3; echo 'Copied ./test.mp3 to /app/'; fi
RUN if [ -f ./data/test.mp3 ]; then cp ./data/test.mp3 /app/test.mp3; echo 'Copied ./data/test.mp3 to /app/'; fi

# Set proper permissions for the app directory, including the potentially copied test file
RUN chown -R appuser:appuser /app

# Switch back to non-root user
USER appuser

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=60s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1