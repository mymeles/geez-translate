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
    # Force transformers to use local files
    TRANSFORMERS_OFFLINE=1

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

# Copy application code
USER root
COPY app.py .

# Set proper permissions for the app directory
RUN chown -R appuser:appuser /app

# Switch back to non-root user
USER appuser

# Add user's .local/bin to PATH for installed packages
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=60s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1
