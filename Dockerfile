FROM python:3.10-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface \
    HF_HOME=/home/appuser/.cache/huggingface \
    LOG_LEVEL=DEBUG

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
    && update-ca-certificates

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

# Create and properly permission the cache directory for Hugging Face
RUN mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser/.cache

# Copy application code
COPY app.py .
COPY scripts ./scripts/

# Copy startup script
COPY start-api.sh .
RUN chmod +x start-api.sh

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port
EXPOSE 8000

# Health check - increased timeout for model loading
HEALTHCHECK --interval=30s --timeout=60s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application with proper workers
CMD ["./start-api.sh"]