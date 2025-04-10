#!/bin/bash
# start-api.sh - Improved startup script for Geez Translate API

# Set environment variables for logging
export LOG_LEVEL=${LOG_LEVEL:-DEBUG}
export PYTHONUNBUFFERED=1 

# Print diagnostic information
echo "===== Starting Geez Translate API Service ====="
echo "Workers: ${WORKERS:-1}"
echo "Log Level: ${LOG_LEVEL}"
echo "Batch Size: ${BATCH_SIZE:-8}"
echo "Max Audio Length: ${MAX_AUDIO_LENGTH:-300}s"
echo "Platform: $(uname -a)"
echo "Python: $(python --version)"

# Print available disk space
echo "Disk space:"
df -h

# Print memory information
echo "Memory information:"
free -h || echo "free command not available"

# Check for temp directory
echo "Temp directory:"
ls -la /tmp

# Check for existing Hugging Face cache
echo "Checking HF cache:"
ls -la ${HF_HOME:-/home/appuser/.cache/huggingface} || echo "HF cache not yet created"

# Check GPU if available
if python -c "import torch; print(torch.cuda.is_available())"; then
  echo "CUDA is available"
  echo "GPU info:"
  python -c "import torch; print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')"
else
  echo "Running in CPU mode"
fi

# Start Gunicorn with proper logging configuration
echo "Starting Gunicorn with ${WORKERS:-1} workers..."
exec gunicorn app:app \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers ${WORKERS:-1} \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 300 \
    --log-level debug \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance