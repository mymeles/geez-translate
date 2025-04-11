#!/bin/bash
# deploy.sh - Simplified deployment script for Geez Translate API

set -e

# Get model size parameter (default to large if not specified)
export MODEL_SIZE=${1:-large}

# Validate model size
if [[ "$MODEL_SIZE" != "medium" && "$MODEL_SIZE" != "large" ]]; then
    echo "Error: MODEL_SIZE must be either 'medium' or 'large'"
    echo "Usage: ./deploy.sh [medium|large]"
    exit 1
fi

echo "===== Deploying Geez Translate API with ${MODEL_SIZE} model ====="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running. Please start Docker first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python first."
    exit 1
fi

# Check if model directory exists
if [ ! -d "./models/seamless-m4t-v2-${MODEL_SIZE}" ]; then
    echo "Model directory not found: ./models/seamless-m4t-v2-${MODEL_SIZE}"
    echo "Downloading model using Python environment..."
    
    # Check if requirements file exists
    if [ ! -f "requirements-model-download.txt" ]; then
        echo "Error: requirements-model-download.txt not found."
        exit 1
    fi
    
    # Create Python virtual environment if it doesn't exist
    if [ ! -d "model-venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv model-venv
    fi
    
    # Activate virtual environment
    echo "Activating Python virtual environment..."
    source model-venv/bin/activate
    
    # Install dependencies from requirements file
    echo "Installing required packages from requirements-model-download.txt..."
    pip install --upgrade pip
    pip install -r requirements-model-download.txt
    
    # Run the download_model.py script
    echo "Running download_model.py..."
    python download_model.py --model-id "facebook/seamless-m4t-v2-${MODEL_SIZE}" --cache-dir "./models/seamless-m4t-v2-${MODEL_SIZE}"
    
    # Deactivate virtual environment
    deactivate
    
    echo "Model download complete."
fi

# Check for GPU and set batch size
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected - will use GPU acceleration"
    
    # Get GPU memory for batch size optimization
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1}')
    echo "  - GPU Memory: ${GPU_MEM}MB"
    
    # Set batch size based on GPU memory
    if [ "$GPU_MEM" -ge 16000 ]; then
        export BATCH_SIZE=16
    elif [ "$GPU_MEM" -ge 12000 ]; then
        export BATCH_SIZE=12
    elif [ "$GPU_MEM" -ge 8000 ]; then
        export BATCH_SIZE=8
    else
        export BATCH_SIZE=4
    fi
    
    # Adjust batch size based on model
    if [ "$MODEL_SIZE" = "large" ]; then
        # Reduce batch size for large model
        export BATCH_SIZE=$((BATCH_SIZE / 2))
    fi
    
    echo "  - Using batch size: $BATCH_SIZE"
else
    echo "No NVIDIA GPU detected - will use CPU mode"
    export BATCH_SIZE=2
    
    # For CPU mode, further reduce batch size for large model
    if [ "$MODEL_SIZE" = "large" ]; then
        export BATCH_SIZE=1
    fi
fi

# Build and start containers
echo "Building and starting containers for ${MODEL_SIZE} model..."
docker compose build && docker compose up -d

echo ""
echo "✓ Geez Translate API is now running with ${MODEL_SIZE} model!"
echo "  API is available at: http://localhost:8000"
echo "  Health check: http://localhost:8000/health"
echo ""
echo "  To check logs:"
echo "  docker-compose logs -f api"
echo ""
echo "  To stop the service:"
echo "  docker-compose down"
echo ""