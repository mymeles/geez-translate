#!/bin/bash

# Simple script to run Geez Translate with Docker Compose

set -e

echo "Starting Geez Translate service..."

# Download the model first
echo "Checking if model is downloaded..."
if [ ! -d "models/cache" ]; then
    echo "Model not found. Downloading model..."
    # Make sure pip and venv are available
    if ! command -v pip &> /dev/null; then
        echo "Error: pip is not installed. Please install Python and pip first."
        exit 1
    fi
    
    # Create a virtual environment if needed
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python -m venv venv
    fi
    
    # Activate virtual environment and install transformers
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install transformers torch torchaudio numpy
    
    # Run the download script
    echo "Downloading model..."
    python download_model.py
    
    echo "Model downloaded successfully!"
else
    echo "Model already downloaded. Using cached version."
fi

# Check if CUDA is available (for NVIDIA GPU support)
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Enabling GPU support..."
    # Uncomment GPU support in docker-compose.yml
    sed -i.bak 's/# deploy:/deploy:/g' docker-compose.yml
    sed -i.bak 's/#   resources:/  resources:/g' docker-compose.yml
    sed -i.bak 's/#     reservations:/    reservations:/g' docker-compose.yml
    sed -i.bak 's/#       devices:/      devices:/g' docker-compose.yml
    sed -i.bak 's/#         - driver: nvidia/        - driver: nvidia/g' docker-compose.yml
    sed -i.bak 's/#           count: 1/          count: 1/g' docker-compose.yml
    sed -i.bak 's/#           capabilities: \[gpu\]/          capabilities: [gpu]/g' docker-compose.yml
    echo "GPU support enabled."
else
    echo "No NVIDIA GPU detected. Running with CPU."
fi

# Check architecture and set environment variable
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
    echo "ARM64 architecture detected (e.g., Apple Silicon M1/M2)."
    # You might want to adjust workers for M1/M2
    sed -i.bak 's/WORKERS=2/WORKERS=1/g' docker-compose.yml
    echo "Adjusted worker count for ARM64 architecture."
fi

# Build and start containers
echo "Building and starting containers..."
docker-compose up -d

# Clean up backup files
find . -name "*.bak" -type f -delete

echo ""
echo "Geez Translate service is now running!"
echo "API is available at: http://localhost:8000"
echo "Health check: http://localhost:8000/health"
echo ""
echo "To stop the service, run: docker-compose down" 