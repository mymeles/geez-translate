#!/bin/bash

# Improved script to run Geez Translate with Docker Compose

set -e

echo "Starting Geez Translate service..."

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running. Please start Docker first."
    exit 1
fi

# Check for docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "Warning: docker-compose not found. Trying with 'docker compose' instead..."
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Check if CUDA is available (for NVIDIA GPU support)
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected."
    
    # Check GPU memory
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1}')
    echo "  - GPU Memory: ${GPU_MEM}MB"
    
    # Adjust batch size based on GPU memory
    if [ "$GPU_MEM" -ge 16000 ]; then
        echo "  - Setting batch size to 16 for 16GB+ GPU"
        BATCH_SIZE=16
    elif [ "$GPU_MEM" -ge 12000 ]; then
        echo "  - Setting batch size to 12 for 12GB+ GPU"
        BATCH_SIZE=12
    elif [ "$GPU_MEM" -ge 8000 ]; then
        echo "  - Setting batch size to 8 for 8GB+ GPU"
        BATCH_SIZE=8
    else
        echo "  - Setting batch size to 4 for lower memory GPU"
        BATCH_SIZE=4
    fi
    
    # Update batch size in docker-compose.yml
    sed -i.bak "s/BATCH_SIZE=.*/BATCH_SIZE=$BATCH_SIZE/g" docker-compose.yml
else
    echo "No NVIDIA GPU detected. Using CPU mode."
    # Set a modest batch size for CPU
    sed -i.bak "s/BATCH_SIZE=.*/BATCH_SIZE=2/g" docker-compose.yml
    
    # Comment out GPU configuration 
    sed -i.bak 's/deploy:/#deploy:/g' docker-compose.yml
    sed -i.bak 's/  resources:/#  resources:/g' docker-compose.yml
    sed -i.bak 's/    reservations:/#    reservations:/g' docker-compose.yml
    sed -i.bak 's/      devices:/#      devices:/g' docker-compose.yml
    sed -i.bak 's/        - driver: nvidia/#        - driver: nvidia/g' docker-compose.yml
    sed -i.bak 's/          count: 1/#          count: 1/g' docker-compose.yml
    sed -i.bak 's/          capabilities: \[gpu\]/#          capabilities: \[gpu\]/g' docker-compose.yml
fi

# Clean up backup files
find . -name "*.bak" -type f -delete

echo "Starting Docker containers..."
$DOCKER_COMPOSE down || true  # Bring down if already running
$DOCKER_COMPOSE up -d

echo ""
echo "✓ Geez Translate service is now running!"
echo "  API is available at: http://localhost:8000"
echo "  Health check: http://localhost:8000/health"
echo ""
echo "  The model will be automatically downloaded when first needed."
echo "  This can take a few minutes for the first run."
echo ""
echo "  To stop the service, run: docker-compose down"
echo "  To view logs, run: docker-compose logs -f api"