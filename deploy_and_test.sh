#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "====================================================="
echo "Geez Translate API Deployment and Test Script"
echo "====================================================="

# Parse command line arguments
FORCE_DOWNLOAD=false
MODEL_SIZE=${MODEL_SIZE:-large}
AUDIO_FILE=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --force-download)
      FORCE_DOWNLOAD=true
      shift
      ;;
    --model-size)
      MODEL_SIZE="$2"
      shift
      shift
      ;;
    --audio)
      AUDIO_FILE="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--force-download] [--model-size medium|large] [--audio audio_file.wav]"
      exit 1
      ;;
  esac
done

echo "Using model size: $MODEL_SIZE"
echo "Force download: $FORCE_DOWNLOAD"

# Stop existing containers
echo "Stopping existing containers..."
docker-compose down

# Clear the model directory if requested
if [ "$FORCE_DOWNLOAD" = true ]; then
    echo "Clearing model directory..."
    MODEL_DIR="./models/seamless-m4t-v2-${MODEL_SIZE}"
    
    if [ -d "$MODEL_DIR" ]; then
        # Move to a backup instead of deleting
        BACKUP_DIR="${MODEL_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
        echo "Moving $MODEL_DIR to $BACKUP_DIR"
        mv "$MODEL_DIR" "$BACKUP_DIR"
    fi
    
    # Create empty directory
    mkdir -p "$MODEL_DIR"
    echo "Model directory cleared and recreated"
fi

# Rebuild the Docker image to include the latest changes
echo "Building Docker image..."
docker-compose build

# Start the containers
echo "Starting containers..."
MODEL_SIZE=$MODEL_SIZE docker-compose up -d

# Wait for the API to fully start
echo "Waiting for API to start (60 seconds)..."
sleep 60  # Give time for model loading

# Test the API health endpoint
echo "Testing API health..."
curl -s http://localhost/health | jq .

# Run the test script if audio file was provided
if [ -n "$AUDIO_FILE" ]; then
    echo "Installing test script dependencies..."
    pip install requests
    
    echo "Running test script with provided audio file..."
    python test_api.py --url http://localhost --audio "$AUDIO_FILE" --test all --poll
fi

echo "====================================================="
echo "Deployment complete!"
echo ""
echo "To test file uploads manually, use:"
echo "curl -F 'file=@your_audio.wav' http://localhost/test-upload"
echo ""
echo "To check logs, use: docker-compose logs -f"
echo "=====================================================" 