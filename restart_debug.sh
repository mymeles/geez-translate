#!/bin/bash

echo "====================================================="
echo "Restarting Geez Translate API with debug mode"
echo "====================================================="

# Stop running containers
echo "Stopping containers..."
docker-compose down

# Start with debug mode enabled
echo "Starting containers with debug logging..."
LOG_LEVEL=debug MODEL_SIZE=${MODEL_SIZE:-large} docker-compose up -d

# Show logs
echo "Showing logs (Ctrl+C to exit)..."
docker-compose logs -f

echo "====================================================="
echo "To test a file upload directly to the API container, use:"
echo "docker-compose exec api curl -F 'file=@/path/to/audio/file.wav' http://localhost:8000/test-upload"
echo "=====================================================" 