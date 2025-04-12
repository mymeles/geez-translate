#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "====================================================="
echo "Restarting Geez Translate API service"
echo "====================================================="

# Stop the running containers
echo "Stopping containers..."
docker-compose down

# Restart the containers with the fixed configuration
echo "Starting containers with fixed configuration..."
MODEL_SIZE=${MODEL_SIZE:-large} docker compose up -d

# Show the logs to verify it's working
echo "Showing logs (press Ctrl+C to exit)..."
docker-compose logs -f

echo "====================================================="
echo "API service has been restarted"
echo "=====================================================" 