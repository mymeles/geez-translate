#!/bin/bash

echo "====================================================="
echo "Testing sample translation functionality"
echo "====================================================="

# Restart the service with debug mode
echo "Restarting service with debug logs..."
LOG_LEVEL=debug docker-compose down && LOG_LEVEL=debug docker-compose up -d

# Wait for the service to start and load the model
echo "Waiting for the service to initialize (120 seconds)..."
sleep 120

# Check the health endpoint to see if sample translation is included
echo "Checking health endpoint for sample translation..."
curl -s http://localhost/health | jq '.'

# Explicitly test the translation endpoint
echo "Triggering sample audio translation test..."
curl -s -X POST http://localhost/test-translation | jq '.'

echo "====================================================="
echo "Tests completed"
echo "====================================================="

# Show logs
echo "Showing recent logs (Ctrl+C to exit)..."
docker-compose logs --tail=100 -f 