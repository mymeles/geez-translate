#!/bin/bash

echo "====================================================="
echo "Geez Translate API Diagnostic Tool"
echo "====================================================="

echo "1. Checking container status..."
docker-compose ps

echo -e "\n2. Checking container logs for errors..."
docker-compose logs --tail=100 api | grep -i "error\|exception\|traceback"

echo -e "\n3. Checking Nginx logs for errors..."
docker-compose logs --tail=20 nginx

echo -e "\n4. Testing direct API container access..."
docker-compose exec api curl -v http://localhost:8000/health

echo -e "\n5. Testing model and processor files..."
echo "Model files in container:"
docker-compose exec api ls -la /app/models/seamless-m4t-v2-large

echo -e "\n6. Testing network connectivity..."
docker-compose exec api curl -v -I https://huggingface.co || echo "Unable to reach HuggingFace"

echo -e "\n7. Checking GPU status in container..."
docker-compose exec api python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}');" || echo "Failed to check GPU status"

echo -e "\n8. Memory usage..."
docker stats --no-stream

echo -e "\n====================================================="
echo "Diagnostic complete. Use './restart_api.sh' to restart the service if needed."
echo "======================================================" 