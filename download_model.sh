#!/bin/bash
# download_model.sh - Downloads the Seamless M4T model for local use

set -e

# Get model size parameter (default to large if not specified)
MODEL_SIZE=${1:-large}

# Validate model size
if [[ "$MODEL_SIZE" != "medium" && "$MODEL_SIZE" != "large" ]]; then
    echo "Error: MODEL_SIZE must be either 'medium' or 'large'"
    echo "Usage: ./download_model.sh [medium|large]"
    exit 1
fi

echo "===== Downloading Seamless M4T v2 $MODEL_SIZE Model ====="

# Check if Python and transformers are installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH. Please install Python first."
    exit 1
fi

# Create models directory if it doesn't exist
mkdir -p models/seamless-m4t-v2-$MODEL_SIZE

echo "Installing required packages..."
pip install -q transformers torch tqdm

echo "Downloading model..."
python3 -c "
import os
import sys
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
from tqdm import tqdm

model_size = '$MODEL_SIZE'
model_id = f'facebook/seamless-m4t-v2-{model_size}'
output_dir = f'./models/seamless-m4t-v2-{model_size}'

print(f'Using model: {model_id}')
print(f'Downloading processor...')
processor = AutoProcessor.from_pretrained(model_id)
processor.save_pretrained(output_dir)
print('Processor saved successfully')

print(f'Downloading model (this may take a while)...')
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_id)
model.save_pretrained(output_dir)
print('Model saved successfully')

print('Model files:')
for root, _, files in os.walk(output_dir):
    for f in files:
        print(f'  {os.path.join(root, f)}')
"

echo ""
echo "✓ Model downloaded successfully to ./models/seamless-m4t-v2-$MODEL_SIZE"
echo ""
echo "You can now run the following commands to build and start the service:"
echo "  MODEL_SIZE=$MODEL_SIZE docker-compose build"
echo "  MODEL_SIZE=$MODEL_SIZE docker-compose up -d"
echo ""
echo "Or to use the model without specifying MODEL_SIZE each time, run:"
echo "  export MODEL_SIZE=$MODEL_SIZE"
echo "  docker-compose build"
echo "  docker-compose up -d"
echo "" 