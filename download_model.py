#!/usr/bin/env python3
"""
Download the Seamless M4T v2 model from Hugging Face to a local directory.
This script will download the model files and save them to a local cache.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download Seamless M4T v2 model from Hugging Face")
    parser.add_argument("--model-id", default="facebook/seamless-m4t-v2-large", 
                        help="Hugging Face model ID to download (default: facebook/seamless-m4t-v2-large)")
    parser.add_argument("--cache-dir", default="./models/cache", 
                        help="Directory to store downloaded models")
    parser.add_argument("--force", action="store_true", 
                        help="Force re-download even if model exists")
    args = parser.parse_args()

    # Create the cache directory if it doesn't exist
    cache_dir = Path(args.cache_dir).absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model {args.model_id} to {cache_dir}")
    
    # Set the cache directory environment variable
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_HOME"] = str(cache_dir)
    
    try:
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    except ImportError:
        print("Error: transformers library not found.")
        print("Please install the required packages with: pip install transformers")
        sys.exit(1)

    try:
        # Download the processor
        print("Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            args.model_id, 
            cache_dir=cache_dir, 
            force_download=args.force
        )
        print("Processor downloaded successfully.")
        
        # Download the model - using specific model class for Seamless M4T v2
        print("Downloading model...")
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            args.model_id, 
            cache_dir=cache_dir, 
            force_download=args.force,
            low_cpu_mem_usage=True
        )
        print("Model downloaded successfully.")
        
        print("\nModel files are stored at:")
        print(f"  {cache_dir}")
        
        print("\nTo use this model in Docker, the docker-compose.yml has been configured to mount this directory.")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 