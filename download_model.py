#!/usr/bin/env python3
"""
Download the Seamless M4T v2 model from Hugging Face to a local directory.
This script will download the model files and save them to a local cache.

This script expects the dependencies to be pre-installed using:
pip install -r requirements-model-download.txt
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model-downloader")

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import torch
        import transformers
        import tqdm
        
        # Print versions for debugging
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA is not available, using CPU (this will be slower)")
        
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install dependencies using: pip install -r requirements-model-download.txt")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Seamless M4T v2 model from Hugging Face")
    parser.add_argument("--model-id", default="facebook/seamless-m4t-v2-large", 
                        help="Hugging Face model ID to download (default: facebook/seamless-m4t-v2-large)")
    parser.add_argument("--cache-dir", default="./models/cache", 
                        help="Directory to store downloaded models")
    parser.add_argument("--force", action="store_true", 
                        help="Force re-download even if model exists")
    args = parser.parse_args()

    # Check dependencies first
    if not check_dependencies():
        return 1

    # Extract model size from model ID
    model_id = args.model_id
    model_size = "large"  # Default
    if "medium" in model_id:
        model_size = "medium"
    
    # Create the cache directory if it doesn't exist
    cache_dir = Path(args.cache_dir).absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading model {model_id} to {cache_dir}")
    
    # Set the cache directory environment variable
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_HOME"] = str(cache_dir)
    
    try:
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    except ImportError as e:
        logger.error(f"Error importing transformers components: {e}")
        logger.error("Please install the required packages with: pip install -r requirements-model-download.txt")
        return 1

    try:
        # Start the overall timer
        overall_start_time = time.time()
        
        # Download the processor
        logger.info("Downloading processor...")
        processor_start_time = time.time()
        processor = AutoProcessor.from_pretrained(
            model_id, 
            force_download=args.force
        )
        # Save processor directly to the specified directory
        processor.save_pretrained(cache_dir)
        processor_time = time.time() - processor_start_time
        logger.info(f"Processor downloaded and saved successfully in {processor_time:.2f} seconds.")
        
        # Download the model - using specific model class for Seamless M4T v2
        logger.info("Downloading model (this may take a while)...")
        model_start_time = time.time()
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            model_id, 
            force_download=args.force,
            low_cpu_mem_usage=True
        )
        # Save model directly to the specified directory
        model.save_pretrained(cache_dir)
        model_time = time.time() - model_start_time
        logger.info(f"Model downloaded and saved successfully in {model_time:.2f} seconds.")
        
        # Print model directory information
        logger.info("\nModel files are stored at:")
        logger.info(f"  {cache_dir}")
        
        # Calculate total size
        total_size_bytes = 0
        file_count = 0
        for path, _, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(path, file)
                size = os.path.getsize(file_path)
                total_size_bytes += size
                file_count += 1
        
        total_size_mb = total_size_bytes / (1024 * 1024)
        total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
        
        # Calculate overall time
        overall_time = time.time() - overall_start_time
        
        logger.info(f"Total files: {file_count}")
        logger.info(f"Total size: {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")
        logger.info(f"Total download and save time: {overall_time:.2f} seconds")
        
        logger.info("\nTo use this model in Docker, run:")
        logger.info(f"  MODEL_SIZE={model_size} docker-compose build")
        logger.info(f"  MODEL_SIZE={model_size} docker-compose up -d")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 