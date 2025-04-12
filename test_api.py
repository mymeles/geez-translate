#!/usr/bin/env python3
"""
Test script for the Geez Translate API.
This script tests different endpoints with various parameters.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import requests
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api-tester")

def test_health(base_url):
    """Test the health endpoint"""
    logger.info("Testing health endpoint...")
    url = f"{base_url}/health"
    response = requests.get(url)
    logger.info(f"Status code: {response.status_code}")
    logger.info(f"Response: {response.json()}")
    return response.status_code == 200

def test_upload(base_url, audio_file):
    """Test the debug file upload endpoint"""
    logger.info(f"Testing file upload with {audio_file}...")
    
    if not os.path.exists(audio_file):
        logger.error(f"File not found: {audio_file}")
        return False
    
    url = f"{base_url}/test-upload"
    
    # Get file size
    file_size = os.path.getsize(audio_file)
    logger.info(f"File size: {file_size} bytes")
    
    # Prepare the files
    with open(audio_file, "rb") as f:
        files = {"file": (os.path.basename(audio_file), f, "audio/wav")}
        
        # Send the request
        logger.info(f"Sending POST request to {url}")
        try:
            response = requests.post(url, files=files)
            logger.info(f"Status code: {response.status_code}")
            logger.info(f"Response: {response.text[:500]}")  # Print first 500 chars
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return False

def test_transcribe(base_url, audio_file, target_language="amh"):
    """Test the transcribe endpoint"""
    logger.info(f"Testing transcribe with {audio_file}...")
    
    if not os.path.exists(audio_file):
        logger.error(f"File not found: {audio_file}")
        return False
    
    url = f"{base_url}/transcribe"
    
    # Get file size
    file_size = os.path.getsize(audio_file)
    logger.info(f"File size: {file_size} bytes")
    
    # Prepare the files and form data
    with open(audio_file, "rb") as f:
        files = {"file": (os.path.basename(audio_file), f, "audio/wav")}
        data = {"target_language": target_language}
        
        # Send the request
        logger.info(f"Sending POST request to {url}")
        try:
            response = requests.post(url, files=files, data=data)
            logger.info(f"Status code: {response.status_code}")
            logger.info(f"Response: {response.text}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info(f"Job ID: {result.get('id')}")
                    return result.get('id')
                except Exception as e:
                    logger.error(f"Failed to parse response JSON: {str(e)}")
                    return None
            return None
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None

def check_status(base_url, job_id):
    """Check the status of a transcription job"""
    logger.info(f"Checking status of job {job_id}...")
    url = f"{base_url}/status/{job_id}"
    
    try:
        response = requests.get(url)
        logger.info(f"Status code: {response.status_code}")
        logger.info(f"Response: {response.text}")
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test the Geez Translate API")
    parser.add_argument("--url", default="http://localhost", help="Base URL for the API")
    parser.add_argument("--audio", help="Path to an audio file for transcription testing")
    parser.add_argument("--test", choices=["health", "upload", "transcribe", "all"], 
                        default="all", help="Which test to run")
    parser.add_argument("--language", default="amh", help="Target language for transcription")
    parser.add_argument("--poll", action="store_true", help="Poll for results after transcription")
    parser.add_argument("--job-id", help="Job ID to check status for")
    args = parser.parse_args()
    
    base_url = args.url.rstrip("/")
    logger.info(f"Using API at {base_url}")
    
    if args.job_id:
        # Just check the status of a specific job
        result = check_status(base_url, args.job_id)
        sys.exit(0 if result else 1)
    
    if args.test in ["health", "all"]:
        if not test_health(base_url):
            logger.error("Health check failed")
            if args.test != "all":
                sys.exit(1)
    
    if args.test in ["upload", "all"] and args.audio:
        if not test_upload(base_url, args.audio):
            logger.error("Upload test failed")
            if args.test != "all":
                sys.exit(1)
    
    if args.test in ["transcribe", "all"] and args.audio:
        job_id = test_transcribe(base_url, args.audio, args.language)
        if not job_id:
            logger.error("Transcription test failed")
            if args.test != "all":
                sys.exit(1)
        
        if args.poll and job_id:
            logger.info(f"Polling for results of job {job_id}...")
            for _ in range(30):  # Poll for up to 5 minutes (30 * 10 seconds)
                result = check_status(base_url, job_id)
                if result and result.get("status") in ["completed", "error"]:
                    logger.info(f"Final result: {json.dumps(result, indent=2)}")
                    break
                logger.info(f"Job still processing, waiting 10 seconds...")
                time.sleep(10)

if __name__ == "__main__":
    main() 