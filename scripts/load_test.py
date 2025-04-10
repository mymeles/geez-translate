#!/usr/bin/env python3
"""
Load test script for the Seamless M4T API
"""

import argparse
import asyncio
import aiohttp
import time
import os
import random
from pathlib import Path
from typing import List, Dict, Any

# Test parameters
DEFAULT_URL = "http://localhost:8000"
DEFAULT_CONCURRENCY = 10
DEFAULT_REQUESTS = 100

async def upload_file(session: aiohttp.ClientSession, url: str, file_path: str, target_language: str = "amh") -> Dict[str, Any]:
    """Upload a file and get the job ID"""
    start_time = time.time()
    
    data = aiohttp.FormData()
    data.add_field('file', 
                  open(file_path, 'rb'),
                  filename=os.path.basename(file_path),
                  content_type='audio/wav')
    data.add_field('target_language', target_language)
    
    async with session.post(f"{url}/transcribe", data=data) as response:
        response_time = time.time() - start_time
        if response.status != 200:
            print(f"Error: {response.status} - {await response.text()}")
            return {"success": False, "response_time": response_time}
        
        response_data = await response.json()
        return {
            "success": True, 
            "job_id": response_data["id"],
            "response_time": response_time
        }

async def check_status(session: aiohttp.ClientSession, url: str, job_id: str) -> Dict[str, Any]:
    """Check the status of a job"""
    start_time = time.time()
    async with session.get(f"{url}/status/{job_id}") as response:
        response_time = time.time() - start_time
        if response.status != 200:
            print(f"Error checking status: {response.status} - {await response.text()}")
            return {"success": False, "response_time": response_time}
        
        response_data = await response.json()
        return {
            "success": True,
            "status": response_data["status"],
            "text": response_data.get("text", ""),
            "response_time": response_time,
            "processing_time": response_data.get("processing_time", 0)
        }

async def complete_job(session: aiohttp.ClientSession, url: str, file_path: str, target_language: str = "amh") -> Dict[str, Any]:
    """Complete a full job cycle - upload, poll until complete, return results"""
    # Upload file
    upload_result = await upload_file(session, url, file_path, target_language)
    if not upload_result["success"]:
        return {"success": False, "stage": "upload"}
    
    job_id = upload_result["job_id"]
    start_time = time.time()
    
    # Poll for completion
    while True:
        await asyncio.sleep(1)  # Don't hammer the API
        status_result = await check_status(session, url, job_id)
        
        if not status_result["success"]:
            return {"success": False, "stage": "status_check"}
        
        if status_result["status"] == "completed":
            elapsed_time = time.time() - start_time
            return {
                "success": True,
                "job_id": job_id,
                "upload_time": upload_result["response_time"],
                "processing_time": status_result["processing_time"],
                "total_time": elapsed_time,
                "text": status_result["text"]
            }
        elif status_result["status"] == "error":
            return {"success": False, "stage": "processing", "error": status_result.get("error", "Unknown error")}
        
        # Check for timeout
        if time.time() - start_time > 180:  # 3 minute timeout
            return {"success": False, "stage": "timeout"}

async def run_load_test(url: str, audio_files: List[str], concurrency: int, total_requests: int):
    """Run a load test with specified concurrency"""
    if not audio_files:
        print("Error: No audio files provided")
        return
    
    # Results storage
    results = {
        "successful_requests": 0,
        "failed_requests": 0,
        "upload_times": [],
        "processing_times": [],
        "total_times": [],
    }
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_one_request(session):
        async with semaphore:
            # Select a random audio file
            file_path = random.choice(audio_files)
            result = await complete_job(session, url, file_path)
            
            if result["success"]:
                results["successful_requests"] += 1
                results["upload_times"].append(result["upload_time"])
                results["processing_times"].append(result["processing_time"])
                results["total_times"].append(result["total_time"])
                print(f"✓ Request completed in {result['total_time']:.2f}s (upload: {result['upload_time']:.2f}s, processing: {result['processing_time']:.2f}s)")
            else:
                results["failed_requests"] += 1
                print(f"✗ Request failed at stage: {result.get('stage', 'unknown')}")
    
    # Start timing
    overall_start = time.time()
    
    # Create tasks
    async with aiohttp.ClientSession() as session:
        tasks = [process_one_request(session) for _ in range(total_requests)]
        await asyncio.gather(*tasks)
    
    # Calculate results
    overall_time = time.time() - overall_start
    
    # Print summary
    print("\n=== Load Test Results ===")
    print(f"URL: {url}")
    print(f"Concurrency: {concurrency}")
    print(f"Total Requests: {total_requests}")
    print(f"Overall Time: {overall_time:.2f} seconds")
    print(f"Requests/second: {total_requests / overall_time:.2f}")
    print(f"Successful: {results['successful_requests']} ({results['successful_requests'] / total_requests * 100:.1f}%)")
    print(f"Failed: {results['failed_requests']} ({results['failed_requests'] / total_requests * 100:.1f}%)")
    
    if results["upload_times"]:
        print("\nTiming Statistics:")
        print(f"  Upload Time (avg): {sum(results['upload_times']) / len(results['upload_times']):.3f}s")
        print(f"  Processing Time (avg): {sum(results['processing_times']) / len(results['processing_times']):.3f}s")
        print(f"  Total Time (avg): {sum(results['total_times']) / len(results['total_times']):.3f}s")
        print(f"  Total Time (min): {min(results['total_times']):.3f}s")
        print(f"  Total Time (max): {max(results['total_times']):.3f}s")

def main():
    parser = argparse.ArgumentParser(description="Load test the Seamless M4T API")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"API base URL (default: {DEFAULT_URL})")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, 
                        help=f"Number of concurrent requests (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS,
                        help=f"Total number of requests to make (default: {DEFAULT_REQUESTS})")
    parser.add_argument("--audio-dir", required=True, help="Directory containing .wav audio files for testing")
    
    args = parser.parse_args()
    
    # Find audio files
    audio_dir = Path(args.audio_dir)
    audio_files = list(str(p) for p in audio_dir.glob("*.wav"))
    
    if not audio_files:
        print(f"No .wav files found in {audio_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files for testing")
    print(f"Running load test with {args.concurrency} concurrent requests...")
    
    # Run the load test
    asyncio.run(run_load_test(args.url, audio_files, args.concurrency, args.requests))

if __name__ == "__main__":
    main()