# Geez Translate - Amharic Speech-to-Text Translation Service

This is a service that uses Facebook's Seamless M4T v2 model to translate Amharic speech to text.

## Quick Start

The easiest way to run the service is using the provided shell script:

```bash
./run.sh
```

This script will automatically detect your system architecture (M1/M2, ARM64, x86_64) and whether CUDA is available, and will start the appropriate Docker container.

### Apple Silicon (M1/M2 Mac) Users

If you're on an Apple Silicon Mac (M1/M2), the script will automatically use the ARM64-optimized container. The service is configured to work efficiently on M1/M2 Macs with:

- Memory limits configured for M1 performance
- Compatible PyTorch versions for ARM64
- Worker settings optimized for Apple Silicon

## Manual Setup with Docker Compose

You can also manually start the service using Docker Compose. Several different configurations are available:

### 1. ARM64 Version (Optimized for Apple Silicon M1/M2)

```bash
docker compose up -d api-arm64
```

### 2. Alternative Conda-based Version for Apple Silicon

If the standard ARM64 version has issues:

```bash
docker compose up -d api-m1-conda
```

### 3. CUDA Version with pip (for NVIDIA GPUs)

```bash
docker compose up -d api-cuda-pip
```

### 4. CUDA Version with conda (alternative for NVIDIA GPUs)

```bash
docker compose up -d api-cuda-conda
```

## Health Check

Once the service is running, you can check its status at:

```
http://localhost:8000/health
```

## API Usage

### Transcribe Audio

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@your-audio-file.wav" \
  -F "target_language=amh"
```

This will return a job ID, which you can use to check the status of the transcription.

### Check Status

```bash
curl "http://localhost:8000/status/{job_id}"
```

### Batch Processing

```bash
curl -X POST "http://localhost:8000/batch" \
  -F "files=@file1.wav" \
  -F "files=@file2.wav" \
  -F "target_language=amh"
```

## Environment Variables

You can adjust the following environment variables in the docker-compose.yml file:

- `BATCH_SIZE`: Number of batch items to process at once (default: 8 for GPU, 4 for CPU)
- `MAX_AUDIO_LENGTH`: Maximum audio length in seconds (default: 300)

## Troubleshooting

### Apple Silicon (M1/M2) Specific Issues

- **Package Download Errors**: If you see package download errors, try the conda-based image:
  ```bash
  docker compose down && docker compose up -d api-m1-conda
  ```

- **Memory Issues**: Reduce the batch size by editing the `BATCH_SIZE` environment variable in docker-compose.yml (try 1 instead of 2)

- **Slow Performance**: This is expected as the model is running on CPU. The model is large, and without GPU acceleration, inference will be slower.

### General Issues

- **SSL/Download Errors**: If you encounter SSL/download errors during Docker build:
  1. Try a different Dockerfile by adjusting the docker-compose.yml file
  2. Check your internet connection and proxy settings 
  3. Ensure Docker Desktop has enough resources allocated (especially on Apple Silicon)
  4. If still having issues, manually download and install PyTorch outside the Docker container

## System Requirements

- Docker and Docker Compose
- 4GB+ RAM
- For GPU acceleration: NVIDIA GPU with CUDA support 

## Docker Deployment (Simplified)

This project now includes a consolidated Docker setup that works across different platforms (x86_64 with NVIDIA GPUs and ARM64 like M1/M2 Macs).

### Prerequisites

- Docker and Docker Compose installed
- For GPU support: NVIDIA Docker runtime installed (for x86_64 systems with NVIDIA GPUs)

### Build and Run

1. Clone this repository and navigate to the project directory:
   ```bash
   git clone https://github.com/yourusername/geez_translate.git
   cd geez_translate
   ```

2. Start the Docker container:
   ```bash
   docker-compose up -d
   ```
   
   For systems with NVIDIA GPUs, uncomment the GPU section in `docker-compose.yml` first.

3. Check the API is running:
   ```bash
   curl http://localhost:8000/health
   ```

### Configuration

The Docker setup uses the following environment variables that can be adjusted in `docker-compose.yml`:

- `WORKERS`: Number of worker processes (default: 2)
- `BATCH_SIZE`: Batch size for processing (default: 8)
- `MAX_AUDIO_LENGTH`: Maximum audio length in seconds (default: 300)
- `MODEL_PATH`: Huggingface model ID (default: facebook/seamless-m4t-v2-large)
- `LOAD_MODEL_ON_STARTUP`: Whether to load the model on startup (default: true)

### Building for Different Platforms

The Dockerfile automatically detects the platform and installs the appropriate PyTorch version:
- For ARM64 (M1/M2 Macs): Uses PyTorch CPU version
- For x86_64: Uses the PyTorch version specified in requirements.txt

### Model Caching

The model will be automatically downloaded from Huggingface on first run and cached in a Docker volume for subsequent runs. This ensures you don't need to download the model repeatedly and provides the following benefits:

- Persistent model storage between container restarts
- Automatic handling of model downloads
- No need for manual model management

> **Note**: If you have an existing `models` directory from a previous setup, it's no longer needed with this new configuration as the model will be downloaded directly from Huggingface. You can safely remove it if you want to free up disk space. 