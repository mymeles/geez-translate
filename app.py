import os
import time
import tempfile
import uuid
from typing import Optional, List
import sys
import platform
import traceback
import logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# Import numpy first and set environment variable to prevent _ARRAY_API error on M1 Macs
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
try:
    import numpy as np
except ImportError:
    print("NumPy not installed. Please install required dependencies.")
    sys.exit(1)

# Import torch with additional error handling for M1 Macs
try:
    import torch
    if "arm64" in platform.machine().lower() and torch.__version__ < "2.0.0":
        print("Warning: You're using an older version of PyTorch on Apple Silicon.")
        print("For best compatibility, use PyTorch 2.0.0+ on M1/M2 Macs.")
except ImportError:
    print("PyTorch not installed. Please install required dependencies.")
    sys.exit(1)

import librosa
import uvicorn
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback

# Enhanced logging configuration
log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()  # Set default to DEBUG for more visibility
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("app.log")  # Also log to a file
    ]
)
logger = logging.getLogger("geez-translate")

# Log startup information immediately
logger.info("=====================================================")
logger.info("APPLICATION STARTING - Enhanced Logging Enabled")
logger.info(f"Log level set to: {log_level}")
logger.info("=====================================================")

# Configure GPU memory growth to avoid OOM errors
if torch.cuda.is_available():
    # Optional - configure GPU memory growth
    torch.cuda.empty_cache()
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    logger.info("GPU not available, using CPU")

# Create FastAPI app
app = FastAPI(
    title="Seamless M4T API",
    description="API for Amharic speech-to-text translation using Seamless M4T v2",
    version="1.0.0"
)

# Middleware for logging requests
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        logger.info(f"Request received: {request.method} {request.url.path}")
        
        # Log headers if needed (be careful with sensitive info)
        # logger.debug(f"Request headers: {dict(request.headers)}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(f"Request completed: {request.method} {request.url.path} - Status: {response.status_code} - Took: {process_time:.4f}s")
            
            # Log response headers if needed
            # logger.debug(f"Response headers: {dict(response.headers)}")
            
            return response
        except Exception as e:
            logger.error(f"Request failed: {request.method} {request.url.path} - Error: {str(e)}")
            logger.error(traceback.format_exc())
            # Re-raise the exception to let FastAPI handle it
            raise e

# Add the logging middleware *early* in the middleware stack
app.add_middleware(LoggingMiddleware)

# Enable CORS *after* the logging middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and response models
class TranscriptionRequest(BaseModel):
    target_language: str = "amh"

class TranscriptionResponse(BaseModel):
    id: str
    text: str
    processing_time: float
    status: str

class BatchRequest(BaseModel):
    file_ids: List[str]
    target_language: str = "amh"

class QueuedResponse(BaseModel):
    id: str
    status: str
    estimated_completion_time: Optional[float] = None

# Global variables
DEFAULT_MODEL_ID = "./models/seamless-m4t-v2-large"
MODEL_ID = os.getenv("MODEL_PATH", DEFAULT_MODEL_ID)  # Can be set to a local path
LOCAL_MODEL = not MODEL_ID.startswith(("facebook/", "http://", "https://"))  # Check if it's a local path
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))  # Adjust based on GPU memory
MAX_AUDIO_LENGTH = int(os.getenv("MAX_AUDIO_LENGTH", "300"))  # Max audio length in seconds
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL_ON_STARTUP = os.getenv("LOAD_MODEL_ON_STARTUP", "false").lower() == "true"

# Log important configuration settings
logger.info(f"MODEL_ID: {MODEL_ID}")
logger.info(f"LOCAL_MODEL: {LOCAL_MODEL}")
logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
logger.info(f"MAX_AUDIO_LENGTH: {MAX_AUDIO_LENGTH}")
logger.info(f"DEVICE: {DEVICE}")
logger.info(f"LOAD_MODEL_ON_STARTUP: {LOAD_MODEL_ON_STARTUP}")

# In-memory queues for processing and results
processing_queue = {}
completed_jobs = {}

# For storing the model and processor
processor = None
model = None
model_loading_lock = False

# Lazy load function for the model
async def load_model():
    global processor, model, model_loading_lock
    
    if model_loading_lock:
        logger.warning("Model loading already in progress")
        return {"status": "loading_in_progress"}
    
    if model is not None and processor is not None:
        logger.info("Model already loaded")
        return {"status": "already_loaded"}
        
    try:
        model_loading_lock = True
        logger.info(f"Loading model from {MODEL_ID} on {DEVICE}...")
        
        # Import here to avoid initial import errors
        try:
            from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
            logger.info("Successfully imported transformers modules")
        except ImportError as e:
            logger.error(f"Failed to import from transformers: {str(e)}")
            logger.error(traceback.format_exc())
            model_loading_lock = False
            return {"status": "error", "message": f"Failed to import required libraries: {str(e)}"}
        
        # Load processor first
        try:
            logger.info("Loading processor...")
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            logger.info("Processor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading processor: {str(e)}")
            logger.error(traceback.format_exc())
            model_loading_lock = False
            return {"status": "error", "message": f"Error loading processor: {str(e)}"}
        
        # Then load model with improved GPU handling
        try:
            logger.info(f"Loading model on {DEVICE}...")
            # For M1 Mac compatibility, use chunked loading
            if "arm64" in platform.machine().lower() and DEVICE == "cpu":
                logger.info("Detected ARM64 platform, using float32 precision")
                model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                    MODEL_ID, 
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32  # Use float32 instead of half precision on M1
                ).to(DEVICE)
            else:
                # For GPU, use half precision and better memory optimization
                logger.info("Using GPU optimization with float16 precision")
                model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else None,
                    low_cpu_mem_usage=True
                ).to(DEVICE)
                    
                # Optional: optimize for inference on GPU
                if DEVICE == "cuda":
                    # Already loaded with float16 dtype
                    torch.cuda.empty_cache()  # Clear any residual memory
                    logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                    logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            
            logger.info("Model loaded successfully")
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            model_loading_lock = False
            return {"status": "error", "message": f"Error loading model: {str(e)}"}
    finally:
        model_loading_lock = False

# Load model and processor
@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    
    # Print system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    
    if LOAD_MODEL_ON_STARTUP:
        logger.info("Loading model on startup as configured...")
        result = await load_model()
        logger.info(f"Model loading result: {result}")
    else:
        logger.info("Model loading on startup disabled. Use /load-model endpoint to load it.")
    
    logger.info("Startup complete")

# Helper functions
def process_audio(audio_path, target_language="amh"):
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if model is None or processor is None:
            logger.error("Model or processor not loaded yet")
            raise ValueError("Model or processor not loaded yet. Please try again later.")
            
        # Log some information about the audio file
        logger.info(f"Processing audio file: {audio_path}, target language: {target_language}")
        
        # Check if file exists and is readable
        if not os.path.exists(audio_path):
            logger.error(f"Audio file does not exist: {audio_path}")
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
            
        if not os.access(audio_path, os.R_OK):
            logger.error(f"Cannot read audio file (permission denied): {audio_path}")
            raise PermissionError(f"Cannot read audio file: {audio_path}")
            
        # Get file size
        file_size = os.path.getsize(audio_path)
        logger.info(f"Audio file size: {file_size} bytes")
        
        # Load audio file
        try:
            logger.info(f"Loading audio file: {audio_path}")
            audio, sample_rate = librosa.load(audio_path, sr=None)
            logger.info(f"Audio loaded successfully: length={len(audio)}, sample_rate={sample_rate}")
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Could not load audio file: {str(e)}")
        
        # Check audio length
        audio_length = len(audio) / sample_rate
        logger.info(f"Audio length: {audio_length:.2f}s")
        if audio_length > MAX_AUDIO_LENGTH:
            logger.warning(f"Audio too long: {audio_length:.2f}s > {MAX_AUDIO_LENGTH}s")
            raise ValueError(f"Audio is too long ({audio_length:.2f}s). Maximum allowed is {MAX_AUDIO_LENGTH}s")
        
        # Resample if needed
        if sample_rate != 16000:
            logger.info(f"Resampling from {sample_rate} to 16000 Hz")
            audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Process the audio
        logger.info("Creating input tensors...")
        try:
            inputs = processor(audios=audio, sampling_rate=sample_rate, return_tensors="pt", padding=True).to(DEVICE)
            logger.info(f"Input created successfully: {inputs.keys()}")
        except Exception as e:
            logger.error(f"Error creating inputs: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to process audio: {str(e)}")
        
        # Generate transcription with improved error handling
        logger.info("Generating transcription...")
        try:
            # Set a timeout for generation to prevent hangs
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=DEVICE=="cuda"):
                logger.info("Starting model generation with no_grad and autocast")
                output_tokens = model.generate(
                    **inputs,
                    tgt_lang=target_language,
                    num_beams=1  # Use greedy decoding for faster inference
                )
                logger.info("Model generation completed")
            
            torch.cuda.empty_cache()  # Clear CUDA cache after generation
            logger.info(f"Generation successful: output shape={output_tokens.shape}")
        except RuntimeError as e:
            # Handle CUDA out of memory errors gracefully
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory error - try reducing batch size")
                torch.cuda.empty_cache()  # Clear CUDA cache
                raise ValueError("GPU memory exceeded. Try reducing audio length or batch size.")
            else:
                logger.error(f"Runtime error during model generation: {str(e)}")
                logger.error(traceback.format_exc())
                raise ValueError(f"Failed to generate transcription: {str(e)}")
        except Exception as e:
            logger.error(f"Error during model generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to generate transcription: {str(e)}")
        
        # Decode the tokens
        try:
            logger.info("Decoding tokens...")
            transcribed_text = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
            logger.info(f"Transcription result: '{transcribed_text[:50]}...' (length={len(transcribed_text)})")
        except Exception as e:
            logger.error(f"Error decoding tokens: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to decode transcription: {str(e)}")
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f}s")
        
        return {
            "text": transcribed_text,
            "processing_time": processing_time,
            "status": "completed"
        }
    except Exception as e:
        processing_time = time.time() - start_time
        error_message = str(e)
        logger.error(f"Failed to process audio: {error_message}")
        logger.error(traceback.format_exc())
        
        return {
            "text": "",
            "processing_time": processing_time,
            "status": "error",
            "error": error_message
        }

async def process_audio_task(file_path, job_id, target_language):
    """Background task to process audio and store results"""
    logger.info(f"--- STARTING BACKGROUND TASK ---")
    logger.info(f"Background task for job {job_id} with file {file_path}")
    
    try:
        # Check if file exists and is readable before processing
        if not os.path.exists(file_path):
            logger.error(f"Audio file does not exist: {file_path}")
            completed_jobs[job_id] = {
                "id": job_id,
                "text": "",
                "processing_time": 0,
                "status": "error",
                "error": f"Audio file not found: {file_path}"
            }
            return
        
        logger.info(f"Processing audio for job {job_id}")
        result = process_audio(file_path, target_language)
        result["id"] = job_id
        
        if result["status"] == "error":
            logger.error(f"Job {job_id} failed: {result.get('error', 'Unknown error')}")
        else:
            logger.info(f"Job {job_id} completed successfully")
        
        # Store the result
        logger.info(f"Storing result for job {job_id}")
        completed_jobs[job_id] = result
        logger.info(f"Result stored for job {job_id}")
        
    except Exception as e:
        logger.error(f"Unexpected error in process_audio_task for job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        completed_jobs[job_id] = {
            "id": job_id,
            "text": "",
            "processing_time": 0,
            "status": "error",
            "error": f"Task execution failed: {str(e)}"
        }
    finally:
        # Clean up temp file ONLY if processing was successful
        try:
            if job_id in completed_jobs and completed_jobs[job_id]["status"] != "error":
                if os.path.exists(file_path):
                    logger.info(f"Removing temporary file {file_path}")
                    os.remove(file_path)
                    logger.info(f"Temporary file {file_path} removed")
            else:
                if os.path.exists(file_path):
                     logger.warning(f"Processing failed for job {job_id}. Keeping temporary file for inspection: {file_path}")
        except Exception as e:
            logger.warning(f"Error during temporary file cleanup for {file_path}: {str(e)}")
        
        # Remove from processing queue
        if job_id in processing_queue:
            logger.info(f"Removing job {job_id} from processing queue")
            del processing_queue[job_id]
            logger.info(f"Job {job_id} removed from processing queue")
        
        logger.info(f"--- BACKGROUND TASK COMPLETED FOR JOB {job_id} --- ({time.time() - task_start_time:.2f}s total task time)")

# API Endpoints
@app.post("/load-model")
async def load_model_endpoint():
    """Endpoint to manually load the model"""
    logger.info("Manual model loading requested")
    if model is not None and processor is not None:
        logger.info("Model is already loaded")
        return {"status": "already_loaded", "message": "Model is already loaded"}
    
    result = await load_model()
    logger.info(f"Manual model loading result: {result}")
    return result

@app.post("/transcribe", response_model=QueuedResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: str = Form("amh")
):
    """
    Transcribe an audio file to text.
    Returns job ID immediately and processes in the background.
    """
    logger.info(f"Transcribe request received for file: {file.filename}, target: {target_language}")
    
    # Check if model is loaded
    if model is None or processor is None:
        logger.info("Model not loaded, attempting to load it now")
        # Try to load the model if it's not loaded yet
        load_result = await load_model()
        if load_result["status"] != "success" and load_result["status"] != "already_loaded":
            logger.error(f"Model could not be loaded: {load_result}")
            raise HTTPException(
                status_code=503, 
                detail="Model could not be loaded. Please try again later or use the /load-model endpoint first."
            )
        
    # Generate a unique ID for this job
    job_id = str(uuid.uuid4())
    logger.info(f"Generated job ID: {job_id}")
    
    # --- Get original file extension ---
    original_filename = file.filename
    file_extension = ""
    if original_filename and '.' in original_filename:
        file_extension = os.path.splitext(original_filename)[1].lower()
    # Use a default if no extension or unknown
    if not file_extension:
        file_extension = ".tmpaudio" # Use a generic suffix if unknown
    logger.info(f"Original filename: {original_filename}, Using suffix: {file_extension}")
    # -------------------------------------

    # Save the uploaded file temporarily using the original extension
    logger.info(f"Saving uploaded file to temp location with suffix {file_extension}")
    # Note: delete=False is important so the background task can access it
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        logger.info(f"Created temp file: {temp_file.name}")
        content = await file.read()
        logger.info(f"Read {len(content)} bytes from uploaded file")
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    logger.info(f"File saved to: {temp_file_path}")
    
    # Check if the file was saved correctly
    if os.path.exists(temp_file_path):
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Temporary file exists, size: {file_size} bytes")
    else:
        logger.error(f"Temporary file was not created successfully: {temp_file_path}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    # Add to processing queue
    logger.info(f"Adding job {job_id} to processing queue")
    processing_queue[job_id] = {
        "file_path": temp_file_path,
        "target_language": target_language,
        "start_time": time.time(),
        "status": "queued"
    }
    
    # Process in background
    logger.info(f"Adding background task for job {job_id}")
    background_tasks.add_task(process_audio_task, temp_file_path, job_id, target_language)
    logger.info(f"Background task added for job {job_id}")
    
    # Return the job ID immediately
    return QueuedResponse(
        id=job_id,
        status="processing",
        estimated_completion_time=30.0  # Estimate based on file size could be added
    )

@app.get("/status/{job_id}", response_model=None)  # Remove response_model to allow additional fields
async def check_status(job_id: str, debug: bool = False):
    """Check the status of a transcription job"""
    logger.info(f"Status check for job {job_id}, debug={debug}")
    
    # Check if job is completed
    if job_id in completed_jobs:
        result = completed_jobs[job_id]
        logger.info(f"Job {job_id} found in completed jobs, status: {result['status']}")
        
        # If debug mode is enabled and there was an error, include stack trace
        if debug and result["status"] == "error":
            logger.info("Returning detailed error information due to debug flag")
            return result
        
        # Otherwise, return the standard response
        return result
    
    # Check if job is in queue
    if job_id in processing_queue:
        logger.info(f"Job {job_id} is still in processing queue")
        time_in_queue = time.time() - processing_queue[job_id]["start_time"]
        logger.info(f"Job {job_id} has been in queue for {time_in_queue:.2f} seconds")
        
        return TranscriptionResponse(
            id=job_id,
            text="",
            processing_time=0.0,
            status="processing"
        )
    
    # Job not found
    logger.warning(f"Job {job_id} not found in either completed jobs or processing queue")
    raise HTTPException(status_code=404, detail="Job not found")

@app.post("/batch", response_model=List[QueuedResponse])
async def batch_process(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    target_language: str = Form("amh")
):
    """Process multiple audio files in batch"""
    logger.info(f"Batch process request received for {len(files)} files")
    
    # Check if model is loaded
    if model is None or processor is None:
        logger.info("Model not loaded for batch processing, attempting to load it now")
        # Try to load the model if it's not loaded yet
        load_result = await load_model()
        if load_result["status"] != "success" and load_result["status"] != "already_loaded":
            logger.error(f"Model could not be loaded for batch processing: {load_result}")
            raise HTTPException(
                status_code=503, 
                detail="Model could not be loaded. Please try again later or use the /load-model endpoint first."
            )
        
    responses = []
    
    for i, file in enumerate(files):
        logger.info(f"Processing batch file {i+1}/{len(files)}: {file.filename}")
        
        # Generate a unique ID for this job
        job_id = str(uuid.uuid4())
        logger.info(f"Generated job ID for batch item: {job_id}")
        
        # Save the uploaded file temporarily
        logger.info(f"Saving batch file to temp location")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            logger.info(f"Created temp file for batch item: {temp_file.name}")
            content = await file.read()
            logger.info(f"Read {len(content)} bytes from batch file")
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Add to processing queue
        logger.info(f"Adding batch job {job_id} to processing queue")
        processing_queue[job_id] = {
            "file_path": temp_file_path,
            "target_language": target_language,
            "start_time": time.time(),
            "status": "queued"
        }
        
        # Process in background
        logger.info(f"Adding background task for batch job {job_id}")
        background_tasks.add_task(process_audio_task, temp_file_path, job_id, target_language)
        logger.info(f"Background task added for batch job {job_id}")
        
        responses.append(QueuedResponse(
            id=job_id,
            status="processing"
        ))
    
    logger.info(f"Returning {len(responses)} batch job responses")
    return responses

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    
    # Get system information
    system_info = {
        "python_version": sys.version,
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
        }
    
    response = {
        "status": "healthy",
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "loading_complete": model is not None and processor is not None,
        "model_loading_in_progress": model_loading_lock,
        "jobs_in_queue": len(processing_queue),
        "completed_jobs": len(completed_jobs),
        "device": DEVICE,
        "system_info": system_info,
        "gpu_info": gpu_info,
        "timestamp": time.time()
    }
    
    # Log queue details for debugging
    if processing_queue:
        logger.info(f"Current processing queue ({len(processing_queue)} jobs):")
        for job_id, job_info in processing_queue.items():
            queue_time = time.time() - job_info["start_time"]
            logger.info(f"  - Job {job_id}: in queue for {queue_time:.2f}s")
    
    return response

@app.get("/queue-status")
async def queue_status():
    """Get detailed information about the processing queue"""
    logger.info("Queue status requested")
    
    queue_info = []
    for job_id, job_data in processing_queue.items():
        queue_time = time.time() - job_data["start_time"]
        queue_info.append({
            "job_id": job_id,
            "file_path": job_data["file_path"],
            "target_language": job_data["target_language"],
            "queue_time_seconds": queue_time,
            "status": job_data["status"]
        })
    
    completed_info = []
    for job_id, job_data in completed_jobs.items():
        completed_info.append({
            "job_id": job_id,
            "status": job_data["status"],
            "processing_time": job_data.get("processing_time", 0)
        })
    
    return {
        "queued_jobs": queue_info,
        "completed_jobs": completed_info,
        "queue_length": len(processing_queue),
        "completed_count": len(completed_jobs)
    }

@app.get("/debug-info")
async def debug_info():
    """Get detailed debug information"""
    logger.info("Debug info requested")
    
    # Check for temp files
    temp_dir = tempfile.gettempdir()
    temp_files = []
    try:
        for f in os.listdir(temp_dir):
            if f.endswith('.wav'):
                file_path = os.path.join(temp_dir, f)
                temp_files.append({
                    "file": f,
                    "size": os.path.getsize(file_path),
                    "mtime": os.path.getmtime(file_path)
                })
    except Exception as e:
        logger.error(f"Error checking temp files: {str(e)}")
    
    # System info
    sys_info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "pid": os.getpid(),
        "memory_info": {"Not available": "Use docker stats to check memory usage"},
        "temp_dir": temp_dir,
        "temp_wav_files": temp_files,
        "cwd": os.getcwd(),
        "env": {k: v for k, v in os.environ.items() if not k.lower().contains("key") and not k.lower().contains("secret") and not k.lower().contains("token")}
    }
    
    # Process info
    import psutil
    try:
        process = psutil.Process(os.getpid())
        process_info = {
            "cpu_percent": process.cpu_percent(),
            "memory_info": str(process.memory_info()),
            "create_time": process.create_time(),
            "status": process.status(),
            "threads": len(process.threads())
        }
    except Exception as e:
        process_info = {"error": str(e)}
    
    return {
        "system_info": sys_info,
        "process_info": process_info,
        "model_info": {
            "model_loaded": model is not None,
            "processor_loaded": processor is not None,
            "model_loading_lock": model_loading_lock,
            "device": DEVICE,
            "model_id": MODEL_ID
        },
        "queue_info": {
            "processing_queue_size": len(processing_queue),
            "completed_jobs_size": len(completed_jobs)
        }
    }

@app.get("/")
async def root():
    """API information"""
    logger.info("Root endpoint accessed")
    return {
        "name": "Seamless M4T API",
        "version": "1.0.0",
        "description": "API for Amharic speech-to-text translation",
        "endpoints": [
            {"path": "/load-model", "method": "POST", "description": "Manually load the model"},
            {"path": "/transcribe", "method": "POST", "description": "Transcribe audio to text"},
            {"path": "/status/{job_id}", "method": "GET", "description": "Check job status"},
            {"path": "/batch", "method": "POST", "description": "Batch process multiple files"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/queue-status", "method": "GET", "description": "View all jobs in queue"},
            {"path": "/debug-info", "method": "GET", "description": "Get detailed debug information"}
        ]
    }

# Run the server if executed directly
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    logger.info(f"Starting uvicorn server on port {port} with {workers} workers")
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=workers)