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

# Set multiprocessing start method to 'spawn' for CUDA compatibility
import multiprocessing

# Initialize multiprocessing method at module level
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # Already set

# Define a worker initialization function for Gunicorn
def _mp_fn(server=None):
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        multiprocessing.set_start_method('spawn')
        logger.info("Set multiprocessing start method to 'spawn' for CUDA compatibility")
    except RuntimeError:
        logger.info("Multiprocessing start method already set or couldn't be changed")
    
    # Initialize CUDA in the worker process if available
    if torch.cuda.is_available():
        try:
            # Initialize CUDA context for this worker
            torch.cuda.init()
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Worker initialized with CUDA: {device_count} device(s), using {device_name}")
            # Empty cache to start clean
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error initializing CUDA in worker: {str(e)}")

# Use spawn method for main process when run directly
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

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

# Use soundfile for faster audio loading instead of librosa where possible
try:
    import soundfile as sf
    USE_SOUNDFILE = True
except ImportError:
    USE_SOUNDFILE = False
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
    # Configure GPU memory optimization
    torch.cuda.empty_cache()
    # Enable TF32 precision for faster computation on Ampere GPUs (A100, A10G, A6000)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cuDNN benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info("GPU optimizations enabled: TF32, cuDNN benchmark")
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
        request_id = str(uuid.uuid4())[:8]  # Short unique ID for request tracking
        
        # Log request details
        logger.info(f"[{request_id}] Request received: {request.method} {request.url.path}")
        
        # Log additional information for debugging
        client_host = request.client.host if request.client else "unknown"
        headers = dict(request.headers.items())
        # Remove sensitive headers if any
        if "authorization" in headers:
            headers["authorization"] = "[REDACTED]"
        
        logger.info(f"[{request_id}] Client: {client_host}, Headers: {headers}")
        
        # For multipart uploads, log form fields and file details
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            logger.info(f"[{request_id}] Detected multipart/form-data upload request")
            try:
                # We can't consume the body yet, as it would prevent processing by the route handlers
                logger.info(f"[{request_id}] Content-Length: {request.headers.get('content-length')}")
            except Exception as e:
                logger.error(f"[{request_id}] Error examining multipart request: {str(e)}")
        
        try:
            # Process the request
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response details
            logger.info(f"[{request_id}] Request completed: {request.method} {request.url.path} - " 
                        f"Status: {response.status_code} - Took: {process_time:.4f}s")
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"[{request_id}] Request failed: {request.method} {request.url.path} - " 
                         f"Error: {str(e)} - Took: {process_time:.4f}s")
            logger.error(f"[{request_id}] {traceback.format_exc()}")
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
# Instead of large model, consider smaller model for faster inference
DEFAULT_MODEL_ID = os.getenv("MODEL_PATH", "./models/seamless-m4t-v2-large")
MODEL_ID = DEFAULT_MODEL_ID
LOCAL_MODEL = not MODEL_ID.startswith(("facebook/", "http://", "https://"))  # Check if it's a local path
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))  # Increased batch size for better throughput
MAX_AUDIO_LENGTH = int(os.getenv("MAX_AUDIO_LENGTH", "300"))  # Max audio length in seconds
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL_ON_STARTUP = os.getenv("LOAD_MODEL_ON_STARTUP", "false").lower() == "true"
USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "true").lower() == "true"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "5"))  # Chunk size in seconds
CHUNK_OVERLAP = float(os.getenv("CHUNK_OVERLAP", "0.5"))  # Chunk overlap in seconds

# Log important configuration settings
logger.info(f"MODEL_ID: {MODEL_ID}")
logger.info(f"LOCAL_MODEL: {LOCAL_MODEL}")
logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
logger.info(f"MAX_AUDIO_LENGTH: {MAX_AUDIO_LENGTH}")
logger.info(f"DEVICE: {DEVICE}")
logger.info(f"LOAD_MODEL_ON_STARTUP: {LOAD_MODEL_ON_STARTUP}")
logger.info(f"USE_QUANTIZATION: {USE_QUANTIZATION}")
logger.info(f"USE_TORCH_COMPILE: {USE_TORCH_COMPILE}")
logger.info(f"CHUNK_SIZE: {CHUNK_SIZE}s, CHUNK_OVERLAP: {CHUNK_OVERLAP}s")

# In-memory queues for processing and results
processing_queue = {}
completed_jobs = {}

# For storing the model and processor
processor = None
model = None
model_loading_lock = False
model_graph_captured = False
cuda_graph = None
dummy_input = None

# Initialize CUDA streams for parallel processing
cuda_streams = {}
if torch.cuda.is_available():
    cuda_streams = {
        "preprocessing": torch.cuda.Stream(),
        "inference": torch.cuda.Stream()
    }

async def load_model():
    global processor, model, model_loading_lock, model_graph_captured, cuda_graph, dummy_input
    
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
        
        # Check if we're in offline mode
        is_offline = os.getenv("TRANSFORMERS_OFFLINE", "0").lower() in ("1", "true")
        local_files_only = is_offline
        logger.info(f"Operating in {'offline' if is_offline else 'online'} mode (local_files_only={local_files_only})")
        
        # Load processor first
        try:
            logger.info("Loading processor...")
            # Handle local model loading properly
            if LOCAL_MODEL:
                logger.info(f"Loading processor from local path: {MODEL_ID}")
                
                # Check for processor files
                model_files = os.listdir(MODEL_ID) if os.path.exists(MODEL_ID) else []
                logger.info(f"Found {len(model_files)} files in model directory: {', '.join(model_files[:5])}{'...' if len(model_files) > 5 else ''}")
                
                # Check if there's a processor config or model files
                processor_config_path = os.path.join(MODEL_ID, "processor_config.json")
                tokenizer_config_path = os.path.join(MODEL_ID, "tokenizer_config.json")
                feature_extractor_path = os.path.join(MODEL_ID, "feature_extractor_config.json")
                
                # First try: Check if main processor configs exist
                if os.path.exists(processor_config_path):
                    logger.info(f"Found processor config, loading from {MODEL_ID}")
                    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
                # Second try: Check if tokenizer config exists
                elif os.path.exists(tokenizer_config_path) or os.path.exists(feature_extractor_path):
                    logger.info(f"Found tokenizer/feature extractor config, loading from {MODEL_ID}")
                    # Additional debug info about the directory contents
                    logger.info(f"Model directory contents: {', '.join(model_files)}")
                    
                    try:
                        from transformers import SeamlessM4TProcessor, AutoTokenizer, AutoFeatureExtractor
                        
                        # Use specific processor class if possible
                        if 'seamless-m4t' in MODEL_ID:
                            logger.info("Using SeamlessM4TProcessor for better compatibility")
                            processor = SeamlessM4TProcessor.from_pretrained(MODEL_ID, local_files_only=True)
                        else:
                            # Manually construct processor from components
                            logger.info("Attempting to manually construct processor from components")
                            tokenizer = None
                            feature_extractor = None
                            
                            if os.path.exists(tokenizer_config_path):
                                logger.info("Loading tokenizer...")
                                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
                            
                            if os.path.exists(feature_extractor_path):
                                logger.info("Loading feature extractor...")
                                feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, local_files_only=True)
                            
                            if tokenizer and feature_extractor:
                                # Create processor
                                processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
                            else:
                                raise ValueError("Could not load tokenizer and feature extractor components")
                    except Exception as component_err:
                        logger.error(f"Error loading processor components: {str(component_err)}")
                        if not is_offline:
                            # Try downloading if we're online
                            logger.warning("Attempting to download processor from HuggingFace...")
                            processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large", local_files_only=False)
                            processor.save_pretrained(MODEL_ID)
                        else:
                            raise
                elif not is_offline:
                    # Try the facebook model ID as fallback if we're not in offline mode
                    logger.warning(f"No processor config found at {MODEL_ID}, downloading from facebook/seamless-m4t-v2-large")
                    try:
                        # Try downloading from HF with trace
                        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large", local_files_only=False)
                        # Save it locally for future use
                        processor.save_pretrained(MODEL_ID)
                        logger.info(f"Saved processor to {MODEL_ID} for future use")
                    except Exception as e:
                        logger.error(f"Error downloading processor: {str(e)}")
                        raise ValueError(f"No processor config found at {MODEL_ID} and could not download: {str(e)}")
                else:
                    # We're in offline mode with no processor config
                    raise ValueError(
                        f"No processor config found at {MODEL_ID} and offline mode is enabled. "
                        f"Please run download_model.py to fetch the model before running in offline mode."
                    )
            else:
                # Using online model ID
                processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=local_files_only)
                
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
                    torch_dtype=torch.float32,  # Use float32 instead of half precision on M1
                    local_files_only=local_files_only
                ).to(DEVICE)
            else:
                # For GPU, use half precision and better memory optimization
                logger.info("Using GPU optimization with float16 precision")
                
                # Handle the case where a locally saved model might not exist yet
                model_config_path = os.path.join(MODEL_ID, "config.json")
                if os.path.exists(model_config_path):
                    logger.info(f"Loading model from local path: {MODEL_ID}")
                    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                        MODEL_ID,
                        torch_dtype=torch.float16 if DEVICE == "cuda" else None,
                        low_cpu_mem_usage=True,
                        local_files_only=True
                    ).to(DEVICE)
                elif not is_offline:
                    # Try the facebook model ID as fallback if we're not in offline mode
                    logger.warning(f"No model config found at {model_config_path}, using facebook/seamless-m4t-v2-medium")
                    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                        "facebook/seamless-m4t-v2-medium",
                        torch_dtype=torch.float16 if DEVICE == "cuda" else None,
                        low_cpu_mem_usage=True,
                        local_files_only=False
                    ).to(DEVICE)
                    # Save the model to the local path for future use
                    model.save_pretrained(MODEL_ID)
                    logger.info(f"Saved model to {MODEL_ID} for future use")
                else:
                    # We're in offline mode with no model config
                    raise ValueError(
                        f"No model config found at {MODEL_ID} and offline mode is enabled. "
                        f"Please run download_model.py to fetch the model before running in offline mode."
                    )
                    
                # Apply INT8 quantization if enabled (on supported GPUs)
                if USE_QUANTIZATION and DEVICE == "cuda":
                    try:
                        logger.info("Applying dynamic INT8 quantization to model")
                        model = torch.quantization.quantize_dynamic(
                            model, 
                            {torch.nn.Linear}, 
                            dtype=torch.qint8
                        )
                        logger.info("Quantization applied successfully")
                    except Exception as e:
                        logger.warning(f"Quantization failed, continuing with FP16 model: {str(e)}")
                
                # Apply model compilation with torch.compile if available (PyTorch 2.0+)
                if USE_TORCH_COMPILE and DEVICE == "cuda" and hasattr(torch, 'compile'):
                    try:
                        logger.info("Applying torch.compile optimization")
                        model = torch.compile(model, mode='reduce-overhead')
                        logger.info("Model compilation successful")
                    except Exception as e:
                        logger.warning(f"Model compilation failed: {str(e)}")
                        
                        # Fallback to TorchScript JIT compilation
                        try:
                            logger.info("Falling back to TorchScript JIT compilation")
                            model = torch.jit.script(model)
                            logger.info("TorchScript compilation successful")
                        except Exception as e2:
                            logger.warning(f"TorchScript compilation failed: {str(e2)}")
                
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

# Optimized audio loading function
def load_audio(audio_path, target_sr=16000):
    """Load audio file with the fastest available method"""
    try:
        if USE_SOUNDFILE:
            # Use soundfile for faster loading
            audio, sample_rate = sf.read(audio_path)
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            # Convert stereo to mono if needed
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = audio.mean(axis=1)
        else:
            # Fallback to librosa
            audio, sample_rate = librosa.load(audio_path, sr=None)
        
        # Resample if needed
        if sample_rate != target_sr:
            if USE_SOUNDFILE:
                # Faster resampling with numpy
                from scipy import signal
                audio = signal.resample_poly(audio, target_sr, sample_rate)
            else:
                audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
            
        return audio, sample_rate
    except Exception as e:
        logger.error(f"Error loading audio: {str(e)}")
        raise ValueError(f"Could not load audio file: {str(e)}")

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
        
        # Load audio file with optimized method
        try:
            logger.info(f"Loading audio file: {audio_path}")
            audio, sample_rate = load_audio(audio_path, target_sr=16000)
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
        
        # Process audio in chunks for long files
        if audio_length > CHUNK_SIZE:
            return process_audio_in_chunks(audio, sample_rate, target_language)
        
        # Convert audio to tensor and move directly to GPU with pinned memory for faster transfer
        if DEVICE == "cuda":
            with torch.cuda.stream(cuda_streams["preprocessing"]):
                # Use pinned memory for faster CPU-to-GPU transfer
                audio_tensor = torch.tensor(audio).pin_memory().to(DEVICE, non_blocking=True)
        else:
            audio_tensor = torch.tensor(audio).to(DEVICE)
        
        # Process the audio
        logger.info("Creating input tensors...")
        try:
            if DEVICE == "cuda":
                with torch.cuda.stream(cuda_streams["preprocessing"]):
                    inputs = processor(audios=audio_tensor, sampling_rate=sample_rate, return_tensors="pt", padding=True).to(DEVICE)
            else:
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
            if DEVICE == "cuda":
                # Wait for preprocessing to complete
                torch.cuda.current_stream().wait_stream(cuda_streams["preprocessing"])
                
            # Use inference mode (faster than no_grad) and autocast for mixed precision
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=DEVICE=="cuda"):
                logger.info("Starting model generation with inference_mode and autocast")
                
                # Try to use CUDA graph if previously captured (for repeated similar-sized inputs)
                if model_graph_captured and cuda_graph is not None and dummy_input is not None:
                    try:
                        # Check if current input is compatible with captured graph
                        if all(inputs[k].shape == dummy_input[k].shape for k in inputs if k in dummy_input):
                            logger.info("Using captured CUDA graph for inference")
                            # Copy inputs to dummy inputs
                            for k in inputs:
                                if k in dummy_input:
                                    dummy_input[k].copy_(inputs[k])
                            # Replay graph
                            cuda_graph.replay()
                            # Extract result
                            output_tokens = model.generate(
                                **inputs,
                                tgt_lang=target_language,
                                num_beams=1  # Use greedy decoding for faster inference
                            )
                        else:
                            logger.info("Input shape mismatch, falling back to standard inference")
                            output_tokens = model.generate(
                                **inputs,
                                tgt_lang=target_language,
                                num_beams=1  # Use greedy decoding for faster inference
                            )
                    except Exception as e:
                        logger.warning(f"CUDA graph replay failed: {str(e)}, falling back to standard inference")
                        output_tokens = model.generate(
                            **inputs,
                            tgt_lang=target_language,
                            num_beams=1  # Use greedy decoding for faster inference
                        )
                else:
                    # Standard inference path
                    output_tokens = model.generate(
                        **inputs,
                        tgt_lang=target_language,
                        num_beams=1  # Use greedy decoding for faster inference
                    )
                
                # Try to capture CUDA graph for future runs
                if not model_graph_captured and DEVICE == "cuda" and torch.cuda.is_available():
                    try:
                        logger.info("Attempting to capture CUDA graph for future inference")
                        # Clone inputs for graph capture
                        dummy_input = {k: v.clone() for k, v in inputs.items()}
                        # Create CUDA graph
                        cuda_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(cuda_graph):
                            model.generate(
                                **dummy_input,
                                tgt_lang=target_language,
                                num_beams=1
                            )
                        model_graph_captured = True
                        logger.info("CUDA graph captured successfully")
                    except Exception as e:
                        logger.warning(f"Failed to capture CUDA graph: {str(e)}")
                        model_graph_captured = False
                
                logger.info("Model generation completed")
            
            # Clear CUDA cache after generation
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            
            logger.info(f"Generation successful: output shape={output_tokens.shape}")
        except RuntimeError as e:
            # Handle CUDA out of memory errors gracefully
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory error - try reducing batch size")
                if DEVICE == "cuda":
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

def process_audio_in_chunks(audio, sample_rate, target_language="amh"):
    """Process longer audio files in chunks with overlap for better results"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing audio in chunks: length={len(audio)/sample_rate:.2f}s")
        
        # Calculate chunk sizes in samples
        chunk_size_samples = int(CHUNK_SIZE * sample_rate)
        overlap_samples = int(CHUNK_OVERLAP * sample_rate)
        stride = chunk_size_samples - overlap_samples
        
        # Split audio into overlapping chunks
        audio_chunks = []
        for i in range(0, len(audio), stride):
            chunk = audio[i:i + chunk_size_samples]
            # Only process chunks that are long enough
            if len(chunk) > 0.5 * sample_rate:  # At least 0.5 seconds
                audio_chunks.append(chunk)
        
        logger.info(f"Split audio into {len(audio_chunks)} chunks")
        
        # Process each chunk and collect results
        results = []
        for i, chunk in enumerate(audio_chunks):
            logger.info(f"Processing chunk {i+1}/{len(audio_chunks)}")
            
            # Convert audio chunk to tensor
            if DEVICE == "cuda":
                with torch.cuda.stream(cuda_streams["preprocessing"]):
                    # Use pinned memory for faster transfer
                    chunk_tensor = torch.tensor(chunk).pin_memory().to(DEVICE, non_blocking=True)
            else:
                chunk_tensor = torch.tensor(chunk).to(DEVICE)
            
            # Process chunk
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=DEVICE=="cuda"):
                # Create inputs
                inputs = processor(audios=chunk_tensor, sampling_rate=sample_rate, return_tensors="pt").to(DEVICE)
                
                # Generate
                output_tokens = model.generate(
                    **inputs,
                    tgt_lang=target_language,
                    num_beams=1  # Use greedy decoding for faster inference
                )
                
                # Decode
                chunk_text = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
                results.append(chunk_text)
        
        # Combine results
        # This is a simple concatenation - in production you might want 
        # more sophisticated text merging to handle overlaps
        transcribed_text = " ".join(results)
        
        processing_time = time.time() - start_time
        logger.info(f"Chunk processing completed in {processing_time:.2f}s")
        
        return {
            "text": transcribed_text,
            "processing_time": processing_time,
            "status": "completed"
        }
    except Exception as e:
        processing_time = time.time() - start_time
        error_message = str(e)
        logger.error(f"Failed to process audio chunks: {error_message}")
        logger.error(traceback.format_exc())
        
        return {
            "text": "",
            "processing_time": processing_time,
            "status": "error",
            "error": error_message
        }

# Define task start time as a global variable
task_start_time = 0

async def process_audio_task(file_path, job_id, target_language):
    """Background task to process audio and store results"""
    global task_start_time
    task_start_time = time.time()
    
    logger.info(f"===================== BACKGROUND TASK STARTED FOR JOB {job_id} =====================")
    logger.info(f"Background task for job {job_id} with file {file_path}")
    
    try:
        # Check if file exists and is readable before processing
        if not os.path.exists(file_path):
            logger.error(f"CRITICAL: Audio file does not exist: {file_path}")
            completed_jobs[job_id] = {
                "id": job_id,
                "text": "",
                "processing_time": 0,
                "status": "error",
                "error": f"Audio file not found: {file_path}"
            }
            return
        
        # Get file information
        file_size = os.path.getsize(file_path)
        file_permissions = oct(os.stat(file_path).st_mode)
        logger.info(f"File details: Size={file_size} bytes, Permissions={file_permissions}")
        
        # Check if file is accessible
        if not os.access(file_path, os.R_OK):
            logger.error(f"CRITICAL: Cannot read audio file (permission denied): {file_path}")
            completed_jobs[job_id] = {
                "id": job_id,
                "text": "",
                "processing_time": 0,
                "status": "error",
                "error": f"Cannot read audio file (permission denied): {file_path}"
            }
            return
        
        # Check model state in the background task
        logger.info(f"Model state in background task - Model: {'Loaded' if model is not None else 'Not loaded'}, Processor: {'Loaded' if processor is not None else 'Not loaded'}")
        
        # Print GPU memory status before processing
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU memory before processing: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB")
        
        logger.info(f"Processing audio for job {job_id}")
        processing_start_time = time.time()
        result = process_audio(file_path, target_language)
        processing_time = time.time() - processing_start_time
        logger.info(f"Audio processing completed in {processing_time:.2f} seconds")
        result["id"] = job_id
        
        if result["status"] == "error":
            logger.error(f"Job {job_id} failed: {result.get('error', 'Unknown error')}")
        else:
            logger.info(f"Job {job_id} completed successfully")
            # Log a sample of the transcription
            transcription = result.get("text", "")
            transcription_sample = transcription[:100] + "..." if len(transcription) > 100 else transcription
            logger.info(f"Transcription sample: {transcription_sample}")
        
        # Store the result
        logger.info(f"Storing result for job {job_id}")
        completed_jobs[job_id] = result
        logger.info(f"Result stored for job {job_id}")
        
    except Exception as e:
        logger.error(f"CRITICAL: Unexpected error in process_audio_task for job {job_id}: {str(e)}")
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
            if job_id in completed_jobs:
                if completed_jobs[job_id]["status"] != "error":
                    if os.path.exists(file_path):
                        logger.info(f"Removing temporary file {file_path}")
                        os.remove(file_path)
                        logger.info(f"Temporary file {file_path} removed")
                else:
                    if os.path.exists(file_path):
                        logger.warning(f"Processing failed for job {job_id}. Keeping temporary file for inspection: {file_path}")
            else:
                logger.warning(f"Job {job_id} not in completed_jobs, something went wrong")
        except Exception as e:
            logger.warning(f"Error during temporary file cleanup for {file_path}: {str(e)}")
            logger.warning(traceback.format_exc())
        
        # Remove from processing queue
        if job_id in processing_queue:
            logger.info(f"Removing job {job_id} from processing queue")
            del processing_queue[job_id]
            logger.info(f"Job {job_id} removed from processing queue")
        else:
            logger.warning(f"Job {job_id} not found in processing queue when trying to remove it")
        
        # Print GPU memory status after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clean up GPU memory
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU memory after processing: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB")
        
        logger.info(f"===================== BACKGROUND TASK COMPLETED FOR JOB {job_id} =====================")
        logger.info(f"Total task time: {time.time() - task_start_time:.2f}s")

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

# Debug endpoint for testing file uploads
@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Simple endpoint to test if file uploads are working properly"""
    logger.info(f"Test upload received for file: {file.filename}")
    
    try:
        # Get file info
        content = await file.read()
        file_size = len(content)
        
        # Get content type and extension
        content_type = file.content_type or "unknown"
        file_extension = ""
        if file.filename and '.' in file.filename:
            file_extension = os.path.splitext(file.filename)[1].lower()
        
        # Log detailed information
        logger.info(f"Received file upload:")
        logger.info(f"  Filename: {file.filename}")
        logger.info(f"  Size: {file_size} bytes")
        logger.info(f"  Content-Type: {content_type}")
        logger.info(f"  Extension: {file_extension}")
        
        # Save the file temporarily to verify file system access
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"test_upload_{int(time.time())}{file_extension}")
        
        with open(temp_path, "wb") as temp_file:
            # Reset file pointer and write the content
            await file.seek(0)
            content = await file.read()
            temp_file.write(content)
        
        logger.info(f"Successfully saved test file to: {temp_path}")
        
        # Get system file info
        if os.path.exists(temp_path):
            stat_info = os.stat(temp_path)
            logger.info(f"  File saved size: {stat_info.st_size} bytes")
            logger.info(f"  File permissions: {oct(stat_info.st_mode)}")
            # Clean up
            os.remove(temp_path)
            logger.info(f"  Test file removed")
        
        return {
            "status": "success",
            "message": "File upload test successful",
            "file_info": {
                "filename": file.filename,
                "size": file_size,
                "content_type": content_type,
                "extension": file_extension
            }
        }
    except Exception as e:
        logger.error(f"Error in test-upload endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"File upload test failed: {str(e)}",
            "error_details": str(e)
        }

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
    logger.info("===================== TRANSCRIBE REQUEST STARTED =====================")
    logger.info(f"Transcribe request received - HTTP Headers: {file.headers}")
    logger.info(f"File info - Filename: {file.filename}, Content-Type: {file.content_type}")
    logger.info(f"Target language: {target_language}")
    
    try:
        # Check if model is loaded
        logger.info("Checking if model is loaded...")
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
            logger.info("Model loaded successfully")
        else:
            logger.info("Model already loaded")
        
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

        # Try to read file data with detailed logging
        logger.info("Attempting to read file data...")
        try:
            # Read first 100 bytes to check if file is readable
            file_preview = await file.read(100)
            file_preview_len = len(file_preview)
            logger.info(f"Successfully read first {file_preview_len} bytes from file")
            
            # Reset file position to beginning
            await file.seek(0)
            logger.info("File seek(0) successful")
        except Exception as read_error:
            logger.error(f"CRITICAL: Error reading file data: {str(read_error)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(read_error)}")

        # Save the uploaded file temporarily using the original extension
        logger.info(f"Saving uploaded file to temp location with suffix {file_extension}")
        try:
            # Note: delete=False is important so the background task can access it
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                logger.info(f"Created temp file: {temp_file.name}")
                # Read the full file content
                content = await file.read()
                content_len = len(content)
                logger.info(f"Read {content_len} bytes from uploaded file")
                
                # Write the content to the temp file
                temp_file.write(content)
                temp_file_path = temp_file.name
                logger.info(f"File saved to: {temp_file_path}")
                
                # Verify the file was written correctly
                temp_file.flush()
                actual_size = os.path.getsize(temp_file_path)
                logger.info(f"Verified file size on disk: {actual_size} bytes")
                
                if actual_size != content_len:
                    logger.warning(f"WARNING: File size mismatch! Read {content_len} bytes but saved {actual_size} bytes")
        except Exception as save_error:
            logger.error(f"CRITICAL: Error saving file: {str(save_error)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {str(save_error)}")
        
        # Check if the file was saved correctly
        if os.path.exists(temp_file_path):
            file_size = os.path.getsize(temp_file_path)
            file_permissions = oct(os.stat(temp_file_path).st_mode)
            logger.info(f"Temporary file exists, size: {file_size} bytes, permissions: {file_permissions}")
            
            # Try to get file MIME type if possible
            try:
                import magic
                file_mime = magic.from_file(temp_file_path, mime=True)
                logger.info(f"File MIME type: {file_mime}")
            except ImportError:
                logger.info("python-magic not installed, skipping MIME type detection")
            except Exception as e:
                logger.warning(f"Could not detect MIME type: {str(e)}")
        else:
            logger.error(f"CRITICAL: Temporary file was not created successfully: {temp_file_path}")
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
        try:
            background_tasks.add_task(process_audio_task, temp_file_path, job_id, target_language)
            logger.info(f"Background task added for job {job_id}")
        except Exception as task_error:
            logger.error(f"CRITICAL: Error adding background task: {str(task_error)}")
            logger.error(traceback.format_exc())
            # Clean up the queue entry
            if job_id in processing_queue:
                del processing_queue[job_id]
            raise HTTPException(status_code=500, detail=f"Error scheduling audio processing: {str(task_error)}")
        
        logger.info("===================== TRANSCRIBE REQUEST COMPLETED =====================")
        # Return the job ID immediately
        return QueuedResponse(
            id=job_id,
            status="processing",
            estimated_completion_time=15.0  # Reduced estimation with optimizations
        )
    except HTTPException:
        logger.error("Transcribe request failed with HTTPException")
        raise
    except Exception as e:
        logger.error(f"CRITICAL UNHANDLED EXCEPTION in transcribe endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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
    
    # Process files in chunks to allow for better memory management
    batch_size = min(BATCH_SIZE, len(files))
    logger.info(f"Using batch size of {batch_size} for {len(files)} files")
    
    for i, file in enumerate(files):
        logger.info(f"Processing batch file {i+1}/{len(files)}: {file.filename}")
        
        # Generate a unique ID for this job
        job_id = str(uuid.uuid4())
        logger.info(f"Generated job ID for batch item: {job_id}")
        
        # Get original file extension
        original_filename = file.filename
        file_extension = ""
        if original_filename and '.' in original_filename:
            file_extension = os.path.splitext(original_filename)[1].lower()
        if not file_extension:
            file_extension = ".tmpaudio"
        
        # Save the uploaded file temporarily
        logger.info(f"Saving batch file to temp location with suffix {file_extension}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
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
            status="processing",
            estimated_completion_time=15.0  # Reduced estimation with optimizations
        ))
    
    logger.info(f"Returning {len(responses)} batch job responses")
    return responses

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    
    try:
        # Simplified response to avoid potential errors
        basic_health = {
            "status": "healthy",
            "model_loaded": model is not None,
            "processor_loaded": processor is not None,
            "device": DEVICE,
            "timestamp": time.time()
        }
        
        # Only add GPU info if it's available and not causing errors
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                basic_health["gpu"] = gpu_name
            except Exception as e:
                logger.error(f"Error getting GPU info: {str(e)}")
                basic_health["gpu_error"] = str(e)
        
        logger.info(f"Health check response: {basic_health}")
        return basic_health
    except Exception as e:
        # Log any unexpected errors
        logger.error(f"CRITICAL ERROR in health endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return a simplified error response
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

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
            if f.endswith(('.wav', '.mp3', '.tmpaudio')):
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
        "temp_audio_files": temp_files,
        "cwd": os.getcwd(),
        "env": {k: v for k, v in os.environ.items() 
                if not any(sensitive in k.lower() for sensitive in ["key", "secret", "token", "pass"])}
    }
    
    # Process info
    try:
        import psutil
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
    
    # GPU specific info
    gpu_info = {}
    if torch.cuda.is_available():
        try:
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
                "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
                "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
                "device_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cuda_graph_captured": model_graph_captured,
                "quantization_enabled": USE_QUANTIZATION,
                "torch_compile_enabled": USE_TORCH_COMPILE
            }
        except Exception as e:
            gpu_info = {"error": str(e)}
    
    # Model info
    model_info = {
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "model_loading_lock": model_loading_lock,
        "device": DEVICE,
        "model_id": MODEL_ID,
        "model_type": "unknown" if model is None else type(model).__name__,
        "model_parameters": "unknown" if model is None else sum(p.numel() for p in model.parameters())
    }
    
    return {
        "system_info": sys_info,
        "process_info": process_info,
        "gpu_info": gpu_info,
        "model_info": model_info,
        "optimization_settings": {
            "use_quantization": USE_QUANTIZATION,
            "use_torch_compile": USE_TORCH_COMPILE,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "batch_size": BATCH_SIZE
        },
        "queue_info": {
            "processing_queue_size": len(processing_queue),
            "completed_jobs_size": len(completed_jobs)
        }
    }

@app.get("/optimization-info")
async def optimization_info():
    """Get information about implemented optimizations"""
    logger.info("Optimization info requested")
    
    return {
        "model_optimizations": {
            "model_size": f"Using {MODEL_ID} model variant",
            "quantization": f"INT8 quantization {'enabled' if USE_QUANTIZATION else 'disabled'}",
            "model_compilation": f"PyTorch compilation {'enabled' if USE_TORCH_COMPILE else 'disabled'}"
        },
        "gpu_optimizations": {
            "cuda_streams": "Using separate CUDA streams for preprocessing and inference",
            "cuda_graphs": f"CUDA graph capture {'active' if model_graph_captured else 'inactive'}",
            "precision": "Using mixed precision (FP16) on GPU" if DEVICE == "cuda" else "N/A",
            "memory_pinning": "Using pinned memory for fast transfers" if DEVICE == "cuda" else "N/A",
            "tf32_enabled": torch.backends.cuda.matmul.allow_tf32 if DEVICE == "cuda" else "N/A",
            "cudnn_benchmark": torch.backends.cudnn.benchmark if DEVICE == "cuda" else "N/A"
        },
        "audio_processing": {
            "fast_loading": f"Using {'SoundFile' if USE_SOUNDFILE else 'librosa'} for audio loading",
            "chunked_processing": f"Enabled for files > {CHUNK_SIZE}s with {CHUNK_OVERLAP}s overlap"
        },
        "batch_processing": {
            "batch_size": BATCH_SIZE,
            "worker_count": int(os.getenv("WORKERS", "1"))
        },
        "expected_performance": {
            "estimated_speedup": "4-10x compared to baseline",
            "estimated_processing_time": "2-5 seconds for 7-second audio clips"
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
        "optimization_status": "Enhanced with performance optimizations",
        "endpoints": [
            {"path": "/load-model", "method": "POST", "description": "Manually load the model"},
            {"path": "/transcribe", "method": "POST", "description": "Transcribe audio to text"},
            {"path": "/status/{job_id}", "method": "GET", "description": "Check job status"},
            {"path": "/batch", "method": "POST", "description": "Batch process multiple files"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/queue-status", "method": "GET", "description": "View all jobs in queue"},
            {"path": "/debug-info", "method": "GET", "description": "Get detailed debug information"},
            {"path": "/optimization-info", "method": "GET", "description": "View implemented optimizations"}
        ]
    }

# Run the server if executed directly
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    logger.info(f"Starting uvicorn server on port {port} with {workers} workers")
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=workers)