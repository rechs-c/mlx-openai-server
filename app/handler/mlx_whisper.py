import gc
import os
import time
import uuid
import asyncio
import tempfile
from http import HTTPStatus
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from loguru import logger

from app.models.mlx_whisper import MLX_Whisper
from app.core.queue import RequestQueue
from app.schemas.openai import TranscriptionRequest, TranscriptionResponse, TranscriptionUsageAudio
from app.utils.errors import create_error_response


class MLXWhisperHandler:
    """
    Handler class for making requests to the underlying MLX Whisper model service.
    Provides request queuing, metrics tracking, and robust error handling for audio transcription.
    """

    def __init__(self, model_path: str, max_concurrency: int = 1):
        """
        Initialize the handler with the specified model path.
        
        Args:
            model_path (str): Path to the model directory.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
        """
        self.model_path = model_path
        self.model = MLX_Whisper(model_path)
        self.model_created = int(time.time())  # Store creation time when model is loaded
        
        # Initialize request queue for audio transcription tasks
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)

        logger.info(f"Initialized MLXWhisperHandler with model path: {model_path}")
    
    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models with their metadata.
        """
        return [{
            "id": self.model_path,
            "object": "model",
            "created": self.model_created,
            "owned_by": "local"
        }]
    
    async def initialize(self, queue_config: Optional[Dict[str, Any]] = None):
        """Initialize the handler and start the request queue."""
        if not queue_config:
            queue_config = {
                "max_concurrency": 1,
                "timeout": 600,  # Longer timeout for audio processing
                "queue_size": 50
            }
        self.request_queue = RequestQueue(
            max_concurrency=queue_config.get("max_concurrency"),
            timeout=queue_config.get("timeout"),
            queue_size=queue_config.get("queue_size")
        )
        await self.request_queue.start(self._process_request)
        logger.info("Initialized MLXWhisperHandler and started request queue")

    async def transcribe_audio(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """
        Transcribe audio from the provided file.
        Uses the request queue for handling concurrent requests.
        
        Args:
            request: TranscriptionRequest object containing the audio file and parameters.
        
        Returns:
            TranscriptionResponse: Response containing the transcribed text and usage info.
        """
        request_id = f"transcription-{uuid.uuid4()}"
        temp_audio_path = None
        
        try:
            # Save uploaded file to a temporary location
            temp_audio_path = await self._save_uploaded_file(request.file)
            
            # Prepare request data
            request_data = await self._prepare_transcription_request(request, temp_audio_path)
            
            # Submit to the request queue
            response = await self.request_queue.submit(request_id, request_data)
            
            # Calculate audio duration for usage tracking
            audio_duration = self._calculate_audio_duration(temp_audio_path)
            
            # Create response
            transcription_response = TranscriptionResponse(
                text=response["text"],
                usage=TranscriptionUsageAudio(
                    type="duration",
                    seconds=audio_duration
                )
            )
            
            return transcription_response
            
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response(
                "Too many requests. Service is at capacity.", 
                "rate_limit_exceeded", 
                HTTPStatus.TOO_MANY_REQUESTS
            )
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            logger.error(f"Error in audio transcription for request {request_id}: {str(e)}")
            content = create_error_response(
                f"Failed to transcribe audio: {str(e)}", 
                "server_error", 
                HTTPStatus.INTERNAL_SERVER_ERROR
            )
            raise HTTPException(status_code=500, detail=content)
        finally:
            # Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    logger.debug(f"Cleaned up temporary audio file: {temp_audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_audio_path}: {str(e)}")

    async def _process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an audio transcription request. This is the worker function for the request queue.
        
        Args:
            request_data: Dictionary containing the request data.
            
        Returns:
            Dict: The model's response containing transcribed text.
        """
        try:
            # Extract request parameters
            audio_path = request_data.pop("audio_path")
            
            # Call the model with the audio file
            result = self.model(
                audio_path=audio_path,
                **request_data
            )
            
            # Force garbage collection after model inference
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio transcription request: {str(e)}")
            # Clean up on error
            gc.collect()
            raise

    async def _save_uploaded_file(self, file) -> str:
        """
        Save the uploaded file to a temporary location.
        
        Args:
            file: The uploaded file object.
            
        Returns:
            str: Path to the temporary file.
        """
        try:
            # Create a temporary file with the same extension as the uploaded file
            file_extension = os.path.splitext(file.filename)[1] if file.filename else ".wav"
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                # Read and write the file contents
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name
            
            logger.debug(f"Saved uploaded file to temporary location: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            raise

    async def _prepare_transcription_request(
        self, 
        request: TranscriptionRequest, 
        audio_path: str
    ) -> Dict[str, Any]:
        """
        Prepare a transcription request by parsing model parameters.
        
        Args:
            request: TranscriptionRequest object.
            audio_path: Path to the audio file.
        
        Returns:
            Dict containing the request data ready for the model.
        """
        try:
            request_data = {
                "audio_path": audio_path,
                "verbose": False,
            }
            
            # Add optional parameters if provided
            if request.temperature is not None:
                request_data["temperature"] = request.temperature
            
            if request.language is not None:
                request_data["language"] = request.language
            
            if request.prompt is not None:
                request_data["initial_prompt"] = request.prompt
            
            # Map additional parameters if they exist
            decode_options = {}
            if request.language is not None:
                decode_options["language"] = request.language
            
            # Add decode options to request data
            request_data.update(decode_options)
            
            logger.debug(f"Prepared transcription request: {request_data}")
            
            return request_data
            
        except Exception as e:
            logger.error(f"Failed to prepare transcription request: {str(e)}")
            content = create_error_response(
                f"Failed to process request: {str(e)}", 
                "bad_request", 
                HTTPStatus.BAD_REQUEST
            )
            raise HTTPException(status_code=400, detail=content)

    def _calculate_audio_duration(self, audio_path: str) -> int:
        """
        Calculate the duration of the audio file in seconds.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            int: Duration in seconds.
        """
        try:
            # Try to import librosa for accurate duration calculation
            try:
                import librosa
                duration = librosa.get_duration(path=audio_path)
                return int(duration)
            except ImportError:
                # Fallback: estimate based on file size (very rough approximation)
                # Assume ~32kbps for compressed audio
                file_size = os.path.getsize(audio_path)
                estimated_duration = file_size / (32000 / 8)  # bytes / (bits_per_sec / 8)
                logger.warning(
                    "librosa not available, using rough file-size-based duration estimate. "
                    "Install librosa for accurate duration calculation."
                )
                return int(estimated_duration)
        except Exception as e:
            logger.warning(f"Failed to calculate audio duration: {str(e)}, defaulting to 0")
            return 0

    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the request queue and performance metrics.
        
        Returns:
            Dict with queue and performance statistics.
        """
        queue_stats = self.request_queue.get_queue_stats()
        
        return {
            "queue_stats": queue_stats,
        }
        
    async def cleanup(self):
        """
        Cleanup resources and stop the request queue before shutdown.
        
        This method ensures all pending requests are properly cancelled
        and resources are released.
        """
        try:
            logger.info("Cleaning up MLXWhisperHandler resources")
            if hasattr(self, 'request_queue'):
                await self.request_queue.stop()
            logger.info("MLXWhisperHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXWhisperHandler cleanup: {str(e)}")
            raise

