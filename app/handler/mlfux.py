import asyncio
import base64
import time
import uuid
from http import HTTPStatus
from typing import Any, Dict, List, Optional
from io import BytesIO
import gc

from fastapi import HTTPException
from loguru import logger
from PIL import Image

from app.core.queue import RequestQueue
from app.models.mflux import MLXFlux
from app.schemas.openai import ImageGenerationRequest, ImageGenerationResponse, ImageData
from app.utils.errors import create_error_response


class MLXFluxHandler:
    """
    Handler class for making image generation requests to the underlying MLX Flux model service.
    Provides request queuing, metrics tracking, and robust error handling.
    """

    def __init__(self, model_path: str, max_concurrency: int = 1, quantize: int = 8):
        """
        Initialize the handler with the specified model path.
        
        Args:
            model_path (str): Path to the model directory or model name for Flux.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
            quantize (int): Quantization level for the model.
        """
        self.model_path = model_path
        self.quantize = quantize
        self.model = MLXFlux(model_name=model_path, quantize=quantize)
        self.model_created = int(time.time())  # Store creation time when model is loaded
        
        # Initialize request queue for image generation tasks
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)

        logger.info(f"Initialized MLXFluxHandler with model path: {model_path}")
    
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
                "timeout": 300,
                "queue_size": 100
            }
        self.request_queue = RequestQueue(
            max_concurrency=queue_config.get("max_concurrency", 1),
            timeout=queue_config.get("timeout", 300),
            queue_size=queue_config.get("queue_size", 100)
        )
        await self.request_queue.start(self._process_request)
        logger.info("Initialized MLXFluxHandler and started request queue")

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """
        Generate an image based on the request parameters.
        Uses the request queue for handling concurrent requests.
        
        Args:
            request: ImageGenerationRequest object containing the generation parameters.
        
        Returns:
            ImageGenerationResponse: Response containing the generated image data.
        """
        request_id = f"image-{uuid.uuid4()}"
        
        try:
            # Prepare request data
            request_data = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "steps": request.steps,
                "seed": request.seed,
                "size": request.size,
                "priority": request.priority,
                "async_mode": request.async_mode
            }
            
            # Submit to the request queue
            image_result = await self.request_queue.submit(request_id, request_data)
            
            # Convert PIL Image to base64
            image_data = self._image_to_base64(image_result)
            
            # Create response
            response = ImageGenerationResponse(
                created=int(time.time()),
                data=[ImageData(b64_json=image_data)]
            )
            
            return response
            
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response(
                "Too many requests. Service is at capacity.", 
                "rate_limit_exceeded", 
                HTTPStatus.TOO_MANY_REQUESTS
            )
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            logger.error(f"Error in image generation for request {request_id}: {str(e)}")
            content = create_error_response(
                f"Failed to generate image: {str(e)}", 
                "server_error", 
                HTTPStatus.INTERNAL_SERVER_ERROR
            )
            raise HTTPException(status_code=500, detail=content)

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image object.
            
        Returns:
            str: Base64 encoded image string.
        """
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')

    async def _process_request(self, request_data: Dict[str, Any]) -> Image.Image:
        """
        Process an image generation request. This is the worker function for the request queue.
        
        Args:
            request_data: Dictionary containing the request data.
            
        Returns:
            Image.Image: The generated PIL Image.
        """
        try:
            # Extract request parameters
            prompt = request_data.get("prompt", "")
            negative_prompt = request_data.get("negative_prompt")
            steps = request_data.get("steps", 4)
            seed = request_data.get("seed", 42)
            size = request_data.get("size", "1024x1024")
            
            # Parse size string to width and height
            if isinstance(size, str) and 'x' in size:
                width, height = map(int, size.split('x'))
            else:
                width, height = 1024, 1024
            
            # Prepare model parameters
            model_params = {
                "num_inference_steps": steps,
                "width": width,
                "height": height,
            }
            
            # Add negative prompt if provided
            if negative_prompt:
                model_params["negative_prompt"] = negative_prompt
            
            # Generate image
            logger.info(f"Generating image with prompt: {prompt[:50]}...")
            image = self.model(
                prompt=prompt,
                seed=seed,
                **model_params
            )
            
            # Force garbage collection after model inference
            gc.collect()
            return image
            
        except Exception as e:
            logger.error(f"Error processing image generation request: {str(e)}")
            # Clean up on error
            gc.collect()
            raise

    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get current queue statistics.
        
        Returns:
            Dict containing queue statistics.
        """
        if not hasattr(self, 'request_queue') or self.request_queue is None:
            return {"error": "Request queue not initialized"}
        
        stats = await self.request_queue.get_stats()
        return {
            "queue_size": stats.get("queue_size", 0),
            "active_requests": stats.get("active_requests", 0),
            "completed_requests": stats.get("completed_requests", 0),
            "failed_requests": stats.get("failed_requests", 0),
            "average_processing_time": stats.get("average_processing_time", 0)
        }

    async def cleanup(self):
        """
        Clean up resources and shut down the request queue.
        """
        if hasattr(self, '_cleaned') and self._cleaned:
            return
        self._cleaned = True
        
        try:
            logger.info("Cleaning up MLXFluxHandler resources")
            if hasattr(self, 'request_queue') and self.request_queue:
                await self.request_queue.stop()
                logger.info("Request queue stopped successfully")
        except Exception as e:
            logger.error(f"Error during MLXFluxHandler cleanup: {str(e)}")
        
        # Force garbage collection
        gc.collect()
        logger.info("MLXFluxHandler cleanup completed")

    def __del__(self):
        """
        Destructor to ensure cleanup on object deletion.
        Note: Async cleanup cannot be reliably performed in __del__.
        Please use 'await cleanup()' explicitly.
        """
        if hasattr(self, '_cleaned') and self._cleaned:
            return
        # Set flag to prevent multiple cleanup attempts
        self._cleaned = True
