import asyncio
import base64
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from http import HTTPStatus
import gc

from fastapi import HTTPException
from loguru import logger

from app.core import ImageProcessor, AudioProcessor
from app.core.queue import RequestQueue
from app.models.mlx_vlm import MLX_VLM
from app.schemas.openai import ChatCompletionRequest, EmbeddingRequest
from app.utils.errors import create_error_response

class MLXVLMHandler:
    """
    Handler class for making requests to the underlying MLX multimodal model service.
    Provides caching, concurrent image processing, audio processing, and robust error handling.
    """

    def __init__(self, model_path: str, max_workers: int = 4, max_concurrency: int = 1):
        """
        Initialize the handler with the specified model path.
        
        Args:
            model_path (str): Path to the model directory.
            max_workers (int): Maximum number of worker threads for image processing.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
        """
        self.model_path = model_path
        self.model = MLX_VLM(model_path)
        self.image_processor = ImageProcessor(max_workers)
        self.audio_processor = AudioProcessor(max_workers)
        self.model_created = int(time.time())  # Store creation time when model is loaded
        
        # Initialize request queue for multimodal and text tasks
        # We use the same queue for both multimodal and text tasks for simplicity
        # and to ensure we don't overload the model with too many concurrent requests
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)
        
        logger.info(f"Initialized MLXHandler with model path: {model_path}")

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
            max_concurrency=queue_config.get("max_concurrency"),
            timeout=queue_config.get("timeout"),
            queue_size=queue_config.get("queue_size")
        )
        await self.request_queue.start(self._process_request)
        logger.info("Initialized MLXHandler and started request queue")

    async def generate_multimodal_stream(self, request: ChatCompletionRequest):
        """
        Generate a streaming response for multimodal chat completion requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            AsyncGenerator: Yields response chunks.
        """
        
        # Create a unique request ID
        request_id = f"multimodal-{uuid.uuid4()}"
        
        try:
            chat_messages, image_paths, audio_paths, model_params = await self._prepare_multimodal_request(request)
            
            # Create a request data object
            request_data = {
                "images": image_paths,
                "audios": audio_paths,
                "messages": chat_messages,
                "stream": True,
                **model_params
            }
            
            # Submit to the multimodal queue and get the generator
            response_generator = await self.request_queue.submit(request_id, request_data)
            
            # Process and yield each chunk asynchronously
            for chunk in response_generator:
                if chunk:
                    chunk = chunk.text
                    yield chunk
        
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response("Too many requests. Service is at capacity.", "rate_limit_exceeded", HTTPStatus.TOO_MANY_REQUESTS)
            raise HTTPException(status_code=429, detail=content)

        except Exception as e:
            logger.error(f"Error in multimodal stream generation for request {request_id}: {str(e)}")
            content = create_error_response(f"Failed to generate multimodal stream: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)

    async def generate_multimodal_response(self, request: ChatCompletionRequest):
        """
        Generate a complete response for multimodal chat completion requests.
        Uses the request queue for handling concurrent requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            str: Complete response.
        """
        try:
            # Create a unique request ID
            request_id = f"multimodal-{uuid.uuid4()}"
            
            # Prepare the multimodal request
            chat_messages, image_paths, audio_paths, model_params = await self._prepare_multimodal_request(request)
            
            # Create a request data object
            request_data = {
                "images": image_paths,
                "audios": audio_paths,
                "messages": chat_messages,
                "stream": False,
                **model_params
            }
        
            response = await self.request_queue.submit(request_id, request_data)
            
            return response
            
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response("Too many requests. Service is at capacity.", "rate_limit_exceeded", HTTPStatus.TOO_MANY_REQUESTS)
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            logger.error(f"Error in multimodal response generation: {str(e)}")
            content = create_error_response(f"Failed to generate multimodal response: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)

    async def generate_text_stream(self, request: ChatCompletionRequest):
        """
        Generate a streaming response for text-only chat completion requests.
        Uses the request queue for handling concurrent requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            AsyncGenerator: Yields response chunks.
        """
        request_id = f"text-{uuid.uuid4()}"
        
        try:
            chat_messages, model_params = await self._prepare_text_request(request)
            request_data = {
                "messages": chat_messages,
                "stream": True,
                **model_params
            }
            response_generator = await self.request_queue.submit(request_id, request_data)
            
            for chunk in response_generator:
                if chunk:
                    yield chunk.text
            
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response("Too many requests. Service is at capacity.", "rate_limit_exceeded", HTTPStatus.TOO_MANY_REQUESTS)
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            logger.error(f"Error in text stream generation for request {request_id}: {str(e)}")
            content = create_error_response(f"Failed to generate text stream: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)

    async def generate_text_response(self, request: ChatCompletionRequest):
        """
        Generate a complete response for text-only chat completion requests.
        Uses the request queue for handling concurrent requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            str: Complete response.
        """
        try:
            # Create a unique request ID
            request_id = f"text-{uuid.uuid4()}"
            
            # Prepare the text request
            chat_messages, model_params = await self._prepare_text_request(request)
            
            # Create a request data object
            request_data = {
                "messages": chat_messages,
                "stream": False,
                **model_params
            }
            
            # Submit to the multimodal queue (reusing the same queue for text requests)
            response = await self.request_queue.submit(request_id, request_data)
            return response
            
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response("Too many requests. Service is at capacity.", "rate_limit_exceeded", HTTPStatus.TOO_MANY_REQUESTS)
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            logger.error(f"Error in text response generation: {str(e)}")
            content = create_error_response(f"Failed to generate text response: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)
        
    async def generate_embeddings_response(self, request: EmbeddingRequest):
        """
        Generate embeddings for a given text input.
        
        Args:
            request: EmbeddingRequest object containing the text input.
        
        Returns:
            List[float]: Embeddings for the input text or images
        """
        try:
            # Create a unique request ID
            image_url = request.image_url
            # Process the image URL to get a local file path
            images = []
            if request.image_url:
                image_path = await self.image_processor.process_image_url(image_url)
                images.append(image_path)
            request_id = f"embeddings-{uuid.uuid4()}"
            request_data = {
                "type": "embeddings",
                "input": request.input,
                "model": request.model,
                "images": images
            }

            # Submit to the request queue
            response = await self.request_queue.submit(request_id, request_data)

            return response

        except Exception as e:
            logger.error(f"Error in embeddings generation: {str(e)}")
            content = create_error_response(f"Failed to generate embeddings: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)


    def __del__(self):
        """Cleanup resources on deletion."""
        # Removed async cleanup from __del__; use close() instead
        pass

    async def close(self):
        """Explicitly cleanup resources asynchronously."""
        if hasattr(self, 'image_processor'):
            await self.image_processor.cleanup()
        if hasattr(self, 'audio_processor'):
            await self.audio_processor.cleanup()

    async def cleanup(self):
        """
        Cleanup resources and stop the request queue before shutdown.
        
        This method ensures all pending requests are properly cancelled
        and resources are released, including the image processor.
        """
        try:
            logger.info("Cleaning up MLXVLMHandler resources")
            if hasattr(self, 'request_queue'):
                await self.request_queue.stop()
            if hasattr(self, 'image_processor'):
                await self.image_processor.cleanup()
            if hasattr(self, 'audio_processor'):
                await self.audio_processor.cleanup()
            # Force garbage collection after cleanup
            gc.collect()
            logger.info("MLXVLMHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXVLMHandler cleanup: {str(e)}")
            raise

    async def _process_request(self, request_data: Dict[str, Any]) -> str:
        """
        Process a multimodal request. This is the worker function for the request queue.
        
        Args:
            request_data: Dictionary containing the request data.
            
        Returns:
            str: The model's response.
        """
        try:
            # Check if the request is for embeddings
            if request_data.get("type") == "embeddings":
                result = self.model.get_embeddings(request_data["input"], request_data["images"])
                # Force garbage collection after embeddings
                gc.collect()
                return result
            
            # Extract request parameters
            images = request_data.get("images", [])
            audios = request_data.get("audios", [])
            messages = request_data.get("messages", [])
            stream = request_data.get("stream", False)
         
            # Remove these keys from model_params
            model_params = request_data.copy()
            model_params.pop("images", None)
            model_params.pop("audios", None)
            model_params.pop("messages", None)
            model_params.pop("stream", None)
            
            # Call the model
            response = self.model(
                images=images,
                audios=audios,
                messages=messages,
                stream=stream,
                **model_params
            )

            # Force garbage collection after model inference
            gc.collect()
            return response
            
        except Exception as e:
            logger.error(f"Error processing multimodal request: {str(e)}")
            # Clean up on error
            gc.collect()
            raise

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

    async def _prepare_text_request(self, request: ChatCompletionRequest) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Prepare a text request by parsing model parameters and verifying the format of messages.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            Tuple containing the formatted chat messages and model parameters.
        """
        chat_messages = []

        try:
            
            # Convert Message objects to dictionaries with 'role' and 'content' keys
            chat_messages = []
            for message in request.messages:
                # Only handle simple string content for text-only requests
                if not isinstance(message.content, str):
                    logger.warning(f"Non-string content in text request will be skipped: {message.role}")
                    continue
                
                chat_messages.append({
                    "role": message.role,
                    "content": message.content
                })

            # Extract model parameters, filtering out None values
            model_params = {
                k: v for k, v in {
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "frequency_penalty": request.frequency_penalty, 
                    "presence_penalty": request.presence_penalty,
                    "stop": request.stop,
                    "n": request.n,
                    "seed": request.seed
                }.items() if v is not None
            }

            # Handle response format
            if request.response_format and request.response_format.get("type") == "json_object":
                model_params["response_format"] = "json"

            # Handle tools and tool choice
            if request.tools:
                model_params["tools"] = request.tools
                if request.tool_choice:
                    model_params["tool_choice"] = request.tool_choice

            # Log processed data
            logger.debug(f"Processed text chat messages: {chat_messages}")
            logger.debug(f"Model parameters: {model_params}")

            return chat_messages, model_params

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to prepare text request: {str(e)}")
            content = create_error_response(f"Failed to process request: {str(e)}", "bad_request", HTTPStatus.BAD_REQUEST)
            raise HTTPException(status_code=400, detail=content)

    async def _prepare_multimodal_request(self, request: ChatCompletionRequest) -> Tuple[List[Dict[str, Any]], List[str], List[str], Dict[str, Any]]:
        """
        Prepare the multimodal request by processing messages with text, images, and audio.
        
        This method:
        1. Extracts text messages, image URLs, and audio data from the request
        2. Processes image URLs and audio data to get local file paths
        3. Prepares model parameters
        4. Returns processed data ready for model inference
        
        Args:
            request (ChatCompletionRequest): The incoming request containing messages and parameters.
            
        Returns:
            Tuple[List[Dict[str, Any]], List[str], List[str], Dict[str, Any]]: A tuple containing:
                - List of processed chat messages
                - List of processed image paths
                - List of processed audio paths
                - Dictionary of model parameters
        """
        chat_messages = []
        image_urls = []
        audio_urls = []

        try:
            # Process each message in the request
            for message in request.messages:
                # Handle system and assistant messages (simple text content)
                if message.role in ["system", "assistant"]:
                    chat_messages.append({"role": message.role, "content": message.content})
                    continue

                # Handle user messages
                if message.role == "user":
                    # Case 1: Simple string content
                    if isinstance(message.content, str):
                        chat_messages.append({"role": "user", "content": message.content})
                        continue
                        
                    # Case 2: Content is a list of dictionaries or objects
                    if isinstance(message.content, list):
                        # Initialize containers for this message
                        texts = []
                        images = []
                        audios = []                 
                        # Process each content item in the list
                        for item in message.content:
                            if item.type == "text":
                                text = getattr(item, "text", "").strip()
                                if text:
                                    texts.append(text)
                                    
                            elif item.type == "image_url":
                                url = getattr(item, "image_url", None)
                                if url and hasattr(url, "url"):
                                    url = url.url
                                    # Validate URL
                                    self._validate_image_url(url)
                                    images.append(url)
                                    
                            elif item.type == "input_audio":
                                audio_input = getattr(item, "input_audio", None)
                                if audio_input and hasattr(audio_input, "data"):
                                    audio_data = audio_input.data
                                    audio_format = getattr(audio_input, "format", "mp3")
                                    # Create data URL from audio data
                                    audio_url = f"data:audio/{audio_format};base64,{audio_data}"
                                    # Validate audio data
                                    self._validate_audio_data(audio_url)
                                    audios.append(audio_url)

                        # Add collected media to global lists
                        if images:
                            image_urls.extend(images)
                            # Validate constraints
                            if len(images) > 4:
                                content = create_error_response("Too many images in a single message (max: 4)", "invalid_request_error", HTTPStatus.BAD_REQUEST)
                                raise HTTPException(status_code=400, detail=content)
                        
                        if audios:
                            audio_urls.extend(audios)
                            # Validate constraints
                            if len(audios) > 2:
                                content = create_error_response("Too many audio files in a single message (max: 2)", "invalid_request_error", HTTPStatus.BAD_REQUEST)
                                raise HTTPException(status_code=400, detail=content)
                            
                        # Add text content if available, otherwise use empty string
                        if texts:
                            chat_messages.append({"role": "user", "content": " ".join(texts)})
                        else:
                           chat_messages.append({"role": "user", "content": ""})
                    else:
                        content = create_error_response("Invalid message content format", "invalid_request_error", HTTPStatus.BAD_REQUEST)
                        raise HTTPException(status_code=400, detail=content)

            # Process images and audio files
            image_paths = await self.image_processor.process_image_urls(image_urls)
            audio_paths = await self.audio_processor.process_audio_urls(audio_urls)
            

            # Get model parameters from the request
            temperature = request.temperature or 0.7
            top_p = request.top_p or 1.0
            frequency_penalty = request.frequency_penalty or 0.0
            presence_penalty = request.presence_penalty or 0.0
            max_tokens = request.max_tokens or 1024
            tools = request.tools or None
            tool_choice = request.tool_choice or None
            
            model_params = {
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_tokens,
                "tools": tools,
                "tool_choice": tool_choice
            }
            
            # Log processed data at debug level
            logger.debug(f"Processed chat messages: {chat_messages}")
            logger.debug(f"Processed image paths: {image_paths}")
            logger.debug(f"Processed audio paths: {audio_paths}")
            logger.debug(f"Model parameters: {model_params}")

            return chat_messages, image_paths, audio_paths, model_params

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to prepare multimodal request: {str(e)}")
            content = create_error_response(f"Failed to process request: {str(e)}", "bad_request", HTTPStatus.BAD_REQUEST)
            raise HTTPException(status_code=400, detail=content)
            
    def _validate_image_url(self, url: str) -> None:
        """
        Validate image URL format.
        
        Args:
            url: The image URL to validate
            
        Raises:
            HTTPException: If URL is invalid
        """
        if not url:
            content = create_error_response("Empty image URL provided", "invalid_request_error", HTTPStatus.BAD_REQUEST)
            raise HTTPException(status_code=400, detail=content)
            
        # Validate base64 images
        if url.startswith("data:"):
            try:
                header, encoded = url.split(",", 1)
                if not header.startswith("data:image/"):
                    raise ValueError("Invalid image format")
                base64.b64decode(encoded)
            except Exception as e:
                content = create_error_response(f"Invalid base64 image: {str(e)}", "invalid_request_error", HTTPStatus.BAD_REQUEST)
                raise HTTPException(status_code=400, detail=content)
                
    def _validate_audio_data(self, url: str) -> None:
        """
        Validate audio data URL format.
        
        Args:
            url: The audio data URL to validate
            
        Raises:
            HTTPException: If audio data is invalid
        """
        if not url:
            content = create_error_response("Empty audio data provided", "invalid_request_error", HTTPStatus.BAD_REQUEST)
            raise HTTPException(status_code=400, detail=content)
            
        # Validate base64 audio
        if url.startswith("data:"):
            try:
                header, encoded = url.split(",", 1)
                if not header.startswith("data:audio/"):
                    raise ValueError("Invalid audio format")
                base64.b64decode(encoded)
            except Exception as e:
                content = create_error_response(f"Invalid base64 audio: {str(e)}", "invalid_request_error", HTTPStatus.BAD_REQUEST)
                raise HTTPException(status_code=400, detail=content)
