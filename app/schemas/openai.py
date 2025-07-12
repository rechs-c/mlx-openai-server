from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator
from typing_extensions import Literal
from loguru import logger


# Configuration
class Config:
    """
    Configuration class holding the default model names for different types of requests.
    """
    TEXT_MODEL = "local-text-model"          # Default model for text-based chat completions
    MULTIMODAL_MODEL = "local-multimodal-model"  # Model used for multimodal requests
    EMBEDDING_MODEL = "local-embedding-model"  # Model used for generating embeddings
    IMAGE_GENERATION_MODEL = "local-image-generation-model"

class ErrorResponse(BaseModel):
    object: str = Field("error", description="The object type, always 'error'.")
    message: str = Field(..., description="The error message.")
    type: str = Field(..., description="The type of error.")
    param: Optional[str] = Field(None, description="The parameter related to the error, if any.")
    code: int = Field(..., description="The error code.")

# Common models used in both streaming and non-streaming contexts
class ImageUrl(BaseModel):
    """
    Represents an image URL in a message.
    """
    url: str = Field(..., description="The image URL.")

class AudioInput(BaseModel):
    """
    Represents an audio URL in a message.
    """
    data: str = Field(..., description="The audio data.")
    format: Literal["mp3", "wav"] = Field(..., description="The audio format.")

class AudioContentItem(BaseModel):
    """
    Represents an audio content item in a message.
    """
    type: str = Field(..., description="The type of content, e.g., 'input_audio'.")
    input_audio: Optional[AudioInput] = Field(None, description="The audio input object, if type is 'input_audio'.")

class MultimodalContentItem(BaseModel):
    """
    Represents a single content item in a message (text, image, or audio).
    """
    type: str = Field(..., description="The type of content, e.g., 'text', 'image_url', or 'input_audio'.")
    text: Optional[str] = Field(None, description="The text content, if type is 'text'.")
    image_url: Optional[ImageUrl] = Field(None, description="The image URL object, if type is 'image_url'.")
    input_audio: Optional[AudioInput] = Field(None, description="The audio input object, if type is 'input_audio'.")

class FunctionCall(BaseModel):
    """
    Represents a function call in a message.
    """
    arguments: str = Field(..., description="The arguments for the function call.")
    name: str = Field(..., description="The name of the function to call.")

class ChatCompletionMessageToolCall(BaseModel):
    """
    Represents a tool call in a message.
    """
    id: str = Field(..., description="The ID of the tool call.")
    function: FunctionCall = Field(..., description="The function call details.")
    type: Literal["function"] = Field(..., description="The type of tool call, always 'function'.")
    index: int = Field(..., description="The index of the tool call.")

class Message(BaseModel):
    """
    Represents a message in a chat completion.
    """
    content: Union[str, List[MultimodalContentItem]] = Field(None, description="The content of the message, either text or a list of content items (vision, audio, or multimodal).")
    refusal: Optional[str] = Field(None, description="The refusal reason, if any.")
    role: Literal["system", "user", "assistant", "tool"] = Field(..., description="The role of the message sender.")
    function_call: Optional[FunctionCall] = Field(None, description="The function call, if any.")
    reasoning_content: Optional[str] = Field(None, description="The reasoning content, if any.")
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = Field(None, description="List of tool calls, if any.")

# Common request base for both streaming and non-streaming
class ChatCompletionRequestBase(BaseModel):
    """
    Base model for chat completion requests.
    """
    model: str = Field(Config.TEXT_MODEL, description="The model to use for completion.")
    messages: List[Message] = Field(..., description="The list of messages in the conversation.")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="List of tools available for the request.")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Tool choice for the request.")
    max_tokens: Optional[int] = Field(None, description="The maximum number of tokens to generate.")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature.")
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling probability.")
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty for token generation.")
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty for token generation.")
    stop: Optional[List[str]] = Field(None, description="List of stop sequences.")
    n: Optional[int] = Field(1, description="Number of completions to generate.")
    response_format: Optional[Dict[str, str]] = Field(None, description="Format for the response.")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility.")
    user: Optional[str] = Field(None, description="User identifier.")

    @validator("messages")
    def check_messages_not_empty(cls, v):
        """
        Ensure that the messages list is not empty and validate message structure.
        """
        if not v:
            raise ValueError("messages cannot be empty")
        
        # Validate message history length
        if len(v) > 100:  # OpenAI's limit is typically around 100 messages
            raise ValueError("message history too long")
            
        # Validate message roles
        valid_roles = {"user", "assistant", "system", "tool"}
        for msg in v:
            if msg.role not in valid_roles:
                raise ValueError(f"invalid role: {msg.role}")
                
        return v

    @validator("temperature")
    def check_temperature(cls, v):
        """
        Validate temperature is between 0 and 2.
        """
        if v is not None and (v < 0 or v > 2):
            raise ValueError("temperature must be between 0 and 2")
        return v

    @validator("max_tokens")
    def check_max_tokens(cls, v):
        """
        Validate max_tokens is positive and within reasonable limits.
        """
        if v is not None:
            if v <= 0:
                raise ValueError("max_tokens must be positive")
        return v
    
    def is_multimodal_request(self) -> bool:
        """
        Check if the request includes image or audio content, indicating a multimodal request.
        """
        for message in self.messages:
            content = message.content
            if isinstance(content, list):
                for item in content:
                    if hasattr(item, 'type'):
                        if item.type == "image_url" and hasattr(item, 'image_url') and item.image_url and item.image_url.url:
                            logger.debug(f"Detected multimodal request with image: {item.image_url.url[:30]}...")
                            return True
                        elif item.type == "input_audio" and hasattr(item, 'input_audio') and item.input_audio and item.input_audio.data:
                            logger.debug(f"Detected multimodal request with audio data")
                            return True
        
        logger.debug(f"No images or audio detected, treating as text-only request")
        return False
    
class ChatTemplateKwargs(BaseModel):
    """
    Represents the arguments for a chat template.
    """
    enable_thinking: bool = Field(False, description="Whether to enable thinking mode.")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="List of tools to use in the request.")
    add_generation_prompt: bool = Field(True, description="Whether to add a generation prompt to the request.")

# Non-streaming request and response
class ChatCompletionRequest(ChatCompletionRequestBase):
    """
    Model for non-streaming chat completion requests.
    """
    stream: bool = Field(False, description="Whether to stream the response.")
    chat_template_kwargs: ChatTemplateKwargs = Field(ChatTemplateKwargs(), description="Arguments for the chat template.")

class Choice(BaseModel):
    """
    Represents a choice in a chat completion response.
    """
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] = Field(..., description="The reason for the choice.")
    index: int = Field(..., description="The index of the choice.")
    message: Message = Field(..., description="The message of the choice.")

class ChatCompletionResponse(BaseModel):
    """
    Represents a complete chat completion response.
    """
    id: str = Field(..., description="The response ID.")
    object: Literal["chat.completion"] = Field(..., description="The object type, always 'chat.completion'.")
    created: int = Field(..., description="The creation timestamp.")
    model: str = Field(..., description="The model used for completion.")
    choices: List[Choice] = Field(..., description="List of choices in the response.")


class ChoiceDeltaFunctionCall(BaseModel):
    """
    Represents a function call delta in a streaming response.
    """
    arguments: Optional[str] = Field(None, description="Arguments for the function call delta.")
    name: Optional[str] = Field(None, description="Name of the function in the delta.")

class ChoiceDeltaToolCall(BaseModel):
    """
    Represents a tool call delta in a streaming response.
    """
    index: Optional[int] = Field(None, description="Index of the tool call delta.")
    id: Optional[str] = Field(None, description="ID of the tool call delta.")
    function: Optional[ChoiceDeltaFunctionCall] = Field(None, description="Function call details in the delta.")
    type: Optional[str] = Field(None, description="Type of the tool call delta.")

class Delta(BaseModel):
    """
    Represents a delta in a streaming response.
    """
    content: Optional[str] = Field(None, description="Content of the delta.")
    function_call: Optional[ChoiceDeltaFunctionCall] = Field(None, description="Function call delta, if any.")
    refusal: Optional[str] = Field(None, description="Refusal reason, if any.")
    role: Optional[Literal["system", "user", "assistant", "tool"]] = Field(None, description="Role in the delta.")
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = Field(None, description="List of tool call deltas, if any.")
    reasoning_content: Optional[str] = Field(None, description="Reasoning content, if any.")

class StreamingChoice(BaseModel):
    """
    Represents a choice in a streaming response.
    """
    delta: Delta = Field(..., description="The delta for this streaming choice.")
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = Field(None, description="The reason for finishing, if any.")
    index: int = Field(..., description="The index of the streaming choice.")
    
class ChatCompletionChunk(BaseModel):
    """
    Represents a chunk in a streaming chat completion response.
    """
    id: str = Field(..., description="The chunk ID.")
    choices: List[StreamingChoice] = Field(..., description="List of streaming choices in the chunk.")
    created: int = Field(..., description="The creation timestamp of the chunk.")
    model: str = Field(..., description="The model used for the chunk.")
    object: Literal["chat.completion.chunk"] = Field(..., description="The object type, always 'chat.completion.chunk'.")

# Embedding models
class EmbeddingRequest(BaseModel):
    """
    Model for embedding requests.
    """
    model: str = Field(Config.EMBEDDING_MODEL, description="The embedding model to use.")
    input: List[str] = Field(..., description="List of text inputs for embedding.")
    image_url: Optional[str] = Field(default=None, description="Image URL to embed.")

class Embedding(BaseModel):
    """
    Represents an embedding object in an embedding response.
    """
    embedding: List[float] = Field(..., description="The embedding vector.")
    index: int = Field(..., description="The index of the embedding in the list.")
    object: str = Field(default="embedding", description="The object type, always 'embedding'.")

class EmbeddingResponse(BaseModel):
    """
    Represents an embedding response.
    """
    object: str = Field("list", description="The object type, always 'list'.")
    data: List[Embedding] = Field(..., description="List of embedding objects.")
    model: str = Field(..., description="The model used for embedding.")

class Model(BaseModel):
    """
    Represents a model in the models list response.
    """
    id: str = Field(..., description="The model ID.")
    object: str = Field("model", description="The object type, always 'model'.")
    created: int = Field(..., description="The creation timestamp.")
    owned_by: str = Field("openai", description="The owner of the model.")

class ModelsResponse(BaseModel):
    """
    Represents the response for the models list endpoint.
    """
    object: str = Field("list", description="The object type, always 'list'.")
    data: List[Model] = Field(..., description="List of models.")


class ImageSize(str, Enum):
    """Available image sizes"""
    SMALL = "256x256"
    MEDIUM = "512x512"
    LARGE = "1024x1024"
    COSMOS_SIZE = "1024x1024"


class Priority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ImageGenerationRequest(BaseModel):
    """Request schema for OpenAI-compatible image generation API"""
    prompt: str = Field(..., description="A text description of the desired image(s). The maximum length is 1000 characters.", max_length=1000)
    model: Optional[str] = Field(default=Config.IMAGE_GENERATION_MODEL, description="The model to use for image generation")
    size: Optional[ImageSize] = Field(default=ImageSize.LARGE, description="The size of the generated images")
    negative_prompt: Optional[str] = Field(None, description="The negative prompt to generate the image from")
    steps: Optional[int] = Field(default=4, ge=1, le=50, description="The number of inference steps (1-50)")
    priority: Optional[Priority] = Field(default=Priority.NORMAL, description="Task priority in queue")
    async_mode: Optional[bool] = Field(default=False, description="Whether to process asynchronously")
    seed: Optional[int] = Field(42, description="Seed for reproducible generation")

class ImageData(BaseModel):
    """Individual image data in the response"""
    url: Optional[str] = Field(None, description="The URL of the generated image, if response_format is url")
    b64_json: Optional[str] = Field(None, description="The base64-encoded JSON of the generated image, if response_format is b64_json")

class ImageGenerationResponse(BaseModel):
    """Response schema for OpenAI-compatible image generation API"""
    created: int = Field(..., description="The Unix timestamp (in seconds) when the image was created")
    data: List[ImageData] = Field(..., description="List of generated images")

class ImageGenerationError(BaseModel):
    """Error response schema"""
    code: str = Field(..., description="Error code (e.g., 'contentFilter', 'generation_error', 'queue_full')")
    message: str = Field(..., description="Human-readable error message")
    type: Optional[str] = Field(None, description="Error type")

class ImageGenerationErrorResponse(BaseModel):
    """Error response wrapper"""
    created: int = Field(..., description="The Unix timestamp (in seconds) when the error occurred")
    error: ImageGenerationError = Field(..., description="Error details")