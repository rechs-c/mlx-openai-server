import json
import random
import time
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from app.handler.mlx_lm import MLXLMHandler
from app.schemas.openai import (ChatCompletionChunk,
                                ChatCompletionMessageToolCall,
                                ChatCompletionRequest, ChatCompletionResponse,
                                Choice, ChoiceDeltaFunctionCall,
                                ChoiceDeltaToolCall, Delta, Embedding,
                                EmbeddingRequest, EmbeddingResponse,
                                FunctionCall, Message, Model, ModelsResponse,
                                StreamingChoice)
from app.utils.errors import create_error_response

router = APIRouter()


@router.post("/health")
async def health(raw_request: Request):
    """
    Health check endpoint.
    """
    try:
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(content=create_error_response("Health check failed", "server_error", 500), status_code=500)

@router.get("/v1/queue/stats")
async def queue_stats(raw_request: Request):
    """
    Get queue statistics.
    """
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)
    
    try:
        stats = await handler.get_queue_stats()
        return {
            "status": "ok",
            "queue_stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get queue stats: {str(e)}")
        return JSONResponse(content=create_error_response("Failed to get queue stats", "server_error", 500), status_code=500)
        
@router.get("/v1/models")
async def models(raw_request: Request):
    """
    Get list of available models.
    """
    handler = raw_request.app.state.handler
    models_data = handler.get_models()
    return ModelsResponse(data=[Model(**model) for model in models_data])

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Handle chat completion requests."""
    
    logger.info("---------------chat_completions>request-----------------")
    logger.info(request)
    logger.info("---------------request<chat_completions-----------------")

    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)
    
    try:
        # Check if this is a vision request
        is_vision_request = request.is_vision_request()
        # If it's a vision request but the handler is MLXLMHandler (text-only), reject it
        if is_vision_request and isinstance(handler, MLXLMHandler):
            return JSONResponse(
                content=create_error_response(
                    "Vision requests are not supported with text-only models. Use a VLM model type instead.", 
                    "unsupported_request", 
                    400
                ), 
                status_code=400
            )
        
        # Process the request based on type
        response = await process_vision_request(handler, request) if is_vision_request \
                   else await process_text_request(handler, request)
        
        # # 只有当 response 不是 StreamingResponse 时才打印
        # if not isinstance(response, StreamingResponse):
        #     logger.info("---------------chat_completions output>-----------------")
        #     logger.info(response)
        #     logger.info("----------------^-----------------")
        # else:
        #     logger.info("---------------chat_completions output (streaming)>-----------------")
        #     logger.info("Streaming response will be handled by handle_stream_response.")
        #     logger.info("----------------^-----------------")
            
        return response
    except Exception as e:
        logger.error(f"Error processing chat completion request: {(e)}", exc_info=True)
        return JSONResponse(content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
    
@router.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest, raw_request: Request):
    """Handle embedding requests."""
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)

    try:
        embeddings = await handler.generate_embeddings_response(request)
        return create_response_embeddings(embeddings, request.model)
    except Exception as e:
        logger.error(f"Error processing embedding request: {str(e)}", exc_info=True)
        return JSONResponse(content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
    
def create_response_embeddings(embeddings: List[float], model: str) -> EmbeddingResponse:
    embeddings_response = []
    for index, embedding in enumerate(embeddings):
        embeddings_response.append(Embedding(embedding=embedding, index=index))
    return EmbeddingResponse(data=embeddings_response, model=model)

def create_response_chunk(chunk: Union[str, Dict[str, Any]], model: str, is_final: bool = False, finish_reason: Optional[str] = "stop", chat_id: Optional[str] = None, created_time: Optional[int] = None) -> ChatCompletionChunk:
    """Create a formatted response chunk for streaming."""
    chat_id = chat_id if chat_id else get_id()
    created_time = created_time if created_time else int(time.time())
    if isinstance(chunk, str):
        return ChatCompletionChunk(
            id=chat_id,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[StreamingChoice(
                index=0,
                delta=Delta(content=chunk, role="assistant"),
                finish_reason=finish_reason if is_final else None
            )]
        )
    if "reasoning_content" in chunk:
        return ChatCompletionChunk(
            id=chat_id,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[StreamingChoice(
                index=0,
                delta=Delta(reasoning_content=chunk["reasoning_content"], role="assistant"),
                finish_reason=finish_reason if is_final else None
            )]
        )

    if "name" in chunk and chunk["name"]:
        tool_chunk = ChoiceDeltaToolCall(
            index=chunk["index"],
            type="function",
            id=get_tool_call_id(),
            function=ChoiceDeltaFunctionCall(
                name=chunk["name"],
                arguments=""
            )
        )
    else:
        tool_chunk = ChoiceDeltaToolCall(
            index=chunk["index"],
            type="function",
            function= ChoiceDeltaFunctionCall(
                arguments=chunk["arguments"]
            )
        )
    delta = Delta(
        content = None,
        role = "assistant",
        tool_calls = [tool_chunk]
    )
    return ChatCompletionChunk(
        id=chat_id,
        object="chat.completion.chunk",
        created=created_time,
        model=model,
        choices=[StreamingChoice(index=0, delta=delta, finish_reason=None)]
    )


async def handle_stream_response(generator: AsyncGenerator, model: str):
    """Handle streaming response generation (OpenAI-compatible)."""
    chat_index = get_id()
    created_time = int(time.time())
    try:
        finish_reason = "stop"
        index = -1
        # First chunk: role-only delta, as per OpenAI
        first_chunk = ChatCompletionChunk(
            id=chat_index,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[StreamingChoice(index=0, delta=Delta(role="assistant"), finish_reason=None)]
        )
        yield f"data: {json.dumps(first_chunk.model_dump())}\n\n"
        async for chunk in generator:
            if chunk:
                if isinstance(chunk, str):
                    response_chunk = create_response_chunk(chunk, model, chat_id=chat_index, created_time=created_time)
                    yield f"data: {json.dumps(response_chunk.model_dump())}\n\n"
                else:
                    finish_reason = "tool_calls"
                    if "name" in chunk and chunk["name"]:
                        index += 1
                    payload = {
                        "index": index,
                        **chunk
                    }
                    response_chunk = create_response_chunk(payload, model, chat_id=chat_index, created_time=created_time)
                    yield f"data: {json.dumps(response_chunk.model_dump())}\n\n"
    except Exception as e:
        logger.error(f"Error in stream wrapper: {str(e)}")
        error_response = create_error_response(str(e), "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
        # Yield error as last chunk before [DONE]
        yield f"data: {json.dumps(error_response)}\n\n"
    finally:
        # Final chunk: finish_reason and [DONE], as per OpenAI
        final_chunk = create_response_chunk('', model, is_final=True, finish_reason=finish_reason, chat_id=chat_index)
        yield f"data: {json.dumps(final_chunk.model_dump())}\n\n"
        yield "data: [DONE]\n\n"

async def process_vision_request(handler, request: ChatCompletionRequest):
    """Process vision-specific requests."""
    if request.stream:
        return StreamingResponse(
            handle_stream_response(handler.generate_vision_stream(request), request.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )
    return format_final_response(await handler.generate_vision_response(request), request.model)

async def process_text_request(handler, request: ChatCompletionRequest):
    """Process text-only requests."""
    if request.stream:
        return StreamingResponse(
            handle_stream_response(handler.generate_text_stream(request), request.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )
    return format_final_response(await handler.generate_text_response(request), request.model)

def get_id():
    """
    Generate a unique ID for chat completions with timestamp and random component.
    """
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"chatcmpl_{timestamp}{random_suffix:06d}"

def get_tool_call_id():
    """
    Generate a unique ID for tool calls with timestamp and random component.
    """
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"call_{timestamp}{random_suffix:06d}"

def format_final_response(response: Union[str, Dict[str, Any]], model: str) -> ChatCompletionResponse:
    """Format the final non-streaming response."""
    
    if isinstance(response, str):
        return ChatCompletionResponse(
            id=get_id(),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[Choice(
                index=0,
                message=Message(role="assistant", content=response),
                finish_reason="stop"
            )]
        )
    
    # If not a string, assume it's a dictionary
    reasoning_content = response.get("reasoning_content", None)
    tool_calls = response.get("tool_calls", [])
    tool_call_responses = []
    for idx, tool_call in enumerate(tool_calls):
        function_call = FunctionCall(
            name=tool_call.get("name"),
            arguments=json.dumps(tool_call.get("arguments"))
        )
        tool_call_response = ChatCompletionMessageToolCall(
            id=get_tool_call_id(),
            type="function",
            function=function_call,
            index=idx
        )
        tool_call_responses.append(tool_call_response)
    
    if len(tool_calls) > 0:
        message = Message(role="assistant", reasoning_content=reasoning_content, tool_calls=tool_call_responses)
        finish_reason = "tool_calls"
    else:
        # If no tool calls, content should be present in the dictionary response
        message = Message(role="assistant", content=response.get("content", ""), reasoning_content=reasoning_content)
        finish_reason = "stop"
    
    return ChatCompletionResponse(
        id=get_id(),
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[Choice(
            index=0,
            message=message,
            finish_reason=finish_reason
        )]
    )
