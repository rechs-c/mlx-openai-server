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

    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)
    
    logger.debug(f"Incoming chat completion request: {request.model_dump_json()}") # Added logging
    try:
        # Check if this is a vision request
        is_vision_request = request.is_vision_request()
        # If it's a vision request but the handler is MLXLMHandler (text-only), reject it
        if is_vision_request and isinstance(handler, MLXLMHandler):
            logger.warning("Vision request received for text-only model.") # Added logging
            return JSONResponse(
                content=create_error_response(
                    "Vision requests are not supported with text-only models. Use a VLM model type instead.", 
                    "unsupported_request", 
                    400
                ), 
                status_code=400
            )
        
        # Process the request based on type
        return await process_vision_request(handler, request) if is_vision_request \
               else await process_text_request(handler, request)
    except Exception as e:
        logger.error(f"Error processing chat completion request: {str(e)}", exc_info=True)
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

    # 处理工具调用 delta
    # chunk 结构现在是 {"index": ..., "type": "function", "id": ..., "name": "...", "arguments": "..."}
    # 或者 {"index": ..., "type": "function", "arguments": "..."}
    tool_chunk_id = chunk.get("id") # 从 chunk 中获取 id
    tool_chunk_name = chunk.get("name")
    tool_chunk_arguments = chunk.get("arguments", "")

    if tool_chunk_name: # 初始工具调用 delta (包含 name)
        tool_chunk = ChoiceDeltaToolCall(
            index=chunk["index"],
            type="function",
            id=tool_chunk_id, # 使用传递过来的 id
            function=ChoiceDeltaFunctionCall(
                name=tool_chunk_name,
                arguments=tool_chunk_arguments # 初始为空，或包含第一个片段
            )
        )
    else: # 后续工具调用 delta (只包含 arguments 片段)
        tool_chunk = ChoiceDeltaToolCall(
            index=chunk["index"],
            type="function",
            function= ChoiceDeltaFunctionCall(
                arguments=tool_chunk_arguments
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
            logger.debug(f"Processing stream chunk from handler: {chunk}") # Added logging
            if chunk:
                if "content" in chunk:
                    # 普通文本
                    response_chunk = create_response_chunk(chunk["content"], model, chat_id=chat_index, created_time=created_time)
                    logger.debug(f"Sending content stream chunk to client: {response_chunk.model_dump()}") # Added logging
                    yield f"data: {json.dumps(response_chunk.model_dump())}\n\n"
                elif "reasoning_content" in chunk:
                    # 推理内容
                    response_chunk = create_response_chunk({"reasoning_content": chunk["reasoning_content"]}, model, chat_id=chat_index, created_time=created_time)
                    logger.debug(f"Sending reasoning content stream chunk to client: {response_chunk.model_dump()}") # Added logging
                    yield f"data: {json.dumps(response_chunk.model_dump())}\n\n"
                elif "function" in chunk: # 检查是否有 function 键，表示是工具调用
                    # mlx_lm.py 已经将工具调用分段，这里直接传递给 create_response_chunk
                    # chunk 结构现在是 {"index": ..., "type": "function", "id": ..., "function": {"name": "...", "arguments": "..."}}
                    # 或者 {"index": ..., "type": "function", "function": {"arguments": "..."}}
                    payload = {
                        "index": chunk.get("index", 0), # 确保有 index
                        "type": chunk.get("type", "function"),
                        "id": chunk.get("id"), # 只有初始 delta 有 id
                        "name": chunk["function"].get("name"), # 只有初始 delta 有 name
                        "arguments": chunk["function"].get("arguments", "")
                    }
                    response_chunk = create_response_chunk(payload, model, chat_id=chat_index, created_time=created_time)
                    logger.debug(f"Sending tool call stream chunk to client: {response_chunk.model_dump()}") # Added logging
                    yield f"data: {json.dumps(response_chunk.model_dump())}\n\n"
                    finish_reason = "tool_calls" # 如果有工具调用，结束原因为 tool_calls
                else:
                    logger.warning(f"Unknown chunk format from handler: {chunk}")
                    # 兜底处理，避免中断流
                    response_chunk = create_response_chunk(str(chunk), model, chat_id=chat_index, created_time=created_time)
                    logger.debug(f"Sending unknown stream chunk to client: {response_chunk.model_dump()}") # Added logging
                    yield f"data: {json.dumps(response_chunk.model_dump())}\n\n"
    except Exception as e:
        logger.error(f"Error in stream wrapper: {str(e)}", exc_info=True) # Added exc_info
        error_response = create_error_response(str(e), "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
        # Yield error as last chunk before [DONE]
        yield f"data: {json.dumps(error_response)}\n\n"
    finally:
        # Final chunk: finish_reason and [DONE], as per OpenAI
        final_chunk = create_response_chunk('', model, is_final=True, finish_reason=finish_reason, chat_id=chat_index)
        logger.debug(f"Sending final stream chunk to client: {final_chunk.model_dump()}") # Added logging
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

def format_final_response(response: Union[str, List[Dict[str, Any]]], model: str) -> ChatCompletionResponse:
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
    
    reasoning_content = response.get("reasoning_content", None)
    tool_calls = response.get("tool_calls", [])
    tool_call_responses = []
    for idx, tool_call in enumerate(tool_calls):
        # 确定 tool_call 结构为 {"function": {"name": "...", "arguments": "{...}"}, "type": "function", "id": null}
        function_data = tool_call.get("function", {})
        function_call = FunctionCall(
            name=function_data.get("name"),
            arguments=function_data.get("arguments") # arguments 已经是 JSON 字符串，无需再次 dumps
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
    else:
        message = Message(role="assistant", content=response, reasoning_content=reasoning_content, tool_calls=tool_call_responses)
    
    final_response = ChatCompletionResponse( # Store in variable for logging
        id=get_id(),
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[Choice(
            index=0,
            message=message,
            finish_reason="tool_calls"
        )]
    )
    logger.debug(f"Final non-streaming response: {final_response.model_dump_json()}") # Added logging
    return final_response
