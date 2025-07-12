import argparse
import asyncio
import gc
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.handler.mlx_vlm import MLXVLMHandler
from app.handler.mlx_lm import MLXLMHandler
from app.handler.mlfux import MLXFluxHandler 
from app.api.endpoints import router
from app.version import __version__

# Try to import MLX for memory management
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
logger.add(lambda msg: print(msg), level="INFO")  # Also print to console

def parse_args():
    parser = argparse.ArgumentParser(description="OAI-compatible proxy")
    parser.add_argument("--model-path", type=str, help="Path to the model (required for lm and multimodal model types)")
    parser.add_argument("--model-name", type=str, choices=["dev", "schnell"], help="Name of the model (required for image-generation model type).")
    parser.add_argument("--model-type", type=str, default="lm", choices=["lm", "multimodal", "image-generation"], help="Model type")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Maximum number of concurrent requests")
    parser.add_argument("--queue-timeout", type=int, default=300, help="Request timeout in seconds")
    parser.add_argument("--queue-size", type=int, default=100, help="Maximum queue size for pending requests")
    
    args = parser.parse_args()
    
    # Validate model arguments
    if args.model_type == "image-generation":
        if not args.model_name:
            parser.error("--model-name is required for image-generation model type. Available options: 'dev', 'schnell'")
        if args.model_path:
            parser.error("--model-path cannot be used with image-generation model type. Use --model-name instead.")
    else:
        if not args.model_path:
            parser.error("--model-path is required for lm and multimodal model types")
        if args.model_name:
            parser.error("--model-name can only be used with image-generation model type. Use --model-path instead.")
    
    return args


def get_model_identifier(args):
    """Get the appropriate model identifier based on model type."""
    if args.model_type == "image-generation":
        return args.model_name
    else:
        return args.model_path

def create_lifespan(config_args):
    """Factory function to create a lifespan context manager with access to config args."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            model_identifier = get_model_identifier(config_args)
            if config_args.model_type == "image-generation":
                logger.info(f"Initializing MLX handler with model name: {model_identifier}")
            else:
                logger.info(f"Initializing MLX handler with model path: {model_identifier}")
            
            if config_args.model_type == "multimodal":
                handler = MLXVLMHandler(
                    model_path=model_identifier,
                    max_concurrency=config_args.max_concurrency
                )
            elif config_args.model_type == "image-generation":
                handler = MLXFluxHandler(
                    model_path=model_identifier,
                    max_concurrency=config_args.max_concurrency
                )
            else:
                handler = MLXLMHandler(
                    model_path=model_identifier,
                    max_concurrency=config_args.max_concurrency
                )       
            # Initialize queue
            await handler.initialize({
                "max_concurrency": config_args.max_concurrency,
                "timeout": config_args.queue_timeout,
                "queue_size": config_args.queue_size
            })
            logger.info("MLX handler initialized successfully")
            app.state.handler = handler
        except Exception as e:
            logger.error(f"Failed to initialize MLX handler: {str(e)}")
            raise
        
        # Initial memory cleanup
        if MLX_AVAILABLE:
            mx.clear_cache()
        gc.collect()
        
        yield
        
        # Shutdown
        logger.info("Shutting down application")
        if hasattr(app.state, "handler") and app.state.handler:
            try:
                # Use the proper cleanup method which handles both request queue and image processor
                logger.info("Cleaning up resources")
                await app.state.handler.cleanup()
                logger.info("Resources cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during shutdown: {str(e)}")
        
        # Final memory cleanup
        if MLX_AVAILABLE:
            mx.clear_cache()
        gc.collect()
    
    return lifespan

# App instance will be created during setup with the correct lifespan
app = None

async def setup_server(args) -> uvicorn.Config:
    global app
    
    # Create FastAPI app with the configured lifespan
    app = FastAPI(
        title="OpenAI-compatible API",
        description="API for OpenAI-compatible chat completion and text embedding",
        version=__version__,
        lifespan=create_lifespan(args)
    )
    
    app.include_router(router)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Periodic memory cleanup for long-running processes
        if hasattr(request.app.state, 'request_count'):
            request.app.state.request_count += 1
        else:
            request.app.state.request_count = 1
        
        # Clean up memory every 50 requests
        if request.app.state.request_count % 50 == 0:
            if MLX_AVAILABLE:
                mx.clear_cache()
            gc.collect()
            logger.debug(f"Performed memory cleanup after {request.app.state.request_count} requests")
        
        return response
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global exception handler caught: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error"}}
        )
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True
    )
    return config

if __name__ == "__main__":
    args = parse_args()
    config = asyncio.run(setup_server(args))
    uvicorn.Server(config).run()