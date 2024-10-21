"""FastAPI server for demonstrating AsyncEngine usage and benchmarking.

Note: This is a demonstration server not intended for production use.
For production deployments, use the OpenAI compatible server.
"""

import json
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.engine.async_llm_engine import AsyncLLMEngine
from sarathi.entrypoints.config import APIServerConfig
from sarathi.utils import random_uuid

# Constants
TIMEOUT_KEEP_ALIVE = 5  # seconds
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

logger = logging.getLogger(__name__)

@dataclass
class GenerationRequest:
    """Structured request data for text generation."""
    prompt: str
    stream: bool
    sampling_params: SamplingParams

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationRequest':
        """Create a GenerationRequest from a dictionary."""
        prompt = data.pop("prompt", None)
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt is required"
            )
            
        stream = data.pop("stream", False)
        try:
            sampling_params = SamplingParams(**data)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sampling parameters: {str(e)}"
            )
            
        return cls(prompt=prompt, stream=stream, sampling_params=sampling_params)

@dataclass
class GenerationResponse:
    """Structured response for text generation."""
    text: str

    def to_json(self) -> str:
        """Convert response to JSON string."""
        return json.dumps({"text": self.text})

class TextGenerationServer:
    """FastAPI server for text generation."""

    def __init__(self, config: APIServerConfig):
        """Initialize the server with configuration."""
        self.config = config
        self.app = FastAPI(
            title="Text Generation API",
            description="API for text generation using AsyncEngine",
            version="1.0.0",
        )
        self.engine: Optional[AsyncLLMEngine] = None
        self._setup_routes()
        self._setup_middleware()

    def _setup_middleware(self) -> None:
        """Configure middleware for the FastAPI application."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        """Configure API routes."""
        self.app.get("/health")(self.health_check)
        self.app.post("/generate")(self.generate)

    async def health_check(self) -> Response:
        """Health check endpoint."""
        if not self.engine:
            return Response(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content="Engine not initialized"
            )
        return Response(status_code=status.HTTP_200_OK)

    async def generate(self, request: Request) -> Response:
        """Generate text completion for the request."""
        if not self.engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Engine not initialized"
            )

        try:
            request_data = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON request"
            )

        try:
            gen_request = GenerationRequest.from_dict(request_data)
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )

        request_id = random_uuid()
        results_generator = self.engine.generate(
            request_id,
            gen_request.prompt,
            gen_request.sampling_params
        )

        if gen_request.stream:
            return StreamingResponse(
                self._stream_results(results_generator),
                media_type="text/event-stream"
            )

        return await self._generate_complete_response(
            request, request_id, results_generator
        )

    async def _stream_results(
        self,
        results_generator: AsyncGenerator
    ) -> AsyncGenerator[bytes, None]:
        """Stream generation results."""
        try:
            async for request_output in results_generator:
                text_output = request_output.prompt + request_output.text
                response = GenerationResponse(text=text_output)
                yield (response.to_json() + "\0").encode("utf-8")
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Streaming error occurred"
            )

    async def _generate_complete_response(
        self,
        request: Request,
        request_id: str,
        results_generator: AsyncGenerator
    ) -> JSONResponse:
        """Generate complete response for non-streaming requests."""
        final_output = None
        try:
            async for request_output in results_generator:
                if await request.is_disconnected():
                    await self.engine.abort(request_id)
                    return Response(status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST)
                final_output = request_output

            if final_output is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No output generated"
                )

            text_output = final_output.prompt + final_output.text
            return JSONResponse({"text": text_output})

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Generation error occurred"
            )

    def initialize_engine(self) -> None:
        """Initialize the AsyncLLMEngine."""
        try:
            self.engine = AsyncLLMEngine.from_system_config(
                self.config.create_system_config(),
                verbose=(self.config.log_level == "debug")
            )
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            raise

    def run(self) -> None:
        """Run the server."""
        self.initialize_engine()
        self.app.root_path = self.config.server_root_path
        
        uvicorn.run(
            self.app,
            host=self.config.host or DEFAULT_HOST,
            port=self.config.port or DEFAULT_PORT,
            log_level=self.config.log_level,
            ssl_keyfile=self.config.ssl_keyfile,
            ssl_certfile=self.config.ssl_certfile,
            ssl_ca_certs=self.config.ssl_ca_certs,
            ssl_cert_reqs=self.config.ssl_cert_reqs,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        )

if __name__ == "__main__":
    config = APIServerConfig.create_from_cli_args()
    server = TextGenerationServer(config)
    server.run()
