"""FastAPI application for the quantized inference service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.common.config import ServiceConfig
from app.common.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.quantized_service.adapters import create_backend
from app.quantized_service.engine import QuantizedEngine

engine: QuantizedEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global engine
    backend = create_backend()
    engine = QuantizedEngine(backend)
    engine.initialize()
    yield


app = FastAPI(title="Quantized Inference Service", version="0.1.0", lifespan=lifespan)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    assert engine is not None
    return engine.generate(request)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    loaded = engine is not None
    device = "unknown"
    if engine:
        device = engine.backend.get_metadata().get("device", "unknown")
    return HealthResponse(service="quantized", model_loaded=loaded, device=device)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=ServiceConfig.quantized_port)
