"""FastAPI application for the baseline inference service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.common.config import ServiceConfig
from app.common.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.baseline_service.engine import BaselineEngine
from app.baseline_service.model_loader import load_model

engine: BaselineEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global engine
    handle = load_model()
    engine = BaselineEngine(handle)
    yield


app = FastAPI(title="Baseline Inference Service", version="0.1.0", lifespan=lifespan)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    assert engine is not None
    return engine.generate(request)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    loaded = engine is not None
    device = engine.handle.device if engine else "unknown"
    return HealthResponse(service="baseline", model_loaded=loaded, device=device)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ServiceConfig.baseline_port)
