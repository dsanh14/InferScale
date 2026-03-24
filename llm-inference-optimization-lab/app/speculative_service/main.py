"""FastAPI application for the speculative decoding service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.common.config import ServiceConfig
from app.common.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.speculative_service.draft_target import DraftTargetPair
from app.speculative_service.engine import SpeculativeEngine

engine: SpeculativeEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global engine
    pair = DraftTargetPair()
    engine = SpeculativeEngine(pair)
    yield


app = FastAPI(title="Speculative Decoding Service", version="0.1.0", lifespan=lifespan)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    assert engine is not None
    return engine.generate(request)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    loaded = engine is not None
    device = engine.pair.device if engine else "unknown"
    return HealthResponse(service="speculative", model_loaded=loaded, device=device)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ServiceConfig.speculative_port)
