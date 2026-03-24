"""FastAPI application for the decode service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel

from app.common.config import ServiceConfig
from app.common.schemas import GenerateRequest, HealthResponse
from app.baseline_service.model_loader import load_model
from app.decode_service.engine import DecodeEngine
from app.prefill_service.cache_artifact import CacheArtifact

engine: DecodeEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global engine
    handle = load_model()
    engine = DecodeEngine(handle)
    yield


app = FastAPI(title="Decode Service", version="0.1.0", lifespan=lifespan)


class DecodePayload(BaseModel):
    request: GenerateRequest
    cache_artifact: dict[str, Any]


@app.post("/decode")
async def decode(payload: DecodePayload) -> dict:
    assert engine is not None
    artifact = CacheArtifact(**payload.cache_artifact)
    return engine.decode(payload.request, artifact)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    loaded = engine is not None
    device = engine.handle.device if engine else "unknown"
    return HealthResponse(service="decode", model_loaded=loaded, device=device)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ServiceConfig.decode_port)
