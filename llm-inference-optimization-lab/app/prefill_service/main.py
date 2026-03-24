"""FastAPI application for the prefill service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.common.config import ServiceConfig
from app.common.schemas import GenerateRequest, HealthResponse
from app.baseline_service.model_loader import load_model
from app.prefill_service.engine import PrefillEngine

engine: PrefillEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global engine
    handle = load_model()
    engine = PrefillEngine(handle)
    yield


app = FastAPI(title="Prefill Service", version="0.1.0", lifespan=lifespan)


@app.post("/prefill")
async def prefill(request: GenerateRequest) -> dict:
    assert engine is not None
    artifact = engine.prefill(request)
    return {"cache_artifact": artifact.model_dump()}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    loaded = engine is not None
    device = engine.handle.device if engine else "unknown"
    return HealthResponse(service="prefill", model_loaded=loaded, device=device)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ServiceConfig.prefill_port)
