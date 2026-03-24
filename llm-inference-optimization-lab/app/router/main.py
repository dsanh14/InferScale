"""FastAPI application for the inference router."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from app.common.config import ServiceConfig
from app.common.metrics import get_collector, prometheus_metrics_text
from app.common.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.router.service import RouterService

router_service: RouterService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global router_service
    router_service = RouterService()
    yield
    await router_service.close()


app = FastAPI(
    title="LLM Inference Router",
    description="Routes inference requests to baseline, quantized, speculative, or disaggregated backends.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    assert router_service is not None
    try:
        return await router_service.handle(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="router", status="ok")


@app.get("/metrics")
async def metrics() -> dict:
    return get_collector().summary()


@app.get("/metrics/prometheus")
async def prom_metrics():
    from fastapi.responses import Response
    return Response(content=prometheus_metrics_text(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ServiceConfig.router_port)
