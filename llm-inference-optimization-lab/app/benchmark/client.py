"""HTTP client for sending benchmark requests to the router."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from app.common.logging_utils import setup_logging
from app.common.schemas import GenerateRequest, GenerateResponse

logger = setup_logging("benchmark.client")


async def send_request(
    client: httpx.AsyncClient,
    router_url: str,
    request: GenerateRequest,
) -> dict[str, Any]:
    """Send a single generate request and return the response with client-side timing."""
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{router_url}/generate", json=request.model_dump())
        resp.raise_for_status()
        payload = resp.json()
        t1 = time.perf_counter()
        result = {
            "request_id": request.request_id,
            "mode": request.mode,
            "client_latency_ms": (t1 - t0) * 1000.0,
            "success": True,
            **payload.get("metrics", {}),
        }
        result["output_text_len"] = len(payload.get("output_text", ""))
        return result
    except Exception as exc:
        t1 = time.perf_counter()
        logger.error(f"Request {request.request_id} failed: {exc}")
        return {
            "request_id": request.request_id,
            "mode": request.mode,
            "client_latency_ms": (t1 - t0) * 1000.0,
            "success": False,
            "error_message": str(exc),
        }


async def send_batch(
    router_url: str,
    requests: list[GenerateRequest],
    concurrency: int = 1,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """Send a batch of requests with the specified concurrency level."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def _bounded(req: GenerateRequest) -> dict[str, Any]:
            async with semaphore:
                return await send_request(client, router_url, req)

        tasks = [_bounded(r) for r in requests]
        results = await asyncio.gather(*tasks)

    return list(results)
