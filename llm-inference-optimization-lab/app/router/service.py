"""Router service logic — forwards requests and handles disaggregated flow."""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.common.config import ServiceConfig
from app.common.events import EventLogger
from app.common.logging_utils import setup_logging
from app.common.metrics import get_collector, record_prometheus
from app.common.schemas import GenerateRequest, GenerateResponse
from app.router.policy import ExplicitModePolicy, RoutingPolicy

logger = setup_logging("router")
event_logger = EventLogger("router")


class RouterService:
    """Stateless request router with support for direct and disaggregated flows."""

    def __init__(self, policy: RoutingPolicy | None = None) -> None:
        self.policy = policy or ExplicitModePolicy()
        self._client = httpx.AsyncClient(timeout=120.0)

    async def handle(self, request: GenerateRequest) -> GenerateResponse:
        """Route a request to the appropriate backend and return the response."""
        request_id = request.request_id or "unknown"
        event_logger.log(request_id, "routing", "request_received", {"mode": request.mode})

        if request.mode == "disaggregated":
            return await self._disaggregated_flow(request)

        return await self._direct_flow(request)

    async def _direct_flow(self, request: GenerateRequest) -> GenerateResponse:
        """Forward to a single backend (baseline / quantized / speculative)."""
        request_id = request.request_id or "unknown"
        target_url = self.policy.resolve(request)
        event_logger.log(
            request_id, "routing", "dispatching", {"target_url": target_url}
        )

        t0 = time.perf_counter()
        resp = await self._client.post(
            f"{target_url}/generate", json=request.model_dump()
        )
        resp.raise_for_status()
        payload = resp.json()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        event_logger.log(
            request_id, "routing", "response_received",
            {"latency_ms": round(elapsed_ms, 2)},
        )

        response = GenerateResponse(**payload)
        self._record_metrics(response)
        return response

    async def _disaggregated_flow(self, request: GenerateRequest) -> GenerateResponse:
        """Two-phase prefill -> decode flow with explicit handoff."""
        request_id = request.request_id or "unknown"
        event_logger.log(request_id, "routing", "disagg_start")

        # Phase 1: prefill
        t_prefill_start = time.perf_counter()
        prefill_resp = await self._client.post(
            f"{ServiceConfig.prefill_url}/prefill",
            json=request.model_dump(),
        )
        prefill_resp.raise_for_status()
        prefill_data = prefill_resp.json()
        t_prefill_end = time.perf_counter()

        prefill_ms = (t_prefill_end - t_prefill_start) * 1000.0
        event_logger.log(
            request_id, "routing", "prefill_done",
            {"prefill_ms": round(prefill_ms, 2)},
        )

        # Phase 2: decode — send the cache artifact from prefill
        t_decode_start = time.perf_counter()
        decode_payload = {
            "request": request.model_dump(),
            "cache_artifact": prefill_data.get("cache_artifact", {}),
        }
        decode_resp = await self._client.post(
            f"{ServiceConfig.decode_url}/decode",
            json=decode_payload,
        )
        decode_resp.raise_for_status()
        decode_data = decode_resp.json()
        t_decode_end = time.perf_counter()

        decode_ms = (t_decode_end - t_decode_start) * 1000.0
        transfer_ms = (t_decode_start - t_prefill_end) * 1000.0

        event_logger.log(
            request_id, "routing", "decode_done",
            {"decode_ms": round(decode_ms, 2), "transfer_ms": round(transfer_ms, 2)},
        )

        metrics = decode_data.get("metrics", {})
        metrics["prefill_ms"] = round(prefill_ms, 2)
        metrics["decode_ms"] = round(decode_ms, 2)
        metrics["transfer_latency_ms"] = round(transfer_ms, 2)
        metrics["total_latency_ms"] = round(prefill_ms + decode_ms + transfer_ms, 2)

        response = GenerateResponse(
            request_id=request_id,
            mode="disaggregated",
            output_text=decode_data.get("output_text", ""),
            metrics=metrics,
            debug={"prefill": prefill_data, "decode": decode_data},
        )
        self._record_metrics(response)
        return response

    def _record_metrics(self, response: GenerateResponse) -> None:
        collector = get_collector()
        collector.record_request(response.mode, response.metrics)
        record_prometheus(response.mode, response.metrics)

    async def close(self) -> None:
        await self._client.aclose()
