"""Health check endpoints.

GET /health         — deep check: binary + Supabase + Engine 2 probes (503 when degraded)
GET /health/shallow — lightweight liveness check for ALB (always 200 if process is up)
GET /health/engine2 — Engine 2 (world model) operational status
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..api_models import HealthResponse
from .state import AppState

logger = logging.getLogger("sentinel.routes.health")


def _check_supabase(state: AppState) -> dict:
    """Probe Supabase connectivity. Returns {"ok": bool, "detail": str}."""
    client = state.supabase_client
    if client is None:
        return {"ok": True, "detail": "not_configured"}
    try:
        # Lightweight RPC — supabase-py client exposes .table().select()
        # Just verify we can reach the server with a trivial query.
        client.table("digest_subscriptions").select("email").limit(1).execute()
        return {"ok": True, "detail": "connected"}
    except Exception as e:
        logger.warning("Supabase health probe failed: %s", e)
        return {"ok": False, "detail": str(e)}


def create_health_router(state: AppState) -> APIRouter:
    router = APIRouter(tags=["health"])

    @router.get("/health/shallow")
    async def shallow_health():
        """Lightweight liveness probe for ALB / load balancer.

        Always returns 200 if the process is running. No dependency checks.
        """
        return {"status": "ok"}

    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        """Deep health check — probes binary, Supabase, and Engine 2.

        Returns 200 when all critical dependencies are healthy.
        Returns 503 when any critical dependency is unavailable.
        """
        binary_found = state.engine is not None
        binary_path = str(state.engine.binary) if state.engine else None

        # Dependency probes
        supabase_probe = _check_supabase(state)
        engine2_ok = state.world_model is not None

        # Critical = binary must be present.
        # Supabase is informational — a DNS/connection error to a dev URL
        # should not take down the health check.
        all_ok = binary_found
        status = "ok" if all_ok else "degraded"

        payload = HealthResponse(
            status=status,
            binary_found=binary_found,
            binary_path=binary_path,
            dev_mode=state.settings.sidecar_dev_mode,
        )

        # Build response with dependency details
        response_data = payload.model_dump()
        response_data["dependencies"] = {
            "supabase": supabase_probe,
            "engine2": {
                "ok": engine2_ok,
                "detail": "active" if engine2_ok else "not_initialized",
            },
        }

        if not all_ok:
            return JSONResponse(content=response_data, status_code=503)

        return JSONResponse(content=response_data, status_code=200)

    @router.get("/health/engine2")
    async def engine2_status() -> dict:
        """Engine 2 (world model) health check — no auth required.

        Returns the world model's operational status, observation count,
        and subsystem summary. Used by monitoring and the frontend to
        show Engine 2 warmup progress.
        """
        if state.world_model is None:
            return {
                "status": "not_initialized",
                "observations": 0,
                "stores_tracked": 0,
                "entities_tracked": 0,
                "predictions_active": 0,
                "transfer_stores": 0,
            }

        try:
            pipeline = state.world_model
            n_observations = sum(len(h) for h in pipeline.entity_history.values())
            n_entities = len(pipeline.entity_history)

            # Predictions
            predictions_active = 0
            if hasattr(pipeline, "predictive"):
                pipeline.predictive.expire_stale_interventions()
                predictions_active = len(pipeline.predictive.active_interventions)

            # Transfer matcher
            transfer_stores = 0
            if state.transfer_matcher is not None:
                transfer_stores = len(state.transfer_matcher.agents)

            status = "active" if n_observations > 0 else "warming_up"

            return {
                "status": status,
                "observations": n_observations,
                "stores_tracked": len(pipeline.stores),
                "entities_tracked": n_entities,
                "predictions_active": predictions_active,
                "transfer_stores": transfer_stores,
                "pipeline_status": pipeline.pipeline_status(),
            }

        except Exception as e:
            logger.warning("Engine 2 health check failed: %s", e)
            return {
                "status": "error",
                "error": str(e),
            }

    return router
