"""Health check endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..api_models import HealthResponse
from .state import AppState

logger = logging.getLogger("sentinel.routes.health")


def create_health_router(state: AppState) -> APIRouter:
    router = APIRouter(tags=["health"])

    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check — no auth required.

        Returns 200 when the engine binary is available, 503 when degraded.
        """
        binary_found = state.engine is not None
        binary_path = str(state.engine.binary) if state.engine else None

        status = "ok" if binary_found else "degraded"
        payload = HealthResponse(
            status=status,
            binary_found=binary_found,
            binary_path=binary_path,
            dev_mode=state.settings.sidecar_dev_mode,
        )

        if not binary_found:
            return JSONResponse(
                content=payload.model_dump(),
                status_code=503,
            )

        return payload

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
