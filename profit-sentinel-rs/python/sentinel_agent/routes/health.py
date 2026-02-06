"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from ..api_models import HealthResponse
from .state import AppState


def create_health_router(state: AppState) -> APIRouter:
    router = APIRouter(tags=["health"])

    @router.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check — no auth required."""
        binary_found = state.engine is not None
        binary_path = str(state.engine.binary) if state.engine else None

        return HealthResponse(
            status="ok" if binary_found else "degraded",
            binary_found=binary_found,
            binary_path=binary_path,
            dev_mode=state.settings.sidecar_dev_mode,
        )

    return router
