"""Predictive inventory alert endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..api_models import PredictiveReportResponse
from ..predictive_alerts import predict_inventory
from .state import AppState


def create_predictions_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["predictions"])

    @router.get(
        "/predictions",
        response_model=PredictiveReportResponse,
        dependencies=[Depends(require_auth)],
    )
    async def get_predictions(
        store_id: str | None = Query(default=None),
        horizon_days: int = Query(default=30, ge=7, le=90),
    ) -> PredictiveReportResponse:
        """Predict stockouts and overstock situations."""
        digest = state.get_or_run_digest(
            [store_id] if store_id else None
        )
        report = predict_inventory(
            digest, store_id=store_id, horizon_days=horizon_days
        )
        return PredictiveReportResponse(**report.to_dict())

    return router
