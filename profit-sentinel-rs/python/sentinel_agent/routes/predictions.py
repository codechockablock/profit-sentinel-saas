"""Predictive inventory alert endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from ..api_models import PredictiveReportResponse
from ..engine import PipelineError
from ..predictive_alerts import predict_inventory
from .state import AppState

logger = logging.getLogger("sentinel.routes.predictions")


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
        try:
            digest = state.get_or_run_digest([store_id] if store_id else None)
        except (HTTPException, PipelineError, FileNotFoundError) as exc:
            logger.info("No data for predictions (returning empty): %s", exc)
            return PredictiveReportResponse(
                store_id=store_id or "all",
                total_predictions=0,
                critical_alerts=0,
                warning_alerts=0,
                total_revenue_at_risk=0.0,
                total_carrying_cost_at_risk=0.0,
                stockout_predictions=[],
                overstock_predictions=[],
                velocity_alerts=[],
                top_recommendation="Upload inventory data to generate predictions.",
            )
        report = predict_inventory(digest, store_id=store_id, horizon_days=horizon_days)
        return PredictiveReportResponse(**report.to_dict())

    return router
