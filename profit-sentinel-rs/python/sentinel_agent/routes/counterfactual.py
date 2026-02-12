"""Engine 3 — Counterfactual API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from .state import AppState

logger = logging.getLogger("sentinel.routes.counterfactual")


def create_counterfactual_router(state: AppState) -> APIRouter:
    router = APIRouter(prefix="/counterfactual", tags=["Engine 3"])

    @router.get("/summary")
    async def get_counterfactual_summary():
        """Get aggregate cost-of-inaction summary across all cached findings."""
        if state.counterfactual_engine is None:
            raise HTTPException(503, "Engine 3 not available")

        # Pull from the most recent analysis result if available
        # For now, return from any cached digest
        for entry in state.digest_cache.values():
            if not entry.is_expired:
                digest = entry.digest
                # Build items from digest issues for Engine 3
                fake_result = {
                    "leaks": {
                        issue.issue_type.value: {
                            "items": [
                                {
                                    "sku": s.sku_id,
                                    "quantity": s.qty_on_hand,
                                    "cost": s.unit_cost,
                                    "revenue": s.retail_price,
                                    "sold": s.sales_last_30d,
                                    "margin": s.margin_pct,
                                    "on_order": s.on_order_qty,
                                }
                                for s in issue.skus
                            ],
                            "count": len(issue.skus),
                        }
                        for issue in digest.issues
                    }
                }

                cfs = state.counterfactual_engine.analyze(fake_result)
                return state.counterfactual_engine.get_aggregate_summary(cfs)

        return {"message": "No analysis data available. Run an analysis first."}

    @router.get("/health")
    async def engine3_health():
        """Engine 3 health check."""
        return {
            "engine": "counterfactual",
            "status": "ok" if state.counterfactual_engine else "unavailable",
            "strategies": (
                [type(s).__name__ for s in state.counterfactual_engine.strategies]
                if state.counterfactual_engine
                else []
            ),
        }

    return router
