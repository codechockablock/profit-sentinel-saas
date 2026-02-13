"""Transfer recommendations endpoint.

GET /api/v1/transfers — list transfer recommendations for dead stock

Wired to TransferMatcher from the world model. Requires multi-store
data to generate recommendations. Returns empty list gracefully when
Engine 2 is not initialized or has insufficient data.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from ..dual_auth import UserContext
from .state import AppState

logger = logging.getLogger("sentinel.routes.transfers")


def create_transfers_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["transfers"])

    @router.get("/transfers")
    async def list_transfers(
        source_store: str = Query(None, description="Filter by source store ID"),
        min_benefit: float = Query(0.0, description="Minimum net benefit ($)"),
        max_results: int = Query(20, ge=1, le=100, description="Max recommendations"),
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """List transfer recommendations for dead stock items.

        Returns recommendations from the per-user TransferMatcher when available.
        Each recommendation includes:
        - Source and destination store
        - SKU details (item, quantity, capital at risk)
        - Match type (exact, subcategory, category)
        - Financial comparison (clearance vs transfer recovery)

        Query params:
            source_store: Filter to a specific source store
            min_benefit: Only show transfers with benefit above this threshold
            max_results: Maximum number of recommendations to return
        """
        # Get per-user transfer matcher
        matcher = state.get_user_transfer_matcher(ctx.user_id)

        # Engine 2 not initialized — return empty gracefully
        if matcher is None:
            return {
                "recommendations": [],
                "total": 0,
                "engine2_status": "not_initialized",
                "message": (
                    "Transfer matching requires Engine 2 (world model). "
                    "Contact support if this persists."
                ),
            }

        try:
            n_stores = len(matcher.agents)

            # Need at least 2 stores for cross-store matching
            if n_stores < 2:
                return {
                    "recommendations": [],
                    "total": 0,
                    "stores_registered": n_stores,
                    "engine2_status": "warming_up",
                    "message": (
                        "Transfer recommendations require multi-store data. "
                        f"Currently tracking {n_stores} store(s). "
                        "Upload inventory from at least 2 stores to enable "
                        "cross-location transfer matching."
                    ),
                }

            # Run transfer matching
            if source_store:
                recs = matcher.find_transfers(
                    source_store, max_recommendations=max_results
                )
            else:
                # Search all stores
                all_recs_by_store = matcher.find_all_transfers(
                    max_per_store=max_results
                )
                recs = []
                for store_recs in all_recs_by_store.values():
                    recs.extend(store_recs)
                # Re-sort combined results by net benefit
                recs.sort(key=lambda r: -r.net_benefit)
                recs = recs[:max_results]

            # Apply min_benefit filter
            if min_benefit > 0:
                recs = [r for r in recs if r.net_benefit >= min_benefit]

            return {
                "recommendations": [r.to_dict() for r in recs],
                "total": len(recs),
                "stores_registered": n_stores,
                "engine2_status": "active",
            }

        except Exception as e:
            # Sovereign collapse: transfer failure is non-fatal
            logger.warning("Transfer matching failed (non-fatal): %s", e)
            return {
                "recommendations": [],
                "total": 0,
                "engine2_status": "error",
                "message": "Transfer matching encountered an error. Engine 1 findings are unaffected.",
            }

    return router
