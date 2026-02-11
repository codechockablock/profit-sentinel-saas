"""Transfer recommendations endpoint.

GET /api/v1/transfers — list transfer recommendations for dead stock

STUB: Returns empty recommendations until Engine 1→2 wiring is
      complete and TransferMatcher has multi-store data. The endpoint
      structure is ready for integration.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from .state import AppState

logger = logging.getLogger("sentinel.routes.transfers")


def create_transfers_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["transfers"])

    @router.get("/transfers")
    async def list_transfers(
        source_store: str = Query(None, description="Filter by source store ID"),
        min_benefit: float = Query(0.0, description="Minimum net benefit ($)"),
        _user=Depends(require_auth),
    ) -> dict:
        """List transfer recommendations for dead stock items.

        Returns recommendations from the TransferMatcher when available.
        Each recommendation includes:
        - Source and destination store
        - SKU details (item, quantity, capital at risk)
        - Match type (exact, subcategory, category)
        - Financial comparison (clearance vs transfer recovery)

        Query params:
            source_store: Filter to a specific source store
            min_benefit: Only show transfers with benefit above this threshold
        """
        # TODO: Wire to TransferMatcher once Engine 1→2 integration is complete
        #
        # Integration plan:
        # 1. AppState gains a `world_model: SentinelPipeline` field
        # 2. After each /analysis/analyze call, Engine 1 results are fed
        #    to Engine 2 via record_observation()
        # 3. TransferMatcher.find_recommendations() runs against the
        #    world model's learned store patterns
        # 4. Results cached and returned here
        #
        # For now, return empty recommendations with the expected shape.

        return {
            "recommendations": [],
            "total": 0,
            "message": (
                "Transfer recommendations require multi-store data. "
                "Upload inventory from at least 2 stores to enable "
                "cross-location transfer matching."
            ),
        }

    return router
