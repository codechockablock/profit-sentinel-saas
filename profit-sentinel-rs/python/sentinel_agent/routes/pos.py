"""POS integration endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..api_models import PosConnectionRequest
from ..dual_auth import UserContext
from ..pos_integrations import (
    create_pos_connection,
    delete_pos_connection,
    disconnect_pos,
    get_supported_systems,
    list_pos_connections,
    trigger_sync,
)


def create_pos_router(require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["pos"])

    @router.get("/pos/systems")
    async def get_pos_systems() -> dict:
        """List all supported POS systems with setup instructions."""
        systems = get_supported_systems()
        return {"systems": [s.to_dict() for s in systems]}

    @router.post(
        "/pos/connections",
        dependencies=[Depends(require_auth)],
    )
    async def create_connection(
        body: PosConnectionRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Create a new POS connection."""
        try:
            conn = create_pos_connection(
                user_id=ctx.user_id,
                pos_system=body.pos_system,
                store_name=body.store_name,
                sync_frequency=body.sync_frequency,
                location_id=body.location_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return conn.to_dict()

    @router.get(
        "/pos/connections",
        dependencies=[Depends(require_auth)],
    )
    async def list_connections(
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """List all POS connections for the current user."""
        connections = list_pos_connections(ctx.user_id)
        return {
            "connections": [c.to_dict() for c in connections],
            "total": len(connections),
        }

    @router.post(
        "/pos/connections/{connection_id}/sync",
        dependencies=[Depends(require_auth)],
    )
    async def trigger_pos_sync(
        connection_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Trigger a manual data sync for a POS connection."""
        result = trigger_sync(connection_id, ctx.user_id)
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=result.errors[0] if result.errors else "Sync failed",
            )
        return result.to_dict()

    @router.post(
        "/pos/connections/{connection_id}/disconnect",
        dependencies=[Depends(require_auth)],
    )
    async def disconnect_connection(
        connection_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Disconnect a POS integration."""
        result = disconnect_pos(connection_id, ctx.user_id)
        if not result:
            raise HTTPException(status_code=404, detail="Connection not found")
        return {"message": "Disconnected", "connection_id": connection_id}

    @router.delete(
        "/pos/connections/{connection_id}",
        dependencies=[Depends(require_auth)],
    )
    async def delete_connection_endpoint(
        connection_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Delete a POS connection."""
        result = delete_pos_connection(connection_id, ctx.user_id)
        if not result:
            raise HTTPException(status_code=404, detail="Connection not found")
        return {"message": "Connection deleted", "connection_id": connection_id}

    return router
