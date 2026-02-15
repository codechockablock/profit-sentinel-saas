"""Region management endpoints.

POST   /api/v1/regions              — Create region
GET    /api/v1/regions              — List regions in org
PUT    /api/v1/regions/{region_id}  — Update region
DELETE /api/v1/regions/{region_id}  — Delete region (must be empty)

All endpoints require auth. Organization resolved from user_roles.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..dual_auth import UserContext
from .state import AppState

logger = logging.getLogger("sentinel.routes.regions")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateRegionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class UpdateRegionRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)


class RegionResponse(BaseModel):
    id: str
    org_id: str
    name: str
    created_at: str


# ---------------------------------------------------------------------------
# Region persistence layer
# ---------------------------------------------------------------------------


class RegionStore:
    """Supabase-backed region persistence (PostgREST API)."""

    TABLE = "regions"

    def __init__(self, supabase_url: str = "", service_key: str = "") -> None:
        self._use_supabase = bool(supabase_url and service_key)
        if self._use_supabase:
            self._base_url = f"{supabase_url}/rest/v1/{self.TABLE}"
            self._headers = {
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            }
            logger.info("RegionStore: using Supabase (%s)", supabase_url)
        else:
            self._memory: dict[str, dict[str, Any]] = {}
            logger.info("RegionStore: using in-memory (non-persistent)")

    def _request(
        self,
        method: str,
        path: str = "",
        *,
        params: dict | None = None,
        json_data: dict | list | None = None,
        extra_headers: dict | None = None,
    ) -> httpx.Response:
        url = f"{self._base_url}{path}"
        headers = {**self._headers, **(extra_headers or {})}
        return httpx.request(
            method, url, headers=headers, params=params, json=json_data, timeout=15.0
        )

    # -- Public API --

    def create(self, org_id: str, name: str) -> dict:
        if self._use_supabase:
            resp = self._request(
                "POST",
                json_data={"org_id": org_id, "name": name},
            )
            if resp.status_code in (200, 201):
                rows = resp.json()
                return rows[0] if rows else {}
            if resp.status_code == 409 or "duplicate" in resp.text.lower():
                raise HTTPException(409, f"A region named '{name}' already exists")
            logger.error(
                "RegionStore.create failed (%s): %s", resp.status_code, resp.text
            )
            raise HTTPException(502, "Failed to create region")
        else:
            import uuid

            for rec in self._memory.values():
                if rec["org_id"] == org_id and rec["name"] == name:
                    raise HTTPException(409, f"A region named '{name}' already exists")
            region_id = str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()
            record = {
                "id": region_id,
                "org_id": org_id,
                "name": name,
                "created_at": now,
            }
            self._memory[region_id] = record
            return record

    def list_for_org(self, org_id: str) -> list[dict]:
        if self._use_supabase:
            resp = self._request(
                "GET",
                params={"org_id": f"eq.{org_id}", "order": "name.asc"},
            )
            if resp.status_code == 200:
                return resp.json()
            logger.error(
                "RegionStore.list failed (%s): %s", resp.status_code, resp.text
            )
            return []
        else:
            return sorted(
                [r for r in self._memory.values() if r["org_id"] == org_id],
                key=lambda r: r["name"],
            )

    def get(self, region_id: str, org_id: str) -> dict | None:
        if self._use_supabase:
            resp = self._request(
                "GET",
                params={
                    "id": f"eq.{region_id}",
                    "org_id": f"eq.{org_id}",
                    "limit": "1",
                },
            )
            if resp.status_code == 200:
                rows = resp.json()
                return rows[0] if rows else None
            return None
        else:
            rec = self._memory.get(region_id)
            if rec and rec["org_id"] == org_id:
                return rec
            return None

    def update(self, region_id: str, org_id: str, updates: dict) -> dict | None:
        if self._use_supabase:
            resp = self._request(
                "PATCH",
                params={
                    "id": f"eq.{region_id}",
                    "org_id": f"eq.{org_id}",
                },
                json_data=updates,
            )
            if resp.status_code == 200:
                rows = resp.json()
                return rows[0] if rows else None
            if resp.status_code == 409 or "duplicate" in resp.text.lower():
                raise HTTPException(409, "A region with that name already exists")
            logger.error(
                "RegionStore.update failed (%s): %s", resp.status_code, resp.text
            )
            return None
        else:
            rec = self._memory.get(region_id)
            if rec and rec["org_id"] == org_id:
                rec.update(updates)
                return rec
            return None

    def delete(self, region_id: str, org_id: str) -> bool:
        if self._use_supabase:
            resp = self._request(
                "DELETE",
                params={
                    "id": f"eq.{region_id}",
                    "org_id": f"eq.{org_id}",
                },
            )
            if resp.status_code == 200:
                rows = resp.json()
                return len(rows) > 0
            logger.error(
                "RegionStore.delete failed (%s): %s", resp.status_code, resp.text
            )
            return False
        else:
            rec = self._memory.get(region_id)
            if rec and rec["org_id"] == org_id:
                del self._memory[region_id]
                return True
            return False


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_regions_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["regions"])

    region_store = RegionStore(
        supabase_url=state.settings.supabase_url,
        service_key=state.settings.supabase_service_key,
    )

    # Attach to AppState so other routes can access it
    state.region_store = region_store  # type: ignore[attr-defined]

    def _resolve_org_id(ctx: UserContext) -> str:
        org_store = getattr(state, "org_store", None)
        if org_store is None:
            raise HTTPException(500, "Organization service not initialized")
        org = org_store.ensure_default_org(ctx.user_id)
        return org["id"]

    @router.post("/regions", status_code=201)
    async def create_region(
        body: CreateRegionRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> RegionResponse:
        org_id = _resolve_org_id(ctx)
        record = region_store.create(org_id, body.name)
        return RegionResponse(**record)

    @router.get("/regions")
    async def list_regions(
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        org_id = _resolve_org_id(ctx)
        regions = region_store.list_for_org(org_id)
        return {
            "regions": [RegionResponse(**r).model_dump() for r in regions],
            "total": len(regions),
        }

    @router.put("/regions/{region_id}")
    async def update_region(
        region_id: str,
        body: UpdateRegionRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> RegionResponse:
        org_id = _resolve_org_id(ctx)
        existing = region_store.get(region_id, org_id)
        if not existing:
            raise HTTPException(404, "Region not found")

        updates = {}
        if body.name is not None:
            updates["name"] = body.name

        if not updates:
            return RegionResponse(**existing)

        updated = region_store.update(region_id, org_id, updates)
        if not updated:
            raise HTTPException(500, "Failed to update region")
        return RegionResponse(**updated)

    @router.delete("/regions/{region_id}")
    async def delete_region(
        region_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        org_id = _resolve_org_id(ctx)
        existing = region_store.get(region_id, org_id)
        if not existing:
            raise HTTPException(404, "Region not found")

        # Check if region has stores
        org_store_store = getattr(state, "org_store_store", None)
        if org_store_store:
            stores = org_store_store.list_for_org(org_id, region_id_filter=region_id)
            if stores:
                raise HTTPException(
                    400,
                    f"Cannot delete region '{existing['name']}': "
                    f"it has {len(stores)} store(s). Move or delete them first.",
                )

        deleted = region_store.delete(region_id, org_id)
        if not deleted:
            raise HTTPException(500, "Failed to delete region")
        return {"message": f"Region '{existing['name']}' deleted"}

    return router
