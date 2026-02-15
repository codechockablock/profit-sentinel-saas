"""Org-scoped store management endpoints.

POST   /api/v1/stores              — Create a store in the user's org
GET    /api/v1/stores              — List stores (optional region_id filter)
GET    /api/v1/stores/{store_id}   — Get store detail with latest snapshot
PUT    /api/v1/stores/{store_id}   — Update store
DELETE /api/v1/stores/{store_id}   — Soft delete (set inactive)

Replaces the user-scoped stores router (routes/stores.py).
All operations scoped by organization membership via user_roles.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..dual_auth import UserContext
from .state import AppState

logger = logging.getLogger("sentinel.routes.org_stores")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateOrgStoreRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    address: str = Field("", max_length=500)
    region_id: str | None = None
    store_type: str = Field("hardware_store", max_length=100)


class UpdateOrgStoreRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    address: str | None = Field(None, max_length=500)
    region_id: str | None = None
    store_type: str | None = None
    status: str | None = None


class OrgStoreResponse(BaseModel):
    id: str
    org_id: str
    region_id: str | None = None
    name: str
    address: str
    store_type: str = "hardware_store"
    created_at: str
    updated_at: str
    last_upload_at: str | None = None
    item_count: int = 0
    total_impact: float = 0.0
    exposure_trend: float = 0.0
    status: str = "active"


# ---------------------------------------------------------------------------
# Org-scoped store persistence layer
# ---------------------------------------------------------------------------


class OrgStoreStore:
    """Supabase-backed org-scoped store persistence (PostgREST API).

    Uses the service role key for server-side access (bypasses RLS).
    Falls back gracefully to in-memory when Supabase is unavailable.
    """

    TABLE = "org_stores"

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
            logger.info("OrgStoreStore: using Supabase (%s)", supabase_url)
        else:
            self._memory: dict[str, dict[str, Any]] = {}
            logger.info("OrgStoreStore: using in-memory (non-persistent)")

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

    def _reactivate_inactive(self, org_id: str, name: str) -> dict | None:
        """Try to find an inactive store with this name and reactivate it."""
        if self._use_supabase:
            resp = self._request(
                "GET",
                params={
                    "org_id": f"eq.{org_id}",
                    "name": f"eq.{name}",
                    "status": "eq.inactive",
                    "limit": "1",
                },
            )
            if resp.status_code == 200:
                rows = resp.json()
                if rows:
                    reactivated = self.update(
                        rows[0]["id"], org_id, {"status": "active"}
                    )
                    if reactivated:
                        logger.info(
                            "Reactivated inactive store '%s' in org %s", name, org_id
                        )
                        return reactivated
        else:
            for rec in self._memory.values():
                if (
                    rec["org_id"] == org_id
                    and rec["name"] == name
                    and rec.get("status") == "inactive"
                ):
                    rec["status"] = "active"
                    rec["updated_at"] = datetime.now(UTC).isoformat()
                    return rec
        return None

    def create(
        self,
        org_id: str,
        name: str,
        address: str = "",
        region_id: str | None = None,
        store_type: str = "hardware_store",
    ) -> dict:
        if self._use_supabase:
            data: dict[str, Any] = {
                "org_id": org_id,
                "name": name,
                "address": address,
                "store_type": store_type,
            }
            if region_id:
                data["region_id"] = region_id

            resp = self._request("POST", json_data=data)
            if resp.status_code in (200, 201):
                rows = resp.json()
                return rows[0] if rows else {}
            if resp.status_code == 409 or "duplicate" in resp.text.lower():
                # Check if an inactive store with this name exists and reactivate it
                reactivated = self._reactivate_inactive(org_id, name)
                if reactivated:
                    return reactivated
                raise HTTPException(409, f"A store named '{name}' already exists")
            logger.error(
                "OrgStoreStore.create failed (%s): %s", resp.status_code, resp.text
            )
            raise HTTPException(502, "Failed to create store")
        else:
            import uuid

            for rec in self._memory.values():
                if rec["org_id"] == org_id and rec["name"] == name:
                    if rec.get("status") == "inactive":
                        rec["status"] = "active"
                        rec["updated_at"] = datetime.now(UTC).isoformat()
                        return rec
                    raise HTTPException(409, f"A store named '{name}' already exists")
            store_id = str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()
            record = {
                "id": store_id,
                "org_id": org_id,
                "region_id": region_id,
                "name": name,
                "address": address,
                "store_type": store_type,
                "created_at": now,
                "updated_at": now,
                "last_upload_at": None,
                "item_count": 0,
                "total_impact": 0,
                "exposure_trend": 0,
                "status": "active",
            }
            self._memory[store_id] = record
            return record

    def list_for_org(
        self, org_id: str, region_id_filter: str | None = None
    ) -> list[dict]:
        if self._use_supabase:
            params: dict[str, str] = {
                "org_id": f"eq.{org_id}",
                "status": "eq.active",
                "order": "name.asc",
            }
            if region_id_filter:
                params["region_id"] = f"eq.{region_id_filter}"
            resp = self._request("GET", params=params)
            if resp.status_code == 200:
                return resp.json()
            logger.error(
                "OrgStoreStore.list failed (%s): %s", resp.status_code, resp.text
            )
            return []
        else:
            results = [
                r
                for r in self._memory.values()
                if r["org_id"] == org_id and r.get("status", "active") == "active"
            ]
            if region_id_filter:
                results = [r for r in results if r.get("region_id") == region_id_filter]
            return sorted(results, key=lambda r: r["name"])

    def get(self, store_id: str, org_id: str) -> dict | None:
        if self._use_supabase:
            resp = self._request(
                "GET",
                params={
                    "id": f"eq.{store_id}",
                    "org_id": f"eq.{org_id}",
                    "limit": "1",
                },
            )
            if resp.status_code == 200:
                rows = resp.json()
                return rows[0] if rows else None
            return None
        else:
            rec = self._memory.get(store_id)
            if rec and rec["org_id"] == org_id:
                return rec
            return None

    def update(self, store_id: str, org_id: str, updates: dict) -> dict | None:
        updates["updated_at"] = datetime.now(UTC).isoformat()
        if self._use_supabase:
            resp = self._request(
                "PATCH",
                params={
                    "id": f"eq.{store_id}",
                    "org_id": f"eq.{org_id}",
                },
                json_data=updates,
            )
            if resp.status_code == 200:
                rows = resp.json()
                return rows[0] if rows else None
            if resp.status_code == 409 or "duplicate" in resp.text.lower():
                raise HTTPException(409, "A store with that name already exists")
            logger.error(
                "OrgStoreStore.update failed (%s): %s", resp.status_code, resp.text
            )
            return None
        else:
            rec = self._memory.get(store_id)
            if rec and rec["org_id"] == org_id:
                rec.update(updates)
                return rec
            return None

    def soft_delete(self, store_id: str, org_id: str) -> bool:
        """Soft delete by setting status to inactive."""
        result = self.update(store_id, org_id, {"status": "inactive"})
        return result is not None

    def update_metadata(self, store_id: str, org_id: str, metadata: dict) -> None:
        """Update store metadata after analysis."""
        self.update(store_id, org_id, metadata)

    def ensure_default_store(self, org_id: str) -> dict:
        """Ensure org has at least one store. Auto-creates 'Main Store' if none."""
        stores = self.list_for_org(org_id)
        if stores:
            return stores[0]
        return self.create(org_id, "Main Store")


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_org_stores_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["stores"])

    org_store_store = OrgStoreStore(
        supabase_url=state.settings.supabase_url,
        service_key=state.settings.supabase_service_key,
    )

    # Attach to AppState so other routes can access it
    state.org_store_store = org_store_store  # type: ignore[attr-defined]

    def _resolve_org_id(ctx: UserContext) -> str:
        org_store = getattr(state, "org_store", None)
        if org_store is None:
            raise HTTPException(500, "Organization service not initialized")
        org = org_store.ensure_default_org(ctx.user_id)
        return org["id"]

    @router.post("/stores", status_code=201)
    async def create_store(
        body: CreateOrgStoreRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> OrgStoreResponse:
        """Create a new store in the user's organization."""
        org_id = _resolve_org_id(ctx)
        record = org_store_store.create(
            org_id, body.name, body.address, body.region_id, body.store_type
        )
        return OrgStoreResponse(**record)

    @router.get("/stores")
    async def list_stores(
        region_id: str | None = Query(default=None, description="Filter by region"),
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """List all active stores in the user's organization."""
        org_id = _resolve_org_id(ctx)
        stores = org_store_store.list_for_org(org_id, region_id_filter=region_id)
        if not stores:
            default = org_store_store.create(org_id, "Main Store")
            stores = [default]
        return {
            "stores": [OrgStoreResponse(**s).model_dump() for s in stores],
            "total": len(stores),
        }

    @router.get("/stores/{store_id}")
    async def get_store(
        store_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> OrgStoreResponse:
        """Get a single store by ID."""
        org_id = _resolve_org_id(ctx)
        record = org_store_store.get(store_id, org_id)
        if not record:
            raise HTTPException(404, "Store not found")
        return OrgStoreResponse(**record)

    @router.put("/stores/{store_id}")
    async def update_store(
        store_id: str,
        body: UpdateOrgStoreRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> OrgStoreResponse:
        """Update a store's details."""
        org_id = _resolve_org_id(ctx)
        existing = org_store_store.get(store_id, org_id)
        if not existing:
            raise HTTPException(404, "Store not found")

        updates: dict[str, Any] = {}
        if body.name is not None:
            updates["name"] = body.name
        if body.address is not None:
            updates["address"] = body.address
        if body.region_id is not None:
            updates["region_id"] = body.region_id if body.region_id else None
        if body.store_type is not None:
            updates["store_type"] = body.store_type
        if body.status is not None:
            if body.status not in ("active", "inactive"):
                raise HTTPException(400, "Status must be 'active' or 'inactive'")
            updates["status"] = body.status

        if not updates:
            return OrgStoreResponse(**existing)

        updated = org_store_store.update(store_id, org_id, updates)
        if not updated:
            raise HTTPException(500, "Failed to update store")
        return OrgStoreResponse(**updated)

    @router.delete("/stores/{store_id}")
    async def delete_store(
        store_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Soft-delete a store (set inactive)."""
        org_id = _resolve_org_id(ctx)
        existing = org_store_store.get(store_id, org_id)
        if not existing:
            raise HTTPException(404, "Store not found")

        deleted = org_store_store.soft_delete(store_id, org_id)
        if not deleted:
            raise HTTPException(500, "Failed to delete store")
        return {"message": f"Store '{existing['name']}' deactivated"}

    return router
