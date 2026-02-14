"""Store management endpoints.

POST   /api/v1/stores              — Create a store
GET    /api/v1/stores              — List user's stores
GET    /api/v1/stores/{store_id}   — Get store details
PUT    /api/v1/stores/{store_id}   — Update store (name, address)
DELETE /api/v1/stores/{store_id}   — Delete store (soft: only if no recent data)

All endpoints require auth. All operations scoped by ctx.user_id.
Stores are persisted to Supabase via PostgREST (same pattern as analysis_store.py).
Falls back to an in-memory store when Supabase is not configured.
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

logger = logging.getLogger("sentinel.routes.stores")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateStoreRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    address: str = Field("", max_length=500)


class UpdateStoreRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    address: str | None = Field(None, max_length=500)


class StoreResponse(BaseModel):
    id: str
    user_id: str
    name: str
    address: str
    created_at: str
    updated_at: str
    last_upload_at: str | None = None
    item_count: int = 0
    total_impact: float = 0.0


# ---------------------------------------------------------------------------
# Store persistence layer
# ---------------------------------------------------------------------------


class StoreStore:
    """Supabase-backed store persistence (PostgREST API).

    Uses the service role key for server-side access (bypasses RLS).
    Falls back gracefully to in-memory when Supabase is unavailable.
    """

    TABLE = "stores"

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
            logger.info("StoreStore: using Supabase (%s)", supabase_url)
        else:
            self._memory: dict[str, dict[str, Any]] = {}
            self._counter = 0
            logger.info("StoreStore: using in-memory (non-persistent)")

    # -- Supabase helpers --

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

    def create(self, user_id: str, name: str, address: str = "") -> dict:
        if self._use_supabase:
            resp = self._request(
                "POST",
                json_data={"user_id": user_id, "name": name, "address": address},
            )
            if resp.status_code in (200, 201):
                rows = resp.json()
                return rows[0] if rows else {}
            if resp.status_code == 409 or "duplicate" in resp.text.lower():
                raise HTTPException(409, f"A store named '{name}' already exists")
            logger.error(
                "StoreStore.create failed (%s): %s", resp.status_code, resp.text
            )
            raise HTTPException(502, "Failed to create store")
        else:
            # In-memory
            import uuid

            for rec in self._memory.values():
                if rec["user_id"] == user_id and rec["name"] == name:
                    raise HTTPException(409, f"A store named '{name}' already exists")
            store_id = str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()
            record = {
                "id": store_id,
                "user_id": user_id,
                "name": name,
                "address": address,
                "created_at": now,
                "updated_at": now,
                "last_upload_at": None,
                "item_count": 0,
                "total_impact": 0,
            }
            self._memory[store_id] = record
            return record

    def list_for_user(self, user_id: str) -> list[dict]:
        if self._use_supabase:
            resp = self._request(
                "GET",
                params={
                    "user_id": f"eq.{user_id}",
                    "order": "created_at.asc",
                },
            )
            if resp.status_code == 200:
                return resp.json()
            logger.error("StoreStore.list failed (%s): %s", resp.status_code, resp.text)
            return []
        else:
            return sorted(
                [r for r in self._memory.values() if r["user_id"] == user_id],
                key=lambda r: r["created_at"],
            )

    def get(self, store_id: str, user_id: str) -> dict | None:
        if self._use_supabase:
            resp = self._request(
                "GET",
                params={
                    "id": f"eq.{store_id}",
                    "user_id": f"eq.{user_id}",
                    "limit": "1",
                },
            )
            if resp.status_code == 200:
                rows = resp.json()
                return rows[0] if rows else None
            return None
        else:
            rec = self._memory.get(store_id)
            if rec and rec["user_id"] == user_id:
                return rec
            return None

    def update(self, store_id: str, user_id: str, updates: dict) -> dict | None:
        updates["updated_at"] = datetime.now(UTC).isoformat()
        if self._use_supabase:
            resp = self._request(
                "PATCH",
                params={
                    "id": f"eq.{store_id}",
                    "user_id": f"eq.{user_id}",
                },
                json_data=updates,
            )
            if resp.status_code == 200:
                rows = resp.json()
                return rows[0] if rows else None
            if resp.status_code == 409 or "duplicate" in resp.text.lower():
                raise HTTPException(409, "A store with that name already exists")
            logger.error(
                "StoreStore.update failed (%s): %s", resp.status_code, resp.text
            )
            return None
        else:
            rec = self._memory.get(store_id)
            if rec and rec["user_id"] == user_id:
                rec.update(updates)
                return rec
            return None

    def delete(self, store_id: str, user_id: str) -> bool:
        if self._use_supabase:
            resp = self._request(
                "DELETE",
                params={
                    "id": f"eq.{store_id}",
                    "user_id": f"eq.{user_id}",
                },
            )
            if resp.status_code == 200:
                rows = resp.json()
                return len(rows) > 0
            logger.error(
                "StoreStore.delete failed (%s): %s", resp.status_code, resp.text
            )
            return False
        else:
            rec = self._memory.get(store_id)
            if rec and rec["user_id"] == user_id:
                del self._memory[store_id]
                return True
            return False

    def update_metadata(self, store_id: str, user_id: str, metadata: dict) -> None:
        """Update store metadata after analysis (last_upload_at, item_count, total_impact)."""
        self.update(store_id, user_id, metadata)

    def ensure_default_store(self, user_id: str) -> dict:
        """Ensure user has at least one store. Auto-creates 'Main Store' if none exist."""
        stores = self.list_for_user(user_id)
        if stores:
            return stores[0]
        return self.create(user_id, "Main Store")


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_stores_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["stores"])

    store_store = StoreStore(
        supabase_url=state.settings.supabase_url,
        service_key=state.settings.supabase_service_key,
    )

    # Attach to AppState so other routes can access it
    state.store_store = store_store  # type: ignore[attr-defined]

    @router.post("/stores", status_code=201)
    async def create_store(
        body: CreateStoreRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> StoreResponse:
        """Create a new store for the current user."""
        record = store_store.create(ctx.user_id, body.name, body.address)
        return StoreResponse(**record)

    @router.get("/stores")
    async def list_stores(
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """List all stores for the current user.

        Auto-creates 'Main Store' on first call if user has no stores.
        """
        stores = store_store.list_for_user(ctx.user_id)
        if not stores:
            default = store_store.create(ctx.user_id, "Main Store")
            stores = [default]
        return {
            "stores": [StoreResponse(**s).model_dump() for s in stores],
            "total": len(stores),
        }

    @router.get("/stores/{store_id}")
    async def get_store(
        store_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> StoreResponse:
        """Get a single store by ID."""
        record = store_store.get(store_id, ctx.user_id)
        if not record:
            raise HTTPException(404, "Store not found")
        return StoreResponse(**record)

    @router.put("/stores/{store_id}")
    async def update_store(
        store_id: str,
        body: UpdateStoreRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> StoreResponse:
        """Update a store's name or address."""
        existing = store_store.get(store_id, ctx.user_id)
        if not existing:
            raise HTTPException(404, "Store not found")

        updates = {}
        if body.name is not None:
            updates["name"] = body.name
        if body.address is not None:
            updates["address"] = body.address

        if not updates:
            return StoreResponse(**existing)

        updated = store_store.update(store_id, ctx.user_id, updates)
        if not updated:
            raise HTTPException(500, "Failed to update store")
        return StoreResponse(**updated)

    @router.delete("/stores/{store_id}")
    async def delete_store(
        store_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Delete a store.

        Users must have at least one store, so the last store cannot be deleted.
        """
        existing = store_store.get(store_id, ctx.user_id)
        if not existing:
            raise HTTPException(404, "Store not found")

        # Prevent deleting the last store
        all_stores = store_store.list_for_user(ctx.user_id)
        if len(all_stores) <= 1:
            raise HTTPException(
                400, "Cannot delete your only store. Create another store first."
            )

        deleted = store_store.delete(store_id, ctx.user_id)
        if not deleted:
            raise HTTPException(500, "Failed to delete store")
        return {"message": f"Store '{existing['name']}' deleted"}

    return router
