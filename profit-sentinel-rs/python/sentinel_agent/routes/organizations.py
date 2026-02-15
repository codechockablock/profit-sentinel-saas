"""Organization management endpoints.

POST   /api/v1/org              — Create organization (first-time setup)
GET    /api/v1/org              — Get user's organization
PUT    /api/v1/org              — Update organization name

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

logger = logging.getLogger("sentinel.routes.organizations")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateOrgRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class UpdateOrgRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)


class OrgResponse(BaseModel):
    id: str
    name: str
    owner_user_id: str
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Organization persistence layer
# ---------------------------------------------------------------------------


class OrgStore:
    """Supabase-backed organization persistence (PostgREST API).

    Uses the service role key for server-side access (bypasses RLS).
    Falls back gracefully to in-memory when Supabase is unavailable.
    """

    ORG_TABLE = "organizations"
    ROLE_TABLE = "user_roles"

    def __init__(self, supabase_url: str = "", service_key: str = "") -> None:
        self._use_supabase = bool(supabase_url and service_key)
        if self._use_supabase:
            self._base_url = f"{supabase_url}/rest/v1"
            self._headers = {
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            }
            logger.info("OrgStore: using Supabase (%s)", supabase_url)
        else:
            self._orgs: dict[str, dict[str, Any]] = {}
            self._roles: dict[str, dict[str, Any]] = {}
            logger.info("OrgStore: using in-memory (non-persistent)")

    def _request(
        self,
        method: str,
        table: str,
        path: str = "",
        *,
        params: dict | None = None,
        json_data: dict | list | None = None,
        extra_headers: dict | None = None,
    ) -> httpx.Response:
        url = f"{self._base_url}/{table}{path}"
        headers = {**self._headers, **(extra_headers or {})}
        return httpx.request(
            method, url, headers=headers, params=params, json=json_data, timeout=15.0
        )

    # -- Public API --

    def create(self, user_id: str, name: str) -> dict:
        """Create an organization and assign the user as owner."""
        if self._use_supabase:
            # Check if user already has an org
            existing = self.get_for_user(user_id)
            if existing:
                raise HTTPException(409, "User already has an organization")

            # Create org
            resp = self._request(
                "POST",
                self.ORG_TABLE,
                json_data={"owner_user_id": user_id, "name": name},
            )
            if resp.status_code not in (200, 201):
                logger.error(
                    "OrgStore.create failed (%s): %s", resp.status_code, resp.text
                )
                detail = "Failed to create organization"
                if resp.status_code == 404 or "does not exist" in resp.text:
                    detail = (
                        "Organization tables not found. "
                        "Please run the Eagle's Eye SQL migration (011_eagle_eye_schema.sql) in Supabase."
                    )
                raise HTTPException(502, detail)

            rows = resp.json()
            org = rows[0] if rows else {}
            org_id = org.get("id")

            # Create owner role
            role_resp = self._request(
                "POST",
                self.ROLE_TABLE,
                json_data={
                    "user_id": user_id,
                    "org_id": org_id,
                    "role": "owner",
                    "scope_type": "business",
                },
            )
            if role_resp.status_code not in (200, 201):
                logger.error(
                    "OrgStore: role creation failed (%s): %s",
                    role_resp.status_code,
                    role_resp.text,
                )
                # Org was created but role failed — still return org
            return org
        else:
            import uuid

            existing = self.get_for_user(user_id)
            if existing:
                raise HTTPException(409, "User already has an organization")

            org_id = str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()
            record = {
                "id": org_id,
                "name": name,
                "owner_user_id": user_id,
                "created_at": now,
                "updated_at": now,
            }
            self._orgs[org_id] = record

            role_id = str(uuid.uuid4())
            self._roles[role_id] = {
                "id": role_id,
                "user_id": user_id,
                "org_id": org_id,
                "role": "owner",
                "scope_type": "business",
                "scope_id": None,
                "created_at": now,
            }
            return record

    def get_for_user(self, user_id: str) -> dict | None:
        """Get the organization for a user via user_roles lookup."""
        if self._use_supabase:
            # First find the user's role to get org_id
            resp = self._request(
                "GET",
                self.ROLE_TABLE,
                params={
                    "user_id": f"eq.{user_id}",
                    "limit": "1",
                    "select": "org_id",
                },
            )
            if resp.status_code != 200 or not resp.json():
                return None

            org_id = resp.json()[0]["org_id"]

            # Now fetch the org
            org_resp = self._request(
                "GET",
                self.ORG_TABLE,
                params={"id": f"eq.{org_id}", "limit": "1"},
            )
            if org_resp.status_code == 200 and org_resp.json():
                return org_resp.json()[0]
            return None
        else:
            for role in self._roles.values():
                if role["user_id"] == user_id:
                    org_id = role["org_id"]
                    return self._orgs.get(org_id)
            return None

    def get_by_id(self, org_id: str) -> dict | None:
        """Get organization by ID."""
        if self._use_supabase:
            resp = self._request(
                "GET",
                self.ORG_TABLE,
                params={"id": f"eq.{org_id}", "limit": "1"},
            )
            if resp.status_code == 200 and resp.json():
                return resp.json()[0]
            return None
        else:
            return self._orgs.get(org_id)

    def update(self, org_id: str, user_id: str, updates: dict) -> dict | None:
        """Update org (owner only)."""
        updates["updated_at"] = datetime.now(UTC).isoformat()
        if self._use_supabase:
            resp = self._request(
                "PATCH",
                self.ORG_TABLE,
                params={
                    "id": f"eq.{org_id}",
                    "owner_user_id": f"eq.{user_id}",
                },
                json_data=updates,
            )
            if resp.status_code == 200 and resp.json():
                return resp.json()[0]
            logger.error("OrgStore.update failed (%s): %s", resp.status_code, resp.text)
            return None
        else:
            org = self._orgs.get(org_id)
            if org and org["owner_user_id"] == user_id:
                org.update(updates)
                return org
            return None

    def get_user_role(self, user_id: str, org_id: str) -> str | None:
        """Get the user's role in an organization."""
        if self._use_supabase:
            resp = self._request(
                "GET",
                self.ROLE_TABLE,
                params={
                    "user_id": f"eq.{user_id}",
                    "org_id": f"eq.{org_id}",
                    "limit": "1",
                    "select": "role",
                },
            )
            if resp.status_code == 200 and resp.json():
                return resp.json()[0]["role"]
            return None
        else:
            for role in self._roles.values():
                if role["user_id"] == user_id and role["org_id"] == org_id:
                    return role["role"]
            return None

    def ensure_default_org(self, user_id: str) -> dict:
        """Ensure user has an organization. Auto-creates one if none exist."""
        org = self.get_for_user(user_id)
        if org:
            return org
        return self.create(user_id, "My Organization")


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_org_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["organization"])

    org_store = OrgStore(
        supabase_url=state.settings.supabase_url,
        service_key=state.settings.supabase_service_key,
    )

    # Attach to AppState so other routes can access it
    state.org_store = org_store  # type: ignore[attr-defined]

    @router.post("/org", status_code=201)
    async def create_org(
        body: CreateOrgRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> OrgResponse:
        """Create an organization. Users can only have one organization."""
        record = org_store.create(ctx.user_id, body.name)
        return OrgResponse(**record)

    @router.get("/org")
    async def get_org(
        ctx: UserContext = Depends(require_auth),
    ) -> OrgResponse:
        """Get the user's organization. Auto-creates if none exists."""
        org = org_store.ensure_default_org(ctx.user_id)
        return OrgResponse(**org)

    @router.put("/org")
    async def update_org(
        body: UpdateOrgRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> OrgResponse:
        """Update organization name (owner only)."""
        org = org_store.get_for_user(ctx.user_id)
        if not org:
            raise HTTPException(404, "Organization not found")

        updates = {}
        if body.name is not None:
            updates["name"] = body.name

        if not updates:
            return OrgResponse(**org)

        updated = org_store.update(org["id"], ctx.user_id, updates)
        if not updated:
            raise HTTPException(403, "Only the organization owner can update it")
        return OrgResponse(**updated)

    return router
