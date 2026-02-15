"""Agent action queue endpoints.

GET    /api/v1/actions                — List actions (filterable)
POST   /api/v1/actions                — Create manual action
POST   /api/v1/actions/{id}/approve   — Approve a pending action
POST   /api/v1/actions/{id}/defer     — Defer with optional until-date
POST   /api/v1/actions/{id}/reject    — Reject with reason
POST   /api/v1/actions/{id}/complete  — Mark completed with outcome notes

Full audit trail for agent-recommended and manual actions.
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

logger = logging.getLogger("sentinel.routes.actions")

# ---------------------------------------------------------------------------
# Valid status transitions
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"approved", "deferred", "rejected"},
    "approved": {"completed", "deferred"},
    "deferred": {"pending", "approved", "rejected"},
    "auto_approved": {"completed"},
}

VALID_ACTION_TYPES = {
    "transfer",
    "clearance",
    "reorder",
    "price_adjustment",
    "vendor_contact",
    "threshold_change",
    "custom",
}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateActionRequest(BaseModel):
    store_id: str | None = None
    action_type: str = Field("custom", description="Type of action")
    description: str = Field(..., min_length=1, max_length=2000)
    reasoning: str = Field("", max_length=2000)
    financial_impact: float = 0.0
    confidence: float = 0.5


class DeferRequest(BaseModel):
    deferred_until: str | None = None


class RejectRequest(BaseModel):
    reason: str = Field("", max_length=1000)


class CompleteRequest(BaseModel):
    outcome_notes: str = Field("", max_length=2000)


class ActionResponse(BaseModel):
    id: str
    org_id: str
    store_id: str | None = None
    user_id: str
    action_type: str
    description: str
    reasoning: str | None = None
    financial_impact: float = 0.0
    confidence: float = 0.0
    status: str
    source: str = "agent"
    linked_finding_id: str | None = None
    created_at: str
    decided_at: str | None = None
    completed_at: str | None = None
    deferred_until: str | None = None
    outcome_notes: str | None = None


# ---------------------------------------------------------------------------
# Action persistence layer
# ---------------------------------------------------------------------------


class ActionStore:
    """Supabase-backed agent_actions persistence (PostgREST API)."""

    TABLE = "agent_actions"

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
            logger.info("ActionStore: using Supabase (%s)", supabase_url)
        else:
            self._memory: dict[str, dict[str, Any]] = {}
            logger.info("ActionStore: using in-memory (non-persistent)")

    def _request(
        self,
        method: str,
        path: str = "",
        *,
        params: dict | None = None,
        json_data: dict | list | None = None,
    ) -> httpx.Response:
        url = f"{self._base_url}{path}"
        return httpx.request(
            method,
            url,
            headers=self._headers,
            params=params,
            json=json_data,
            timeout=15.0,
        )

    # -- Public API --

    def create(
        self,
        org_id: str,
        store_id: str | None = None,
        user_id: str = "",
        action_type: str = "custom",
        description: str = "",
        reasoning: str = "",
        financial_impact: float = 0.0,
        confidence: float = 0.5,
        source: str = "agent",
        linked_finding_id: str | None = None,
    ) -> dict:
        data: dict[str, Any] = {
            "org_id": org_id,
            "user_id": user_id,
            "action_type": action_type,
            "description": description,
            "reasoning": reasoning,
            "financial_impact": financial_impact,
            "confidence": confidence,
            "source": source,
            "status": "pending",
        }
        if store_id:
            data["store_id"] = store_id
        if linked_finding_id:
            data["linked_finding_id"] = linked_finding_id

        if self._use_supabase:
            resp = self._request("POST", json_data=data)
            if resp.status_code in (200, 201):
                rows = resp.json()
                return rows[0] if rows else {}
            logger.error(
                "ActionStore.create failed (%s): %s", resp.status_code, resp.text
            )
            raise HTTPException(502, "Failed to create action")
        else:
            import uuid

            action_id = str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()
            data["id"] = action_id
            data["created_at"] = now
            data["decided_at"] = None
            data["completed_at"] = None
            data["deferred_until"] = None
            data["outcome_notes"] = None
            self._memory[action_id] = data
            return data

    def list_for_org(
        self,
        org_id: str,
        status: str | None = None,
        store_id: str | None = None,
        action_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        if self._use_supabase:
            params: dict[str, str] = {
                "org_id": f"eq.{org_id}",
                "order": "created_at.desc",
                "limit": str(limit),
                "offset": str(offset),
            }
            if status:
                params["status"] = f"eq.{status}"
            if store_id:
                params["store_id"] = f"eq.{store_id}"
            if action_type:
                params["action_type"] = f"eq.{action_type}"

            resp = self._request("GET", params=params)
            if resp.status_code == 200:
                return resp.json()
            logger.error(
                "ActionStore.list failed (%s): %s", resp.status_code, resp.text
            )
            return []
        else:
            results = [r for r in self._memory.values() if r.get("org_id") == org_id]
            if status:
                results = [r for r in results if r.get("status") == status]
            if store_id:
                results = [r for r in results if r.get("store_id") == store_id]
            if action_type:
                results = [r for r in results if r.get("action_type") == action_type]
            results.sort(key=lambda r: r.get("created_at", ""), reverse=True)
            return results[offset : offset + limit]

    def get(self, action_id: str, org_id: str) -> dict | None:
        if self._use_supabase:
            resp = self._request(
                "GET",
                params={
                    "id": f"eq.{action_id}",
                    "org_id": f"eq.{org_id}",
                    "limit": "1",
                },
            )
            if resp.status_code == 200:
                rows = resp.json()
                return rows[0] if rows else None
            return None
        else:
            rec = self._memory.get(action_id)
            if rec and rec.get("org_id") == org_id:
                return rec
            return None

    def update_status(
        self,
        action_id: str,
        org_id: str,
        new_status: str,
        *,
        outcome_notes: str | None = None,
        deferred_until: str | None = None,
    ) -> dict | None:
        now = datetime.now(UTC).isoformat()
        updates: dict[str, Any] = {"status": new_status}

        if new_status in ("approved", "deferred", "rejected"):
            updates["decided_at"] = now
        if new_status == "completed":
            updates["completed_at"] = now
        if outcome_notes is not None:
            updates["outcome_notes"] = outcome_notes
        if deferred_until is not None:
            updates["deferred_until"] = deferred_until

        if self._use_supabase:
            resp = self._request(
                "PATCH",
                params={
                    "id": f"eq.{action_id}",
                    "org_id": f"eq.{org_id}",
                },
                json_data=updates,
            )
            if resp.status_code == 200:
                rows = resp.json()
                return rows[0] if rows else None
            logger.error(
                "ActionStore.update_status failed (%s): %s",
                resp.status_code,
                resp.text,
            )
            return None
        else:
            rec = self._memory.get(action_id)
            if rec and rec.get("org_id") == org_id:
                rec.update(updates)
                return rec
            return None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_actions_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["actions"])

    action_store = ActionStore(
        supabase_url=state.settings.supabase_url,
        service_key=state.settings.supabase_service_key,
    )

    # Attach to AppState so briefing generator can create actions
    state.action_store = action_store  # type: ignore[attr-defined]

    def _resolve_org_id(ctx: UserContext) -> str:
        org_store = getattr(state, "org_store", None)
        if org_store is None:
            raise HTTPException(500, "Organization service not initialized")
        org = org_store.ensure_default_org(ctx.user_id)
        return org["id"]

    @router.get("/actions")
    async def list_actions(
        status: str | None = Query(default=None, description="Filter by status"),
        store_id: str | None = Query(default=None, description="Filter by store"),
        action_type: str | None = Query(default=None, description="Filter by type"),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """List actions for the user's organization."""
        org_id = _resolve_org_id(ctx)
        actions = action_store.list_for_org(
            org_id,
            status=status,
            store_id=store_id,
            action_type=action_type,
            limit=limit,
            offset=offset,
        )
        return {
            "actions": actions,
            "total": len(actions),
        }

    @router.post("/actions", status_code=201)
    async def create_action(
        body: CreateActionRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> ActionResponse:
        """Create a manual action."""
        org_id = _resolve_org_id(ctx)
        if body.action_type not in VALID_ACTION_TYPES:
            raise HTTPException(
                400,
                f"Invalid action_type. Must be one of: {', '.join(sorted(VALID_ACTION_TYPES))}",
            )
        record = action_store.create(
            org_id=org_id,
            store_id=body.store_id,
            user_id=ctx.user_id,
            action_type=body.action_type,
            description=body.description,
            reasoning=body.reasoning,
            financial_impact=body.financial_impact,
            confidence=body.confidence,
            source="manual",
        )
        return ActionResponse(**record)

    @router.post("/actions/{action_id}/approve")
    async def approve_action(
        action_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> ActionResponse:
        """Approve a pending action."""
        org_id = _resolve_org_id(ctx)
        action = action_store.get(action_id, org_id)
        if not action:
            raise HTTPException(404, "Action not found")

        current = action.get("status", "")
        if "approved" not in VALID_TRANSITIONS.get(current, set()):
            raise HTTPException(400, f"Cannot approve action in '{current}' status")

        updated = action_store.update_status(action_id, org_id, "approved")
        if not updated:
            raise HTTPException(500, "Failed to approve action")
        return ActionResponse(**updated)

    @router.post("/actions/{action_id}/defer")
    async def defer_action(
        action_id: str,
        body: DeferRequest | None = None,
        ctx: UserContext = Depends(require_auth),
    ) -> ActionResponse:
        """Defer an action with optional until-date."""
        org_id = _resolve_org_id(ctx)
        action = action_store.get(action_id, org_id)
        if not action:
            raise HTTPException(404, "Action not found")

        current = action.get("status", "")
        if "deferred" not in VALID_TRANSITIONS.get(current, set()):
            raise HTTPException(400, f"Cannot defer action in '{current}' status")

        deferred_until = body.deferred_until if body else None
        updated = action_store.update_status(
            action_id, org_id, "deferred", deferred_until=deferred_until
        )
        if not updated:
            raise HTTPException(500, "Failed to defer action")
        return ActionResponse(**updated)

    @router.post("/actions/{action_id}/reject")
    async def reject_action(
        action_id: str,
        body: RejectRequest | None = None,
        ctx: UserContext = Depends(require_auth),
    ) -> ActionResponse:
        """Reject an action with optional reason."""
        org_id = _resolve_org_id(ctx)
        action = action_store.get(action_id, org_id)
        if not action:
            raise HTTPException(404, "Action not found")

        current = action.get("status", "")
        if "rejected" not in VALID_TRANSITIONS.get(current, set()):
            raise HTTPException(400, f"Cannot reject action in '{current}' status")

        reason = body.reason if body else ""
        updated = action_store.update_status(
            action_id, org_id, "rejected", outcome_notes=reason
        )
        if not updated:
            raise HTTPException(500, "Failed to reject action")
        return ActionResponse(**updated)

    @router.post("/actions/{action_id}/complete")
    async def complete_action(
        action_id: str,
        body: CompleteRequest | None = None,
        ctx: UserContext = Depends(require_auth),
    ) -> ActionResponse:
        """Mark an action as completed with optional outcome notes."""
        org_id = _resolve_org_id(ctx)
        action = action_store.get(action_id, org_id)
        if not action:
            raise HTTPException(404, "Action not found")

        current = action.get("status", "")
        if "completed" not in VALID_TRANSITIONS.get(current, set()):
            raise HTTPException(400, f"Cannot complete action in '{current}' status")

        notes = body.outcome_notes if body else ""
        updated = action_store.update_status(
            action_id, org_id, "completed", outcome_notes=notes
        )
        if not updated:
            raise HTTPException(500, "Failed to complete action")
        return ActionResponse(**updated)

    return router
