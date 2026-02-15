"""Eagle's Eye executive endpoint.

GET /api/v1/eagle-eye — Full business view in one call.

Aggregates data from org_stores, store_snapshots, agent_actions,
and regions into a single response for the executive dashboard.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..dual_auth import UserContext
from .state import AppState

logger = logging.getLogger("sentinel.routes.eagle_eye")

# ---------------------------------------------------------------------------
# Health thresholds (configurable later)
# ---------------------------------------------------------------------------

GREEN_THRESHOLD = 25_000  # total_impact < $25K = green
YELLOW_THRESHOLD = 75_000  # total_impact < $75K = yellow, >= $75K = red
STALE_DATA_DAYS = 7  # data older than 7 days → red


# ---------------------------------------------------------------------------
# Store snapshots persistence layer
# ---------------------------------------------------------------------------


class SnapshotStore:
    """Supabase-backed store_snapshots persistence (PostgREST API)."""

    TABLE = "store_snapshots"

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
            logger.info("SnapshotStore: using Supabase (%s)", supabase_url)
        else:
            self._memory: list[dict[str, Any]] = []
            logger.info("SnapshotStore: using in-memory (non-persistent)")

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

    def create(self, snapshot_data: dict) -> dict:
        """Insert a new store snapshot."""
        if self._use_supabase:
            resp = self._request("POST", json_data=snapshot_data)
            if resp.status_code in (200, 201):
                rows = resp.json()
                return rows[0] if rows else {}
            logger.error(
                "SnapshotStore.create failed (%s): %s", resp.status_code, resp.text
            )
            return {}
        else:
            import uuid

            snapshot_data["id"] = str(uuid.uuid4())
            snapshot_data.setdefault("snapshot_at", datetime.now(UTC).isoformat())
            self._memory.append(snapshot_data)
            return snapshot_data

    def latest_for_org(self, org_id: str) -> list[dict]:
        """Get latest snapshot per store for an org."""
        if self._use_supabase:
            resp = self._request(
                "GET",
                params={
                    "org_id": f"eq.{org_id}",
                    "order": "snapshot_at.desc",
                    "limit": "500",
                },
            )
            if resp.status_code == 200:
                rows = resp.json()
                # Dedup: keep only latest per store_id
                seen: dict[str, dict] = {}
                for row in rows:
                    sid = row.get("store_id")
                    if sid and sid not in seen:
                        seen[sid] = row
                return list(seen.values())
            logger.error(
                "SnapshotStore.latest_for_org failed (%s): %s",
                resp.status_code,
                resp.text,
            )
            return []
        else:
            org_snaps = [s for s in self._memory if s.get("org_id") == org_id]
            org_snaps.sort(key=lambda s: s.get("snapshot_at", ""), reverse=True)
            seen: dict[str, dict] = {}
            for s in org_snaps:
                sid = s.get("store_id")
                if sid and sid not in seen:
                    seen[sid] = s
            return list(seen.values())

    def latest_for_store(self, store_id: str) -> dict | None:
        """Get latest snapshot for a single store."""
        if self._use_supabase:
            resp = self._request(
                "GET",
                params={
                    "store_id": f"eq.{store_id}",
                    "order": "snapshot_at.desc",
                    "limit": "1",
                },
            )
            if resp.status_code == 200:
                rows = resp.json()
                return rows[0] if rows else None
            return None
        else:
            store_snaps = [s for s in self._memory if s.get("store_id") == store_id]
            store_snaps.sort(key=lambda s: s.get("snapshot_at", ""), reverse=True)
            return store_snaps[0] if store_snaps else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_store_status(store: dict, snapshot: dict | None) -> str:
    """Compute store health status: healthy, attention, or critical."""
    impact = float(snapshot.get("total_impact_high", 0)) if snapshot else 0
    last_upload = store.get("last_upload_at")

    # Stale data check
    if last_upload:
        try:
            upload_dt = datetime.fromisoformat(last_upload.replace("Z", "+00:00"))
            if datetime.now(UTC) - upload_dt > timedelta(days=STALE_DATA_DAYS):
                return "critical"
        except (ValueError, TypeError):
            pass

    if impact >= YELLOW_THRESHOLD:
        return "critical"
    if impact >= GREEN_THRESHOLD:
        return "attention"
    return "healthy"


def _top_issue_for_snapshot(snapshot: dict | None) -> str | None:
    """Determine the top issue for a store based on its snapshot."""
    if not snapshot:
        return None

    issues = []
    dead_stock = float(snapshot.get("dead_stock_capital", 0))
    if dead_stock > 0:
        issues.append(("Dead stock", dead_stock))
    margin = float(snapshot.get("margin_erosion_impact", 0))
    if margin > 0:
        issues.append(("Margin erosion", margin))
    shrinkage = float(snapshot.get("shrinkage_impact", 0))
    if shrinkage > 0:
        issues.append(("Shrinkage", shrinkage))

    if not issues:
        return None

    top = max(issues, key=lambda x: x[1])
    return f"{top[0]}: ${top[1]:,.0f}"


# ---------------------------------------------------------------------------
# Eagle-eye data assembly
# ---------------------------------------------------------------------------


def build_eagle_eye_data(state: AppState, user_id: str) -> dict:
    """Build the full eagle-eye response.

    This function is also called internally by the briefing generator
    to provide context data for Claude.
    """
    org_store = getattr(state, "org_store", None)
    if not org_store:
        raise HTTPException(500, "Organization service not initialized")

    org = org_store.get_for_user(user_id)
    if not org:
        return {
            "org": None,
            "regions": [],
            "unassigned_stores": [],
            "network_alerts": [],
        }

    org_id = org["id"]

    # Fetch data from all stores
    org_store_store = getattr(state, "org_store_store", None)
    stores = org_store_store.list_for_org(org_id) if org_store_store else []

    region_store = getattr(state, "region_store", None)
    regions = region_store.list_for_org(org_id) if region_store else []

    snapshot_store = getattr(state, "snapshot_store", None)
    snapshots = snapshot_store.latest_for_org(org_id) if snapshot_store else []
    snapshot_map = {s["store_id"]: s for s in snapshots}

    # Fetch pending actions count
    action_store = getattr(state, "action_store", None)
    pending_actions = []
    completed_actions_30d = 0
    if action_store:
        try:
            all_actions = action_store.list_for_org(org_id, limit=500)
            pending_actions = [a for a in all_actions if a.get("status") == "pending"]
            cutoff = (datetime.now(UTC) - timedelta(days=30)).isoformat()
            completed_actions_30d = sum(
                1
                for a in all_actions
                if a.get("status") == "completed"
                and (a.get("completed_at") or "") > cutoff
            )
        except Exception as e:
            logger.warning("Failed to fetch actions for eagle-eye: %s", e)

    # Build store summaries
    store_summaries = []
    for store in stores:
        snap = snapshot_map.get(store["id"])
        status = _compute_store_status(store, snap)
        store_pending = sum(
            1 for a in pending_actions if a.get("store_id") == store["id"]
        )

        summary = {
            "id": store["id"],
            "name": store["name"],
            "region_id": store.get("region_id"),
            "status": status,
            "total_impact": float(snap.get("total_impact_high", 0)) if snap else 0,
            "exposure_trend": float(store.get("exposure_trend", 0)),
            "item_count": (
                int(snap.get("item_count", 0))
                if snap
                else int(store.get("item_count", 0))
            ),
            "flagged_count": int(snap.get("flagged_count", 0)) if snap else 0,
            "last_upload_at": store.get("last_upload_at"),
            "pending_actions": store_pending,
            "top_issue": _top_issue_for_snapshot(snap),
        }
        store_summaries.append(summary)

    # Group by region
    stores_by_region: dict[str | None, list[dict]] = defaultdict(list)
    for s in store_summaries:
        stores_by_region[s.get("region_id")].append(s)

    region_summaries = []
    for region in regions:
        region_stores = stores_by_region.get(region["id"], [])
        total_exposure = sum(s["total_impact"] for s in region_stores)
        region_pending = sum(s["pending_actions"] for s in region_stores)

        region_summaries.append(
            {
                "id": region["id"],
                "name": region["name"],
                "store_count": len(region_stores),
                "total_exposure": total_exposure,
                "exposure_trend": (
                    sum(s["exposure_trend"] for s in region_stores) / len(region_stores)
                    if region_stores
                    else 0
                ),
                "pending_actions": region_pending,
                "stores": region_stores,
            }
        )

    unassigned = stores_by_region.get(None, [])

    # Network totals
    total_exposure = sum(s["total_impact"] for s in store_summaries)
    avg_trend = (
        sum(s["exposure_trend"] for s in store_summaries) / len(store_summaries)
        if store_summaries
        else 0
    )
    # Network alerts (stores that need immediate attention)
    network_alerts = []
    for s in store_summaries:
        if s["status"] == "critical" and s["total_impact"] > 0:
            network_alerts.append(
                {
                    "type": "high_exposure",
                    "description": f"{s['name']}: {s['top_issue'] or 'High exposure'}",
                    "affected_stores": 1,
                    "total_impact": s["total_impact"],
                    "recommended_action": "Review findings and take action",
                }
            )

    return {
        "org": {
            "id": org["id"],
            "name": org["name"],
            "total_stores": len(store_summaries),
            "total_exposure": total_exposure,
            "exposure_trend": avg_trend,
            "total_pending_actions": len(pending_actions),
            "total_completed_actions_30d": completed_actions_30d,
        },
        "regions": region_summaries,
        "unassigned_stores": unassigned,
        "network_alerts": network_alerts[:10],
    }


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_eagle_eye_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["eagle-eye"])

    snapshot_store = SnapshotStore(
        supabase_url=state.settings.supabase_url,
        service_key=state.settings.supabase_service_key,
    )

    # Attach to AppState so upload_routes can create snapshots
    state.snapshot_store = snapshot_store  # type: ignore[attr-defined]

    @router.get("/eagle-eye")
    async def get_eagle_eye(
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Get the full executive business overview.

        Returns organization, regions, stores with health status,
        pending actions, and network alerts.
        """
        return build_eagle_eye_data(state, ctx.user_id)

    return router
