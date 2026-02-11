"""Findings endpoint — paginated findings with acknowledge support.

GET  /api/v1/findings       — list findings for current user
POST /api/v1/findings/{id}/acknowledge — mark a finding as acknowledged
POST /api/v1/findings/{id}/restore     — restore an acknowledged finding

STUB: Responses use in-memory store. Production should read from
      Supabase analysis results and persist acknowledge state.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from .state import AppState

logger = logging.getLogger("sentinel.routes.findings")


def create_findings_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["findings"])

    # In-memory acknowledge store (production: move to Supabase)
    acknowledged_ids: set[str] = set()

    @router.get("/findings")
    async def list_findings(
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        status: str | None = Query(None, regex="^(active|acknowledged|all)$"),
        sort_by: str | None = Query(
            "dollar_impact", regex="^(dollar_impact|priority|date)$"
        ),
        department: str | None = Query(None),
        _user=Depends(require_auth),
    ) -> dict:
        """List findings with pagination, filtering, and sort.

        Query params:
            page: Page number (1-indexed)
            page_size: Items per page (max 100)
            status: Filter — active, acknowledged, or all
            sort_by: Sort order — dollar_impact, priority, or date
            department: Filter by department/category
        """
        # Collect all issues from cached digests
        all_findings = []
        for entry in state.digest_cache.values():
            if entry.is_expired:
                continue
            for issue in entry.digest.issues:
                finding = {
                    "id": issue.id,
                    "type": issue.issue_type,
                    "title": issue.title,
                    "description": issue.description,
                    "severity": issue.severity,
                    "dollar_impact": getattr(issue, "dollar_impact", 0.0),
                    "department": getattr(issue, "department", None),
                    "recommended_action": getattr(issue, "recommendation", None),
                    "acknowledged": issue.id in acknowledged_ids,
                }
                all_findings.append(finding)

        # Filter by status
        filter_status = status or "active"
        if filter_status == "active":
            all_findings = [f for f in all_findings if not f["acknowledged"]]
        elif filter_status == "acknowledged":
            all_findings = [f for f in all_findings if f["acknowledged"]]
        # "all" returns everything

        # Filter by department
        if department:
            all_findings = [
                f
                for f in all_findings
                if f.get("department", "").lower() == department.lower()
            ]

        # Sort
        if sort_by == "dollar_impact":
            all_findings.sort(key=lambda f: f.get("dollar_impact", 0), reverse=True)
        elif sort_by == "priority":
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            all_findings.sort(
                key=lambda f: severity_order.get(f.get("severity", "low"), 4)
            )

        # Paginate
        total = len(all_findings)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = all_findings[start:end]

        return {
            "findings": page_items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": max(1, (total + page_size - 1) // page_size),
            },
        }

    @router.post("/findings/{finding_id}/acknowledge")
    async def acknowledge_finding(
        finding_id: str,
        _user=Depends(require_auth),
    ) -> dict:
        """Mark a finding as acknowledged."""
        acknowledged_ids.add(finding_id)
        logger.info(f"Finding acknowledged: {finding_id}")
        return {"id": finding_id, "acknowledged": True}

    @router.post("/findings/{finding_id}/restore")
    async def restore_finding(
        finding_id: str,
        _user=Depends(require_auth),
    ) -> dict:
        """Restore an acknowledged finding to active."""
        acknowledged_ids.discard(finding_id)
        logger.info(f"Finding restored: {finding_id}")
        return {"id": finding_id, "acknowledged": False}

    return router
