"""Findings endpoint — paginated findings with acknowledge support.

GET  /api/v1/findings       — list findings for current user
POST /api/v1/findings/{id}/acknowledge — mark a finding as acknowledged
POST /api/v1/findings/{id}/restore     — restore an acknowledged finding

Engine 1 findings always display. Engine 2 enriches findings with
tier classification, prediction data, and confidence scores when
available. If Engine 2 is unavailable, findings show without enrichment.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query

from ..dual_auth import UserContext
from .state import AppState

logger = logging.getLogger("sentinel.routes.findings")


def create_findings_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["findings"])

    # In-memory acknowledge store keyed by user_id (production: move to Supabase)
    acknowledged_ids: dict[str, set[str]] = defaultdict(set)

    def _enrich_finding(finding: dict, user_id: str) -> dict:
        """Add Engine 2 enrichment to a finding if world model is available.

        This is additive — it never removes Engine 1 data, only adds
        Engine 2 insights (tier, predictions, confidence) on top.
        """
        pipeline = state.get_user_world_model(user_id)
        if pipeline is None:
            return finding

        try:
            entity_id = finding.get("id", "")
            sku = finding.get("sku", entity_id)

            # Check if we have observation history for this entity
            # entity_history keys are "store_id:entity_id"
            matching_keys = [
                k for k in pipeline.entity_history if k.endswith(f":{sku}")
            ]

            if matching_keys:
                history_key = matching_keys[0]
                history = pipeline.entity_history[history_key]
                store_id = history_key.split(":")[0]

                # Add observation count (shows Engine 2 is tracking)
                finding["engine2_observations"] = len(history)

                # Generate predictions if enough history
                if len(history) >= 7:
                    interventions = pipeline.predict_interventions(
                        store_id=store_id,
                        entity_id=sku,
                    )
                    if interventions:
                        top = interventions[0]
                        finding["prediction"] = top.to_dict()
            else:
                finding["engine2_observations"] = 0

        except Exception as e:
            # Enrichment failure is silent — Engine 1 data is preserved
            logger.debug("Finding enrichment failed for %s: %s", finding.get("id"), e)

        return finding

    @router.get("/findings")
    async def list_findings(
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        status: str | None = Query(None, pattern="^(active|acknowledged|all)$"),
        sort_by: str | None = Query(
            "dollar_impact", pattern="^(dollar_impact|priority|date)$"
        ),
        department: str | None = Query(None),
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """List findings with pagination, filtering, and sort.

        Query params:
            page: Page number (1-indexed)
            page_size: Items per page (max 100)
            status: Filter — active, acknowledged, or all
            sort_by: Sort order — dollar_impact, priority, or date
            department: Filter by department/category
        """
        # Collect all issues from cached digests for THIS USER (Engine 1 data)
        user_ack = acknowledged_ids[ctx.user_id]
        user_cache = state.digest_cache.get(ctx.user_id, {})
        all_findings = []
        for entry in user_cache.values():
            if entry.is_expired:
                continue
            for issue in entry.digest.issues:
                # Derive severity from priority_score
                if issue.priority_score >= 8.0:
                    severity = "critical"
                elif issue.priority_score >= 5.0:
                    severity = "high"
                elif issue.priority_score >= 3.0:
                    severity = "medium"
                else:
                    severity = "low"

                finding = {
                    "id": issue.id,
                    "type": issue.issue_type,
                    "title": issue.issue_type.display_name,
                    "description": issue.context,
                    "severity": severity,
                    "dollar_impact": issue.dollar_impact,
                    "department": getattr(issue, "department", None),
                    "recommended_action": issue.root_cause_display,
                    "acknowledged": issue.id in user_ack,
                    "sku": issue.skus[0].sku_id if issue.skus else None,
                }
                # Engine 2 enrichment (additive, never blocks)
                finding = _enrich_finding(finding, ctx.user_id)
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

        # Engine 2 status summary (per-user pipeline)
        engine2_status = "not_initialized"
        user_pipeline = state.get_user_world_model(ctx.user_id)
        if user_pipeline is not None:
            try:
                n_obs = sum(len(h) for h in user_pipeline.entity_history.values())
                engine2_status = "active" if n_obs > 0 else "warming_up"
            except Exception:
                engine2_status = "error"

        return {
            "findings": page_items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": max(1, (total + page_size - 1) // page_size),
            },
            "engine2_status": engine2_status,
        }

    @router.post("/findings/{finding_id}/acknowledge")
    async def acknowledge_finding(
        finding_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Mark a finding as acknowledged for the current user."""
        acknowledged_ids[ctx.user_id].add(finding_id)
        logger.info("Finding acknowledged: %s (user=%s)", finding_id, ctx.user_id)
        return {"id": finding_id, "acknowledged": True}

    @router.post("/findings/{finding_id}/restore")
    async def restore_finding(
        finding_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Restore an acknowledged finding to active for the current user."""
        acknowledged_ids[ctx.user_id].discard(finding_id)
        logger.info("Finding restored: %s (user=%s)", finding_id, ctx.user_id)
        return {"id": finding_id, "acknowledged": False}

    return router
