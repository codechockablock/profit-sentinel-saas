"""Dashboard summary endpoint.

GET /api/v1/dashboard — pre-computed dashboard summary with recovery
                        amounts, department status, finding counts,
                        and Engine 2 predictions.

Engine 1 findings always display. Engine 2 predictions are additive —
if the world model is unhealthy or warming up, prediction_count = 0
and the dashboard still works fine from Engine 1 data alone.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from fastapi import APIRouter, Depends

from ..dual_auth import UserContext
from .state import AppState

logger = logging.getLogger("sentinel.routes.dashboard")


def create_dashboard_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["dashboard"])

    @router.get("/dashboard")
    async def dashboard_summary(
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Pre-computed dashboard summary.

        Returns:
            recovery_total: Total potential recovery across all active findings
            finding_count: Number of active findings
            department_status: Per-department traffic-light (green/yellow/red)
            top_findings: Top 5 findings by dollar impact
            prediction_count: Number of active predictions (confidence > 0.7)
            engine2_status: World model status (active/warming_up/not_initialized)
        """
        # ---------------------------------------------------------------
        # Engine 1 data: from this user's digest cache only
        # ---------------------------------------------------------------
        user_cache = state.digest_cache.get(ctx.user_id, {})
        all_issues = []
        for entry in user_cache.values():
            if entry.is_expired:
                continue
            for issue in entry.digest.issues:
                all_issues.append(issue)

        # Compute recovery total
        recovery_total = sum(
            getattr(issue, "dollar_impact", 0.0) for issue in all_issues
        )

        # Department status (traffic-light)
        dept_findings: dict[str, list] = defaultdict(list)
        for issue in all_issues:
            dept = getattr(issue, "department", "Uncategorized") or "Uncategorized"
            dept_findings[dept].append(issue)

        department_status = {}
        for dept, issues in dept_findings.items():
            max_impact = max(
                (getattr(i, "dollar_impact", 0.0) for i in issues), default=0
            )
            if max_impact > 500 or len(issues) >= 3:
                status = "red"
            elif len(issues) > 0:
                status = "yellow"
            else:
                status = "green"
            department_status[dept] = {
                "status": status,
                "finding_count": len(issues),
                "total_impact": sum(getattr(i, "dollar_impact", 0.0) for i in issues),
            }

        # Top findings by dollar impact
        sorted_issues = sorted(
            all_issues,
            key=lambda i: getattr(i, "dollar_impact", 0.0),
            reverse=True,
        )
        top_findings = []
        for issue in sorted_issues[:5]:
            # Derive severity from priority_score (Issue model has no severity field)
            if issue.priority_score >= 8.0:
                severity = "critical"
            elif issue.priority_score >= 5.0:
                severity = "high"
            elif issue.priority_score >= 3.0:
                severity = "medium"
            else:
                severity = "low"

            top_findings.append(
                {
                    "id": issue.id,
                    "type": issue.issue_type,
                    "title": issue.issue_type.display_name,
                    "severity": severity,
                    "dollar_impact": issue.dollar_impact,
                    "department": getattr(issue, "department", None),
                }
            )

        # ---------------------------------------------------------------
        # Engine 2 data: predictions from per-user PredictiveEngine (additive)
        # ---------------------------------------------------------------
        prediction_count = 0
        top_predictions = []
        engine2_status = "not_initialized"
        engine2_summary = {}

        pipeline = state.get_user_world_model(ctx.user_id)
        if pipeline is not None:
            try:
                # Count active interventions with confidence > 0.7
                if hasattr(pipeline, "predictive"):
                    # Expire stale interventions first
                    pipeline.predictive.expire_stale_interventions()

                    high_conf = [
                        i
                        for i in pipeline.predictive.active_interventions.values()
                        if i.confidence > 0.7
                    ]
                    prediction_count = len(high_conf)

                    # Top 3 most urgent predictions
                    sorted_interventions = (
                        pipeline.predictive.prioritized_interventions(top_k=3)
                    )
                    top_predictions = [i.to_dict() for i in sorted_interventions]

                # Pipeline status summary
                engine2_summary = pipeline.pipeline_status()

                # Determine overall status
                n_observations = sum(len(h) for h in pipeline.entity_history.values())
                if n_observations == 0:
                    engine2_status = "warming_up"
                else:
                    engine2_status = "active"

            except (KeyError, ValueError, TypeError) as e:
                logger.warning("Engine 2 dashboard data error (non-fatal): %s", e)
                engine2_status = "error"
            except Exception as e:
                logger.error(
                    "Engine 2 dashboard data failed (non-fatal): %s", e, exc_info=True
                )
                engine2_status = "error"

        # ---------------------------------------------------------------
        # Transfer matching stats (additive, per-user)
        # ---------------------------------------------------------------
        transfer_stats = {"stores_registered": 0, "total_recommendations": 0}
        matcher = state.get_user_transfer_matcher(ctx.user_id)
        if matcher is not None:
            try:
                transfer_stats["stores_registered"] = len(matcher.agents)
            except (AttributeError, TypeError) as e:
                logger.debug("Transfer stats check failed: %s", e)

        return {
            # Engine 1 (always available)
            "recovery_total": round(recovery_total, 2),
            "finding_count": len(all_issues),
            "department_count": len(dept_findings),
            "department_status": department_status,
            "top_findings": top_findings,
            # Engine 2 (additive — 0/empty if unavailable)
            "prediction_count": prediction_count,
            "top_predictions": top_predictions,
            "engine2_status": engine2_status,
            "engine2_summary": engine2_summary,
            "transfer_stats": transfer_stats,
        }

    return router
