"""Dashboard summary endpoint.

GET /api/v1/dashboard — pre-computed dashboard summary with recovery
                        amounts, department status, and finding counts.

STUB: Computes from cached digest data. Production should add
      time-series tracking for trends.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from fastapi import APIRouter, Depends

from .state import AppState

logger = logging.getLogger("sentinel.routes.dashboard")


def create_dashboard_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["dashboard"])

    @router.get("/dashboard")
    async def dashboard_summary(
        _user=Depends(require_auth),
    ) -> dict:
        """Pre-computed dashboard summary.

        Returns:
            recovery_total: Total potential recovery across all active findings
            finding_count: Number of active findings
            department_status: Per-department traffic-light (green/yellow/red)
            top_findings: Top 5 findings by dollar impact
            prediction_count: Number of active predictions (confidence > 0.7)
        """
        all_issues = []
        for entry in state.digest_cache.values():
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
            top_findings.append(
                {
                    "id": issue.id,
                    "type": issue.issue_type,
                    "title": issue.title,
                    "severity": issue.severity,
                    "dollar_impact": getattr(issue, "dollar_impact", 0.0),
                    "department": getattr(issue, "department", None),
                }
            )

        return {
            "recovery_total": round(recovery_total, 2),
            "finding_count": len(all_issues),
            "department_count": len(dept_findings),
            "department_status": department_status,
            "top_findings": top_findings,
            "prediction_count": 0,  # TODO: wire to PredictiveEngine
        }

    return router
