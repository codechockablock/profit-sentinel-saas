"""Vendor call prep and scoring endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..api_models import (
    CoopReportResponse,
    VendorCallResponse,
    VendorScoresResponse,
)
from ..llm_layer import (
    render_call_prep,
    render_coop_report,
    render_inventory_health_summary,
)
from ..vendor_scoring import score_vendors
from .state import AppState


def create_vendor_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["vendor"])

    @router.get(
        "/vendor-call/{issue_id}",
        response_model=VendorCallResponse,
        dependencies=[Depends(require_auth)],
    )
    async def vendor_call_prep(issue_id: str) -> VendorCallResponse:
        """Prepare a vendor call brief for an issue."""
        issue = state.find_issue(issue_id)
        prep = state.vendor_assistant.prepare_call(issue)
        rendered = render_call_prep(prep)

        return VendorCallResponse(
            call_prep=prep,
            rendered_text=rendered,
        )

    @router.get(
        "/coop/{store_id}",
        response_model=CoopReportResponse,
        dependencies=[Depends(require_auth)],
    )
    async def coop_report(store_id: str) -> CoopReportResponse:
        """Generate co-op intelligence report for a store."""
        cache_key = f"{store_id}:5"
        entry = state.digest_cache.get(cache_key)

        if not entry or entry.is_expired:
            digest = state.get_or_run_digest([store_id])
        else:
            digest = entry.digest

        report = state.generator.generate_coop_report(
            digest,
            store_id=store_id,
        )

        rendered = render_coop_report(report)
        health_text = None
        if report.health_report:
            health_text = render_inventory_health_summary(report.health_report)

        rebate_rendered = []
        if report.rebate_statuses:
            rebate_rendered = report.rebate_statuses

        return CoopReportResponse(
            report=report,
            rendered_text=rendered,
            health_summary=health_text,
            rebate_statuses=rebate_rendered,
            total_opportunity=report.total_opportunity,
        )

    @router.get(
        "/vendor-scores",
        response_model=VendorScoresResponse,
        dependencies=[Depends(require_auth)],
    )
    async def get_vendor_scores(
        store_id: str | None = Query(default=None),
    ) -> VendorScoresResponse:
        """Score all vendors on quality, delivery, pricing, and compliance."""
        digest = state.get_or_run_digest([store_id] if store_id else None)
        report = score_vendors(digest, store_id=store_id)
        return VendorScoresResponse(**report.to_dict())

    return router
