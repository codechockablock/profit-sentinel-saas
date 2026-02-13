"""Digest and subscription endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

logger = logging.getLogger("sentinel.routes.digest")

from ..api_models import (
    DigestResponse,
    DigestSendRequest,
    DigestSendResponse,
    SchedulerStatusResponse,
    SubscribeRequest,
    SubscribeResponse,
    SubscriptionListResponse,
)
from ..digest_scheduler import (
    add_subscription,
    list_subscriptions,
    remove_subscription,
)
from ..dual_auth import UserContext
from ..llm_layer import render_digest
from .state import AppState


def create_digest_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["digest"])

    @router.get(
        "/digest",
        response_model=DigestResponse,
    )
    async def get_digest(
        stores: str | None = Query(
            default=None,
            description="Comma-separated store IDs (e.g. store-7,store-12)",
        ),
        top_k: int = Query(default=5, ge=1, le=20),
        ctx: UserContext = Depends(require_auth),
    ) -> DigestResponse:
        """Run pipeline and return morning digest."""
        store_list = [s.strip() for s in stores.split(",")] if stores else None
        digest = state.get_or_run_digest(ctx.user_id, store_list, top_k)

        return DigestResponse(
            digest=digest,
            rendered_text=render_digest(digest),
            generated_at=digest.generated_at,
            store_filter=digest.store_filter,
            issue_count=digest.summary.total_issues,
            total_dollar_impact=digest.summary.total_dollar_impact,
        )

    # IMPORTANT: These must be registered BEFORE /digest/{store_id}
    # to prevent FastAPI from matching "subscriptions" as a store_id.

    @router.post(
        "/digest/subscribe",
        response_model=SubscribeResponse,
        dependencies=[Depends(require_auth)],
    )
    async def subscribe_digest(body: SubscribeRequest) -> SubscribeResponse:
        """Subscribe to morning digest emails."""
        sub = add_subscription(
            body.email,
            stores=body.stores,
            send_hour=body.send_hour,
            tz=body.timezone,
        )
        return SubscribeResponse(subscription=sub, message="Subscription created")

    @router.get(
        "/digest/subscriptions",
        response_model=SubscriptionListResponse,
        dependencies=[Depends(require_auth)],
    )
    async def get_subscriptions() -> SubscriptionListResponse:
        """List all active digest subscriptions."""
        subs = list_subscriptions()
        return SubscriptionListResponse(subscriptions=subs, total=len(subs))

    @router.delete(
        "/digest/subscribe/{email}",
        dependencies=[Depends(require_auth)],
    )
    async def unsubscribe_digest(email: str) -> dict:
        """Remove a digest subscription."""
        removed = remove_subscription(email)
        if not removed:
            raise HTTPException(
                status_code=404, detail=f"Subscription for '{email}' not found"
            )
        return {"message": f"Unsubscribed {email}"}

    @router.post(
        "/digest/send",
        response_model=DigestSendResponse,
        dependencies=[Depends(require_auth)],
    )
    async def send_digest_now(body: DigestSendRequest) -> DigestSendResponse:
        """Send a digest email immediately (on-demand)."""
        if not state.settings.resend_api_key:
            raise HTTPException(
                status_code=503,
                detail="Email delivery not configured (RESEND_API_KEY not set)",
            )
        try:
            result = await state.digest_scheduler.send_now(body.email)
            return DigestSendResponse(
                email_id=result.get("id"),
                message=f"Digest sent to {body.email}",
            )
        except Exception as exc:
            logger.warning("Digest email send failed: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="Unable to send digest email. Please try again later.",
            )

    @router.get(
        "/digest/scheduler-status",
        response_model=SchedulerStatusResponse,
        dependencies=[Depends(require_auth)],
    )
    async def scheduler_status() -> SchedulerStatusResponse:
        """Get digest scheduler status."""
        subs = list_subscriptions()
        return SchedulerStatusResponse(
            enabled=state.settings.digest_email_enabled,
            running=state.digest_scheduler.is_running,
            subscribers=len(subs),
            send_hour=state.settings.digest_send_hour,
        )

    # Single-store digest (parameterized — must come AFTER specific routes)
    @router.get(
        "/digest/{store_id}",
        response_model=DigestResponse,
    )
    async def get_store_digest(
        store_id: str,
        top_k: int = Query(default=5, ge=1, le=20),
        ctx: UserContext = Depends(require_auth),
    ) -> DigestResponse:
        """Single-store digest view."""
        digest = state.get_or_run_digest(ctx.user_id, [store_id], top_k)

        return DigestResponse(
            digest=digest,
            rendered_text=render_digest(digest),
            generated_at=digest.generated_at,
            store_filter=digest.store_filter,
            issue_count=digest.summary.total_issues,
            total_dollar_impact=digest.summary.total_dollar_impact,
        )

    return router
