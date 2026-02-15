"""Agent briefing endpoints.

GET  /api/v1/briefing         — Get or generate agent briefing
POST /api/v1/briefing/refresh — Force regenerate briefing

Returns a cached briefing if available (non-expired, matching data hash).
Otherwise generates a new one using the Anthropic API.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..dual_auth import UserContext
from ..services.briefing_generator import BriefingGenerator
from .eagle_eye import build_eagle_eye_data
from .state import AppState

logger = logging.getLogger("sentinel.routes.agent_briefing")


def create_briefing_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["briefing"])

    generator = BriefingGenerator(
        anthropic_api_key=state.settings.anthropic_api_key,
        supabase_url=state.settings.supabase_url,
        service_key=state.settings.supabase_service_key,
    )

    # Attach to AppState
    state.briefing_generator = generator  # type: ignore[attr-defined]

    def _generate_briefing(ctx: UserContext, force: bool = False) -> dict:
        """Core briefing logic — shared by GET and POST."""
        # Build eagle-eye data for context
        business_data = build_eagle_eye_data(state, ctx.user_id)

        if not business_data.get("org"):
            return {
                "briefing": (
                    "Welcome to Profit Sentinel. Set up your organization and "
                    "add stores to get started with executive briefings."
                ),
                "action_items": [],
                "generated_at": None,
                "expires_at": None,
            }

        org = business_data["org"]
        org_id = org["id"]
        org_name = org["name"]

        # Determine user role
        org_store = getattr(state, "org_store", None)
        role = "owner"
        if org_store:
            user_role = org_store.get_user_role(ctx.user_id, org_id)
            if user_role:
                role = user_role

        # Check cache (unless forced refresh)
        data_hash = generator.compute_data_hash(business_data)
        if not force:
            cached = generator.get_cached(org_id, ctx.user_id, data_hash)
            if cached:
                logger.info(
                    "Returning cached briefing for user=%s org=%s",
                    ctx.user_id,
                    org_id,
                )
                return cached

        # Generate new briefing
        logger.info(
            "Generating new briefing for user=%s org=%s (force=%s)",
            ctx.user_id,
            org_id,
            force,
        )
        result = generator.generate(
            org_id=org_id,
            user_id=ctx.user_id,
            role=role,
            org_name=org_name,
            business_data=business_data,
        )

        # Create agent_actions for each action item
        action_store = getattr(state, "action_store", None)
        if action_store and result.get("action_items"):
            for item in result["action_items"]:
                try:
                    action_store.create(
                        org_id=org_id,
                        store_id=item.get("store_id"),
                        user_id=ctx.user_id,
                        action_type=item.get("type", "custom"),
                        description=item.get("description", ""),
                        reasoning=item.get("reasoning", ""),
                        financial_impact=float(item.get("financial_impact", 0)),
                        confidence=float(item.get("confidence", 0.5)),
                        source="agent",
                    )
                except Exception as e:
                    logger.warning("Failed to create action from briefing: %s", e)

        return result

    @router.get("/briefing")
    async def get_briefing(
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Get the current briefing (cached or newly generated)."""
        return _generate_briefing(ctx, force=False)

    @router.post("/briefing/refresh")
    async def refresh_briefing(
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Force-regenerate the briefing (ignores cache)."""
        return _generate_briefing(ctx, force=True)

    return router
