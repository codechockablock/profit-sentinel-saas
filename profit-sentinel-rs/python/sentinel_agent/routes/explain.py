"""Symbolic reasoning and explain endpoints (Phase 13)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..api_models import (
    BackwardChainRequest,
    BackwardChainResponse,
    ExplainResponse,
)
from .state import AppState


def create_explain_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["explain"])

    @router.get(
        "/explain/{issue_id}",
        response_model=ExplainResponse,
        dependencies=[Depends(require_auth)],
    )
    async def explain_issue(issue_id: str) -> ExplainResponse:
        """Generate full proof tree explaining an issue's root cause."""
        issue = state.find_issue(issue_id)
        proof = state.reasoner.explain(issue)

        return ExplainResponse(
            issue_id=issue_id,
            proof_tree=proof.to_dict(),
            rendered_text=proof.render(),
        )

    @router.post(
        "/explain/{issue_id}/why",
        response_model=BackwardChainResponse,
        dependencies=[Depends(require_auth)],
    )
    async def backward_chain_issue(
        issue_id: str,
        body: BackwardChainRequest,
    ) -> BackwardChainResponse:
        """Backward-chain from a goal to explain how it was derived."""
        issue = state.find_issue(issue_id)
        steps = state.reasoner.backward_chain(issue, body.goal)

        return BackwardChainResponse(
            issue_id=issue_id,
            goal=body.goal,
            reasoning_steps=steps,
        )

    return router
