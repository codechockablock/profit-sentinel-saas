"""Analysis history and comparison endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..analysis_store import (
    delete_analysis,
    get_analysis,
    get_comparison_pair,
    list_user_analyses,
    rename_analysis,
)
from ..api_models import (
    AnalysisCompareRequest,
    AnalysisCompareResponse,
    AnalysisDetailResponse,
    AnalysisListResponse,
    AnalysisRenameRequest,
)
from ..cross_report import compare_analyses
from ..dual_auth import UserContext


def create_analyses_router(require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["analyses"])

    @router.get(
        "/analyses",
        response_model=AnalysisListResponse,
        dependencies=[Depends(require_auth)],
    )
    async def get_analyses(
        request: Request,
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        ctx: UserContext = Depends(require_auth),
    ) -> AnalysisListResponse:
        """List saved analyses for the current user."""
        analyses = list_user_analyses(ctx.user_id, limit=limit, offset=offset)
        return AnalysisListResponse(analyses=analyses, total=len(analyses))

    # IMPORTANT: compare must be registered BEFORE /analyses/{analysis_id}
    @router.post(
        "/analyses/compare",
        response_model=AnalysisCompareResponse,
        dependencies=[Depends(require_auth)],
    )
    async def compare_analyses_endpoint(
        body: AnalysisCompareRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> AnalysisCompareResponse:
        """Compare two saved analyses for cross-report pattern detection."""
        current = get_analysis(body.current_id, ctx.user_id)
        if not current:
            raise HTTPException(
                status_code=404,
                detail=f"Current analysis '{body.current_id}' not found",
            )
        previous = get_analysis(body.previous_id, ctx.user_id)
        if not previous:
            raise HTTPException(
                status_code=404,
                detail=f"Previous analysis '{body.previous_id}' not found",
            )

        comparison = compare_analyses(current, previous)
        return AnalysisCompareResponse(**comparison.to_dict())

    @router.get(
        "/analyses/{analysis_id}",
        response_model=AnalysisDetailResponse,
        dependencies=[Depends(require_auth)],
    )
    async def get_analysis_detail(
        analysis_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> AnalysisDetailResponse:
        """Get a single saved analysis with full result."""
        record = get_analysis(analysis_id, ctx.user_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis '{analysis_id}' not found",
            )
        return AnalysisDetailResponse(**record)

    @router.patch(
        "/analyses/{analysis_id}",
        dependencies=[Depends(require_auth)],
    )
    async def rename_analysis_endpoint(
        analysis_id: str,
        body: AnalysisRenameRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Rename a saved analysis."""
        updated = rename_analysis(analysis_id, ctx.user_id, body.label)
        if not updated:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis '{analysis_id}' not found",
            )
        return {"message": "Analysis renamed", "analysis_id": analysis_id}

    @router.delete(
        "/analyses/{analysis_id}",
        dependencies=[Depends(require_auth)],
    )
    async def delete_analysis_endpoint(
        analysis_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Delete a saved analysis."""
        deleted = delete_analysis(analysis_id, ctx.user_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis '{analysis_id}' not found",
            )
        return {"message": "Analysis deleted", "analysis_id": analysis_id}

    return router
