"""Conversational diagnostic session endpoints (Phase 11)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..api_models import (
    DiagnosticAnswerRequest,
    DiagnosticAnswerResponse,
    DiagnosticQuestionResponse,
    DiagnosticReportResponse,
    DiagnosticStartRequest,
    DiagnosticStartResponse,
    DiagnosticSummaryResponse,
)
from ..diagnostics import (
    DiagnosticSession,
    enhance_question_with_llm,
    narrate_diagnostic_report,
    render_diagnostic_report,
)
from .state import AppState


def create_diagnostic_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["diagnostic"])

    def _get_session(session_id: str) -> dict:
        """Validate and return diagnostic session data."""
        data = state.diagnostic_sessions.get(session_id)
        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Diagnostic session '{session_id}' not found.",
            )
        return data

    @router.post(
        "/diagnostic/start",
        response_model=DiagnosticStartResponse,
        dependencies=[Depends(require_auth)],
    )
    async def start_diagnostic(
        body: DiagnosticStartRequest,
    ) -> DiagnosticStartResponse:
        """Start a new conversational diagnostic session."""
        session = state.diagnostic_engine.start_session(body.items)

        state.diagnostic_sessions[session.session_id] = {
            "engine": state.diagnostic_engine,
            "session": session,
            "store_name": body.store_name,
            "status": "in_progress",
        }

        return DiagnosticStartResponse(
            session_id=session.session_id,
            store_name=body.store_name,
            total_items=session.items_analyzed,
            negative_items=session.negative_items,
            total_shrinkage=session.total_shrinkage,
            patterns_detected=len(session.patterns),
        )

    @router.get(
        "/diagnostic/{session_id}/question",
        response_model=DiagnosticQuestionResponse | None,
        dependencies=[Depends(require_auth)],
    )
    async def get_diagnostic_question(
        session_id: str,
    ) -> DiagnosticQuestionResponse | None:
        """Get the current question for a diagnostic session."""
        data = _get_session(session_id)
        question = state.diagnostic_engine.get_current_question(data["session"])

        if not question:
            return None

        # Optional: Claude-enhanced conversational question
        if state.settings.anthropic_api_key:
            question = await enhance_question_with_llm(
                question, state.settings.anthropic_api_key
            )

        return DiagnosticQuestionResponse(**question)

    @router.post(
        "/diagnostic/{session_id}/answer",
        response_model=DiagnosticAnswerResponse,
        dependencies=[Depends(require_auth)],
    )
    async def answer_diagnostic(
        session_id: str,
        body: DiagnosticAnswerRequest,
    ) -> DiagnosticAnswerResponse:
        """Submit an answer to the current diagnostic question."""
        data = _get_session(session_id)
        result = state.diagnostic_engine.answer_question(
            data["session"],
            body.classification,
            body.note,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        if result.get("is_complete"):
            data["status"] = "complete"

        next_q = result.get("next_question")
        next_q_model = DiagnosticQuestionResponse(**next_q) if next_q else None

        return DiagnosticAnswerResponse(
            answered=result["answered"],
            progress=result["progress"],
            running_totals=result["running_totals"],
            is_complete=result["is_complete"],
            next_question=next_q_model,
        )

    @router.get(
        "/diagnostic/{session_id}/summary",
        response_model=DiagnosticSummaryResponse,
        dependencies=[Depends(require_auth)],
    )
    async def get_diagnostic_summary(
        session_id: str,
    ) -> DiagnosticSummaryResponse:
        """Get current diagnostic session summary with running totals."""
        data = _get_session(session_id)
        session: DiagnosticSession = data["session"]
        summary = session.get_summary()

        return DiagnosticSummaryResponse(
            session_id=session_id,
            store_name=data["store_name"],
            status=data["status"],
            total_items=session.items_analyzed,
            negative_items=session.negative_items,
            total_shrinkage=summary["total_shrinkage"],
            explained_value=summary["explained_value"],
            unexplained_value=summary["unexplained_value"],
            reduction_percent=summary["reduction_percent"],
            patterns_total=summary["patterns_total"],
            patterns_answered=summary["patterns_answered"],
        )

    @router.get(
        "/diagnostic/{session_id}/report",
        response_model=DiagnosticReportResponse,
        dependencies=[Depends(require_auth)],
    )
    async def get_diagnostic_report(
        session_id: str,
    ) -> DiagnosticReportResponse:
        """Generate the final diagnostic report."""
        data = _get_session(session_id)
        session: DiagnosticSession = data["session"]

        report = state.diagnostic_engine.get_final_report(session)
        rendered = render_diagnostic_report(report)

        # Optional: Claude-narrated closing summary
        if state.settings.anthropic_api_key:
            report = await narrate_diagnostic_report(
                report, state.settings.anthropic_api_key
            )

        return DiagnosticReportResponse(
            session_id=session_id,
            summary=report["summary"],
            by_classification=report["by_classification"],
            items_to_investigate=report["items_to_investigate"],
            journey=report["journey"],
            rendered_text=rendered,
        )

    return router
