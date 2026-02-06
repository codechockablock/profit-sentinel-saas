"""Sentinel Sidecar API.

FastAPI application serving the Profit Sentinel mobile interface.
Bridges the Rust pipeline and Python agent layer to REST endpoints
for the 6 AM executive mobile experience.

Endpoints:
    GET  /health                      — health check
    GET  /api/v1/digest               — morning digest
    GET  /api/v1/digest/{store_id}    — single-store digest
    POST /api/v1/delegate             — create task from issue
    GET  /api/v1/tasks                — list delegated tasks
    GET  /api/v1/tasks/{task_id}      — single task detail
    PATCH /api/v1/tasks/{task_id}     — update task status
    GET  /api/v1/vendor-call/{issue_id} — vendor call prep
    GET  /api/v1/coop/{store_id}      — co-op intelligence report
    GET  /api/v1/explain/{issue_id}   — proof tree for issue root cause
    POST /api/v1/explain/{issue_id}/why — backward-chain from a goal
    GET  /                            — mobile web UI (static)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api_models import (
    BackwardChainRequest,
    BackwardChainResponse,
    CoopReportResponse,
    DelegateRequest,
    DelegateResponse,
    DiagnosticAnswerRequest,
    DiagnosticAnswerResponse,
    DiagnosticQuestionResponse,
    DiagnosticReportResponse,
    DiagnosticStartRequest,
    DiagnosticStartResponse,
    DiagnosticSummaryResponse,
    DigestResponse,
    ErrorResponse,
    ExplainResponse,
    HealthResponse,
    TaskListResponse,
    TaskResponse,
    TaskStatus,
    TaskStatusUpdate,
    VendorCallResponse,
)
from .delegation import DelegationManager
from .diagnostics import (
    DiagnosticEngine,
    DiagnosticSession,
    render_diagnostic_report,
    render_diagnostic_summary,
)
from .digest import MorningDigestGenerator
from .engine import PipelineError, SentinelEngine
from .llm_layer import (
    render_call_prep,
    render_coop_report,
    render_digest,
    render_inventory_health_summary,
    render_rebate_status,
    render_task_for_manager,
)
from .models import Digest, Issue
from .sidecar_config import SidecarSettings, get_settings
from .symbolic_reasoning import SymbolicReasoner
from .vendor_assist import VendorCallAssistant

logger = logging.getLogger("sentinel.sidecar")

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------


class _DigestCacheEntry:
    """Cached digest with TTL."""

    def __init__(self, digest: Digest, ttl_seconds: int):
        self.digest = digest
        self.created_at = time.monotonic()
        self.ttl_seconds = ttl_seconds

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds


_digest_cache: dict[str, _DigestCacheEntry] = {}
_task_store: dict[str, TaskResponse] = {}
_diagnostic_sessions: dict[str, dict] = (
    {}
)  # session_id -> {engine, session, store_name, status}


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def _get_auth_dependency(settings: SidecarSettings):
    """Create auth dependency based on settings.

    In dev mode, returns a dummy user. In production, validates
    Supabase JWT tokens following the pattern from
    apps/api/src/dependencies.py.
    """

    async def verify_token(request: Request) -> str:
        if settings.sidecar_dev_mode:
            return "dev-user"

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid Authorization header",
            )

        token = auth_header[7:]

        try:
            from supabase import create_client

            supabase = create_client(
                settings.supabase_url,
                settings.supabase_service_key,
            )
            user_response = supabase.auth.get_user(token)

            if not user_response or not user_response.user:
                raise HTTPException(status_code=401, detail="Invalid token")

            return user_response.user.id
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Auth error: %s", e)
            raise HTTPException(
                status_code=401,
                detail="Authentication failed",
            )

    return verify_token


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: SidecarSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="Profit Sentinel Sidecar",
        version="0.13.0",
        description=(
            "Mobile-first API for Profit Sentinel — "
            "inventory intelligence for Do It Best operations."
        ),
    )

    # CORS
    origins = (
        ["*"]
        if settings.sidecar_dev_mode
        else [
            "https://*.profitsentinel.com",
            "http://localhost:3000",
        ]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auth dependency
    verify_token = _get_auth_dependency(settings)

    # Services
    try:
        engine = SentinelEngine(
            binary_path=settings.sentinel_bin if settings.sentinel_bin else None,
        )
    except PipelineError:
        engine = None  # Binary not found — health endpoint will report this

    generator = MorningDigestGenerator(engine=engine)
    delegation_mgr = DelegationManager()
    vendor_assistant = VendorCallAssistant()

    # -----------------------------------------------------------------
    # Global exception handler
    # -----------------------------------------------------------------

    @app.exception_handler(PipelineError)
    async def pipeline_error_handler(
        request: Request,
        exc: PipelineError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content=ErrorResponse(
                code="PIPELINE_ERROR",
                message="Rust pipeline execution failed",
                detail=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(
        request: Request,
        exc: FileNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                code="FILE_NOT_FOUND",
                message="Data file not found",
                detail=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred",
                detail=str(exc) if settings.sidecar_dev_mode else None,
            ).model_dump(),
        )

    # -----------------------------------------------------------------
    # Helper: get or refresh digest
    # -----------------------------------------------------------------

    def _get_or_run_digest(
        stores: list[str] | None = None,
        top_k: int = 5,
    ) -> Digest:
        """Get cached digest or run pipeline."""
        if engine is None:
            raise PipelineError(
                "sentinel-server binary not found. "
                "Run 'cargo build --release -p sentinel-server' first."
            )

        cache_key = f"{','.join(sorted(stores or []))}:{top_k}"
        entry = _digest_cache.get(cache_key)

        if entry and not entry.is_expired:
            return entry.digest

        digest = generator.generate(
            settings.csv_path,
            stores=stores,
            top_k=top_k,
        )

        _digest_cache[cache_key] = _DigestCacheEntry(
            digest,
            settings.digest_cache_ttl_seconds,
        )
        return digest

    def _find_issue(issue_id: str) -> Issue:
        """Find an issue across all cached digests."""
        for entry in _digest_cache.values():
            if entry.is_expired:
                continue
            for issue in entry.digest.issues:
                if issue.id == issue_id:
                    return issue
        raise HTTPException(
            status_code=404,
            detail=f"Issue '{issue_id}' not found. Run digest first.",
        )

    # -----------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check — no auth required."""
        binary_found = engine is not None
        binary_path = str(engine.binary) if engine else None

        return HealthResponse(
            status="ok" if binary_found else "degraded",
            binary_found=binary_found,
            binary_path=binary_path,
            dev_mode=settings.sidecar_dev_mode,
        )

    @app.get(
        "/api/v1/digest",
        response_model=DigestResponse,
        dependencies=[Depends(verify_token)],
    )
    async def get_digest(
        stores: str | None = Query(
            default=None,
            description="Comma-separated store IDs (e.g. store-7,store-12)",
        ),
        top_k: int = Query(default=5, ge=1, le=20),
    ) -> DigestResponse:
        """Run pipeline and return morning digest."""
        store_list = [s.strip() for s in stores.split(",")] if stores else None

        digest = _get_or_run_digest(store_list, top_k)

        return DigestResponse(
            digest=digest,
            rendered_text=render_digest(digest),
            generated_at=digest.generated_at,
            store_ids=digest.store_filter,
            issue_count=digest.summary.total_issues,
            total_dollar_impact=digest.summary.total_dollar_impact,
        )

    @app.get(
        "/api/v1/digest/{store_id}",
        response_model=DigestResponse,
        dependencies=[Depends(verify_token)],
    )
    async def get_store_digest(
        store_id: str,
        top_k: int = Query(default=5, ge=1, le=20),
    ) -> DigestResponse:
        """Single-store digest view."""
        digest = _get_or_run_digest([store_id], top_k)

        return DigestResponse(
            digest=digest,
            rendered_text=render_digest(digest),
            generated_at=digest.generated_at,
            store_ids=digest.store_filter,
            issue_count=digest.summary.total_issues,
            total_dollar_impact=digest.summary.total_dollar_impact,
        )

    @app.post(
        "/api/v1/delegate",
        response_model=DelegateResponse,
        dependencies=[Depends(verify_token)],
    )
    async def delegate_issue(body: DelegateRequest) -> DelegateResponse:
        """Create a delegated task from a pipeline issue."""
        issue = _find_issue(body.issue_id)

        task = delegation_mgr.create_task(
            issue,
            assignee=body.assignee,
            deadline=body.deadline,
        )

        rendered = render_task_for_manager(task)

        # Store task
        notes = [body.notes] if body.notes else []
        task_resp = TaskResponse(
            task=task,
            status=TaskStatus.OPEN,
            rendered_text=rendered,
            notes=notes,
        )
        _task_store[task.task_id] = task_resp

        return DelegateResponse(
            task=task,
            rendered_text=rendered,
            task_id=task.task_id,
        )

    @app.get(
        "/api/v1/tasks",
        response_model=TaskListResponse,
        dependencies=[Depends(verify_token)],
    )
    async def list_tasks(
        store_id: str | None = Query(default=None),
        priority: str | None = Query(default=None),
        status: str | None = Query(default=None),
    ) -> TaskListResponse:
        """List delegated tasks with optional filtering."""
        tasks = list(_task_store.values())

        if store_id:
            tasks = [t for t in tasks if t.task.store_id == store_id]
        if priority:
            tasks = [t for t in tasks if t.task.priority.value == priority]
        if status:
            tasks = [t for t in tasks if t.status.value == status]

        return TaskListResponse(tasks=tasks, total=len(tasks))

    @app.get(
        "/api/v1/tasks/{task_id}",
        response_model=TaskResponse,
        dependencies=[Depends(verify_token)],
    )
    async def get_task(task_id: str) -> TaskResponse:
        """Get a single task by ID."""
        task_resp = _task_store.get(task_id)
        if not task_resp:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        return task_resp

    @app.patch(
        "/api/v1/tasks/{task_id}",
        response_model=TaskResponse,
        dependencies=[Depends(verify_token)],
    )
    async def update_task(
        task_id: str,
        body: TaskStatusUpdate,
    ) -> TaskResponse:
        """Update a task's status (complete, escalate, etc.)."""
        task_resp = _task_store.get(task_id)
        if not task_resp:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

        # Update status
        task_resp.status = body.status
        if body.notes:
            task_resp.notes.append(body.notes)

        _task_store[task_id] = task_resp
        return task_resp

    @app.get(
        "/api/v1/vendor-call/{issue_id}",
        response_model=VendorCallResponse,
        dependencies=[Depends(verify_token)],
    )
    async def vendor_call_prep(issue_id: str) -> VendorCallResponse:
        """Prepare a vendor call brief for an issue."""
        issue = _find_issue(issue_id)

        prep = vendor_assistant.prepare_call(issue)
        rendered = render_call_prep(prep)

        return VendorCallResponse(
            call_prep=prep,
            rendered_text=rendered,
        )

    @app.get(
        "/api/v1/coop/{store_id}",
        response_model=CoopReportResponse,
        dependencies=[Depends(verify_token)],
    )
    async def coop_report(store_id: str) -> CoopReportResponse:
        """Generate co-op intelligence report for a store.

        Requires a digest to be cached (run /api/v1/digest first).
        """
        # Find cached digest for this store
        cache_key = f"{store_id}:5"
        entry = _digest_cache.get(cache_key)

        if not entry or entry.is_expired:
            # Try to run a fresh digest
            digest = _get_or_run_digest([store_id])
        else:
            digest = entry.digest

        report = generator.generate_coop_report(
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

    # -----------------------------------------------------------------
    # Explain endpoints (Phase 13 — VSA→Symbolic Bridge)
    # -----------------------------------------------------------------

    reasoner = SymbolicReasoner()

    @app.get(
        "/api/v1/explain/{issue_id}",
        response_model=ExplainResponse,
        dependencies=[Depends(verify_token)],
    )
    async def explain_issue(issue_id: str) -> ExplainResponse:
        """Generate full proof tree explaining an issue's root cause.

        Returns the transparent reasoning chain showing which signals
        fired, which evidence rules matched, how each cause hypothesis
        scored, and why the winner was selected.
        """
        issue = _find_issue(issue_id)
        proof = reasoner.explain(issue)

        return ExplainResponse(
            issue_id=issue_id,
            proof_tree=proof.to_dict(),
            rendered_text=proof.render(),
        )

    @app.post(
        "/api/v1/explain/{issue_id}/why",
        response_model=BackwardChainResponse,
        dependencies=[Depends(verify_token)],
    )
    async def backward_chain_issue(
        issue_id: str,
        body: BackwardChainRequest,
    ) -> BackwardChainResponse:
        """Backward-chain from a goal to explain how it was derived.

        Given a goal predicate (e.g. "root_cause(Theft)" or
        "suspect(systematic_shrinkage)"), traces backward through
        domain rules to show what facts and inferences support it.
        """
        issue = _find_issue(issue_id)
        steps = reasoner.backward_chain(issue, body.goal)

        return BackwardChainResponse(
            issue_id=issue_id,
            goal=body.goal,
            reasoning_steps=steps,
        )

    # -----------------------------------------------------------------
    # Diagnostic endpoints (Phase 11)
    # -----------------------------------------------------------------

    diagnostic_engine = DiagnosticEngine()

    @app.post(
        "/api/v1/diagnostic/start",
        response_model=DiagnosticStartResponse,
        dependencies=[Depends(verify_token)],
    )
    async def start_diagnostic(body: DiagnosticStartRequest) -> DiagnosticStartResponse:
        """Start a new conversational diagnostic session.

        Accepts inventory items and detects negative-stock patterns
        for interactive classification.
        """
        session = diagnostic_engine.start_session(body.items)

        _diagnostic_sessions[session.session_id] = {
            "engine": diagnostic_engine,
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

    def _get_diagnostic_session(session_id: str) -> dict:
        """Validate and return diagnostic session data."""
        data = _diagnostic_sessions.get(session_id)
        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Diagnostic session '{session_id}' not found.",
            )
        return data

    @app.get(
        "/api/v1/diagnostic/{session_id}/question",
        response_model=DiagnosticQuestionResponse | None,
        dependencies=[Depends(verify_token)],
    )
    async def get_diagnostic_question(
        session_id: str,
    ) -> DiagnosticQuestionResponse | None:
        """Get the current question for a diagnostic session."""
        data = _get_diagnostic_session(session_id)
        question = diagnostic_engine.get_current_question(data["session"])

        if not question:
            return None

        return DiagnosticQuestionResponse(**question)

    @app.post(
        "/api/v1/diagnostic/{session_id}/answer",
        response_model=DiagnosticAnswerResponse,
        dependencies=[Depends(verify_token)],
    )
    async def answer_diagnostic(
        session_id: str,
        body: DiagnosticAnswerRequest,
    ) -> DiagnosticAnswerResponse:
        """Submit an answer to the current diagnostic question."""
        data = _get_diagnostic_session(session_id)
        result = diagnostic_engine.answer_question(
            data["session"],
            body.classification,
            body.note,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        if result.get("is_complete"):
            data["status"] = "complete"

        # Build response, converting next_question dict if present
        next_q = result.get("next_question")
        next_q_model = DiagnosticQuestionResponse(**next_q) if next_q else None

        return DiagnosticAnswerResponse(
            answered=result["answered"],
            progress=result["progress"],
            running_totals=result["running_totals"],
            is_complete=result["is_complete"],
            next_question=next_q_model,
        )

    @app.get(
        "/api/v1/diagnostic/{session_id}/summary",
        response_model=DiagnosticSummaryResponse,
        dependencies=[Depends(verify_token)],
    )
    async def get_diagnostic_summary(
        session_id: str,
    ) -> DiagnosticSummaryResponse:
        """Get current diagnostic session summary with running totals."""
        data = _get_diagnostic_session(session_id)
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

    @app.get(
        "/api/v1/diagnostic/{session_id}/report",
        response_model=DiagnosticReportResponse,
        dependencies=[Depends(verify_token)],
    )
    async def get_diagnostic_report(
        session_id: str,
    ) -> DiagnosticReportResponse:
        """Generate the final diagnostic report.

        Includes the classification journey, items to investigate,
        and a rendered text summary.
        """
        data = _get_diagnostic_session(session_id)
        session: DiagnosticSession = data["session"]

        report = diagnostic_engine.get_final_report(session)
        rendered = render_diagnostic_report(report)

        return DiagnosticReportResponse(
            session_id=session_id,
            summary=report["summary"],
            by_classification=report["by_classification"],
            items_to_investigate=report["items_to_investigate"],
            journey=report["journey"],
            rendered_text=rendered,
        )

    # -----------------------------------------------------------------
    # Static files (mobile UI) — mounted last so API routes take priority
    # -----------------------------------------------------------------

    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True))

    return app
