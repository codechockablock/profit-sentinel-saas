"""Sentinel Sidecar API.

FastAPI application serving the Profit Sentinel interface.
Bridges the Rust pipeline and Python agent layer to REST endpoints.

Auth model:
    Public (anonymous OK):
        POST /uploads/presign, /uploads/suggest-mapping, /analysis/analyze
        GET  /analysis/primitives, /analysis/supported-pos
        GET  /health
    Authenticated only:
        GET  /api/v1/digest, /api/v1/digest/{store_id}
        POST /api/v1/delegate
        GET  /api/v1/tasks, /api/v1/tasks/{task_id}
        PATCH /api/v1/tasks/{task_id}
        GET  /api/v1/vendor-call/{issue_id}
        GET  /api/v1/coop/{store_id}
        GET  /api/v1/explain/{issue_id}
        POST /api/v1/explain/{issue_id}/why
        POST /api/v1/diagnostic/start
        GET  /api/v1/diagnostic/{id}/question
        POST /api/v1/diagnostic/{id}/answer
        GET  /api/v1/diagnostic/{id}/summary
        GET  /api/v1/diagnostic/{id}/report
        POST /api/v1/digest/subscribe
        GET  /api/v1/digest/subscriptions
        DELETE /api/v1/digest/subscribe/{email}
        POST /api/v1/digest/send
        GET  /api/v1/digest/scheduler-status
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
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
    DigestSendRequest,
    DigestSendResponse,
    ErrorResponse,
    ExplainResponse,
    HealthResponse,
    SchedulerStatusResponse,
    SubscribeRequest,
    SubscribeResponse,
    SubscriptionListResponse,
    TaskListResponse,
    TaskResponse,
    TaskStatus,
    TaskStatusUpdate,
    VendorCallResponse,
)
from .delegation import DelegationManager
from .digest_scheduler import (
    DigestScheduler,
    add_subscription,
    get_subscription,
    init_subscription_store,
    list_subscriptions,
    pause_subscription,
    remove_subscription,
    resume_subscription,
)
from .subscription_store import create_store
from .diagnostics import (
    DiagnosticEngine,
    DiagnosticSession,
    render_diagnostic_report,
    render_diagnostic_summary,
)
from .digest import MorningDigestGenerator
from .dual_auth import make_get_user_context, make_require_auth
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
from .upload_routes import create_upload_router
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
# Auth dependencies — built from dual_auth module
# ---------------------------------------------------------------------------
# Two dependency flavours are created at app startup:
#
#   get_user_context  — allows anonymous; returns UserContext for any request
#   require_auth      — raises 401 for anonymous; used on executive/diagnostic
#                       endpoints that need a logged-in user
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: SidecarSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    # Auth dependencies (dual mode — public + authenticated)
    get_user_context = make_get_user_context(settings)
    require_auth = make_require_auth(settings)

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

    # Subscription store (Supabase if configured, else in-memory)
    sub_store = create_store(
        supabase_url=settings.supabase_url,
        supabase_service_key=settings.supabase_service_key,
    )
    init_subscription_store(sub_store)

    # Digest email scheduler
    digest_scheduler = DigestScheduler(
        resend_api_key=settings.resend_api_key,
        generator=generator,
        csv_path=settings.csv_path,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        if settings.digest_email_enabled:
            digest_scheduler.start()
            logger.info("Digest email scheduler started")
        yield
        # Shutdown
        digest_scheduler.stop()

    app = FastAPI(
        title="Profit Sentinel Sidecar",
        version="0.13.0",
        description=(
            "Mobile-first API for Profit Sentinel — "
            "inventory intelligence for Do It Best operations."
        ),
        lifespan=lifespan,
    )

    # CORS — exact origins required (CORSMiddleware does not support wildcards)
    origins = (
        ["*"]
        if settings.sidecar_dev_mode
        else [
            "https://www.profitsentinel.com",
            "https://profitsentinel.com",
            "https://profit-sentinel-saas.vercel.app",
            "https://profit-sentinel.vercel.app",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

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
        dependencies=[Depends(require_auth)],
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

    # -----------------------------------------------------------------
    # Digest Email endpoints
    # IMPORTANT: These must be registered BEFORE /api/v1/digest/{store_id}
    # to prevent FastAPI from matching "subscriptions" as a store_id.
    # -----------------------------------------------------------------

    @app.post(
        "/api/v1/digest/subscribe",
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

    @app.get(
        "/api/v1/digest/subscriptions",
        response_model=SubscriptionListResponse,
        dependencies=[Depends(require_auth)],
    )
    async def get_subscriptions() -> SubscriptionListResponse:
        """List all active digest subscriptions."""
        subs = list_subscriptions()
        return SubscriptionListResponse(subscriptions=subs, total=len(subs))

    @app.delete(
        "/api/v1/digest/subscribe/{email}",
        dependencies=[Depends(require_auth)],
    )
    async def unsubscribe_digest(email: str) -> dict:
        """Remove a digest subscription."""
        removed = remove_subscription(email)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Subscription for '{email}' not found")
        return {"message": f"Unsubscribed {email}"}

    @app.post(
        "/api/v1/digest/send",
        response_model=DigestSendResponse,
        dependencies=[Depends(require_auth)],
    )
    async def send_digest_now(body: DigestSendRequest) -> DigestSendResponse:
        """Send a digest email immediately (on-demand)."""
        if not settings.resend_api_key:
            raise HTTPException(
                status_code=503,
                detail="Email delivery not configured (RESEND_API_KEY not set)",
            )
        try:
            result = await digest_scheduler.send_now(body.email)
            return DigestSendResponse(
                email_id=result.get("id"),
                message=f"Digest sent to {body.email}",
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Email send failed: {exc}")

    @app.get(
        "/api/v1/digest/scheduler-status",
        response_model=SchedulerStatusResponse,
        dependencies=[Depends(require_auth)],
    )
    async def scheduler_status() -> SchedulerStatusResponse:
        """Get digest scheduler status."""
        subs = list_subscriptions()
        return SchedulerStatusResponse(
            enabled=settings.digest_email_enabled,
            running=digest_scheduler.is_running,
            subscribers=len(subs),
            send_hour=settings.digest_send_hour,
        )

    # -----------------------------------------------------------------
    # Single-store digest (parameterized — must come AFTER specific routes)
    # -----------------------------------------------------------------

    @app.get(
        "/api/v1/digest/{store_id}",
        response_model=DigestResponse,
        dependencies=[Depends(require_auth)],
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
        dependencies=[Depends(require_auth)],
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
        dependencies=[Depends(require_auth)],
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
        dependencies=[Depends(require_auth)],
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
        dependencies=[Depends(require_auth)],
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
        dependencies=[Depends(require_auth)],
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
        dependencies=[Depends(require_auth)],
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
        dependencies=[Depends(require_auth)],
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
        dependencies=[Depends(require_auth)],
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
        dependencies=[Depends(require_auth)],
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
        dependencies=[Depends(require_auth)],
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
        dependencies=[Depends(require_auth)],
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
        dependencies=[Depends(require_auth)],
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
        dependencies=[Depends(require_auth)],
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
    # Legacy-compatible upload & analysis routes (production frontend)
    # -----------------------------------------------------------------

    upload_router = create_upload_router(settings, engine, get_user_context)
    app.include_router(upload_router)

    # -----------------------------------------------------------------
    # Static files (mobile UI) — mounted last so API routes take priority
    # -----------------------------------------------------------------

    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True))

    return app
