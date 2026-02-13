"""Sentinel Sidecar API.

FastAPI application serving the Profit Sentinel interface.
Bridges the Rust pipeline and Python agent layer to REST endpoints.

Route modules are in sentinel_agent.routes.* — this file handles only
app creation, middleware, exception handlers, and lifespan.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .analysis_store import create_analysis_store, init_analysis_store
from .api_keys import init_api_key_store
from .api_models import ErrorResponse
from .counterfactual import CounterfactualEngine
from .delegation import DelegationManager
from .digest import MorningDigestGenerator
from .digest_scheduler import DigestScheduler, init_subscription_store
from .dual_auth import make_get_user_context, make_require_auth
from .engine import PipelineError, SentinelEngine
from .pos_integrations import init_pos_store
from .routes.analyses import create_analyses_router
from .routes.api_keys import create_api_keys_router
from .routes.config import create_config_router
from .routes.counterfactual import create_counterfactual_router
from .routes.dashboard import create_dashboard_router
from .routes.diagnostic import create_diagnostic_router
from .routes.digest import create_digest_router
from .routes.explain import create_explain_router
from .routes.findings import create_findings_router
from .routes.health import create_health_router
from .routes.pos import create_pos_router
from .routes.predictions import create_predictions_router
from .routes.state import AppState
from .routes.tasks import create_tasks_router
from .routes.transfers import create_transfers_router
from .routes.vendor import create_vendor_router
from .sidecar_config import SidecarSettings, get_settings
from .subscription_store import create_store
from .upload_routes import create_upload_router
from .vendor_assist import VendorCallAssistant

# Engine 2 imports — graceful fallback if numpy/world_model not available
try:
    from .world_model import SentinelPipeline
    from .world_model.config import DeadStockConfig
    from .world_model.transfer_matching import EntityHierarchy, TransferMatcher

    _HAS_ENGINE2 = True
except ImportError as e:
    _HAS_ENGINE2 = False
    SentinelPipeline = None  # type: ignore[assignment,misc]
    logging.getLogger("sentinel.sidecar").warning(
        "Engine 2 imports failed (numpy missing?): %s", e
    )

logger = logging.getLogger("sentinel.sidecar")


def create_app(settings: SidecarSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    # -----------------------------------------------------------------
    # Dev mode safety guard — block dev mode in production (ECS/AWS)
    # -----------------------------------------------------------------
    if settings.sidecar_dev_mode:
        _is_production = bool(
            os.environ.get("ECS_CONTAINER_METADATA_URI")
            or os.environ.get("ECS_CONTAINER_METADATA_URI_V4")
            or os.environ.get("AWS_EXECUTION_ENV")
        )
        if _is_production:
            raise RuntimeError(
                "SIDECAR_DEV_MODE=true is not allowed in ECS/production. "
                "Detected production environment via ECS metadata or AWS_EXECUTION_ENV."
            )

    # -----------------------------------------------------------------
    # Auth dependencies (dual mode — public + authenticated)
    # -----------------------------------------------------------------
    get_user_context = make_get_user_context(settings)
    require_auth = make_require_auth(settings)

    # -----------------------------------------------------------------
    # Services
    # -----------------------------------------------------------------
    try:
        engine = SentinelEngine(
            binary_path=settings.sentinel_bin if settings.sentinel_bin else None,
        )
    except PipelineError:
        engine = None  # Binary not found — health endpoint will report this

    generator = MorningDigestGenerator(engine=engine)

    # Subscription store (Supabase if configured, else in-memory)
    sub_store = create_store(
        supabase_url=settings.supabase_url,
        supabase_service_key=settings.supabase_service_key,
    )
    init_subscription_store(sub_store)

    # Analysis store (Supabase if configured, else in-memory)
    analysis_store = create_analysis_store(
        supabase_url=settings.supabase_url,
        supabase_service_key=settings.supabase_service_key,
    )
    init_analysis_store(analysis_store)

    # API key store + POS connection store
    init_api_key_store()
    init_pos_store()

    # Digest email scheduler
    digest_scheduler = DigestScheduler(
        resend_api_key=settings.resend_api_key,
        generator=generator,
        csv_path=settings.csv_path,
        anthropic_api_key=settings.anthropic_api_key,
    )

    # -----------------------------------------------------------------
    # Engine 2: VSA World Model (continuous monitoring + predictions)
    # -----------------------------------------------------------------
    # Initialized eagerly at startup. Engine 1 (Rust pipeline) results
    # flow into this via feed_engine2(). Engine 2 adds predictions,
    # transfer recommendations, and tier classification on top of
    # Engine 1's instant analysis. Engine 1 findings always display —
    # if Engine 2 is unhealthy it goes quiet, not Engine 1.
    world_model = None
    transfer_matcher = None
    if _HAS_ENGINE2:
        try:
            dead_stock_config = DeadStockConfig()
            world_model = SentinelPipeline(
                dim=4096,
                seed=42,
                use_rust=False,  # Start with pure numpy; flip when Rust maturin is built
                dead_stock_config=dead_stock_config,
            )
            # Transfer matching subsystem — shared algebra
            hierarchy = EntityHierarchy(world_model.algebra)
            transfer_matcher = TransferMatcher(
                algebra=world_model.algebra,
                hierarchy=hierarchy,
            )
            logger.info("Engine 2 (VSA world model) initialized: dim=4096")
        except Exception as e:
            logger.warning("Engine 2 initialization failed (non-fatal): %s", e)
            world_model = None
            transfer_matcher = None
    else:
        logger.warning("Engine 2 not available — world_model imports failed")

    # -----------------------------------------------------------------
    # Engine 3: Counterfactual World Model
    # -----------------------------------------------------------------
    # Stateless — no initialization can fail. Always available.
    counterfactual_engine = CounterfactualEngine()
    logger.info("Engine 3 (counterfactual world model) initialized")

    # Singleton Supabase client (shared across all auth checks)
    supabase_client = None
    if settings.supabase_url and settings.supabase_service_key:
        try:
            from supabase import create_client

            supabase_client = create_client(
                settings.supabase_url,
                settings.supabase_service_key,
            )
            logger.info("Singleton Supabase client initialized")
        except Exception as e:
            logger.warning("Failed to create Supabase client (non-fatal): %s", e)

    # Shared state for all route modules
    state = AppState(
        settings=settings,
        engine=engine,
        generator=generator,
        delegation_mgr=DelegationManager(),
        vendor_assistant=VendorCallAssistant(),
        digest_scheduler=digest_scheduler,
        world_model=world_model,
        transfer_matcher=transfer_matcher,
        counterfactual_engine=counterfactual_engine,
        supabase_client=supabase_client,
    )

    # -----------------------------------------------------------------
    # Lifespan
    # -----------------------------------------------------------------
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.digest_email_enabled:
            digest_scheduler.start()
            logger.info("Digest email scheduler started")
        yield
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

    # -----------------------------------------------------------------
    # Middleware
    # -----------------------------------------------------------------
    origins = [
        "https://www.profitsentinel.com",
        "https://profitsentinel.com",
        "https://profit-sentinel-saas.vercel.app",
        "https://profit-sentinel.vercel.app",
    ]
    if settings.sidecar_dev_mode:
        origins.extend(
            [
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
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["*"],
    )

    # -----------------------------------------------------------------
    # Global exception handlers
    # -----------------------------------------------------------------
    @app.exception_handler(PipelineError)
    async def pipeline_error_handler(
        request: Request, exc: PipelineError
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
        request: Request, exc: FileNotFoundError
    ) -> JSONResponse:
        logger.warning("File not found: %s", exc)
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                code="NO_DATA",
                message="No analysis data available. Upload inventory data to get started.",
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred",
                detail=str(exc) if settings.sidecar_dev_mode else None,
            ).model_dump(),
        )

    # Expose state on app for test access (app.state is a Starlette State object)
    app.extra["sentinel_state"] = state

    # -----------------------------------------------------------------
    # Route modules
    # -----------------------------------------------------------------
    app.include_router(create_health_router(state))
    app.include_router(create_digest_router(state, require_auth))
    app.include_router(create_tasks_router(state, require_auth))
    app.include_router(create_vendor_router(state, require_auth))
    app.include_router(create_explain_router(state, require_auth))
    app.include_router(create_diagnostic_router(state, require_auth))
    app.include_router(create_predictions_router(state, require_auth))
    app.include_router(create_api_keys_router(require_auth))
    app.include_router(create_pos_router(require_auth))
    app.include_router(create_analyses_router(require_auth))
    app.include_router(create_findings_router(state, require_auth))
    app.include_router(create_dashboard_router(state, require_auth))
    app.include_router(create_config_router(state, require_auth))
    app.include_router(create_transfers_router(state, require_auth))
    app.include_router(create_counterfactual_router(state, require_auth))

    # Legacy-compatible upload & analysis routes (production frontend)
    # Pass app_state for Engine 1→2 bridging (feeds Rust results to world model)
    upload_router = create_upload_router(
        settings, engine, get_user_context, app_state=state
    )
    app.include_router(upload_router)

    # -----------------------------------------------------------------
    # Static files (mobile UI) — mounted last so API routes take priority
    # -----------------------------------------------------------------
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True))

    return app
