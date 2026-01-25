"""
Profit Sentinel API - FastAPI Application Entry Point

Forensic analysis of POS exports to detect hidden profit leaks.

SECURITY HARDENING:
- Security headers middleware (X-Content-Type-Options, X-Frame-Options, etc.)
- Request ID correlation for audit logging
- CORS restricted to specific origins
- Rate limiting on sensitive endpoints
- Input validation via Pydantic models
"""

import logging
import re
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings
from .routes import api_router

# Rate limiter instance (uses client IP by default)
limiter = Limiter(key_func=get_remote_address)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Vercel preview URL pattern: https://<project>-<hash>-<team>.vercel.app
# Matches: profit-sentinel.vercel.app, profit-sentinel-saas.vercel.app,
#          profit-sentinel-abc123-team.vercel.app, etc.
VERCEL_PREVIEW_PATTERN = re.compile(r"^https://profit-sentinel[-a-z0-9]*\.vercel\.app$")

# Production domain patterns (handles www and non-www)
PRODUCTION_DOMAIN_PATTERN = re.compile(r"^https://(www\.)?profitsentinel\.com$")


def is_allowed_origin(origin: str, allowed_origins: list[str]) -> bool:
    """
    Check if an origin is allowed.
    Supports exact matches, Vercel preview URLs, and production domains.
    """
    if not origin:
        return False

    # Normalize origin (strip trailing slash, lowercase)
    origin = origin.rstrip("/").lower()

    # Exact match check (case-insensitive)
    allowed_lower = [o.rstrip("/").lower() for o in allowed_origins]
    if origin in allowed_lower:
        return True

    # Production domain pattern (www.profitsentinel.com or profitsentinel.com)
    if PRODUCTION_DOMAIN_PATTERN.match(origin):
        return True

    # Vercel preview URL pattern match
    if VERCEL_PREVIEW_PATTERN.match(origin):
        return True

    return False


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.

    SECURITY: These headers protect against common web vulnerabilities:
    - X-Content-Type-Options: Prevents MIME-type sniffing
    - X-Frame-Options: Prevents clickjacking
    - X-XSS-Protection: Legacy XSS protection (modern browsers use CSP)
    - Strict-Transport-Security: Enforces HTTPS
    - Content-Security-Policy: Restricts resource loading
    - Referrer-Policy: Controls referrer information
    - Permissions-Policy: Restricts browser features
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Add request ID to all requests for correlation and audit logging.

    SECURITY: Request IDs enable:
    - Audit trail correlation across services
    - Incident investigation
    - Rate limiting bypass detection
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Use existing request ID from header or generate new one
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    """
    Custom CORS middleware that supports dynamic Vercel preview URLs.
    Falls back to standard CORS for listed origins.

    IMPORTANT: This middleware ensures CORS headers are added even when
    exceptions occur, preventing browsers from blocking error responses.
    """

    def __init__(self, app, allowed_origins: list[str]):
        super().__init__(app)
        self.allowed_origins = allowed_origins

    def _add_cors_headers(self, response: Response, origin: str) -> None:
        """Add CORS headers to response if origin is allowed."""
        if is_allowed_origin(origin, self.allowed_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        origin = request.headers.get("origin", "")

        # Log all incoming requests with origin for debugging
        if origin:
            logger.debug(f"Request from origin: {origin}, path: {request.url.path}")

        # Handle preflight OPTIONS requests
        if request.method == "OPTIONS":
            if is_allowed_origin(origin, self.allowed_origins):
                logger.info(f"CORS preflight APPROVED for origin: {origin}")
                response = Response(status_code=200)
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = (
                    "GET, POST, PUT, DELETE, OPTIONS"
                )
                response.headers["Access-Control-Allow-Headers"] = (
                    "Authorization, Content-Type, Accept, X-Requested-With"
                )
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Max-Age"] = "600"
                return response
            else:
                # Log rejected preflight for debugging
                logger.warning(f"CORS preflight REJECTED for origin: {origin}")

        # Process the actual request with exception handling
        try:
            response = await call_next(request)
        except Exception as e:
            # Ensure CORS headers on error responses so browser can read error details
            logger.error(f"Request failed: {e}")
            from fastapi.responses import JSONResponse

            response = JSONResponse(
                status_code=500, content={"detail": "Internal server error"}
            )
            self._add_cors_headers(response, origin)
            return response

        # Add CORS headers to successful response if origin is allowed
        self._add_cors_headers(response, origin)

        if origin and not is_allowed_origin(origin, self.allowed_origins):
            # Log rejected origin for non-OPTIONS requests
            logger.debug(f"CORS headers not added for origin: {origin}")

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"CORS origins configured: {settings.cors_origins}")

    # Verify critical services on startup
    _verify_sentinel_engine()
    _verify_email_service()

    yield
    logger.info(f"Shutting down {settings.app_name}")


def _verify_sentinel_engine():
    """Verify sentinel engine is available and log status."""
    try:
        from sentinel_engine import _DIAGNOSTIC_AVAILABLE, _DORIAN_AVAILABLE
        from sentinel_engine import __version__ as engine_version

        # Primary: Check Dorian (v5.0.0+)
        if _DORIAN_AVAILABLE and _DIAGNOSTIC_AVAILABLE:
            logger.info(
                f"Sentinel Engine v{engine_version} loaded successfully "
                "(Dorian + Diagnostic available)"
            )
            return True

        # Fallback: Check legacy core
        try:
            from sentinel_engine import _CORE_AVAILABLE, get_all_primitives

            if _CORE_AVAILABLE and get_all_primitives is not None:
                primitives = get_all_primitives()
                logger.info(
                    f"Sentinel Engine v{engine_version} loaded (legacy mode) "
                    f"({len(primitives)} primitives available)"
                )
                return True
        except ImportError:
            pass

        logger.warning(
            f"Sentinel Engine v{engine_version} loaded but modules unavailable. "
            f"Dorian: {_DORIAN_AVAILABLE}, Diagnostic: {_DIAGNOSTIC_AVAILABLE}. "
            "Analysis will use heuristic fallback mode."
        )
        return True
    except ImportError as e:
        logger.error(
            f"CRITICAL: Sentinel Engine NOT AVAILABLE - {e}. "
            "Analysis will use MOCK DATA with placeholder SKUs! "
            "Emails will contain 'SKU-001', 'SKU-002' instead of real data. "
            "To fix: pip install -e packages/sentinel-engine"
        )
        return False


def _verify_email_service():
    """Verify email service configuration and log status."""
    import os

    resend_key = os.getenv("RESEND_API_KEY")
    sendgrid_key = os.getenv("SENDGRID_API_KEY")

    # SECURITY: Never log any part of API keys
    if resend_key:
        logger.info("Email service: Resend configured")
    elif sendgrid_key:
        logger.info("Email service: SendGrid configured")
    else:
        logger.warning(
            "Email service NOT CONFIGURED - no RESEND_API_KEY or SENDGRID_API_KEY found. "
            "Report emails will not be sent."
        )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Forensic analysis of POS exports to detect hidden profit leaks",
        version=settings.app_version,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Health checks"},
            {"name": "uploads", "description": "File upload and mapping"},
            {"name": "analysis", "description": "Profit leak analysis"},
        ],
    )

    # Rate limiting - protects against abuse and brute force
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS middleware - layered approach for maximum compatibility:
    #
    # Layer 1: FastAPI's built-in CORSMiddleware (added first, runs last)
    # - Provides robust CORS handling for all standard origins
    # - Handles preflight OPTIONS requests automatically
    # - Works as a safety net for edge cases
    #
    # Layer 2: Custom DynamicCORSMiddleware (added second, runs first)
    # - Adds support for dynamic Vercel preview URLs (regex matching)
    # - Adds CORS headers to error responses
    #
    # Note: Middleware runs in reverse order of registration (LIFO)

    # Layer 1: Built-in CORS for standard origins (safety net)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
        max_age=600,
    )

    # Layer 2: Custom CORS for dynamic origins (Vercel previews, error handling)
    app.add_middleware(DynamicCORSMiddleware, allowed_origins=settings.cors_origins)

    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)

    # Request ID middleware for audit logging
    app.add_middleware(RequestIDMiddleware)

    # Trusted hosts middleware (production only)
    if settings.env == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=[
                "api.profitsentinel.com",
                "profitsentinel.com",
                "www.profitsentinel.com",
            ],
        )

    # Include routes
    app.include_router(api_router)

    return app


# Application instance
app = create_app()


# For backwards compatibility with existing deployment
@app.get("/", tags=["health"])
async def root():
    """Root endpoint (legacy)."""
    return {"message": "Profit Sentinel backend is running"}


@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint with detailed service status."""
    import os

    # Check sentinel engine
    engine_status = "unavailable"
    engine_version = None
    dorian_available = False
    diagnostic_available = False
    try:
        from sentinel_engine import _DIAGNOSTIC_AVAILABLE, _DORIAN_AVAILABLE
        from sentinel_engine import __version__ as ev

        engine_version = ev
        dorian_available = _DORIAN_AVAILABLE
        diagnostic_available = _DIAGNOSTIC_AVAILABLE
        engine_status = (
            "available" if (dorian_available or diagnostic_available) else "partial"
        )
    except ImportError:
        pass

    # Check email service
    email_status = "not_configured"
    email_provider = None
    if os.getenv("RESEND_API_KEY"):
        email_status = "configured"
        email_provider = "resend"
    elif os.getenv("SENDGRID_API_KEY"):
        email_status = "configured"
        email_provider = "sendgrid"

    return {
        "status": "healthy",
        "services": {
            "sentinel_engine": {
                "status": engine_status,
                "version": engine_version,
                "dorian": dorian_available,
                "diagnostic": diagnostic_available,
                "warning": (
                    "Using mock analysis - emails will contain placeholder SKUs"
                    if engine_status == "unavailable"
                    else None
                ),
            },
            "email": {
                "status": email_status,
                "provider": email_provider,
            },
        },
    }
