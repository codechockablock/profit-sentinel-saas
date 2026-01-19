"""
Profit Sentinel API - FastAPI Application Entry Point

Forensic analysis of POS exports to detect hidden profit leaks.
"""

import logging
import re
from collections.abc import Callable
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings
from .routes import api_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Vercel preview URL pattern: https://<project>-<hash>-<team>.vercel.app
# Matches: profit-sentinel.vercel.app, profit-sentinel-saas.vercel.app,
#          profit-sentinel-abc123-team.vercel.app, etc.
VERCEL_PREVIEW_PATTERN = re.compile(
    r"^https://profit-sentinel[-a-z0-9]*\.vercel\.app$"
)

# Production domain patterns (handles www and non-www)
PRODUCTION_DOMAIN_PATTERN = re.compile(
    r"^https://(www\.)?profitsentinel\.com$"
)


def is_allowed_origin(origin: str, allowed_origins: list[str]) -> bool:
    """
    Check if an origin is allowed.
    Supports exact matches, Vercel preview URLs, and production domains.
    """
    if not origin:
        return False

    # Normalize origin (strip trailing slash, lowercase)
    origin = origin.rstrip('/').lower()

    # Exact match check (case-insensitive)
    allowed_lower = [o.rstrip('/').lower() for o in allowed_origins]
    if origin in allowed_lower:
        return True

    # Production domain pattern (www.profitsentinel.com or profitsentinel.com)
    if PRODUCTION_DOMAIN_PATTERN.match(origin):
        return True

    # Vercel preview URL pattern match
    if VERCEL_PREVIEW_PATTERN.match(origin):
        return True

    return False


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
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept, X-Requested-With"
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
                status_code=500,
                content={"detail": "Internal server error"}
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
    yield
    logger.info(f"Shutting down {settings.app_name}")


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
            {"name": "analysis", "description": "Profit leak analysis"}
        ]
    )

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
    app.add_middleware(
        DynamicCORSMiddleware,
        allowed_origins=settings.cors_origins
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
    """Health check endpoint (legacy)."""
    return {"status": "healthy"}
