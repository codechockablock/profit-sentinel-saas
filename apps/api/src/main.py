"""
Profit Sentinel API - FastAPI Application Entry Point

Forensic analysis of POS exports to detect hidden profit leaks.
"""

import logging
import re
from contextlib import asynccontextmanager
from typing import Callable

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
VERCEL_PREVIEW_PATTERN = re.compile(
    r"^https://profit-sentinel[a-z0-9-]*\.vercel\.app$"
)


def is_allowed_origin(origin: str, allowed_origins: list[str]) -> bool:
    """
    Check if an origin is allowed.
    Supports exact matches and Vercel preview URL patterns.
    """
    if not origin:
        return False

    # Exact match check
    if origin in allowed_origins:
        return True

    # Vercel preview URL pattern match
    if VERCEL_PREVIEW_PATTERN.match(origin):
        return True

    return False


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    """
    Custom CORS middleware that supports dynamic Vercel preview URLs.
    Falls back to standard CORS for listed origins.
    """

    def __init__(self, app, allowed_origins: list[str]):
        super().__init__(app)
        self.allowed_origins = allowed_origins

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        origin = request.headers.get("origin", "")

        # Handle preflight OPTIONS requests
        if request.method == "OPTIONS":
            if is_allowed_origin(origin, self.allowed_origins):
                response = Response(status_code=200)
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Max-Age"] = "600"
                return response

        # Process the actual request
        response = await call_next(request)

        # Add CORS headers to response if origin is allowed
        if is_allowed_origin(origin, self.allowed_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"

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

    # CORS middleware - using custom middleware for dynamic Vercel preview support
    # This handles:
    # 1. Exact origin matches from settings.cors_origins
    # 2. Vercel preview URLs matching pattern: profit-sentinel*.vercel.app
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
