"""
Profit Sentinel API - FastAPI Application Entry Point

Forensic analysis of POS exports to detect hidden profit leaks.
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
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

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
