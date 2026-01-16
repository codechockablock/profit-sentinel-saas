"""
API Routes.

All route modules are registered here.
"""

from fastapi import APIRouter

from .analysis import router as analysis_router
from .health import router as health_router
from .uploads import router as uploads_router

# Main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(health_router, tags=["health"])
api_router.include_router(uploads_router, prefix="/uploads", tags=["uploads"])
api_router.include_router(analysis_router, prefix="/analysis", tags=["analysis"])
