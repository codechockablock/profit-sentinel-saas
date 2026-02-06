"""
API Routes.

All route modules are registered here.
"""

from fastapi import APIRouter

from .analysis import router as analysis_router
from .billing import router as billing_router
from .contact import router as contact_router
from .diagnostic import router as diagnostic_router
from .employee import router as employee_router
from .health import router as health_router
from .metrics import router as metrics_router
from .premium import router as premium_router
from .privacy import router as privacy_router
from .repair import router as repair_router
from .reports import router as reports_router
from .uploads import router as uploads_router

# Main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(health_router, tags=["health"])
api_router.include_router(uploads_router, prefix="/uploads", tags=["uploads"])
api_router.include_router(analysis_router, prefix="/analysis", tags=["analysis"])
api_router.include_router(diagnostic_router, prefix="/diagnostic", tags=["diagnostic"])
api_router.include_router(premium_router, prefix="/premium", tags=["premium"])
api_router.include_router(reports_router, prefix="/reports", tags=["reports"])
api_router.include_router(metrics_router, prefix="/metrics", tags=["metrics"])
api_router.include_router(repair_router, prefix="/repair", tags=["repair"])
api_router.include_router(employee_router, prefix="/employee", tags=["employee"])
api_router.include_router(contact_router, prefix="/contact", tags=["contact"])
api_router.include_router(billing_router, prefix="/billing", tags=["billing"])
api_router.include_router(privacy_router, prefix="/privacy", tags=["privacy"])
