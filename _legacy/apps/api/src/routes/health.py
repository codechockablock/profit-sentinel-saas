"""
Health check endpoints with detailed service status.
"""

import os

from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()

# Rate limiter for health endpoints
limiter = Limiter(key_func=get_remote_address)


@router.get("/")
@limiter.limit("100/minute")
async def root(request: Request):
    """Root endpoint."""
    return {"message": "Profit Sentinel backend is running"}


@router.get("/health")
@limiter.limit("100/minute")
async def health(request: Request):
    """Health check endpoint with detailed service status."""
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
