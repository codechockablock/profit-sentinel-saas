"""
Health check endpoints.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Profit Sentinel backend is running"}


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
