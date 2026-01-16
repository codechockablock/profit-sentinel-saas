"""
FastAPI dependency injection.

Provides injectable dependencies for routes.

SECURITY: Authentication dependencies have explicit error handling.
Invalid tokens raise 401, expired tokens raise 401, service errors raise 503.
"""

import logging
from typing import Optional

import boto3
from fastapi import Header, HTTPException
from openai import OpenAI
from supabase import Client, create_client

# Import auth error types - may vary by supabase version
try:
    from gotrue.errors import AuthApiError
except ImportError:
    try:
        from supabase_auth.errors import AuthApiError
    except ImportError:
        # Fallback: create a placeholder that never matches
        class AuthApiError(Exception):
            pass

from .config import get_settings

logger = logging.getLogger(__name__)


def get_s3_client():
    """Get S3 client instance."""
    return boto3.client("s3")


def get_supabase_client() -> Optional[Client]:
    """Get Supabase client instance."""
    settings = get_settings()
    if settings.supabase_url and settings.supabase_service_key:
        return create_client(settings.supabase_url, settings.supabase_service_key)
    return None


def get_grok_client() -> Optional[OpenAI]:
    """Get Grok AI client (OpenAI-compatible)."""
    settings = get_settings()
    if settings.ai_api_key:
        return OpenAI(
            api_key=settings.ai_api_key,
            base_url="https://api.x.ai/v1"
        )
    return None


async def get_current_user(
    authorization: Optional[str] = Header(None),
) -> Optional[str]:
    """
    Extract current user ID from Supabase JWT token.

    Returns None if no token provided (allows anonymous access).
    Raises HTTPException for invalid/expired tokens or service errors.

    Security:
        - 401 for invalid or expired tokens
        - 503 for auth service unavailable
        - None only for missing Authorization header
    """
    if not authorization:
        return None

    # Validate header format
    if not authorization.startswith("Bearer "):
        logger.warning("Malformed Authorization header (missing Bearer prefix)")
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format"
        )

    token = authorization[7:]  # Remove "Bearer " prefix
    if not token or len(token) < 10:
        logger.warning("Empty or malformed token")
        raise HTTPException(status_code=401, detail="Invalid token")

    supabase = get_supabase_client()
    if not supabase:
        logger.error("Supabase client not configured - auth service unavailable")
        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable"
        )

    try:
        user = supabase.auth.get_user(token)
        if not user or not user.user:
            logger.warning("Token valid but no user returned")
            raise HTTPException(status_code=401, detail="Invalid token")
        return user.user.id

    except AuthApiError as e:
        # Specific Supabase auth errors (invalid token, expired, etc.)
        error_msg = str(e).lower()
        if "expired" in error_msg:
            logger.info(f"Token expired for request")
            raise HTTPException(status_code=401, detail="Token expired")
        elif "invalid" in error_msg or "malformed" in error_msg:
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(status_code=401, detail="Invalid token")
        else:
            logger.warning(f"Auth API error: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")

    except HTTPException:
        # Re-raise our own exceptions
        raise

    except Exception as e:
        # Unexpected errors - log full details but return generic message
        logger.error(f"Unexpected auth error: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=503,
            detail="Authentication service error"
        )


async def require_user(
    authorization: Optional[str] = Header(None),
) -> str:
    """
    Require authenticated user. Raises 401 if not authenticated.

    This is stricter than get_current_user - it requires a valid token.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user_id = await get_current_user(authorization)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user_id
