"""
FastAPI dependency injection.

Provides injectable dependencies for routes.
"""

import logging
from typing import Optional

import boto3
from fastapi import Header, HTTPException
from openai import OpenAI
from supabase import Client, create_client

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

    Returns None if no valid token is provided (allows anonymous access).
    """
    if not authorization:
        return None

    supabase = get_supabase_client()
    if not supabase:
        return None

    try:
        token = authorization.replace("Bearer ", "")
        user = supabase.auth.get_user(token)
        return user.user.id if user else None
    except Exception as e:
        logger.warning(f"Auth failed: {e}")
        return None


async def require_user(
    authorization: Optional[str] = Header(None),
) -> str:
    """
    Require authenticated user. Raises 401 if not authenticated.
    """
    user_id = await get_current_user(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id
