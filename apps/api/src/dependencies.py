"""
FastAPI dependency injection.

Provides injectable dependencies for routes.

SECURITY: Authentication dependencies have explicit error handling.
Invalid tokens raise 401, expired tokens raise 401, service errors raise 503.
"""

import logging

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


def get_supabase_client() -> Client | None:
    """Get Supabase client instance."""
    settings = get_settings()
    if settings.supabase_url and settings.supabase_service_key:
        return create_client(settings.supabase_url, settings.supabase_service_key)
    return None


def get_grok_client() -> OpenAI | None:
    """Get Grok AI client (OpenAI-compatible)."""
    settings = get_settings()
    if settings.ai_api_key:
        return OpenAI(api_key=settings.ai_api_key, base_url="https://api.x.ai/v1")
    return None


async def get_current_user(
    authorization: str | None = Header(None),
) -> str | None:
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
            status_code=401, detail="Invalid authorization header format"
        )

    token = authorization[7:]  # Remove "Bearer " prefix
    if not token or len(token) < 10:
        logger.warning("Empty or malformed token")
        raise HTTPException(status_code=401, detail="Invalid token")

    supabase = get_supabase_client()
    if not supabase:
        logger.error("Supabase client not configured - auth service unavailable")
        raise HTTPException(
            status_code=503, detail="Authentication service unavailable"
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
            logger.info("Token expired for request")
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
        raise HTTPException(status_code=503, detail="Authentication service error")


async def require_user(
    authorization: str | None = Header(None),
) -> str:
    """
    Require authenticated user. Raises 401 if not authenticated.

    This is stricter than get_current_user - it requires a valid token.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = await get_current_user(authorization)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


async def get_user_tier(user_id: str | None) -> str:
    """
    Get subscription tier for a user from Supabase.

    Returns:
        'free', 'pro', or 'enterprise'. Defaults to 'free' for anonymous users.
    """
    if not user_id:
        return "free"

    supabase = get_supabase_client()
    if not supabase:
        logger.warning("Supabase not configured, defaulting to free tier")
        return "free"

    try:
        result = (
            supabase.table("user_profiles")
            .select("subscription_tier")
            .eq("id", user_id)
            .single()
            .execute()
        )
        if result.data and result.data.get("subscription_tier"):
            return result.data["subscription_tier"]
        return "free"
    except Exception as e:
        logger.warning(f"Failed to fetch user tier: {e}, defaulting to free")
        return "free"


async def require_pro_tier(
    authorization: str | None = Header(None),
) -> str:
    """
    Require Pro or Enterprise tier. Raises 403 if not subscribed.

    Returns:
        User ID if Pro/Enterprise tier
    """
    user_id = await require_user(authorization)
    tier = await get_user_tier(user_id)

    if tier not in ("pro", "enterprise"):
        raise HTTPException(
            status_code=403,
            detail="Pro subscription required for this feature",
        )
    return user_id


async def check_subscription_access(user_id: str) -> dict:
    """
    Check if user has active subscription access (trialing or active).

    Uses the database check_user_access function which handles:
    - Active subscriptions
    - Trial periods (within 14 days)
    - Trial expiration
    - Past due grace period

    Returns:
        Dict with:
        - has_access: bool - whether user can access paid features
        - access_reason: str - why access is granted/denied
        - subscription_status: str - current status
        - trial_days_left: int | None - days remaining in trial
        - current_period_end: datetime | None - subscription period end
    """
    supabase = get_supabase_client()
    if not supabase:
        logger.warning("Supabase not configured, denying access")
        return {
            "has_access": False,
            "access_reason": "service_unavailable",
            "subscription_status": "none",
            "trial_days_left": None,
            "current_period_end": None,
        }

    try:
        # Call the database function to check access
        result = supabase.rpc("check_user_access", {"p_user_id": user_id}).execute()

        if not result.data or len(result.data) == 0:
            return {
                "has_access": False,
                "access_reason": "user_not_found",
                "subscription_status": "none",
                "trial_days_left": None,
                "current_period_end": None,
            }

        return result.data[0]

    except Exception as e:
        logger.error(f"Failed to check subscription access: {e}")
        # Fail open for now - if we can't check, allow access
        # This prevents billing issues from blocking paying customers
        return {
            "has_access": True,
            "access_reason": "check_failed_allowed",
            "subscription_status": "unknown",
            "trial_days_left": None,
            "current_period_end": None,
        }


async def require_active_subscription(
    authorization: str | None = Header(None),
) -> str:
    """
    Require active subscription (trialing or paid). Raises 403 if expired.

    This checks both:
    - Active paid subscriptions
    - Active trial periods (within 14 days)

    Returns:
        User ID if subscription is active

    Raises:
        HTTPException 401: Not authenticated
        HTTPException 403: Subscription expired or canceled
    """
    user_id = await require_user(authorization)
    access = await check_subscription_access(user_id)

    if not access.get("has_access", False):
        reason = access.get("access_reason", "subscription_required")

        # Provide helpful error messages based on reason
        if reason == "trial_expired":
            raise HTTPException(
                status_code=403,
                detail="Your trial has expired. Upgrade to continue using Profit Sentinel.",
            )
        elif reason in ("subscription_canceled", "subscription_expired"):
            raise HTTPException(
                status_code=403,
                detail="Your subscription has ended. Please resubscribe to continue.",
            )
        else:
            raise HTTPException(
                status_code=403,
                detail="Active subscription required for this feature.",
            )

    return user_id
