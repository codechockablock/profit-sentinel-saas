"""Dual authentication layer for public demo + authenticated users.

Provides two FastAPI dependencies:
    get_user_context  — allows anonymous; returns UserContext for any request
    require_auth      — rejects anonymous; returns UserContext for auth'd users

Anonymous users get:
    - 5 analyses/hour rate limit
    - 10MB file size limit
    - S3 prefix: uploads/anonymous/{ip_hash}
    - Results include upgrade_prompt
    - No access to diagnostic/explain/delegate/task/coop/vendor-call

Authenticated users get:
    - 100 analyses/hour rate limit
    - 50MB file size limit
    - S3 prefix: uploads/{user_id}
    - Full feature access
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, Request

from .rate_limits import (
    ANON_MAX_FILE_SIZE_MB,
    ANON_RATE_LIMIT,
    AUTH_MAX_FILE_SIZE_MB,
    AUTH_RATE_LIMIT,
)

logger = logging.getLogger("sentinel.dual_auth")

# ---------------------------------------------------------------------------
# In-memory rate-limit store (adequate for single-container ECS)
# NOTE: This rate limiter is per-worker. Under uvicorn with N workers,
# the effective rate limit is N * configured limit. For production at
# scale, migrate to Redis-based rate limiting.
# ---------------------------------------------------------------------------

_rate_limits: dict[str, list[datetime]] = defaultdict(list)
_rate_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# UserContext
# ---------------------------------------------------------------------------


class UserContext:
    """Unified user context for both anonymous and authenticated users."""

    __slots__ = ("user_id", "is_authenticated", "email", "ip_address")

    def __init__(
        self,
        user_id: str,
        is_authenticated: bool,
        email: str | None = None,
        ip_address: str | None = None,
    ):
        self.user_id = user_id
        self.is_authenticated = is_authenticated
        self.email = email
        self.ip_address = ip_address

    @property
    def rate_limit(self) -> int:
        return AUTH_RATE_LIMIT if self.is_authenticated else ANON_RATE_LIMIT

    @property
    def max_file_size_mb(self) -> int:
        return AUTH_MAX_FILE_SIZE_MB if self.is_authenticated else ANON_MAX_FILE_SIZE_MB

    @property
    def s3_prefix(self) -> str:
        if self.is_authenticated:
            return f"uploads/{self.user_id}"
        ip_hash = hashlib.sha256((self.ip_address or "unknown").encode()).hexdigest()[
            :12
        ]
        return f"uploads/anonymous/{ip_hash}"

    def __repr__(self) -> str:
        kind = "auth" if self.is_authenticated else "anon"
        return f"<UserContext {kind} {self.user_id}>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_client_ip(request: Request) -> str:
    """Extract client IP, handling ALB/proxy X-Forwarded-For.

    Trusts the rightmost IP in X-Forwarded-For — this is the one
    appended by the ALB (the trusted proxy), not the leftmost which
    is client-supplied and spoofable.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else "unknown"


async def check_rate_limit(ctx: UserContext) -> None:
    """Raise 429 if the user has exceeded their hourly analysis limit."""
    async with _rate_lock:
        now = datetime.now(UTC)
        window_start = now - timedelta(hours=1)

        # Prune old entries
        _rate_limits[ctx.user_id] = [
            t for t in _rate_limits[ctx.user_id] if t > window_start
        ]

        if len(_rate_limits[ctx.user_id]) >= ctx.rate_limit:
            msg = (
                f"Rate limit exceeded. "
                f"{'Authenticated users' if ctx.is_authenticated else 'Anonymous users'} "
                f"are limited to {ctx.rate_limit} analyses per hour."
            )
            if not ctx.is_authenticated:
                msg += " Sign up for higher limits."
            raise HTTPException(status_code=429, detail=msg)

        _rate_limits[ctx.user_id].append(now)


def build_upgrade_prompt() -> dict:
    """Return the upgrade CTA dict for anonymous analysis results."""
    return {
        "message": (
            "Sign up to save your analyses, access diagnostic tools, "
            "and unlock full features."
        ),
        "cta": "Create Free Account",
        "url": "/signup",
    }


# ---------------------------------------------------------------------------
# Dependency factories — called once at app startup
# ---------------------------------------------------------------------------


def make_get_user_context(settings):
    """Build the get_user_context dependency.

    In dev mode every request returns a dev-user UserContext.
    In production, checks for a Bearer token:
        - valid token  → authenticated UserContext
        - absent/invalid → anonymous UserContext (no 401)
    """

    async def get_user_context(request: Request) -> UserContext:
        ip_address = get_client_ip(request)

        # --- Dev mode shortcut ---
        if settings.sidecar_dev_mode:
            return UserContext(
                user_id="dev-user",
                is_authenticated=True,
                email="dev@localhost",
                ip_address=ip_address,
            )

        # --- Try authenticated path ---
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                # Use singleton Supabase client from AppState if available,
                # otherwise fall back to creating one (shouldn't happen in prod)
                supabase = None
                sentinel_state = getattr(request.app, "extra", {}).get("sentinel_state")
                if sentinel_state is not None:
                    supabase = getattr(sentinel_state, "supabase_client", None)
                if supabase is None:
                    from supabase import create_client

                    supabase = create_client(
                        settings.supabase_url,
                        settings.supabase_service_key,
                    )
                user_response = supabase.auth.get_user(token)

                if user_response and user_response.user:
                    return UserContext(
                        user_id=user_response.user.id,
                        is_authenticated=True,
                        email=getattr(user_response.user, "email", None),
                        ip_address=ip_address,
                    )
            except Exception as exc:
                # Invalid token → fall through to anonymous
                logger.debug("Token validation failed, treating as anonymous: %s", exc)

        # --- Anonymous fallback ---
        ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()[:16]
        return UserContext(
            user_id=f"anon_{ip_hash}",
            is_authenticated=False,
            ip_address=ip_address,
        )

    return get_user_context


def make_require_auth(settings):
    """Build the require_auth dependency.

    Same as get_user_context, but raises 401 for anonymous users.
    Used on endpoints that require authentication (diagnostics, explain, etc.).
    """

    _get_user_context = make_get_user_context(settings)

    async def require_auth(request: Request) -> UserContext:
        ctx = await _get_user_context(request)
        if not ctx.is_authenticated:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Authentication required. " "Please sign in to access this feature."
                ),
            )
        return ctx

    return require_auth
