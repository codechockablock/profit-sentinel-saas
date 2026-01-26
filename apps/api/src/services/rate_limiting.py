"""
Rate Limiting Service - Distributed rate limiting with Redis backend.

Provides:
- In-memory rate limiting for development (default)
- Redis-backed distributed rate limiting for production (multi-instance)

Security Audit Item: H6-H7 - Distributed rate limiting
"""

import logging
from functools import lru_cache

from slowapi import Limiter
from slowapi.util import get_remote_address

from ..config import get_settings

logger = logging.getLogger(__name__)


def _get_redis_storage():
    """
    Get Redis storage backend for slowapi if Redis is configured.

    Returns:
        Redis storage instance or None for in-memory fallback
    """
    settings = get_settings()

    if not settings.has_redis:
        logger.info("Redis not configured, using in-memory rate limiting")
        return None

    try:
        from limits.storage import RedisStorage

        storage = RedisStorage(settings.redis_url)
        logger.info("Redis rate limiting storage initialized")
        return storage
    except ImportError:
        logger.warning(
            "Redis rate limiting requested but 'limits[redis]' not installed. "
            "Install with: pip install limits[redis]"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Redis storage: {e}")
        return None


@lru_cache
def get_limiter() -> Limiter:
    """
    Get the rate limiter instance (singleton).

    Returns distributed Redis-backed limiter if Redis is configured,
    otherwise returns in-memory limiter.

    Returns:
        Configured Limiter instance
    """
    storage = _get_redis_storage()

    if storage:
        limiter = Limiter(
            key_func=get_remote_address,
            storage_uri=get_settings().redis_url,
        )
        logger.info("Distributed rate limiter initialized with Redis backend")
    else:
        limiter = Limiter(key_func=get_remote_address)
        logger.info("In-memory rate limiter initialized (development mode)")

    return limiter


# Convenience export for common rate limits
def create_route_limiter() -> Limiter:
    """
    Create a limiter for route decoration.

    Usage:
        from ..services.rate_limiting import create_route_limiter
        limiter = create_route_limiter()

        @router.get("/endpoint")
        @limiter.limit("10/minute")
        async def endpoint(request: Request):
            ...
    """
    return get_limiter()
