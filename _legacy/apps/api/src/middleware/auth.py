"""
Authentication middleware.

Provides request-level authentication handling.
"""

import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware for authentication handling.

    Extracts user information from JWT tokens and adds to request state.
    """

    async def dispatch(self, request: Request, call_next):
        """Process request and add auth info to state."""
        # Extract token from header
        auth_header = request.headers.get("Authorization")
        user_id = None

        if auth_header and auth_header.startswith("Bearer "):
            # Token validation is handled in dependencies
            # This middleware just logs auth attempts
            logger.debug("Auth header present")

        # Store in request state for access in routes
        request.state.user_id = user_id

        response = await call_next(request)
        return response
