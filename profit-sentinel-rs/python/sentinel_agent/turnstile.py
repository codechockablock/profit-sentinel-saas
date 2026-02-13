"""Cloudflare Turnstile verification.

Validates captcha tokens from the frontend before allowing guest uploads.
Authenticated users bypass captcha entirely.

Turnstile docs: https://developers.cloudflare.com/turnstile/get-started/server-side-validation/
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("sentinel.turnstile")

TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


async def verify_turnstile_token(
    token: str,
    secret_key: str,
    remote_ip: str | None = None,
) -> bool:
    """Verify a Turnstile token with Cloudflare.

    Args:
        token: The cf-turnstile-response token from the frontend.
        secret_key: Cloudflare Turnstile secret key.
        remote_ip: Optional client IP for additional verification.

    Returns:
        True if the token is valid, False otherwise.
    """
    if not secret_key:
        # No secret key configured — skip verification (dev mode)
        logger.warning("Turnstile secret key not configured, skipping verification")
        return True

    if not token:
        logger.warning("Empty Turnstile token received")
        return False

    payload: dict[str, str] = {
        "secret": secret_key,
        "response": token,
    }
    if remote_ip:
        payload["remoteip"] = remote_ip

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(TURNSTILE_VERIFY_URL, data=payload)
            resp.raise_for_status()
            result = resp.json()

        success = result.get("success", False)
        if not success:
            error_codes = result.get("error-codes", [])
            logger.warning("Turnstile verification failed: %s", error_codes)

        return success

    except Exception as e:
        logger.error("Turnstile verification error: %s", e)
        # On network error, fail open to avoid blocking legitimate users
        # TODO: Make this configurable (fail closed in production)
        return True
