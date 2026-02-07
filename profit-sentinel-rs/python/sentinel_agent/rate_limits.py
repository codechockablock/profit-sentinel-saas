"""Centralized rate limit configuration.

Two rate-limiting systems coexist:

1. **Web UI (dual_auth.py)**: Anonymous and Supabase-authenticated users.
   - Anonymous: ANON_RATE_LIMIT per hour, ANON_MAX_FILE_SIZE_MB upload cap
   - Authenticated: AUTH_RATE_LIMIT per hour, AUTH_MAX_FILE_SIZE_MB upload cap

2. **API Keys (api_keys.py)**: Programmatic access via ps_live_/ps_test_ keys.
   - Tiers: free, pro, enterprise — each with hourly and daily limits.
   - See TIER_LIMITS for the full tier schedule.

If you change a limit here, both systems pick it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Web UI rate limits (dual_auth.py)
# ---------------------------------------------------------------------------

ANON_RATE_LIMIT = 5
"""Analyses per hour for anonymous (no account) users."""

AUTH_RATE_LIMIT = 100
"""Analyses per hour for Supabase-authenticated web users."""

ANON_MAX_FILE_SIZE_MB = 10
"""Upload size cap (MB) for anonymous users."""

AUTH_MAX_FILE_SIZE_MB = 50
"""Upload size cap (MB) for authenticated web users."""

# ---------------------------------------------------------------------------
# API key tiers (api_keys.py)
# ---------------------------------------------------------------------------


class ApiTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TierLimits:
    """Rate limits for an API tier."""

    requests_per_hour: int
    requests_per_day: int
    max_file_size_mb: int
    concurrent_analyses: int

    def to_dict(self) -> dict:
        return {
            "requests_per_hour": self.requests_per_hour,
            "requests_per_day": self.requests_per_day,
            "max_file_size_mb": self.max_file_size_mb,
            "concurrent_analyses": self.concurrent_analyses,
        }


TIER_LIMITS: dict[ApiTier, TierLimits] = {
    ApiTier.FREE: TierLimits(
        requests_per_hour=10,
        requests_per_day=100,
        max_file_size_mb=ANON_MAX_FILE_SIZE_MB,
        concurrent_analyses=1,
    ),
    ApiTier.PRO: TierLimits(
        requests_per_hour=AUTH_RATE_LIMIT,
        requests_per_day=2000,
        max_file_size_mb=AUTH_MAX_FILE_SIZE_MB,
        concurrent_analyses=5,
    ),
    ApiTier.ENTERPRISE: TierLimits(
        requests_per_hour=1000,
        requests_per_day=50000,
        max_file_size_mb=200,
        concurrent_analyses=20,
    ),
}
