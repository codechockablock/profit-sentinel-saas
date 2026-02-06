"""Enterprise API Key Management.

Provides API key generation, validation, and rate limiting for enterprise
customers who want to integrate Profit Sentinel into their workflows.

API keys are:
    - Prefixed: "ps_live_" (production) or "ps_test_" (sandbox)
    - Hashed: stored as SHA-256 hash, never in plaintext
    - Scoped: tied to a user_id with tier-based rate limits
    - Rotatable: old keys can be revoked without disruption

Tiers:
    free:       10 requests/hour,   100 requests/day
    pro:        100 requests/hour,  2,000 requests/day
    enterprise: 1,000 requests/hour, 50,000 requests/day
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum

logger = logging.getLogger("sentinel.api_keys")

# ---------------------------------------------------------------------------
# Tier definitions
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
        max_file_size_mb=10,
        concurrent_analyses=1,
    ),
    ApiTier.PRO: TierLimits(
        requests_per_hour=100,
        requests_per_day=2000,
        max_file_size_mb=50,
        concurrent_analyses=5,
    ),
    ApiTier.ENTERPRISE: TierLimits(
        requests_per_hour=1000,
        requests_per_day=50000,
        max_file_size_mb=200,
        concurrent_analyses=20,
    ),
}


# ---------------------------------------------------------------------------
# API Key data classes
# ---------------------------------------------------------------------------


@dataclass
class ApiKeyRecord:
    """Stored API key record (hash only — never store plaintext)."""

    key_id: str
    key_hash: str
    user_id: str
    tier: ApiTier
    name: str
    created_at: datetime
    last_used_at: datetime | None = None
    is_active: bool = True
    is_test: bool = False
    usage_count: int = 0

    def to_dict(self) -> dict:
        return {
            "key_id": self.key_id,
            "user_id": self.user_id,
            "tier": self.tier.value,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "last_used_at": (
                self.last_used_at.isoformat() if self.last_used_at else None
            ),
            "is_active": self.is_active,
            "is_test": self.is_test,
            "usage_count": self.usage_count,
            "limits": TIER_LIMITS[self.tier].to_dict(),
        }


@dataclass
class ApiKeyValidation:
    """Result of validating an API key."""

    is_valid: bool
    user_id: str | None = None
    tier: ApiTier | None = None
    key_id: str | None = None
    error: str | None = None
    limits: TierLimits | None = None

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "user_id": self.user_id,
            "tier": self.tier.value if self.tier else None,
            "key_id": self.key_id,
            "error": self.error,
            "limits": self.limits.to_dict() if self.limits else None,
        }


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------


def _generate_key(test: bool = False) -> str:
    """Generate a new API key with appropriate prefix."""
    prefix = "ps_test_" if test else "ps_live_"
    token = secrets.token_hex(24)  # 48 hex chars
    return f"{prefix}{token}"


def _hash_key(key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def _key_id_from_hash(key_hash: str) -> str:
    """Generate a short key ID from the hash."""
    return f"key_{key_hash[:12]}"


# ---------------------------------------------------------------------------
# In-memory store (production would use Supabase)
# ---------------------------------------------------------------------------


class InMemoryApiKeyStore:
    """In-memory API key store for dev/test."""

    def __init__(self):
        self._keys: dict[str, ApiKeyRecord] = {}  # key_hash -> record
        self._user_keys: dict[str, list[str]] = defaultdict(list)  # user_id -> [key_hash]
        self._hourly_usage: dict[str, list[datetime]] = defaultdict(list)
        self._daily_usage: dict[str, list[datetime]] = defaultdict(list)

    def create_key(
        self,
        user_id: str,
        tier: ApiTier = ApiTier.FREE,
        name: str = "Default",
        test: bool = False,
    ) -> tuple[str, ApiKeyRecord]:
        """Create a new API key.

        Returns:
            Tuple of (plaintext_key, record). The plaintext key is
            returned only once — it cannot be retrieved later.
        """
        plaintext = _generate_key(test=test)
        key_hash = _hash_key(plaintext)
        key_id = _key_id_from_hash(key_hash)

        record = ApiKeyRecord(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            tier=tier,
            name=name,
            created_at=datetime.now(UTC),
            is_test=test,
        )

        self._keys[key_hash] = record
        self._user_keys[user_id].append(key_hash)
        return plaintext, record

    def validate_key(self, plaintext_key: str) -> ApiKeyValidation:
        """Validate an API key and return its context."""
        key_hash = _hash_key(plaintext_key)
        record = self._keys.get(key_hash)

        if not record:
            return ApiKeyValidation(
                is_valid=False, error="Invalid API key"
            )

        if not record.is_active:
            return ApiKeyValidation(
                is_valid=False,
                key_id=record.key_id,
                error="API key has been revoked",
            )

        # Check rate limits
        now = datetime.now(UTC)
        limits = TIER_LIMITS[record.tier]

        # Hourly limit
        hour_ago = now - timedelta(hours=1)
        self._hourly_usage[key_hash] = [
            t for t in self._hourly_usage[key_hash] if t > hour_ago
        ]
        if len(self._hourly_usage[key_hash]) >= limits.requests_per_hour:
            return ApiKeyValidation(
                is_valid=False,
                key_id=record.key_id,
                user_id=record.user_id,
                tier=record.tier,
                error=(
                    f"Hourly rate limit exceeded "
                    f"({limits.requests_per_hour}/hr for {record.tier.value} tier)"
                ),
            )

        # Daily limit
        day_ago = now - timedelta(days=1)
        self._daily_usage[key_hash] = [
            t for t in self._daily_usage[key_hash] if t > day_ago
        ]
        if len(self._daily_usage[key_hash]) >= limits.requests_per_day:
            return ApiKeyValidation(
                is_valid=False,
                key_id=record.key_id,
                user_id=record.user_id,
                tier=record.tier,
                error=(
                    f"Daily rate limit exceeded "
                    f"({limits.requests_per_day}/day for {record.tier.value} tier)"
                ),
            )

        # Record usage
        self._hourly_usage[key_hash].append(now)
        self._daily_usage[key_hash].append(now)
        record.last_used_at = now
        record.usage_count += 1

        return ApiKeyValidation(
            is_valid=True,
            user_id=record.user_id,
            tier=record.tier,
            key_id=record.key_id,
            limits=limits,
        )

    def list_keys(self, user_id: str) -> list[ApiKeyRecord]:
        """List all API keys for a user."""
        hashes = self._user_keys.get(user_id, [])
        return [self._keys[h] for h in hashes if h in self._keys]

    def revoke_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key by ID."""
        for record in self._keys.values():
            if record.key_id == key_id and record.user_id == user_id:
                record.is_active = False
                return True
        return False

    def get_usage_stats(self, key_id: str, user_id: str) -> dict | None:
        """Get usage statistics for an API key."""
        for key_hash, record in self._keys.items():
            if record.key_id == key_id and record.user_id == user_id:
                now = datetime.now(UTC)
                hour_ago = now - timedelta(hours=1)
                day_ago = now - timedelta(days=1)

                hourly = len([
                    t for t in self._hourly_usage.get(key_hash, [])
                    if t > hour_ago
                ])
                daily = len([
                    t for t in self._daily_usage.get(key_hash, [])
                    if t > day_ago
                ])

                limits = TIER_LIMITS[record.tier]
                return {
                    "key_id": record.key_id,
                    "tier": record.tier.value,
                    "usage_count_total": record.usage_count,
                    "usage_last_hour": hourly,
                    "usage_last_day": daily,
                    "limit_hourly": limits.requests_per_hour,
                    "limit_daily": limits.requests_per_day,
                    "remaining_hourly": limits.requests_per_hour - hourly,
                    "remaining_daily": limits.requests_per_day - daily,
                }
        return None


# ---------------------------------------------------------------------------
# Global store pattern (same as analysis_store, subscription_store)
# ---------------------------------------------------------------------------

_store: InMemoryApiKeyStore | None = None


def init_api_key_store(store: InMemoryApiKeyStore | None = None) -> None:
    """Initialize the global API key store."""
    global _store
    _store = store or InMemoryApiKeyStore()


def _get_store() -> InMemoryApiKeyStore:
    """Get the global store, initializing if needed."""
    global _store
    if _store is None:
        _store = InMemoryApiKeyStore()
    return _store


def create_api_key(
    user_id: str,
    tier: ApiTier = ApiTier.FREE,
    name: str = "Default",
    test: bool = False,
) -> tuple[str, ApiKeyRecord]:
    """Create a new API key."""
    return _get_store().create_key(user_id, tier, name, test)


def validate_api_key(plaintext_key: str) -> ApiKeyValidation:
    """Validate an API key."""
    return _get_store().validate_key(plaintext_key)


def list_api_keys(user_id: str) -> list[ApiKeyRecord]:
    """List all API keys for a user."""
    return _get_store().list_keys(user_id)


def revoke_api_key(key_id: str, user_id: str) -> bool:
    """Revoke an API key."""
    return _get_store().revoke_key(key_id, user_id)


def get_key_usage(key_id: str, user_id: str) -> dict | None:
    """Get usage statistics for an API key."""
    return _get_store().get_usage_stats(key_id, user_id)
