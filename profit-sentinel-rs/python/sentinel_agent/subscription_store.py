"""Subscription persistence layer.

Supports two backends:
    - InMemoryStore  — ephemeral, for dev/testing
    - SupabaseStore  — persistent, for production (uses PostgREST API)

Usage:
    store = create_store(supabase_url, supabase_service_key)
    # Automatically picks SupabaseStore if credentials are present,
    # otherwise falls back to InMemoryStore.

Table schema (create via Supabase SQL editor):

    CREATE TABLE IF NOT EXISTS digest_subscriptions (
        email       TEXT PRIMARY KEY,
        stores      TEXT[] DEFAULT '{}',
        enabled     BOOLEAN DEFAULT TRUE,
        send_hour   INTEGER DEFAULT 6 CHECK (send_hour >= 0 AND send_hour <= 23),
        timezone    TEXT DEFAULT 'America/New_York',
        created_at  TIMESTAMPTZ DEFAULT NOW(),
        updated_at  TIMESTAMPTZ DEFAULT NOW()
    );

    -- Enable RLS (optional for service-key access)
    ALTER TABLE digest_subscriptions ENABLE ROW LEVEL SECURITY;
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger("sentinel.subscription_store")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SubscriptionStore(ABC):
    """Abstract subscription store interface."""

    @abstractmethod
    def add(
        self,
        email: str,
        *,
        stores: list[str] | None = None,
        send_hour: int = 6,
        tz: str = "America/New_York",
    ) -> dict: ...

    @abstractmethod
    def remove(self, email: str) -> bool: ...

    @abstractmethod
    def get(self, email: str) -> dict | None: ...

    @abstractmethod
    def list_active(self) -> list[dict]: ...

    @abstractmethod
    def pause(self, email: str) -> bool: ...

    @abstractmethod
    def resume(self, email: str) -> bool: ...


# ---------------------------------------------------------------------------
# In-memory implementation (dev/testing)
# ---------------------------------------------------------------------------


class InMemoryStore(SubscriptionStore):
    """Ephemeral in-memory subscription store."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def add(
        self,
        email: str,
        *,
        stores: list[str] | None = None,
        send_hour: int = 6,
        tz: str = "America/New_York",
    ) -> dict:
        sub = {
            "email": email,
            "stores": stores or [],
            "enabled": True,
            "send_hour": send_hour,
            "timezone": tz,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._data[email] = sub
        logger.info("InMemoryStore: subscription added/updated: %s", email)
        return sub

    def remove(self, email: str) -> bool:
        if email in self._data:
            del self._data[email]
            logger.info("InMemoryStore: subscription removed: %s", email)
            return True
        return False

    def get(self, email: str) -> dict | None:
        return self._data.get(email)

    def list_active(self) -> list[dict]:
        return [s for s in self._data.values() if s.get("enabled")]

    def pause(self, email: str) -> bool:
        sub = self._data.get(email)
        if sub:
            sub["enabled"] = False
            return True
        return False

    def resume(self, email: str) -> bool:
        sub = self._data.get(email)
        if sub:
            sub["enabled"] = True
            return True
        return False


# ---------------------------------------------------------------------------
# Supabase implementation (production)
# ---------------------------------------------------------------------------


class SupabaseStore(SubscriptionStore):
    """Persistent subscription store using Supabase PostgREST API.

    Uses the service role key for server-side access (bypasses RLS).
    All operations are synchronous (httpx) to keep the CRUD interface simple.
    """

    TABLE = "digest_subscriptions"

    def __init__(self, url: str, service_key: str) -> None:
        self._base_url = f"{url}/rest/v1/{self.TABLE}"
        self._headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        logger.info("SupabaseStore: initialized with %s", url)

    def _request(
        self,
        method: str,
        path: str = "",
        *,
        params: dict | None = None,
        json: dict | None = None,
    ) -> httpx.Response:
        url = f"{self._base_url}{path}"
        resp = httpx.request(
            method,
            url,
            headers=self._headers,
            params=params,
            json=json,
            timeout=10.0,
        )
        return resp

    def add(
        self,
        email: str,
        *,
        stores: list[str] | None = None,
        send_hour: int = 6,
        tz: str = "America/New_York",
    ) -> dict:
        payload = {
            "email": email,
            "stores": stores or [],
            "enabled": True,
            "send_hour": send_hour,
            "timezone": tz,
            "updated_at": datetime.now(UTC).isoformat(),
        }

        # Upsert (POST with on_conflict)
        headers = {
            **self._headers,
            "Prefer": "return=representation,resolution=merge-duplicates",
        }
        resp = httpx.post(
            self._base_url,
            headers=headers,
            json=payload,
            params={"on_conflict": "email"},
            timeout=10.0,
        )

        if resp.status_code in (200, 201):
            rows = resp.json()
            result = rows[0] if rows else payload
            logger.info("SupabaseStore: subscription upserted: %s", email)
            return result

        # Fallback: if upsert not supported, try insert then patch
        logger.warning(
            "SupabaseStore: upsert returned %s, trying insert/update", resp.status_code
        )
        resp = self._request("POST", json=payload)
        if resp.status_code == 409:  # Conflict — already exists, update
            resp = self._request(
                "PATCH",
                params={"email": f"eq.{email}"},
                json=payload,
            )
        if resp.status_code in (200, 201):
            rows = resp.json()
            result = rows[0] if rows else payload
            logger.info("SupabaseStore: subscription added/updated: %s", email)
            return result

        logger.error("SupabaseStore: add failed (%s): %s", resp.status_code, resp.text)
        return payload  # Return payload even on failure for graceful degradation

    def remove(self, email: str) -> bool:
        resp = self._request("DELETE", params={"email": f"eq.{email}"})
        if resp.status_code == 200:
            rows = resp.json()
            removed = len(rows) > 0
            if removed:
                logger.info("SupabaseStore: subscription removed: %s", email)
            return removed
        logger.error(
            "SupabaseStore: remove failed (%s): %s", resp.status_code, resp.text
        )
        return False

    def get(self, email: str) -> dict | None:
        resp = self._request("GET", params={"email": f"eq.{email}", "limit": "1"})
        if resp.status_code == 200:
            rows = resp.json()
            return rows[0] if rows else None
        return None

    def list_active(self) -> list[dict]:
        resp = self._request(
            "GET", params={"enabled": "eq.true", "order": "created_at.asc"}
        )
        if resp.status_code == 200:
            return resp.json()
        logger.error("SupabaseStore: list failed (%s): %s", resp.status_code, resp.text)
        return []

    def pause(self, email: str) -> bool:
        resp = self._request(
            "PATCH",
            params={"email": f"eq.{email}"},
            json={
                "enabled": False,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        if resp.status_code == 200:
            rows = resp.json()
            return len(rows) > 0
        return False

    def resume(self, email: str) -> bool:
        resp = self._request(
            "PATCH",
            params={"email": f"eq.{email}"},
            json={
                "enabled": True,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        if resp.status_code == 200:
            rows = resp.json()
            return len(rows) > 0
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_store(
    supabase_url: str = "",
    supabase_service_key: str = "",
) -> SubscriptionStore:
    """Create a subscription store.

    Returns SupabaseStore if credentials are provided, InMemoryStore otherwise.
    """
    if supabase_url and supabase_service_key:
        try:
            store = SupabaseStore(supabase_url, supabase_service_key)
            logger.info("Using Supabase-backed subscription store")
            return store
        except Exception:
            logger.exception(
                "Failed to create SupabaseStore, falling back to InMemoryStore"
            )

    logger.info("Using in-memory subscription store (non-persistent)")
    return InMemoryStore()
