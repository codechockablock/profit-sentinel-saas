"""Analysis result persistence layer.

Saves and retrieves analysis results for authenticated users,
enabling cross-report pattern detection and analysis history.

Uses the Supabase PostgREST API (same pattern as subscription_store.py).
Falls back to an in-memory store when Supabase is not configured.

Table: analysis_synopses (see migration 008_analysis_history.sql)
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger("sentinel.analysis_store")


class StorageError(Exception):
    """Raised when a persistence operation fails."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AnalysisStore(ABC):
    """Abstract interface for analysis result persistence."""

    @abstractmethod
    def save(
        self,
        *,
        user_id: str,
        result: dict,
        file_hash: str,
        file_row_count: int,
        file_column_count: int | None = None,
        original_filename: str | None = None,
        analysis_label: str | None = None,
        processing_time_seconds: float | None = None,
    ) -> dict:
        """Save an analysis result. Returns the saved record."""
        ...

    @abstractmethod
    def list_for_user(
        self,
        user_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """List analyses for a user, newest first."""
        ...

    @abstractmethod
    def get_by_id(self, analysis_id: str, user_id: str) -> dict | None:
        """Get a single analysis by ID (scoped to user)."""
        ...

    @abstractmethod
    def delete(self, analysis_id: str, user_id: str) -> bool:
        """Delete an analysis by ID (scoped to user)."""
        ...

    @abstractmethod
    def update_label(self, analysis_id: str, user_id: str, label: str) -> bool:
        """Rename an analysis."""
        ...

    @abstractmethod
    def count_for_user(self, user_id: str) -> int:
        """Count total analyses for a user (for pagination)."""
        ...

    @abstractmethod
    def get_recent_pair(self, user_id: str) -> tuple[dict | None, dict | None]:
        """Get the two most recent analyses for comparison."""
        ...


# ---------------------------------------------------------------------------
# In-memory implementation (dev/testing)
# ---------------------------------------------------------------------------


class InMemoryAnalysisStore(AnalysisStore):
    """Ephemeral in-memory analysis store."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}  # id -> record
        self._counter = 0

    def save(
        self,
        *,
        user_id: str,
        result: dict,
        file_hash: str,
        file_row_count: int,
        file_column_count: int | None = None,
        original_filename: str | None = None,
        analysis_label: str | None = None,
        processing_time_seconds: float | None = None,
    ) -> dict:
        self._counter += 1
        analysis_id = f"analysis-{self._counter:04d}"

        # Extract summary stats from result
        summary = result.get("summary", {})
        detection_counts = {}
        for leak in result.get("leaks", {}):
            leak_data = result["leaks"][leak]
            if isinstance(leak_data, dict):
                detection_counts[leak] = leak_data.get("count", 0)

        total_issues = summary.get("total_items_flagged", 0)

        if not analysis_label:
            name_part = original_filename or "Analysis"
            if "." in name_part:
                name_part = name_part.rsplit(".", 1)[0]
            analysis_label = (
                f"{name_part} — {file_row_count} rows, {total_issues} issues"
            )

        record = {
            "id": analysis_id,
            "user_id": user_id,
            "file_hash": file_hash,
            "file_row_count": file_row_count,
            "file_column_count": file_column_count,
            "original_filename": original_filename,
            "analysis_label": analysis_label,
            "detection_counts": detection_counts,
            "total_impact_estimate_low": (
                summary.get("estimated_impact", {}).get("low")
                or summary.get("estimated_impact", {}).get("low_estimate")
                or 0
            ),
            "total_impact_estimate_high": (
                summary.get("estimated_impact", {}).get("high")
                or summary.get("estimated_impact", {}).get("high_estimate")
                or 0
            ),
            "dataset_stats": summary.get("dataset_stats", {}),
            "processing_time_seconds": processing_time_seconds,
            "full_result": result,
            "created_at": datetime.now(UTC).isoformat(),
        }

        self._data[analysis_id] = record
        logger.info(
            "InMemoryAnalysisStore: saved analysis %s for user %s", analysis_id, user_id
        )
        return record

    def list_for_user(
        self,
        user_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        user_analyses = [r for r in self._data.values() if r["user_id"] == user_id]
        # Sort by created_at descending
        user_analyses.sort(key=lambda r: r["created_at"], reverse=True)

        # Return without full_result for list view (lighter payload)
        results = []
        for r in user_analyses[offset : offset + limit]:
            summary_record = {k: v for k, v in r.items() if k != "full_result"}
            summary_record["has_full_result"] = r.get("full_result") is not None
            results.append(summary_record)
        return results

    def get_by_id(self, analysis_id: str, user_id: str) -> dict | None:
        record = self._data.get(analysis_id)
        if record and record["user_id"] == user_id:
            return record
        return None

    def delete(self, analysis_id: str, user_id: str) -> bool:
        record = self._data.get(analysis_id)
        if record and record["user_id"] == user_id:
            del self._data[analysis_id]
            logger.info("InMemoryAnalysisStore: deleted analysis %s", analysis_id)
            return True
        return False

    def update_label(self, analysis_id: str, user_id: str, label: str) -> bool:
        record = self._data.get(analysis_id)
        if record and record["user_id"] == user_id:
            record["analysis_label"] = label
            return True
        return False

    def count_for_user(self, user_id: str) -> int:
        return sum(1 for r in self._data.values() if r["user_id"] == user_id)

    def get_recent_pair(self, user_id: str) -> tuple[dict | None, dict | None]:
        analyses = self.list_for_user(user_id, limit=2)
        current = None
        previous = None
        if len(analyses) >= 1:
            # Re-fetch with full_result
            current = self.get_by_id(analyses[0]["id"], user_id)
        if len(analyses) >= 2:
            previous = self.get_by_id(analyses[1]["id"], user_id)
        return current, previous


# ---------------------------------------------------------------------------
# Supabase implementation (production)
# ---------------------------------------------------------------------------


class SupabaseAnalysisStore(AnalysisStore):
    """Persistent analysis store using Supabase PostgREST API.

    Uses the service role key for server-side access (bypasses RLS).
    """

    TABLE = "analysis_synopses"

    def __init__(self, url: str, service_key: str) -> None:
        self._base_url = f"{url}/rest/v1/{self.TABLE}"
        self._headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        logger.info("SupabaseAnalysisStore: initialized with %s", url)

    def _request(
        self,
        method: str,
        path: str = "",
        *,
        params: dict | None = None,
        json_data: dict | list | None = None,
        extra_headers: dict | None = None,
    ) -> httpx.Response:
        url = f"{self._base_url}{path}"
        headers = {**self._headers, **(extra_headers or {})}
        resp = httpx.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_data,
            timeout=15.0,
        )
        return resp

    def save(
        self,
        *,
        user_id: str,
        result: dict,
        file_hash: str,
        file_row_count: int,
        file_column_count: int | None = None,
        original_filename: str | None = None,
        analysis_label: str | None = None,
        processing_time_seconds: float | None = None,
    ) -> dict:
        # Extract summary stats
        summary = result.get("summary", {})
        detection_counts = {}
        for leak_key, leak_data in result.get("leaks", {}).items():
            if isinstance(leak_data, dict):
                detection_counts[leak_key] = leak_data.get("count", 0)

        total_issues = summary.get("total_items_flagged", 0)

        if not analysis_label:
            name_part = original_filename or "Analysis"
            if "." in name_part:
                name_part = name_part.rsplit(".", 1)[0]
            analysis_label = (
                f"{name_part} — {file_row_count} rows, {total_issues} issues"
            )

        estimated_impact = summary.get("estimated_impact", {})

        payload = {
            "user_id": user_id,
            "file_hash": file_hash,
            "file_row_count": file_row_count,
            "file_column_count": file_column_count,
            "original_filename": original_filename,
            "analysis_label": analysis_label,
            "detection_counts": detection_counts,
            "total_impact_estimate_low": (
                estimated_impact.get("low") or estimated_impact.get("low_estimate") or 0
            ),
            "total_impact_estimate_high": (
                estimated_impact.get("high")
                or estimated_impact.get("high_estimate")
                or 0
            ),
            "dataset_stats": summary.get("dataset_stats", {}),
            "processing_time_seconds": processing_time_seconds,
            "full_result": result,
            "engine_version": result.get("engine_version", "sidecar"),
        }

        resp = self._request("POST", json_data=payload)

        if resp.status_code in (200, 201):
            rows = resp.json()
            record = rows[0] if rows else payload
            logger.info(
                "SupabaseAnalysisStore: saved analysis %s for user %s",
                record.get("id", "?"),
                user_id,
            )
            return record

        logger.error(
            "SupabaseAnalysisStore: save failed (%s): %s",
            resp.status_code,
            resp.text,
        )
        raise StorageError(
            f"SupabaseAnalysisStore: save failed ({resp.status_code}): {resp.text}"
        )

    def list_for_user(
        self,
        user_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        # Select all columns EXCEPT full_result for list view (lighter payload)
        resp = self._request(
            "GET",
            params={
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc",
                "limit": str(limit),
                "offset": str(offset),
                "select": (
                    "id,user_id,file_hash,file_row_count,file_column_count,"
                    "original_filename,analysis_label,detection_counts,"
                    "total_impact_estimate_low,total_impact_estimate_high,"
                    "dataset_stats,processing_time_seconds,engine_version,"
                    "created_at"
                ),
            },
        )
        if resp.status_code == 200:
            rows = resp.json()
            for row in rows:
                row["has_full_result"] = True  # All Supabase records have it
            return rows
        logger.error(
            "SupabaseAnalysisStore: list failed (%s): %s",
            resp.status_code,
            resp.text,
        )
        return []

    def get_by_id(self, analysis_id: str, user_id: str) -> dict | None:
        resp = self._request(
            "GET",
            params={
                "id": f"eq.{analysis_id}",
                "user_id": f"eq.{user_id}",
                "limit": "1",
            },
        )
        if resp.status_code == 200:
            rows = resp.json()
            return rows[0] if rows else None
        return None

    def delete(self, analysis_id: str, user_id: str) -> bool:
        resp = self._request(
            "DELETE",
            params={
                "id": f"eq.{analysis_id}",
                "user_id": f"eq.{user_id}",
            },
        )
        if resp.status_code == 200:
            rows = resp.json()
            deleted = len(rows) > 0
            if deleted:
                logger.info("SupabaseAnalysisStore: deleted analysis %s", analysis_id)
            return deleted
        logger.error(
            "SupabaseAnalysisStore: delete failed (%s): %s",
            resp.status_code,
            resp.text,
        )
        return False

    def update_label(self, analysis_id: str, user_id: str, label: str) -> bool:
        resp = self._request(
            "PATCH",
            params={
                "id": f"eq.{analysis_id}",
                "user_id": f"eq.{user_id}",
            },
            json_data={"analysis_label": label},
        )
        if resp.status_code == 200:
            rows = resp.json()
            return len(rows) > 0
        return False

    def count_for_user(self, user_id: str) -> int:
        resp = self._request(
            "GET",
            path="",
            params={
                "user_id": f"eq.{user_id}",
                "select": "id",
            },
            extra_headers={"Prefer": "count=exact"},
        )
        if resp.status_code == 200:
            # PostgREST returns count in Content-Range header
            content_range = resp.headers.get("Content-Range", "")
            if "/" in content_range:
                total_str = content_range.split("/")[-1]
                if total_str != "*":
                    return int(total_str)
            # Fallback: count returned rows
            return len(resp.json())
        return 0

    def get_recent_pair(self, user_id: str) -> tuple[dict | None, dict | None]:
        resp = self._request(
            "GET",
            params={
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc",
                "limit": "2",
            },
        )
        if resp.status_code == 200:
            rows = resp.json()
            current = rows[0] if len(rows) >= 1 else None
            previous = rows[1] if len(rows) >= 2 else None
            return current, previous
        return None, None


# ---------------------------------------------------------------------------
# Global store + factory
# ---------------------------------------------------------------------------

_store: AnalysisStore = InMemoryAnalysisStore()


def init_analysis_store(store: AnalysisStore) -> None:
    """Set the global analysis store (called once at app startup)."""
    global _store
    _store = store
    logger.info("Analysis store initialized: %s", type(store).__name__)


def create_analysis_store(
    supabase_url: str = "",
    supabase_service_key: str = "",
) -> AnalysisStore:
    """Create an analysis store.

    Returns SupabaseAnalysisStore if credentials are provided,
    InMemoryAnalysisStore otherwise.
    """
    if supabase_url and supabase_service_key:
        try:
            store = SupabaseAnalysisStore(supabase_url, supabase_service_key)
            logger.info("Using Supabase-backed analysis store")
            return store
        except Exception:
            logger.exception(
                "Failed to create SupabaseAnalysisStore, falling back to InMemoryAnalysisStore"
            )

    logger.info("Using in-memory analysis store (non-persistent)")
    return InMemoryAnalysisStore()


# ---------------------------------------------------------------------------
# Module-level convenience functions (delegate to global store)
# ---------------------------------------------------------------------------


def save_analysis(**kwargs) -> dict:
    """Save an analysis result."""
    return _store.save(**kwargs)


def list_user_analyses(user_id: str, *, limit: int = 20, offset: int = 0) -> list[dict]:
    """List analyses for a user."""
    return _store.list_for_user(user_id, limit=limit, offset=offset)


def count_user_analyses(user_id: str) -> int:
    """Count total analyses for a user."""
    return _store.count_for_user(user_id)


def get_analysis(analysis_id: str, user_id: str) -> dict | None:
    """Get a single analysis."""
    return _store.get_by_id(analysis_id, user_id)


def delete_analysis(analysis_id: str, user_id: str) -> bool:
    """Delete an analysis."""
    return _store.delete(analysis_id, user_id)


def rename_analysis(analysis_id: str, user_id: str, label: str) -> bool:
    """Rename an analysis."""
    return _store.update_label(analysis_id, user_id, label)


def get_comparison_pair(user_id: str) -> tuple[dict | None, dict | None]:
    """Get the two most recent analyses for cross-report comparison."""
    return _store.get_recent_pair(user_id)


def compute_file_hash(content: bytes) -> str:
    """Compute SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()
