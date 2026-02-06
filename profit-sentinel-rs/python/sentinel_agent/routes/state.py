"""Shared application state for route modules.

Created once during app startup in sidecar.create_app() and injected
into each router factory function. This replaces closure-captured
variables with explicit dependency injection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from fastapi import HTTPException

from ..api_models import TaskResponse
from ..delegation import DelegationManager
from ..digest import MorningDigestGenerator
from ..diagnostics import DiagnosticEngine
from ..engine import PipelineError, SentinelEngine
from ..models import Digest, Issue
from ..sidecar_config import SidecarSettings
from ..symbolic_reasoning import SymbolicReasoner
from ..vendor_assist import VendorCallAssistant
from ..digest_scheduler import DigestScheduler


class DigestCacheEntry:
    """Cached digest with TTL."""

    def __init__(self, digest: Digest, ttl_seconds: int):
        self.digest = digest
        self.created_at = time.monotonic()
        self.ttl_seconds = ttl_seconds

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds


@dataclass
class AppState:
    """Shared state created during app startup."""

    settings: SidecarSettings
    engine: SentinelEngine | None
    generator: MorningDigestGenerator
    delegation_mgr: DelegationManager
    vendor_assistant: VendorCallAssistant
    digest_scheduler: DigestScheduler
    reasoner: SymbolicReasoner = field(default_factory=SymbolicReasoner)
    diagnostic_engine: DiagnosticEngine = field(default_factory=DiagnosticEngine)

    # In-memory stores
    digest_cache: dict[str, DigestCacheEntry] = field(default_factory=dict)
    task_store: dict[str, TaskResponse] = field(default_factory=dict)
    diagnostic_sessions: dict[str, dict] = field(default_factory=dict)

    def get_or_run_digest(
        self,
        stores: list[str] | None = None,
        top_k: int = 5,
    ) -> Digest:
        """Get cached digest or run pipeline."""
        if self.engine is None:
            raise PipelineError(
                "sentinel-server binary not found. "
                "Run 'cargo build --release -p sentinel-server' first."
            )

        cache_key = f"{','.join(sorted(stores or []))}:{top_k}"
        entry = self.digest_cache.get(cache_key)

        if entry and not entry.is_expired:
            return entry.digest

        digest = self.generator.generate(
            self.settings.csv_path,
            stores=stores,
            top_k=top_k,
        )

        self.digest_cache[cache_key] = DigestCacheEntry(
            digest,
            self.settings.digest_cache_ttl_seconds,
        )
        return digest

    def find_issue(self, issue_id: str) -> Issue:
        """Find an issue across all cached digests."""
        for entry in self.digest_cache.values():
            if entry.is_expired:
                continue
            for issue in entry.digest.issues:
                if issue.id == issue_id:
                    return issue
        raise HTTPException(
            status_code=404,
            detail=f"Issue '{issue_id}' not found. Run digest first.",
        )
