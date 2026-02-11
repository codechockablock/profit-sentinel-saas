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
from ..diagnostics import DiagnosticEngine
from ..digest import MorningDigestGenerator
from ..digest_scheduler import DigestScheduler
from ..engine import PipelineError, SentinelEngine
from ..models import Digest, Issue
from ..sidecar_config import SidecarSettings
from ..symbolic_reasoning import SymbolicReasoner
from ..vendor_assist import VendorCallAssistant

# Lazy import to avoid circular dependencies — world_model is optional
try:
    from ..world_model import SentinelPipeline

    _HAS_WORLD_MODEL = True
except ImportError:
    _HAS_WORLD_MODEL = False


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

    # Engine 2: VSA World Model (optional — initialized lazily)
    # This is the continuous monitoring engine that learns store patterns.
    # Engine 1 (Rust pipeline) results are fed here via record_observation().
    world_model: object | None = field(default=None)

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

    def feed_engine2(self, analysis_result: dict) -> None:
        """Feed Engine 1 (Rust pipeline) results into Engine 2 (world model).

        This bridges the instant analysis from the Rust pipeline into the
        continuous monitoring world model. Each flagged issue becomes an
        observation that the world model can learn patterns from.

        Args:
            analysis_result: The full result dict from RustResultAdapter.transform()
        """
        if self.world_model is None:
            return  # Engine 2 not initialized — skip silently

        import logging

        logger = logging.getLogger("sentinel.engine_bridge")

        try:
            leaks = analysis_result.get("leaks", {})
            obs_count = 0
            for leak_type, leak_data in leaks.items():
                items = leak_data.get("items", [])
                for item in items:
                    sku = item.get("sku") or item.get("item_id", "unknown")
                    # Build observation dict for the world model
                    obs = {
                        "entity_id": sku,
                        "issue_type": leak_type,
                        "severity": item.get("severity", "medium"),
                        "dollar_impact": item.get("dollar_impact", 0.0),
                        "description": item.get("description", ""),
                    }
                    # If world model has record_observation, use it
                    if hasattr(self.world_model, "record_observation"):
                        self.world_model.record_observation(obs)
                        obs_count += 1

            if obs_count > 0:
                logger.info(
                    "Engine 1→2 bridge: fed %d observations to world model",
                    obs_count,
                )
        except Exception as e:
            logger.warning("Engine 1→2 bridge failed (non-fatal): %s", e)

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
