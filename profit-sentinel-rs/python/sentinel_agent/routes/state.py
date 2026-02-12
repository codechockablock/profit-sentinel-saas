"""Shared application state for route modules.

Created once during app startup in sidecar.create_app() and injected
into each router factory function. This replaces closure-captured
variables with explicit dependency injection.
"""

from __future__ import annotations

import logging
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
    from ..world_model.transfer_matching import TransferMatcher

    _HAS_WORLD_MODEL = True
except ImportError:
    _HAS_WORLD_MODEL = False

logger = logging.getLogger("sentinel.state")


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

    # Engine 2: VSA World Model (initialized eagerly in sidecar.create_app)
    # Consumes Engine 1 (Rust pipeline) results via feed_engine2().
    # Adds predictions, transfer recommendations, tier classification.
    # If None, Engine 2 features are silently disabled — Engine 1 is unaffected.
    world_model: object | None = field(default=None)

    # Transfer matching engine (shares algebra with world_model)
    transfer_matcher: object | None = field(default=None)

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
        cache_key = f"{','.join(sorted(stores or []))}:{top_k}"
        entry = self.digest_cache.get(cache_key)

        if entry and not entry.is_expired:
            return entry.digest

        # Fallback: check any non-expired cache entry (e.g. from /analysis/analyze
        # which may have used a different top_k). Better to show data with a
        # different top_k than to crash on a missing CSV file.
        for key, entry in self.digest_cache.items():
            if not entry.is_expired:
                return entry.digest

        if self.engine is None:
            raise PipelineError(
                "sentinel-server binary not found. "
                "Run 'cargo build --release -p sentinel-server' first."
            )

        try:
            digest = self.generator.generate(
                self.settings.csv_path,
                stores=stores,
                top_k=top_k,
            )
        except FileNotFoundError:
            logger.warning("Data file not found: %s", self.settings.csv_path)
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "NO_DATA",
                    "message": "No analysis data available. Upload inventory data to get started.",
                },
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

        Flow: Engine 1 runs → findings displayed immediately → same data
        fed here as observations → Engine 2 adds predictions/transfers/tier
        classification on top.

        Sovereign collapse rule: if Engine 2 fails here, Engine 1 findings
        are already returned to the user. This is fire-and-forget.

        Args:
            analysis_result: The full result dict from RustResultAdapter.transform()
        """
        if self.world_model is None:
            return  # Engine 2 not initialized — skip silently

        try:
            store_id = analysis_result.get("store_id", "default-store")
            leaks = analysis_result.get("leaks", {})
            obs_count = 0
            timestamp = time.time()

            for leak_type, leak_data in leaks.items():
                items = leak_data.get("items", [])
                for item in items:
                    sku = item.get("sku") or item.get("item_id", "unknown")

                    # Build observation dict matching SentinelPipeline.record_observation
                    # signature: (store_id, entity_id, observation, timestamp)
                    obs = {
                        "issue_type": leak_type,
                        "severity": item.get("severity", "medium"),
                        "dollar_impact": item.get("dollar_impact", 0.0),
                        "velocity": item.get("velocity", item.get("qty_sold", 0.0)),
                        "stock": item.get("quantity", item.get("stock", 0)),
                        "cost": item.get("cost", item.get("unit_cost", 0.0)),
                        "price": item.get("price", item.get("retail", 0.0)),
                        "margin": item.get("margin", item.get("margin_pct", 0.0)),
                        "vendor_id": item.get("vendor_id"),
                        "vendor_name": item.get("vendor_name"),
                        "description": item.get("description", ""),
                        "category": item.get("category", item.get("department", "")),
                        "subcategory": item.get("subcategory", ""),
                    }

                    self.world_model.record_observation(
                        store_id=store_id,
                        entity_id=sku,
                        observation=obs,
                        timestamp=timestamp,
                    )
                    obs_count += 1

            if obs_count > 0:
                logger.info(
                    "Engine 1→2 bridge: fed %d observations from %d leak types "
                    "to world model (store=%s)",
                    obs_count,
                    len(leaks),
                    store_id,
                )
        except Exception as e:
            # Sovereign collapse: Engine 2 failure is non-fatal.
            # Engine 1 findings are already returned to the user.
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
