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
from ..counterfactual import CounterfactualEngine
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

    # Engine 3: Counterfactual World Model
    # Computes alternate timelines for Engine 1 findings.
    # If None, counterfactual features are silently disabled.
    counterfactual_engine: CounterfactualEngine | None = field(default=None)

    # In-memory stores — all keyed by user_id for tenant isolation
    digest_cache: dict[str, dict[str, DigestCacheEntry]] = field(default_factory=dict)
    task_store: dict[str, dict[str, TaskResponse]] = field(default_factory=dict)
    diagnostic_sessions: dict[str, dict[str, dict]] = field(default_factory=dict)

    # Per-user world model pipelines + transfer matchers
    # (world_model / transfer_matcher above are templates; actual per-user
    # instances are stored here, keyed by user_id)
    world_models: dict[str, object] = field(default_factory=dict)
    transfer_matchers: dict[str, object] = field(default_factory=dict)

    def get_or_run_digest(
        self,
        user_id: str,
        stores: list[str] | None = None,
        top_k: int = 5,
    ) -> Digest:
        """Get cached digest or run pipeline, scoped to user_id."""
        user_cache = self.digest_cache.get(user_id, {})
        cache_key = f"{','.join(sorted(stores or []))}:{top_k}"
        entry = user_cache.get(cache_key)

        if entry and not entry.is_expired:
            return entry.digest

        # Fallback: check any non-expired cache entry for THIS USER only
        # (e.g. from /analysis/analyze which may have used a different top_k).
        for key, entry in user_cache.items():
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

        if user_id not in self.digest_cache:
            self.digest_cache[user_id] = {}
        self.digest_cache[user_id][cache_key] = DigestCacheEntry(
            digest,
            self.settings.digest_cache_ttl_seconds,
        )
        return digest

    def get_user_world_model(self, user_id: str) -> object | None:
        """Get or create a per-user world model pipeline.

        Returns None if Engine 2 is not available (world_model template is None).
        """
        if self.world_model is None:
            return None

        if user_id not in self.world_models:
            try:
                if _HAS_WORLD_MODEL:
                    # Create a new pipeline with same config as the template
                    pipeline = SentinelPipeline(
                        dim=4096,
                        seed=42,
                        use_rust=False,
                        dead_stock_config=getattr(self.world_model, "dead_stock_config", None),
                    )
                    self.world_models[user_id] = pipeline
                    logger.info("Created per-user world model for user=%s", user_id)
                else:
                    return None
            except Exception as e:
                logger.warning("Failed to create per-user world model for %s: %s", user_id, e)
                return None

        return self.world_models[user_id]

    def get_user_transfer_matcher(self, user_id: str) -> object | None:
        """Get or create a per-user transfer matcher.

        Returns None if Engine 2 / transfer matching is not available.
        """
        if self.transfer_matcher is None:
            return None

        if user_id not in self.transfer_matchers:
            try:
                if _HAS_WORLD_MODEL:
                    pipeline = self.get_user_world_model(user_id)
                    if pipeline is None:
                        return None
                    from ..world_model.transfer_matching import (
                        EntityHierarchy,
                        TransferMatcher,
                    )

                    hierarchy = EntityHierarchy(pipeline.algebra)
                    matcher = TransferMatcher(
                        algebra=pipeline.algebra,
                        hierarchy=hierarchy,
                    )
                    self.transfer_matchers[user_id] = matcher
                    logger.info("Created per-user transfer matcher for user=%s", user_id)
                else:
                    return None
            except Exception as e:
                logger.warning("Failed to create per-user transfer matcher for %s: %s", user_id, e)
                return None

        return self.transfer_matchers[user_id]

    def feed_engine2(self, user_id: str, analysis_result: dict) -> None:
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
            user_id: The authenticated user's ID (tenant isolation key).
            analysis_result: The full result dict from RustResultAdapter.transform()
        """
        pipeline = self.get_user_world_model(user_id)
        if pipeline is None:
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

                    pipeline.record_observation(
                        store_id=store_id,
                        entity_id=sku,
                        observation=obs,
                        timestamp=timestamp,
                    )
                    obs_count += 1

            if obs_count > 0:
                logger.info(
                    "Engine 1→2 bridge: fed %d observations from %d leak types "
                    "to world model (store=%s, user=%s)",
                    obs_count,
                    len(leaks),
                    store_id,
                    user_id,
                )
        except Exception as e:
            # Sovereign collapse: Engine 2 failure is non-fatal.
            # Engine 1 findings are already returned to the user.
            logger.warning("Engine 1→2 bridge failed (non-fatal): %s", e)

    def find_issue(self, user_id: str, issue_id: str) -> Issue:
        """Find an issue across cached digests for a specific user."""
        user_cache = self.digest_cache.get(user_id, {})
        for entry in user_cache.values():
            if entry.is_expired:
                continue
            for issue in entry.digest.issues:
                if issue.id == issue_id:
                    return issue
        raise HTTPException(
            status_code=404,
            detail=f"Issue '{issue_id}' not found. Run digest first.",
        )
