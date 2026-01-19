"""
sentinel_engine/agent.py - Always-On Agentic System for Profit Sentinel

Event-driven agent with VSA working memory, temporal context, and
probabilistic hypothesis tracking. Continuously monitors for retail
anomalies without polling overhead.

Architecture:
    ┌─────────────────────────────────────────┐
    │         Always-On Agent Core            │
    │  ┌───────────────────────────────────┐  │
    │  │     Event Loop (asyncio)          │  │
    │  │  • File watch (inventory)         │  │
    │  │  • Webhook receiver               │  │
    │  │  • Scheduled scans                │  │
    │  └─────────────────┬─────────────────┘  │
    │                    ▼                    │
    │  ┌───────────────────────────────────┐  │
    │  │   Temporal Working Memory         │  │
    │  │  • Rolling 30-day window          │  │
    │  │  • T-Bind encoded events          │  │
    │  │  • Decay-weighted retrieval       │  │
    │  └─────────────────┬─────────────────┘  │
    │                    ▼                    │
    │  ┌───────────────────────────────────┐  │
    │  │   Hypothesis Engine (P-Sup)       │  │
    │  │  • Multiple competing theories    │  │
    │  │  • Bayesian evidence updates      │  │
    │  │  • Confidence-based collapse      │  │
    │  └─────────────────────────────────────┘  │
    └─────────────────────────────────────────┘

Usage:
    from sentinel_engine.agent import SentinelAgent, AgentConfig

    config = AgentConfig(
        watch_directories=[Path("/data/inventory")],
        dimensions=16384,
    )
    agent = SentinelAgent(config)
    await agent.start()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
)

import torch
from vsa_core import (
    HypothesisBundle,
    bind,
    bundle,
    cw_bundle,
    p_sup,
    p_sup_collapse,
    p_sup_update,
    similarity,
    t_bind,
    t_unbind,
)

from .context import AnalysisContext, create_analysis_context
from .streaming import StreamingResult, process_large_file

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for always-on agent."""

    # Directories to watch for new files
    watch_directories: list[Path] = field(default_factory=list)

    # Webhook settings
    webhook_port: int = 8080
    webhook_enabled: bool = False

    # Scheduled scan interval (hours, 0 to disable)
    scan_interval_hours: float = 24.0

    # Working memory settings
    working_memory_window_days: int = 30
    decay_rate: float = 0.1  # Temporal decay rate

    # Alert thresholds
    alert_threshold: float = 0.7  # Minimum score to consider
    collapse_threshold: float = 0.85  # P-Sup collapse threshold

    # VSA settings
    dimensions: int = 16384

    # Processing settings
    chunk_size: int = 15000

    # Primitives to track
    primitives: list[str] = field(default_factory=lambda: [
        "low_stock",
        "high_margin_leak",
        "dead_item",
        "negative_inventory",
        "overstock",
        "price_discrepancy",
        "shrinkage_pattern",
        "margin_erosion",
    ])


# =============================================================================
# TEMPORAL WORKING MEMORY
# =============================================================================

@dataclass
class TemporalEvent:
    """Single event in temporal memory."""
    timestamp: float
    primitive: str
    sku: str
    score: float
    vector: torch.Tensor


class TemporalWorkingMemory:
    """Rolling temporal window of VSA facts with decay.

    Maintains a bundle of recent events encoded with T-Bind for
    temporal queries like "What anomalies appeared recently?"

    Properties:
        - Events older than window_days are pruned
        - Recent events have higher influence (exponential decay)
        - Supports time-range queries
    """

    def __init__(
        self,
        dimensions: int = 16384,
        window_days: int = 30,
        decay_rate: float = 0.1,
        max_events: int = 10000
    ):
        """Initialize temporal working memory.

        Args:
            dimensions: VSA dimensionality
            window_days: Days to retain events
            window_hours: Hours to retain events
            decay_rate: Temporal decay constant
            max_events: Maximum events before pruning
        """
        self.dimensions = dimensions
        self.window_days = window_days
        self.decay_rate = decay_rate
        self.max_events = max_events

        self.events: list[TemporalEvent] = []
        self.temporal_bundle: torch.Tensor | None = None
        self.reference_time: float = time.time()

    def add_event(
        self,
        primitive: str,
        sku: str,
        score: float,
        fact_vector: torch.Tensor,
        timestamp: float | None = None
    ) -> None:
        """Add timestamped fact to temporal memory.

        Args:
            primitive: Primitive type (e.g., "low_stock")
            sku: SKU identifier
            score: Detection score
            fact_vector: VSA vector for this fact
            timestamp: Event timestamp (default: now)
        """
        timestamp = timestamp or time.time()
        self.reference_time = max(self.reference_time, timestamp)

        # Create event record
        event = TemporalEvent(
            timestamp=timestamp,
            primitive=primitive,
            sku=sku,
            score=score,
            vector=fact_vector
        )
        self.events.append(event)

        # Encode with temporal binding
        t_encoded = t_bind(
            fact_vector,
            timestamp,
            self.reference_time,
            decay_rate=self.decay_rate
        )

        # Update bundle
        if self.temporal_bundle is None:
            self.temporal_bundle = t_encoded
        else:
            self.temporal_bundle = bundle(self.temporal_bundle, t_encoded)

        # Prune if needed
        self._maybe_prune()

    def _maybe_prune(self) -> None:
        """Prune old events and rebuild bundle if needed."""
        cutoff = self.reference_time - (self.window_days * 86400)

        # Check if pruning needed
        old_count = sum(1 for e in self.events if e.timestamp < cutoff)
        if old_count == 0 and len(self.events) <= self.max_events:
            return

        # Filter events
        self.events = [e for e in self.events if e.timestamp >= cutoff]

        # Limit to max_events (keep most recent)
        if len(self.events) > self.max_events:
            self.events = sorted(self.events, key=lambda e: e.timestamp)[-self.max_events:]

        # Rebuild bundle
        self._rebuild_bundle()

    def _rebuild_bundle(self) -> None:
        """Rebuild temporal bundle from events."""
        if not self.events:
            self.temporal_bundle = None
            return

        self.reference_time = max(e.timestamp for e in self.events)

        # Use CW-bundle with scores as confidences
        vectors = []
        confidences = []

        for event in self.events:
            t_encoded = t_bind(
                event.vector,
                event.timestamp,
                self.reference_time,
                decay_rate=self.decay_rate
            )
            vectors.append(t_encoded)
            confidences.append(event.score)

        self.temporal_bundle, _ = cw_bundle(vectors, confidences)

    def query_recent(
        self,
        query_vector: torch.Tensor,
        days_back: int = 7
    ) -> list[tuple[TemporalEvent, float]]:
        """Query for matching events in recent window.

        Args:
            query_vector: Vector to match against
            days_back: How many days back to search

        Returns:
            List of (event, similarity) tuples, sorted by similarity
        """
        cutoff = self.reference_time - (days_back * 86400)
        recent_events = [e for e in self.events if e.timestamp >= cutoff]

        results = []
        for event in recent_events:
            sim = float(similarity(query_vector, event.vector).real)
            results.append((event, sim))

        return sorted(results, key=lambda x: -x[1])

    def query_time_range(
        self,
        query_vector: torch.Tensor,
        start_time: float,
        end_time: float,
        num_samples: int = 20
    ) -> list[tuple[float, float]]:
        """Query for pattern across time range.

        Returns similarity scores at sampled time points.

        Args:
            query_vector: Pattern to search for
            start_time: Start of range (timestamp)
            end_time: End of range (timestamp)
            num_samples: Number of time points to sample

        Returns:
            List of (timestamp, similarity) tuples
        """
        if self.temporal_bundle is None:
            return []

        import numpy as np
        time_points = np.linspace(start_time, end_time, num_samples)

        results = []
        for t in time_points:
            # Unbind at this time point
            probe = t_unbind(
                self.temporal_bundle,
                t,
                self.reference_time,
                decay_rate=self.decay_rate
            )
            sim = float(similarity(probe, query_vector).real)
            results.append((float(t), sim))

        return results

    def get_primitive_trend(
        self,
        primitive_vector: torch.Tensor,
        days_back: int = 14
    ) -> dict[str, Any]:
        """Analyze trend for a primitive over time.

        Returns:
            Dict with trend analysis:
            - direction: "increasing", "decreasing", "stable"
            - strength: Correlation coefficient
            - recent_avg: Average similarity in recent period
        """
        end_time = self.reference_time
        start_time = end_time - (days_back * 86400)

        scores = self.query_time_range(primitive_vector, start_time, end_time)

        if len(scores) < 3:
            return {"direction": "unknown", "strength": 0, "recent_avg": 0}

        import numpy as np
        times = np.array([s[0] for s in scores])
        sims = np.array([s[1] for s in scores])

        # Linear regression for trend
        times_norm = (times - times.mean()) / (times.std() + 1e-10)
        slope = np.corrcoef(times_norm, sims)[0, 1]

        # Determine direction
        if slope > 0.3:
            direction = "increasing"
        elif slope < -0.3:
            direction = "decreasing"
        else:
            direction = "stable"

        # Recent average (last 3 days)
        recent_cutoff = end_time - (3 * 86400)
        recent_scores = [s[1] for s in scores if s[0] >= recent_cutoff]
        recent_avg = float(np.mean(recent_scores)) if recent_scores else 0

        return {
            "direction": direction,
            "strength": float(abs(slope)),
            "recent_avg": recent_avg
        }

    def size(self) -> int:
        """Return number of events in memory."""
        return len(self.events)

    def clear(self) -> None:
        """Clear all events."""
        self.events = []
        self.temporal_bundle = None


# =============================================================================
# HYPOTHESIS ENGINE
# =============================================================================

class HypothesisEngine:
    """Manages probabilistic hypotheses about detected anomalies.

    Uses P-Sup to maintain multiple competing explanations until
    evidence is sufficient to collapse to a conclusion.
    """

    def __init__(
        self,
        ctx: AnalysisContext,
        collapse_threshold: float = 0.85
    ):
        """Initialize hypothesis engine.

        Args:
            ctx: Analysis context with primitives
            collapse_threshold: Probability threshold for collapse
        """
        self.ctx = ctx
        self.collapse_threshold = collapse_threshold
        self.active_hypotheses: dict[str, HypothesisBundle] = {}
        self.collapsed_conclusions: list[dict[str, Any]] = []

    def create_hypothesis_bundle(
        self,
        sku: str,
        candidate_primitives: list[tuple[str, float]]
    ) -> HypothesisBundle:
        """Create hypothesis bundle for SKU anomaly.

        Args:
            sku: SKU identifier
            candidate_primitives: List of (primitive_name, initial_prob)

        Returns:
            New HypothesisBundle
        """
        hypotheses = []
        for prim_name, prob in candidate_primitives:
            prim_vec = self.ctx.get_primitive(prim_name)
            if prim_vec is not None:
                hypotheses.append((prim_name, prim_vec, prob))

        if not hypotheses:
            raise ValueError("No valid primitives for hypothesis bundle")

        return p_sup(hypotheses)

    def update_with_evidence(
        self,
        sku: str,
        evidence_vector: torch.Tensor
    ) -> str | None:
        """Update hypothesis bundle with new evidence.

        Args:
            sku: SKU to update
            evidence_vector: New evidence

        Returns:
            Collapsed conclusion if threshold exceeded, else None
        """
        if sku not in self.active_hypotheses:
            return None

        # Update probabilities
        self.active_hypotheses[sku] = p_sup_update(
            self.active_hypotheses[sku],
            evidence_vector
        )

        # Check for collapse
        winner = p_sup_collapse(
            self.active_hypotheses[sku],
            threshold=self.collapse_threshold
        )

        if winner:
            # Record conclusion
            self.collapsed_conclusions.append({
                "sku": sku,
                "conclusion": winner,
                "confidence": float(self.active_hypotheses[sku].probabilities.max()),
                "timestamp": time.time()
            })
            del self.active_hypotheses[sku]

        return winner

    def add_hypothesis(self, sku: str, bundle: HypothesisBundle) -> None:
        """Add or replace hypothesis bundle for SKU."""
        self.active_hypotheses[sku] = bundle

    def get_hypothesis(self, sku: str) -> HypothesisBundle | None:
        """Get hypothesis bundle for SKU."""
        return self.active_hypotheses.get(sku)

    def get_recent_conclusions(self, n: int = 10) -> list[dict[str, Any]]:
        """Get most recent collapsed conclusions."""
        return self.collapsed_conclusions[-n:]


# =============================================================================
# ALERT DISPATCHER
# =============================================================================

@dataclass
class Alert:
    """Alert generated by the agent."""
    alert_type: str
    sku: str
    confidence: float
    primitive: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


class AlertDispatcher:
    """Dispatches alerts to configured channels."""

    def __init__(self):
        self.handlers: list[Callable[[Alert], None]] = []
        self.alert_history: list[Alert] = []

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register alert handler."""
        self.handlers.append(handler)

    async def dispatch(self, alert: Alert) -> None:
        """Dispatch alert to all handlers."""
        self.alert_history.append(alert)

        for handler in self.handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

    def get_recent_alerts(self, n: int = 20) -> list[Alert]:
        """Get recent alerts."""
        return self.alert_history[-n:]


# =============================================================================
# SENTINEL AGENT
# =============================================================================

class SentinelAgent:
    """Always-on agent for retail anomaly detection.

    Combines:
    - Event-driven file monitoring
    - Temporal working memory
    - Probabilistic hypothesis tracking
    - Alert generation

    Example:
        config = AgentConfig(
            watch_directories=[Path("/data/inventory")],
            webhook_port=8080,
        )
        agent = SentinelAgent(config)

        # Optional: register alert handler
        agent.alert_dispatcher.register_handler(my_handler)

        # Start agent
        await agent.start()
    """

    def __init__(self, config: AgentConfig):
        """Initialize agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.running = False

        # Create analysis context
        self.ctx = create_analysis_context(dimensions=config.dimensions)

        # Initialize components
        self.working_memory = TemporalWorkingMemory(
            dimensions=config.dimensions,
            window_days=config.working_memory_window_days,
            decay_rate=config.decay_rate
        )

        self.hypothesis_engine = HypothesisEngine(
            ctx=self.ctx,
            collapse_threshold=config.collapse_threshold
        )

        self.alert_dispatcher = AlertDispatcher()

        # Track processed files
        self.processed_files: set[Path] = set()

        # Latest analysis results
        self.latest_result: StreamingResult | None = None

    async def start(self) -> None:
        """Start the agent event loop."""
        self.running = True
        logger.info("Starting Sentinel Agent v3.1.0")
        logger.info(f"  Watch directories: {self.config.watch_directories}")
        logger.info(f"  Working memory: {self.config.working_memory_window_days} days")
        logger.info(f"  Dimensions: {self.config.dimensions}")

        tasks = []

        # File watcher
        if self.config.watch_directories:
            tasks.append(self._watch_files())

        # Scheduled scanner
        if self.config.scan_interval_hours > 0:
            tasks.append(self._scheduled_scan())

        # Hypothesis updater
        tasks.append(self._hypothesis_loop())

        # Run all tasks
        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        """Stop the agent."""
        self.running = False
        logger.info("Sentinel Agent stopped")

    async def _watch_files(self) -> None:
        """Watch for new inventory files."""
        try:
            from watchfiles import awatch
        except ImportError:
            logger.warning("watchfiles not installed, file watching disabled")
            return

        paths = [str(p) for p in self.config.watch_directories if p.exists()]
        if not paths:
            logger.warning("No valid watch directories")
            return

        logger.info(f"Watching for files in: {paths}")

        async for changes in awatch(*paths):
            if not self.running:
                break

            for change_type, path in changes:
                path = Path(path)
                if path.suffix in ('.csv', '.tsv') and path not in self.processed_files:
                    await self._process_file(path)

    async def _process_file(self, filepath: Path) -> None:
        """Process new inventory file."""
        logger.info(f"Processing: {filepath}")

        try:
            # Run streaming analysis
            result = process_large_file(
                filepath,
                dimensions=self.config.dimensions,
                chunk_size=self.config.chunk_size,
                primitives=self.config.primitives
            )

            self.latest_result = result
            self.processed_files.add(filepath)

            # Update working memory with findings
            await self._integrate_findings(result)

            logger.info(f"Processed {filepath}: {sum(result.leak_counts.values())} detections")

        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")

    async def _integrate_findings(self, result: StreamingResult) -> None:
        """Integrate analysis findings into working memory and hypotheses."""
        timestamp = time.time()

        for primitive, leaks in result.top_leaks_by_primitive.items():
            prim_vec = self.ctx.get_primitive(primitive)
            if prim_vec is None:
                continue

            for sku, score in leaks:
                if score < self.config.alert_threshold:
                    continue

                # Create fact vector
                sku_vec = self.ctx.get_or_create(sku)
                fact_vec = bind(sku_vec, prim_vec)

                # Add to working memory
                self.working_memory.add_event(
                    primitive=primitive,
                    sku=sku,
                    score=score,
                    fact_vector=fact_vec,
                    timestamp=timestamp
                )

                # Check for hypothesis collapse
                winner = self.hypothesis_engine.update_with_evidence(sku, fact_vec)

                if winner:
                    # Generate alert for collapsed hypothesis
                    alert = Alert(
                        alert_type="hypothesis_confirmed",
                        sku=sku,
                        confidence=score,
                        primitive=winner,
                        timestamp=timestamp,
                        metadata={"source": "hypothesis_collapse"}
                    )
                    await self.alert_dispatcher.dispatch(alert)

    async def _scheduled_scan(self) -> None:
        """Periodic scan of watch directories."""
        interval_seconds = self.config.scan_interval_hours * 3600

        while self.running:
            await asyncio.sleep(interval_seconds)

            if not self.running:
                break

            logger.info("Running scheduled scan")

            for directory in self.config.watch_directories:
                if not directory.exists():
                    continue

                for filepath in directory.glob("*.csv"):
                    if filepath not in self.processed_files:
                        await self._process_file(filepath)

                for filepath in directory.glob("*.tsv"):
                    if filepath not in self.processed_files:
                        await self._process_file(filepath)

    async def _hypothesis_loop(self) -> None:
        """Periodically update hypotheses based on working memory patterns."""
        while self.running:
            await asyncio.sleep(60)  # Check every minute

            if not self.running:
                break

            if self.working_memory.size() == 0:
                continue

            # Detect emerging patterns
            await self._detect_patterns()

    async def _detect_patterns(self) -> None:
        """Detect emerging patterns and create/update hypotheses."""
        for primitive in self.config.primitives:
            prim_vec = self.ctx.get_primitive(primitive)
            if prim_vec is None:
                continue

            # Check trend for this primitive
            trend = self.working_memory.get_primitive_trend(prim_vec, days_back=7)

            if trend["direction"] == "increasing" and trend["recent_avg"] > 0.5:
                # Pattern is emerging - generate alert
                alert = Alert(
                    alert_type="trend_detected",
                    sku="*",  # Category-wide
                    confidence=trend["recent_avg"],
                    primitive=primitive,
                    timestamp=time.time(),
                    metadata={
                        "trend_direction": trend["direction"],
                        "trend_strength": trend["strength"],
                        "source": "pattern_detection"
                    }
                )
                await self.alert_dispatcher.dispatch(alert)

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get current agent status."""
        return {
            "running": self.running,
            "working_memory_size": self.working_memory.size(),
            "active_hypotheses": len(self.hypothesis_engine.active_hypotheses),
            "files_processed": len(self.processed_files),
            "recent_conclusions": self.hypothesis_engine.get_recent_conclusions(5),
            "latest_result": {
                "total_rows": self.latest_result.total_rows,
                "leak_counts": self.latest_result.leak_counts,
            } if self.latest_result else None
        }

    def query_memory(
        self,
        primitive: str,
        days_back: int = 7
    ) -> list[tuple[TemporalEvent, float]]:
        """Query working memory for primitive matches.

        Args:
            primitive: Primitive to search for
            days_back: Days to look back

        Returns:
            List of (event, similarity) tuples
        """
        prim_vec = self.ctx.get_primitive(primitive)
        if prim_vec is None:
            return []

        return self.working_memory.query_recent(prim_vec, days_back)

    async def analyze_file(self, filepath: Path) -> StreamingResult:
        """Manually trigger file analysis.

        Args:
            filepath: File to analyze

        Returns:
            Analysis result
        """
        result = process_large_file(
            filepath,
            dimensions=self.config.dimensions,
            chunk_size=self.config.chunk_size,
            primitives=self.config.primitives
        )

        await self._integrate_findings(result)
        return result
