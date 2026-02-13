"""
Profit Sentinel — Next-Generation Pipeline
=============================================

Five integrated systems that transform Profit Sentinel from a
detection tool into a compounding intelligence engine.

1. FEEDBACK CLOSURE — learn from customer outcomes
2. TEMPORAL DEPTH — multi-scale pattern recognition
3. VENDOR INTELLIGENCE — cross-network supplier analysis
4. PREDICTIVE INTERVENTION — warn before problems form
5. COMPETITIVE MOAT — quality metrics that compound over time

These aren't five separate features. They're one feedback loop
operating at five different timescales:

- Feedback closure:  weeks   (did the transfer work?)
- Predictive:        days    (will this SKU die?)
- Vendor:            months  (is this supplier reliable?)
- Temporal:          seasons (what pattern repeats annually?)
- Moat metrics:      years   (how much better are we getting?)

Each system feeds the others. Transfer outcomes improve temporal
patterns. Temporal patterns improve predictions. Predictions
improve vendor intelligence. Vendor intelligence improves transfer
matching. The loop compounds.

Author: Joseph + Claude
Date: 2026-02-10
"""

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .config import DeadStockConfig, DeadStockTier
from .core import PhasorAlgebra, WorldModelConfig

# Optional Rust-accelerated backend
try:
    from .rust_algebra import RUST_AVAILABLE, RustPhasorAlgebra
except ImportError:
    RUST_AVAILABLE = False


# =============================================================================
# 1. FEEDBACK CLOSURE
# =============================================================================


class OutcomeType(Enum):
    """What happened after we made a recommendation."""

    TRANSFER_EXECUTED = "transfer_executed"
    TRANSFER_SOLD = "transfer_sold"  # Items sold at destination
    TRANSFER_STALLED = "transfer_stalled"  # Items didn't sell
    TRANSFER_REJECTED = "transfer_rejected"  # Customer didn't act
    FINDING_ACTED_ON = "finding_acted_on"  # Customer addressed finding
    FINDING_IGNORED = "finding_ignored"  # Customer saw but ignored
    FINDING_CONFIRMED = "finding_confirmed"  # External verification
    FINDING_DISPUTED = "finding_disputed"  # Customer says we're wrong
    PREDICTION_CORRECT = "prediction_correct"
    PREDICTION_WRONG = "prediction_wrong"


@dataclass
class Outcome:
    """A recorded outcome from a recommendation or finding."""

    outcome_id: str
    outcome_type: OutcomeType
    source_finding_id: str  # Which finding/recommendation
    timestamp: float
    entity_id: str  # SKU involved
    store_id: str

    # For transfers
    dest_store_id: str | None = None
    units_transferred: int = 0
    units_sold: int = 0
    days_to_sell: int = 0
    actual_recovery: float = 0.0
    predicted_recovery: float = 0.0

    # For findings
    dollar_impact_actual: float | None = None
    dollar_impact_predicted: float | None = None

    # Metadata
    notes: str = ""


class FeedbackEngine:
    """
    Closes the loop between recommendations and outcomes.

    When a customer acts on a recommendation, the outcome feeds
    back into the system to improve future recommendations.

    The key mechanism: outcomes are encoded as VSA vectors and
    used to update transition primitives via contrastive learning.
    Good outcomes reinforce the patterns that produced them.
    Bad outcomes weaken those patterns.

    This is how the system gets smarter over time — not by
    accumulating rules, but by adjusting the geometry of its
    pattern space based on what actually worked.
    """

    def __init__(self, algebra: PhasorAlgebra):
        self.algebra = algebra
        self.outcomes: deque[Outcome] = deque(maxlen=10000)

        # Outcome encoding vectors
        self.outcome_roles = {
            "type": algebra.get_or_create("outcome_type"),
            "entity": algebra.get_or_create("outcome_entity"),
            "store": algebra.get_or_create("outcome_store"),
            "success": algebra.get_or_create("outcome_success"),
            "magnitude": algebra.get_or_create("outcome_magnitude"),
        }

        # Success/failure basis vectors
        self.success_vec = algebra.get_or_create("success_positive")
        self.failure_vec = algebra.get_or_create("success_negative")
        self.neutral_vec = algebra.get_or_create("success_neutral")

        # Aggregate metrics
        self.transfer_success_rate = 0.0
        self.finding_accuracy_rate = 0.0
        self.prediction_accuracy_rate = 0.0
        self.total_value_recovered = 0.0
        self.total_value_predicted = 0.0

        # Per-pattern success tracking
        # Maps pattern signature → (successes, attempts)
        self.pattern_success: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))

        # Reward signal history for the moat metrics
        self.reward_history: deque[dict] = deque(maxlen=10000)

    def record_outcome(self, outcome: Outcome) -> dict:
        """
        Record an outcome and generate a learning signal.

        Returns a reward dict that can be fed into the transition
        model's contrastive learning to reinforce or weaken the
        patterns that produced the original recommendation.
        """
        self.outcomes.append(outcome)

        # Compute reward signal
        reward = self._compute_reward(outcome)

        # Update aggregate metrics
        self._update_aggregates(outcome)

        # Store for moat metrics
        self.reward_history.append(
            {
                "timestamp": outcome.timestamp,
                "outcome_type": outcome.outcome_type.value,
                "reward": reward["reward_magnitude"],
                "entity": outcome.entity_id,
                "store": outcome.store_id,
            }
        )

        return reward

    def _compute_reward(self, outcome: Outcome) -> dict:
        """
        Compute a reward signal from an outcome.

        Positive reward → reinforce the pattern (contrastive positive)
        Negative reward → weaken the pattern (contrastive negative)
        Zero reward → no update

        Magnitude scales with dollar impact — big wins and big
        misses produce stronger learning signals than small ones.
        """
        reward_type = "neutral"
        reward_magnitude = 0.0

        if outcome.outcome_type == OutcomeType.TRANSFER_SOLD:
            # Transfer worked — reinforce the cross-store binding
            reward_type = "positive"
            if outcome.predicted_recovery > 0:
                # Scale by how close prediction was
                accuracy = min(
                    outcome.actual_recovery / outcome.predicted_recovery,
                    outcome.predicted_recovery / max(outcome.actual_recovery, 0.01),
                )
                reward_magnitude = accuracy * outcome.actual_recovery
            else:
                reward_magnitude = outcome.actual_recovery

        elif outcome.outcome_type == OutcomeType.TRANSFER_STALLED:
            # Transfer failed — weaken the cross-store binding
            reward_type = "negative"
            reward_magnitude = outcome.predicted_recovery  # Lost opportunity cost

        elif outcome.outcome_type == OutcomeType.FINDING_CONFIRMED:
            reward_type = "positive"
            if outcome.dollar_impact_actual and outcome.dollar_impact_predicted:
                accuracy = 1.0 - abs(
                    outcome.dollar_impact_actual - outcome.dollar_impact_predicted
                ) / max(outcome.dollar_impact_predicted, 1.0)
                reward_magnitude = max(0, accuracy) * outcome.dollar_impact_actual
            else:
                reward_magnitude = outcome.dollar_impact_actual or 100.0

        elif outcome.outcome_type == OutcomeType.FINDING_DISPUTED:
            reward_type = "negative"
            reward_magnitude = outcome.dollar_impact_predicted or 100.0

        elif outcome.outcome_type == OutcomeType.PREDICTION_CORRECT:
            reward_type = "positive"
            reward_magnitude = outcome.dollar_impact_actual or 50.0

        elif outcome.outcome_type == OutcomeType.PREDICTION_WRONG:
            reward_type = "negative"
            reward_magnitude = outcome.dollar_impact_predicted or 50.0

        # Encode as VSA vector for contrastive update
        success_basis = {
            "positive": self.success_vec,
            "negative": self.failure_vec,
            "neutral": self.neutral_vec,
        }[reward_type]

        reward_vector = self.algebra.bind(self.outcome_roles["success"], success_basis)

        return {
            "reward_type": reward_type,
            "reward_magnitude": reward_magnitude,
            "reward_vector": reward_vector,
            "learning_rate_scale": min(reward_magnitude / 1000.0, 2.0),
            "outcome_id": outcome.outcome_id,
        }

    def _update_aggregates(self, outcome: Outcome):
        """Update running aggregate metrics."""
        transfers = [
            o
            for o in self.outcomes
            if o.outcome_type
            in (OutcomeType.TRANSFER_SOLD, OutcomeType.TRANSFER_STALLED)
        ]
        if transfers:
            sold = sum(
                1 for o in transfers if o.outcome_type == OutcomeType.TRANSFER_SOLD
            )
            self.transfer_success_rate = sold / len(transfers)

        findings = [
            o
            for o in self.outcomes
            if o.outcome_type
            in (OutcomeType.FINDING_CONFIRMED, OutcomeType.FINDING_DISPUTED)
        ]
        if findings:
            confirmed = sum(
                1 for o in findings if o.outcome_type == OutcomeType.FINDING_CONFIRMED
            )
            self.finding_accuracy_rate = confirmed / len(findings)

        predictions = [
            o
            for o in self.outcomes
            if o.outcome_type
            in (OutcomeType.PREDICTION_CORRECT, OutcomeType.PREDICTION_WRONG)
        ]
        if predictions:
            correct = sum(
                1
                for o in predictions
                if o.outcome_type == OutcomeType.PREDICTION_CORRECT
            )
            self.prediction_accuracy_rate = correct / len(predictions)

        self.total_value_recovered = sum(
            o.actual_recovery
            for o in self.outcomes
            if o.outcome_type == OutcomeType.TRANSFER_SOLD
        )
        self.total_value_predicted = sum(
            o.predicted_recovery
            for o in self.outcomes
            if o.outcome_type
            in (OutcomeType.TRANSFER_SOLD, OutcomeType.TRANSFER_STALLED)
        )

    def get_pattern_reliability(self, pattern_key: str) -> float:
        """
        How reliable is a specific pattern based on outcome history?

        Returns 0.0-1.0. Used by the orchestrator to calibrate
        confidence language and by the response validator to
        determine confidence level.
        """
        successes, attempts = self.pattern_success.get(pattern_key, (0, 0))
        if attempts == 0:
            return 0.5  # No data — neutral prior
        # Beta distribution mean with Laplace smoothing
        return (successes + 1) / (attempts + 2)

    def feedback_summary(self) -> dict:
        """Summary for the orchestrator's context window."""
        return {
            "total_outcomes": len(self.outcomes),
            "transfer_success_rate": round(self.transfer_success_rate, 3),
            "finding_accuracy_rate": round(self.finding_accuracy_rate, 3),
            "prediction_accuracy_rate": round(self.prediction_accuracy_rate, 3),
            "total_value_recovered": round(self.total_value_recovered, 2),
            "total_value_predicted": round(self.total_value_predicted, 2),
            "recovery_ratio": round(
                self.total_value_recovered / max(self.total_value_predicted, 1.0), 3
            ),
        }


# =============================================================================
# 2. TEMPORAL DEPTH
# =============================================================================


class TimeScale(Enum):
    """Temporal hierarchy levels."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    SEASONAL = "seasonal"  # ~3 months
    ANNUAL = "annual"


@dataclass
class TemporalPrimitive:
    """A learned pattern at a specific time scale."""

    name: str
    scale: TimeScale
    vector: np.ndarray
    observation_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    confidence: float = 0.0  # How reliable is this pattern?

    # For seasonal patterns: which periods is it active?
    active_periods: list[int] = field(default_factory=list)


class TemporalHierarchy:
    """
    Multi-scale temporal pattern recognition.

    Learns patterns at five timescales independently, then
    cross-binds them to detect when patterns at different
    scales interact.

    Example: "This SKU has weekly reorder cycles (scale: weekly)
    that get stronger in spring (scale: seasonal) because of
    a contractor account (detected via velocity primitive)."

    The cross-scale binding is where temporal depth creates
    value that single-scale analysis can't provide.
    """

    def __init__(self, algebra: PhasorAlgebra):
        self.algebra = algebra

        # Scale role vectors
        self.scale_roles = {
            scale: algebra.get_or_create(f"scale_{scale.value}") for scale in TimeScale
        }

        # Period role vectors (day-of-week, month-of-year, etc.)
        self.dow_vectors = {i: algebra.get_or_create(f"dow_{i}") for i in range(7)}
        self.month_vectors = {i: algebra.get_or_create(f"month_{i}") for i in range(12)}
        self.quarter_vectors = {
            i: algebra.get_or_create(f"quarter_{i}") for i in range(4)
        }

        # Learned primitives per scale
        self.primitives: dict[TimeScale, dict[str, TemporalPrimitive]] = {
            scale: {} for scale in TimeScale
        }

        # Observation buffers per scale (for windowed analysis)
        self.buffers: dict[TimeScale, deque] = {
            TimeScale.DAILY: deque(maxlen=90),  # 3 months of daily
            TimeScale.WEEKLY: deque(maxlen=52),  # 1 year of weekly
            TimeScale.MONTHLY: deque(maxlen=24),  # 2 years monthly
            TimeScale.SEASONAL: deque(maxlen=12),  # 3 years seasonal
            TimeScale.ANNUAL: deque(maxlen=5),  # 5 years annual
        }

        # Cross-scale bindings
        self.cross_scale_patterns: list[dict] = []

    def observe(
        self,
        entity_id: str,
        state_vector: np.ndarray,
        timestamp: float,
        metadata: dict = None,
    ):
        """
        Record an observation with temporal context.

        The same observation is encoded at each timescale with
        the appropriate period binding. Daily gets day-of-week,
        weekly gets week-of-year context, etc.
        """
        import datetime

        dt = datetime.datetime.fromtimestamp(timestamp)

        # Encode at each scale
        daily_ctx = self.algebra.bind(state_vector, self.dow_vectors[dt.weekday()])
        self.buffers[TimeScale.DAILY].append(
            {
                "entity": entity_id,
                "vector": daily_ctx,
                "raw": state_vector,
                "timestamp": timestamp,
                "period": dt.weekday(),
            }
        )

        weekly_ctx = self.algebra.bind(
            state_vector,
            self.algebra.permute(
                self.scale_roles[TimeScale.WEEKLY], dt.isocalendar()[1] % 52
            ),
        )
        self.buffers[TimeScale.WEEKLY].append(
            {
                "entity": entity_id,
                "vector": weekly_ctx,
                "raw": state_vector,
                "timestamp": timestamp,
                "period": dt.isocalendar()[1],
            }
        )

        monthly_ctx = self.algebra.bind(state_vector, self.month_vectors[dt.month - 1])
        self.buffers[TimeScale.MONTHLY].append(
            {
                "entity": entity_id,
                "vector": monthly_ctx,
                "raw": state_vector,
                "timestamp": timestamp,
                "period": dt.month,
            }
        )

        seasonal_ctx = self.algebra.bind(
            state_vector, self.quarter_vectors[(dt.month - 1) // 3]
        )
        self.buffers[TimeScale.SEASONAL].append(
            {
                "entity": entity_id,
                "vector": seasonal_ctx,
                "raw": state_vector,
                "timestamp": timestamp,
                "period": (dt.month - 1) // 3,
            }
        )

    def learn_patterns(self, scale: TimeScale, min_observations: int = 10):
        """
        Derive temporal primitives from the observation buffer at
        a given scale.

        Uses the same warmup approach as the world model battery:
        cluster observed patterns, derive centroids as primitives.
        """
        buffer = list(self.buffers[scale])
        if len(buffer) < min_observations:
            return []

        vectors = np.array([obs["vector"] for obs in buffer])
        n = len(vectors)

        # Pairwise similarity — use Rust matrix_similarity when available
        if hasattr(self.algebra, "matrix_similarity") and n >= 10:
            # Rust path: compute full sim matrix via N queries against N rows
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                scores = self.algebra.matrix_similarity(vectors[i], vectors)
                sim_matrix[i, :] = scores
        else:
            # Pure Python fallback
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    s = self.algebra.similarity(vectors[i], vectors[j])
                    sim_matrix[i, j] = s
                    sim_matrix[j, i] = s
                sim_matrix[i, i] = 1.0

        # Greedy clustering (same as warmup in battery.py)
        max_primitives = min(8, n // 3)
        centroids = [0]  # Start with first observation

        for _ in range(max_primitives - 1):
            # Find observation most distant from all existing centroids
            min_sims = np.ones(n)
            for c_idx in centroids:
                min_sims = np.minimum(min_sims, sim_matrix[c_idx])

            # Exclude existing centroids
            for c_idx in centroids:
                min_sims[c_idx] = 2.0

            next_idx = np.argmin(min_sims)
            if min_sims[next_idx] < 0.8:  # Only add if sufficiently different
                centroids.append(next_idx)

        # Create primitives from centroids
        new_primitives = []
        for i, c_idx in enumerate(centroids):
            name = f"{scale.value}_P{i}"

            # Find which observations belong to this cluster
            cluster_periods = []
            cluster_count = 0
            for j, obs in enumerate(buffer):
                if sim_matrix[c_idx, j] > 0.3:
                    cluster_periods.append(obs["period"])
                    cluster_count += 1

            prim = TemporalPrimitive(
                name=name,
                scale=scale,
                vector=vectors[c_idx].copy(),
                observation_count=cluster_count,
                first_seen=buffer[c_idx]["timestamp"],
                last_seen=buffer[-1]["timestamp"],
                confidence=cluster_count / len(buffer),
                active_periods=list(set(cluster_periods)),
            )
            self.primitives[scale][name] = prim
            new_primitives.append(prim)

        return new_primitives

    def detect_cross_scale_patterns(self) -> list[dict]:
        """
        Find patterns that span multiple timescales.

        Cross-binds primitives from different scales and checks
        for high-similarity combinations. These represent things
        like "weekly reorder cycle that intensifies seasonally."
        """
        patterns = []
        scales = list(self.primitives.keys())

        for i, scale_a in enumerate(scales):
            for scale_b in scales[i + 1 :]:
                for name_a, prim_a in self.primitives[scale_a].items():
                    for name_b, prim_b in self.primitives[scale_b].items():
                        # Cross-bind
                        cross = self.algebra.bind(prim_a.vector, prim_b.vector)

                        # Check if this cross-binding is significantly
                        # different from random (similarity to either
                        # component alone should be low if it's a
                        # genuine cross-scale pattern)
                        sim_a = self.algebra.similarity(cross, prim_a.vector)
                        sim_b = self.algebra.similarity(cross, prim_b.vector)

                        # Genuine cross-scale patterns have moderate
                        # similarity to both components
                        if 0.1 < sim_a < 0.6 and 0.1 < sim_b < 0.6:
                            # Check for shared active periods
                            shared_periods = set(prim_a.active_periods) & set(
                                prim_b.active_periods
                            )

                            pattern = {
                                "scales": (scale_a.value, scale_b.value),
                                "primitives": (name_a, name_b),
                                "cross_vector": cross,
                                "confidence": min(prim_a.confidence, prim_b.confidence),
                                "shared_periods": list(shared_periods),
                                "description": (
                                    f"{name_a} ({scale_a.value}) interacts "
                                    f"with {name_b} ({scale_b.value})"
                                ),
                            }
                            patterns.append(pattern)

        self.cross_scale_patterns = patterns
        return patterns

    def predict_seasonal(
        self, entity_id: str, current_state: np.ndarray, target_month: int
    ) -> dict:
        """
        Predict what an entity will look like at a future month
        by applying seasonal primitives.

        This is the "should I stock up for spring?" query.
        """
        target_quarter = (target_month - 1) // 3

        # Find seasonal primitives active in the target period
        active_seasonal = [
            p
            for p in self.primitives.get(TimeScale.SEASONAL, {}).values()
            if target_quarter in p.active_periods
        ]

        if not active_seasonal:
            return {
                "prediction": None,
                "confidence": 0.0,
                "reason": "No seasonal patterns learned for this period yet.",
            }

        # Apply each seasonal primitive and bundle predictions
        predictions = []
        for prim in active_seasonal:
            predicted = self.algebra.bind(current_state, prim.vector)
            predictions.append(
                {
                    "vector": predicted,
                    "primitive": prim.name,
                    "confidence": prim.confidence,
                }
            )

        # Weight by confidence
        if predictions:
            best = max(predictions, key=lambda p: p["confidence"])
            return {
                "prediction": best["vector"],
                "confidence": best["confidence"],
                "primitive_used": best["primitive"],
                "n_patterns_considered": len(predictions),
                "reason": f"Based on {best['primitive']} pattern.",
            }

        return {
            "prediction": None,
            "confidence": 0.0,
            "reason": "Insufficient seasonal data.",
        }


# =============================================================================
# 3. VENDOR INTELLIGENCE
# =============================================================================


@dataclass
class VendorProfile:
    """Aggregated intelligence about a vendor across the network."""

    vendor_id: str
    vendor_name: str

    # Stores that buy from this vendor
    active_stores: set[str] = field(default_factory=set)

    # SKUs supplied
    sku_count: int = 0
    skus: set[str] = field(default_factory=set)

    # Pricing behavior
    cost_changes: deque[dict] = field(default_factory=lambda: deque(maxlen=1000))
    avg_cost_change_pct: float = 0.0
    cost_change_frequency: float = 0.0  # Changes per quarter

    # Delivery behavior
    delivery_scores: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    avg_fill_rate: float = 1.0
    fill_rate_trend: float = 0.0  # Positive = improving

    # Network-wide margin impact
    total_margin_impact: float = 0.0  # Dollars
    margin_impact_by_store: dict[str, float] = field(default_factory=dict)

    # VSA encoding
    behavior_vector: np.ndarray | None = None

    # Risk score (0 = safe, 1 = high risk)
    risk_score: float = 0.0


class VendorIntelligence:
    """
    Cross-network vendor analysis.

    Individual stores see their own vendor interactions. The network
    sees patterns across all stores. This creates intelligence that
    no single store could produce alone.

    Examples:
    - "Vendor X raised prices 8% at all stores but only 6 adjusted retail"
    - "Vendor Y's fill rate dropped from 94% to 87% network-wide"
    - "Three stores are near a volume rebate threshold they don't know about"
    - "Vendor Z's delivery timing correlates with seasonal demand spikes"

    The VSA encoding allows vendor behavior patterns to be compared
    geometrically. Two vendors with similar behavior vectors exhibit
    similar patterns, even if they supply different products.
    """

    def __init__(self, algebra: PhasorAlgebra):
        self.algebra = algebra
        self.vendors: dict[str, VendorProfile] = {}

        # Vendor behavior encoding roles
        self.vendor_roles = {
            "pricing": algebra.get_or_create("vendor_pricing"),
            "delivery": algebra.get_or_create("vendor_delivery"),
            "reliability": algebra.get_or_create("vendor_reliability"),
            "risk": algebra.get_or_create("vendor_risk"),
        }

        # Behavior basis vectors
        self.pricing_behaviors = {
            "stable": algebra.get_or_create("pricing_stable"),
            "gradual_increase": algebra.get_or_create("pricing_gradual_up"),
            "sudden_increase": algebra.get_or_create("pricing_sudden_up"),
            "decrease": algebra.get_or_create("pricing_decrease"),
            "volatile": algebra.get_or_create("pricing_volatile"),
        }

        self.delivery_behaviors = {
            "reliable": algebra.get_or_create("delivery_reliable"),
            "declining": algebra.get_or_create("delivery_declining"),
            "erratic": algebra.get_or_create("delivery_erratic"),
            "seasonal": algebra.get_or_create("delivery_seasonal"),
        }

        # Network-wide alerts (bounded)
        self.alerts: deque[dict] = deque(maxlen=1000)

    def register_vendor(self, vendor_id: str, vendor_name: str):
        """Register a vendor in the intelligence system."""
        if vendor_id not in self.vendors:
            self.vendors[vendor_id] = VendorProfile(
                vendor_id=vendor_id,
                vendor_name=vendor_name,
            )

    def record_cost_change(
        self,
        vendor_id: str,
        store_id: str,
        sku_id: str,
        old_cost: float,
        new_cost: float,
        timestamp: float,
    ):
        """Record a cost change event for a vendor."""
        if vendor_id not in self.vendors:
            return

        profile = self.vendors[vendor_id]
        profile.active_stores.add(store_id)
        profile.skus.add(sku_id)
        profile.sku_count = len(profile.skus)

        pct_change = (new_cost - old_cost) / old_cost if old_cost > 0 else 0

        profile.cost_changes.append(
            {
                "store_id": store_id,
                "sku_id": sku_id,
                "old_cost": old_cost,
                "new_cost": new_cost,
                "pct_change": pct_change,
                "timestamp": timestamp,
            }
        )

        # Recalculate averages
        if profile.cost_changes:
            profile.avg_cost_change_pct = np.mean(
                [c["pct_change"] for c in profile.cost_changes]
            )

        # Check for network-wide alerts
        self._check_cost_alert(vendor_id, sku_id, pct_change, timestamp)

    def record_delivery(
        self,
        vendor_id: str,
        store_id: str,
        ordered_qty: int,
        received_qty: int,
        timestamp: float,
    ):
        """Record a delivery event."""
        if vendor_id not in self.vendors:
            return

        profile = self.vendors[vendor_id]
        profile.active_stores.add(store_id)

        fill_rate = received_qty / ordered_qty if ordered_qty > 0 else 1.0
        profile.delivery_scores.append(fill_rate)

        # Rolling average (last 20 deliveries)
        scores = list(profile.delivery_scores)
        recent = scores[-20:]
        profile.avg_fill_rate = np.mean(recent)

        # Trend (compare last 10 to previous 10)
        if len(profile.delivery_scores) >= 20:
            recent_10 = np.mean(scores[-10:])
            previous_10 = np.mean(scores[-20:-10])
            profile.fill_rate_trend = recent_10 - previous_10

        # Alert on declining fill rate
        if profile.fill_rate_trend < -0.05 and len(profile.delivery_scores) >= 20:
            self.alerts.append(
                {
                    "type": "FILL_RATE_DECLINE",
                    "vendor_id": vendor_id,
                    "vendor_name": profile.vendor_name,
                    "current_rate": round(profile.avg_fill_rate, 3),
                    "trend": round(profile.fill_rate_trend, 3),
                    "timestamp": timestamp,
                    "message": (
                        f"{profile.vendor_name}'s delivery fill rate has "
                        f"dropped to {profile.avg_fill_rate:.0%} "
                        f"(trending {profile.fill_rate_trend:+.1%})"
                    ),
                }
            )

    def _check_cost_alert(
        self, vendor_id: str, sku_id: str, pct_change: float, timestamp: float
    ):
        """Check if a cost change warrants a network alert."""
        profile = self.vendors[vendor_id]

        # Alert: sudden large increase (>5%)
        if pct_change > 0.05:
            # Check if this affects multiple stores
            recent_changes = [
                c
                for c in profile.cost_changes
                if c["timestamp"] > timestamp - 30 * 86400  # Last 30 days
                and c["pct_change"] > 0.03
            ]
            affected_stores = set(c["store_id"] for c in recent_changes)

            if len(affected_stores) >= 2:
                self.alerts.append(
                    {
                        "type": "NETWORK_COST_INCREASE",
                        "vendor_id": vendor_id,
                        "vendor_name": profile.vendor_name,
                        "avg_increase": round(
                            np.mean([c["pct_change"] for c in recent_changes]) * 100, 1
                        ),
                        "affected_stores": list(affected_stores),
                        "affected_skus": list(set(c["sku_id"] for c in recent_changes)),
                        "timestamp": timestamp,
                        "message": (
                            f"{profile.vendor_name} has raised costs across "
                            f"{len(affected_stores)} stores in the last 30 days. "
                            f"Review pricing adjustments."
                        ),
                    }
                )

    def encode_vendor_behavior(self, vendor_id: str) -> np.ndarray | None:
        """
        Encode a vendor's behavior pattern as a VSA vector.

        This allows geometric comparison between vendors.
        Two vendors with similar behavior vectors exhibit
        similar patterns regardless of what products they supply.
        """
        profile = self.vendors.get(vendor_id)
        if not profile:
            return None

        a = self.algebra

        # Pricing behavior classification
        if len(profile.cost_changes) < 3:
            pricing_vec = self.pricing_behaviors["stable"]
        elif profile.avg_cost_change_pct > 0.08:
            pricing_vec = self.pricing_behaviors["sudden_increase"]
        elif profile.avg_cost_change_pct > 0.02:
            pricing_vec = self.pricing_behaviors["gradual_increase"]
        elif profile.avg_cost_change_pct < -0.02:
            pricing_vec = self.pricing_behaviors["decrease"]
        else:
            # Check volatility
            changes = [c["pct_change"] for c in profile.cost_changes]
            if np.std(changes) > 0.05:
                pricing_vec = self.pricing_behaviors["volatile"]
            else:
                pricing_vec = self.pricing_behaviors["stable"]

        # Delivery behavior classification
        if len(profile.delivery_scores) < 5:
            delivery_vec = self.delivery_behaviors["reliable"]
        elif profile.avg_fill_rate > 0.95:
            delivery_vec = self.delivery_behaviors["reliable"]
        elif profile.fill_rate_trend < -0.03:
            delivery_vec = self.delivery_behaviors["declining"]
        elif np.std(list(profile.delivery_scores)[-10:]) > 0.1:
            delivery_vec = self.delivery_behaviors["erratic"]
        else:
            delivery_vec = self.delivery_behaviors["reliable"]

        # Compose behavior vector
        behavior = a.bind(
            a.bind(self.vendor_roles["pricing"], pricing_vec),
            a.bind(self.vendor_roles["delivery"], delivery_vec),
        )

        profile.behavior_vector = behavior

        # Compute risk score
        risk = 0.0
        if profile.avg_cost_change_pct > 0.05:
            risk += 0.3
        if profile.avg_fill_rate < 0.90:
            risk += 0.3
        if profile.fill_rate_trend < -0.03:
            risk += 0.2
        if len(profile.cost_changes) > 0:
            recent_changes = [
                c
                for c in profile.cost_changes
                if c["timestamp"] > time.time() - 90 * 86400
            ]
            if len(recent_changes) > 5:
                risk += 0.2  # Frequent changes = instability

        profile.risk_score = min(risk, 1.0)

        return behavior

    def find_similar_vendors(
        self, vendor_id: str, top_k: int = 3
    ) -> list[tuple[str, float]]:
        """
        Find vendors with similar behavior patterns.

        Useful for: "This vendor is behaving like Vendor X did
        before they went bankrupt / raised prices / improved service."
        """
        source = self.vendors.get(vendor_id)
        if not source or source.behavior_vector is None:
            return []

        # Collect candidates (exclude self and vendors without behavior vectors)
        candidate_ids = []
        candidate_vecs = []
        for vid, profile in self.vendors.items():
            if vid == vendor_id or profile.behavior_vector is None:
                continue
            candidate_ids.append(vid)
            candidate_vecs.append(profile.behavior_vector)

        if not candidate_ids:
            return []

        # Use Rust matrix_similarity when available
        if hasattr(self.algebra, "matrix_similarity") and len(candidate_ids) >= 4:
            matrix = np.array(candidate_vecs)
            scores = self.algebra.matrix_similarity(source.behavior_vector, matrix)
            similarities = list(zip(candidate_ids, scores.tolist()))
        else:
            similarities = []
            for vid, vec in zip(candidate_ids, candidate_vecs):
                sim = self.algebra.similarity(source.behavior_vector, vec)
                similarities.append((vid, float(sim)))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]

    def network_vendor_report(self) -> dict:
        """Generate a network-wide vendor intelligence summary."""
        high_risk = [(vid, v) for vid, v in self.vendors.items() if v.risk_score > 0.5]

        return {
            "total_vendors": len(self.vendors),
            "high_risk_vendors": [
                {
                    "vendor_id": vid,
                    "vendor_name": v.vendor_name,
                    "risk_score": round(v.risk_score, 2),
                    "avg_cost_change": round(v.avg_cost_change_pct * 100, 1),
                    "fill_rate": round(v.avg_fill_rate * 100, 1),
                    "stores_affected": len(v.active_stores),
                }
                for vid, v in high_risk
            ],
            "active_alerts": len(self.alerts),
            "recent_alerts": list(self.alerts)[-5:] if self.alerts else [],
        }


# =============================================================================
# 4. PREDICTIVE INTERVENTION
# =============================================================================


class InterventionType(Enum):
    """Types of predictive interventions."""

    DEAD_STOCK_WARNING = "dead_stock_warning"
    MARGIN_EROSION_TREND = "margin_erosion_trend"
    VELOCITY_DECLINE = "velocity_decline"
    STOCKOUT_RISK = "stockout_risk"
    VENDOR_RISK = "vendor_risk"
    SEASONAL_PREPARATION = "seasonal_preparation"


@dataclass
class Intervention:
    """A predictive intervention — warning before the problem forms."""

    intervention_type: InterventionType
    entity_id: str
    store_id: str

    # Timing
    predicted_days_to_event: int
    confidence: float
    urgency: float  # 0-1, combines time pressure + dollar impact

    # What will happen if no action taken
    predicted_impact: float  # Dollars at risk
    description: str

    # What to do about it
    recommended_action: str
    alternative_actions: list[str] = field(default_factory=list)

    # Evidence
    trend_data: list[float] = field(default_factory=list)
    supporting_signals: list[str] = field(default_factory=list)

    # Tracking
    intervention_id: str = ""
    created_at: float = 0.0
    expires_at: float = 0.0  # When this warning becomes stale

    def to_dict(self) -> dict:
        return {
            "type": self.intervention_type.value,
            "entity": self.entity_id,
            "store": self.store_id,
            "days_to_event": self.predicted_days_to_event,
            "confidence": round(self.confidence, 2),
            "urgency": round(self.urgency, 2),
            "predicted_impact": round(self.predicted_impact, 2),
            "description": self.description,
            "action": self.recommended_action,
        }


class PredictiveEngine:
    """
    Predicts problems before they fully form and generates
    interventions with time-to-act windows.

    Uses three signal sources:
    1. Trajectory analysis — extrapolating current trends
    2. Temporal patterns — seasonal/cyclical expectations
    3. Vendor intelligence — supply-side risk signals

    The key insight: detection tells you what IS wrong.
    Prediction tells you what WILL BE wrong. The difference
    is the intervention window — time the customer has to act
    before the problem materializes.
    """

    def __init__(
        self,
        algebra: PhasorAlgebra,
        temporal: TemporalHierarchy = None,
        vendor_intel: VendorIntelligence = None,
        feedback: FeedbackEngine = None,
        dead_stock_config: DeadStockConfig = None,
    ):
        self.algebra = algebra
        self.temporal = temporal
        self.vendor_intel = vendor_intel
        self.feedback = feedback
        self.dead_stock_config = dead_stock_config or DeadStockConfig()

        # Active interventions (not yet expired)
        self.active_interventions: dict[str, Intervention] = {}
        self.intervention_counter = 0

        # Thresholds — read from config where applicable
        self.velocity_decline_threshold = -0.3  # 30% decline triggers warning
        self.dead_stock_velocity_threshold = self.dead_stock_config.min_healthy_velocity
        self.margin_erosion_threshold = -0.02  # 2% margin decline
        self.stockout_days_threshold = 14  # Warn 2 weeks before stockout

    def analyze_entity(
        self,
        entity_id: str,
        store_id: str,
        history: list[dict],
        current_state: dict,
        vendor_id: str = None,
        current_time: float = None,
    ) -> list[Intervention]:
        """
        Analyze an entity's trajectory and generate interventions.

        history: list of {timestamp, velocity, stock, margin, cost, price}
        current_state: latest observation
        """
        interventions = []

        if len(history) < 7:
            return interventions  # Need at least a week of data

        # --- Dead stock prediction ---
        velocity_trend = self._compute_trend(
            [h.get("velocity", 0) for h in history[-21:]]
        )
        current_velocity = current_state.get("velocity", 0)

        if current_velocity > 0 and velocity_trend < self.velocity_decline_threshold:
            # Velocity is declining — predict when it hits zero
            if velocity_trend < -0.01:  # Avoid division by near-zero
                days_to_zero = (
                    int(-current_velocity / velocity_trend)
                    if velocity_trend != 0
                    else 999
                )
            else:
                days_to_zero = 999

            current_stock = current_state.get("stock", 0)
            unit_cost = current_state.get("cost", 0)
            capital_at_risk = current_stock * unit_cost

            if days_to_zero < 60 and capital_at_risk > 50:
                urgency = min(1.0, (60 - days_to_zero) / 60 * capital_at_risk / 1000)

                intervention = Intervention(
                    intervention_type=InterventionType.DEAD_STOCK_WARNING,
                    entity_id=entity_id,
                    store_id=store_id,
                    predicted_days_to_event=days_to_zero,
                    confidence=min(0.9, 0.5 + len(history) / 60),
                    urgency=urgency,
                    predicted_impact=capital_at_risk,
                    description=(
                        f"Sales velocity declining {abs(velocity_trend):.0%}/week. "
                        f"At current rate, this item stops selling in "
                        f"~{days_to_zero} days. {current_stock} units "
                        f"(${capital_at_risk:,.0f}) at risk of becoming dead stock."
                    ),
                    recommended_action=(
                        f"Consider reducing reorder quantity or transferring "
                        f"{current_stock // 2} units to a store with active demand."
                    ),
                    alternative_actions=[
                        "Run a limited-time promotion to accelerate sell-through",
                        "Bundle with complementary items to increase velocity",
                    ],
                    trend_data=[h.get("velocity", 0) for h in history[-14:]],
                    supporting_signals=[
                        f"Velocity trend: {velocity_trend:+.1%}/week",
                        f"Current velocity: {current_velocity:.1f} units/week",
                    ],
                )
                interventions.append(intervention)

        # --- Margin erosion prediction ---
        margins = [h.get("margin", 0) for h in history[-30:]]
        if len(margins) >= 14:
            margin_trend = self._compute_trend(margins)
            current_margin = current_state.get("margin", 0)

            if margin_trend < self.margin_erosion_threshold:
                # Predict when margin hits zero
                days_to_zero_margin = (
                    int(-current_margin / margin_trend)
                    if margin_trend < -0.001
                    else 999
                )

                velocity = current_state.get("velocity", 0)
                weekly_revenue = velocity * current_state.get("price", 0)
                annual_impact = abs(margin_trend) * weekly_revenue * 52

                if annual_impact > 100:
                    intervention = Intervention(
                        intervention_type=InterventionType.MARGIN_EROSION_TREND,
                        entity_id=entity_id,
                        store_id=store_id,
                        predicted_days_to_event=min(days_to_zero_margin, 180),
                        confidence=min(0.85, 0.4 + len(margins) / 60),
                        urgency=min(1.0, annual_impact / 5000),
                        predicted_impact=annual_impact,
                        description=(
                            f"Margin declining {abs(margin_trend):.1%}/week. "
                            f"Projected annual impact: ${annual_impact:,.0f}."
                        ),
                        recommended_action=(
                            "Review vendor costs and adjust retail pricing."
                        ),
                        trend_data=margins[-14:],
                        supporting_signals=[
                            f"Margin trend: {margin_trend:+.2%}/week",
                            f"Current margin: {current_margin:.1%}",
                        ],
                    )
                    interventions.append(intervention)

        # --- Stockout prediction ---
        current_stock = current_state.get("stock", 0)
        if current_velocity > 0 and current_stock > 0:
            days_of_supply = current_stock / (current_velocity / 7)

            if days_of_supply < self.stockout_days_threshold:
                daily_revenue = current_velocity / 7 * current_state.get("price", 0)
                potential_lost_revenue = daily_revenue * (
                    self.stockout_days_threshold - days_of_supply
                )

                intervention = Intervention(
                    intervention_type=InterventionType.STOCKOUT_RISK,
                    entity_id=entity_id,
                    store_id=store_id,
                    predicted_days_to_event=int(days_of_supply),
                    confidence=0.8,
                    urgency=min(
                        1.0,
                        (self.stockout_days_threshold - days_of_supply)
                        / self.stockout_days_threshold,
                    ),
                    predicted_impact=potential_lost_revenue,
                    description=(
                        f"Only {days_of_supply:.0f} days of supply remaining "
                        f"at current sell rate. Risk of stockout."
                    ),
                    recommended_action="Reorder immediately or request transfer from another store.",
                    trend_data=[h.get("stock", 0) for h in history[-14:]],
                    supporting_signals=[
                        f"Current stock: {current_stock} units",
                        f"Velocity: {current_velocity:.1f}/week",
                        f"Days of supply: {days_of_supply:.0f}",
                    ],
                )
                interventions.append(intervention)

        # --- Vendor risk signals ---
        if vendor_id and self.vendor_intel:
            vendor = self.vendor_intel.vendors.get(vendor_id)
            if vendor and vendor.risk_score > 0.5:
                intervention = Intervention(
                    intervention_type=InterventionType.VENDOR_RISK,
                    entity_id=entity_id,
                    store_id=store_id,
                    predicted_days_to_event=30,  # Vendor issues develop over weeks
                    confidence=min(0.7, vendor.risk_score),
                    urgency=vendor.risk_score * 0.5,
                    predicted_impact=current_stock * unit_cost * 0.1,  # 10% risk
                    description=(
                        f"Vendor {vendor.vendor_name} showing elevated risk "
                        f"(score: {vendor.risk_score:.0%}). "
                        f"Fill rate: {vendor.avg_fill_rate:.0%}, "
                        f"cost trend: {vendor.avg_cost_change_pct:+.1%}."
                    ),
                    recommended_action=(
                        "Consider identifying alternative suppliers or "
                        "increasing safety stock."
                    ),
                    supporting_signals=[
                        f"Vendor risk score: {vendor.risk_score:.0%}",
                        f"Fill rate: {vendor.avg_fill_rate:.0%}",
                        f"Stores affected: {len(vendor.active_stores)}",
                    ],
                )
                interventions.append(intervention)

        # Assign IDs and timestamps
        sim_time = current_time or time.time()
        for intervention in interventions:
            self.intervention_counter += 1
            intervention.intervention_id = f"INT-{self.intervention_counter:06d}"
            intervention.created_at = sim_time
            intervention.expires_at = (
                sim_time + intervention.predicted_days_to_event * 86400
            )

        # Store active interventions
        for intervention in interventions:
            self.active_interventions[intervention.intervention_id] = intervention

        return interventions

    def _compute_trend(self, values: list[float]) -> float:
        """
        Compute linear trend (slope) of a time series.
        Returns rate of change per observation period.
        """
        if len(values) < 3:
            return 0.0

        n = len(values)
        x = np.arange(n, dtype=float)
        y = np.array(values, dtype=float)

        # Avoid NaN from constant series
        if np.std(y) < 1e-10:
            return 0.0

        # Simple linear regression slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator < 1e-10:
            return 0.0

        return float(numerator / denominator)

    def expire_stale_interventions(self, current_time: float = None):
        """Remove interventions that have passed their window."""
        now = current_time or time.time()
        expired = [
            iid
            for iid, intervention in self.active_interventions.items()
            if intervention.expires_at < now
        ]
        for iid in expired:
            del self.active_interventions[iid]

    def prioritized_interventions(
        self, store_id: str = None, top_k: int = 5, current_time: float = None
    ) -> list[Intervention]:
        """Get top interventions ranked by urgency × impact."""
        self.expire_stale_interventions(current_time)

        interventions = list(self.active_interventions.values())
        if store_id:
            interventions = [i for i in interventions if i.store_id == store_id]

        interventions.sort(key=lambda i: i.urgency * i.predicted_impact, reverse=True)
        return interventions[:top_k]


# =============================================================================
# 5. COMPETITIVE MOAT METRICS
# =============================================================================


class MoatMetrics:
    """
    Tracks the metrics that compound over time and constitute
    the competitive moat.

    Three categories:
    1. Quality metrics — how accurate are we getting?
    2. Network metrics — how much intelligence does scale provide?
    3. Value metrics — how much money have we saved customers?

    These metrics serve two purposes:
    - Internal: guide development priorities (what's improving, what's not)
    - External: demonstrate compounding value to customers and investors

    The moat is real when these metrics improve faster than a
    new competitor could replicate from scratch.
    """

    def __init__(self):
        self.snapshots: deque[dict] = deque(maxlen=520)  # ~10 years of weekly
        self.snapshot_interval_days = 7  # Weekly snapshots

    def capture_snapshot(
        self,
        feedback: FeedbackEngine,
        vendor_intel: VendorIntelligence,
        temporal: TemporalHierarchy,
        predictive: PredictiveEngine,
        n_stores: int,
        n_skus: int,
    ):
        """Capture a point-in-time snapshot of all moat metrics."""
        snapshot = {
            "timestamp": time.time(),
            # --- QUALITY METRICS ---
            "transfer_success_rate": feedback.transfer_success_rate,
            "finding_accuracy_rate": feedback.finding_accuracy_rate,
            "prediction_accuracy_rate": feedback.prediction_accuracy_rate,
            # Calibration: are our confidence levels accurate?
            # High confidence findings should be confirmed more often
            # than low confidence findings
            "calibration_score": self._compute_calibration(feedback),
            # --- NETWORK METRICS ---
            "n_stores": n_stores,
            "n_skus": n_skus,
            "n_vendors_tracked": len(vendor_intel.vendors),
            "n_vendor_alerts": len(vendor_intel.alerts),
            "cross_store_findings": self._count_cross_store(feedback),
            # Temporal depth: how many timescales have learned patterns?
            "temporal_scales_active": sum(
                1 for scale in TimeScale if len(temporal.primitives.get(scale, {})) > 0
            ),
            "temporal_patterns_total": sum(
                len(prims) for prims in temporal.primitives.values()
            ),
            "cross_scale_patterns": len(temporal.cross_scale_patterns),
            # --- VALUE METRICS ---
            "total_value_recovered": feedback.total_value_recovered,
            "total_outcomes": len(feedback.outcomes),
            "value_per_outcome": (
                feedback.total_value_recovered / max(len(feedback.outcomes), 1)
            ),
            # Active interventions (predictive value)
            "active_interventions": len(predictive.active_interventions),
            "intervention_predicted_value": sum(
                i.predicted_impact for i in predictive.active_interventions.values()
            ),
        }

        # Compute trends if we have history
        if len(self.snapshots) >= 2:
            prev = self.snapshots[-1]
            snapshot["transfer_rate_trend"] = (
                snapshot["transfer_success_rate"] - prev["transfer_success_rate"]
            )
            snapshot["finding_rate_trend"] = (
                snapshot["finding_accuracy_rate"] - prev["finding_accuracy_rate"]
            )
            snapshot["value_growth"] = (
                snapshot["total_value_recovered"] - prev["total_value_recovered"]
            )

        self.snapshots.append(snapshot)
        return snapshot

    def _compute_calibration(self, feedback: FeedbackEngine) -> float:
        """
        How well-calibrated are confidence levels?

        Perfect calibration: 80% of "High confidence" findings are
        confirmed, 50% of "Medium" are confirmed, 20% of "Low" are
        confirmed. Measures the gap between stated and actual confidence.

        Returns 0-1, where 1 is perfectly calibrated.
        """
        # Simplified — would need confidence level on each outcome
        # For now, use overall accuracy as a proxy
        if feedback.finding_accuracy_rate > 0:
            return min(1.0, feedback.finding_accuracy_rate / 0.8)
        return 0.5

    def _count_cross_store(self, feedback: FeedbackEngine) -> int:
        """Count findings that involved cross-store intelligence."""
        return sum(1 for o in feedback.outcomes if o.dest_store_id is not None)

    def moat_report(self) -> dict:
        """
        Generate the moat report — the evidence that the system
        is getting better over time.
        """
        if not self.snapshots:
            return {"status": "No data yet."}

        latest = self.snapshots[-1]

        report = {
            "current_quality": {
                "transfer_success": f"{latest['transfer_success_rate']:.0%}",
                "finding_accuracy": f"{latest['finding_accuracy_rate']:.0%}",
                "prediction_accuracy": f"{latest['prediction_accuracy_rate']:.0%}",
                "calibration": f"{latest.get('calibration_score', 0):.0%}",
            },
            "network_scale": {
                "stores": latest["n_stores"],
                "skus": latest["n_skus"],
                "vendors": latest["n_vendors_tracked"],
                "temporal_depth": f"{latest['temporal_scales_active']}/5 scales",
                "cross_scale_patterns": latest["cross_scale_patterns"],
            },
            "value_delivered": {
                "total_recovered": f"${latest['total_value_recovered']:,.0f}",
                "per_outcome": f"${latest['value_per_outcome']:,.0f}",
                "total_outcomes": latest["total_outcomes"],
                "active_interventions": latest["active_interventions"],
                "intervention_value": (
                    f"${latest['intervention_predicted_value']:,.0f}"
                ),
            },
        }

        # Add trends if available
        if len(self.snapshots) >= 4:
            first_month = self.snapshots[0]
            report["improvement"] = {
                "transfer_rate_delta": (
                    f"{latest['transfer_success_rate'] - first_month['transfer_success_rate']:+.1%}"
                ),
                "finding_rate_delta": (
                    f"{latest['finding_accuracy_rate'] - first_month['finding_accuracy_rate']:+.1%}"
                ),
                "value_growth": (
                    f"${latest['total_value_recovered'] - first_month['total_value_recovered']:,.0f}"
                ),
                "network_growth": (
                    f"{latest['n_stores'] - first_month['n_stores']:+d} stores"
                ),
            }

        return report


# =============================================================================
# INTEGRATED PIPELINE
# =============================================================================


class SentinelPipeline:
    """
    The complete Profit Sentinel pipeline with all five systems
    integrated into one feedback loop.

    Usage:
        pipeline = SentinelPipeline(dim=4096)

        # Ingest data
        pipeline.ingest_store("store_1", inventory_data)

        # Run analysis
        findings = pipeline.analyze("store_1")

        # Record outcome when customer acts
        pipeline.record_outcome(outcome)

        # Generate predictions
        interventions = pipeline.predict("store_1")

        # Moat snapshot (weekly)
        report = pipeline.moat_snapshot()
    """

    def __init__(
        self,
        dim: int = 4096,
        seed: int = 42,
        use_rust: bool = False,
        dead_stock_config: DeadStockConfig = None,
    ):
        if use_rust and RUST_AVAILABLE:
            self.algebra = RustPhasorAlgebra(dim=dim, seed=seed)
        else:
            self.algebra = PhasorAlgebra(dim=dim, seed=seed)

        self.dead_stock_config = dead_stock_config or DeadStockConfig()

        # Five systems — all share the same algebra instance
        self.feedback = FeedbackEngine(self.algebra)
        self.temporal = TemporalHierarchy(self.algebra)
        self.vendor_intel = VendorIntelligence(self.algebra)
        self.predictive = PredictiveEngine(
            self.algebra,
            temporal=self.temporal,
            vendor_intel=self.vendor_intel,
            feedback=self.feedback,
            dead_stock_config=self.dead_stock_config,
        )
        self.moat = MoatMetrics()

        # Store tracking
        self.stores: dict[str, dict] = {}
        MAX_ENTITY_HISTORY = 500
        self.entity_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=MAX_ENTITY_HISTORY)
        )
        self._max_entities_per_tenant = 10000
        self._entity_last_seen: dict[str, float] = {}

    def prune_stale_entities(self, max_age_seconds: float = 86400 * 30):
        """Remove entities not seen in 30 days."""
        cutoff = time.time() - max_age_seconds
        stale = [k for k, v in self._entity_last_seen.items() if v < cutoff]
        for k in stale:
            self.entity_history.pop(k, None)
            del self._entity_last_seen[k]

    def record_observation(
        self, store_id: str, entity_id: str, observation: dict, timestamp: float = None
    ):
        """
        Record an observation — feeds all five systems.

        observation: {velocity, stock, margin, cost, price, vendor_id, ...}
        """
        timestamp = timestamp or time.time()

        # Store history for predictive engine
        obs_record = {**observation, "timestamp": timestamp}
        history_key = f"{store_id}:{entity_id}"
        self.entity_history[history_key].append(obs_record)
        self._entity_last_seen[history_key] = timestamp

        # Prune if entity count exceeds limit
        if len(self.entity_history) > self._max_entities_per_tenant:
            self.prune_stale_entities()

        # Feed temporal hierarchy
        state_vec = self._encode_observation(observation)
        self.temporal.observe(entity_id, state_vec, timestamp)

        # Feed vendor intelligence
        vendor_id = observation.get("vendor_id")
        if vendor_id:
            vendor_name = observation.get("vendor_name", vendor_id)
            self.vendor_intel.register_vendor(vendor_id, vendor_name)

            # Track cost changes
            history = self.entity_history[history_key]
            if len(history) >= 2:
                prev_cost = history[-2].get("cost", 0)
                curr_cost = observation.get("cost", 0)
                if prev_cost > 0 and abs(curr_cost - prev_cost) > 0.01:
                    self.vendor_intel.record_cost_change(
                        vendor_id, store_id, entity_id, prev_cost, curr_cost, timestamp
                    )

    def predict_interventions(
        self, store_id: str, entity_id: str, current_time: float = None
    ) -> list[Intervention]:
        """Generate predictive interventions for an entity."""
        history_key = f"{store_id}:{entity_id}"
        history = list(self.entity_history.get(history_key, []))

        if not history:
            return []

        current_state = history[-1]
        vendor_id = current_state.get("vendor_id")

        return self.predictive.analyze_entity(
            entity_id,
            store_id,
            history,
            current_state,
            vendor_id,
            current_time=current_time,
        )

    def record_outcome(self, outcome: Outcome) -> dict:
        """Record an outcome — closes the feedback loop."""
        return self.feedback.record_outcome(outcome)

    def moat_snapshot(self, n_stores: int = 0, n_skus: int = 0) -> dict:
        """Capture a moat metrics snapshot."""
        return self.moat.capture_snapshot(
            self.feedback,
            self.vendor_intel,
            self.temporal,
            self.predictive,
            n_stores,
            n_skus,
        )

    def _encode_observation(self, observation: dict) -> np.ndarray:
        """Encode an observation dict as a VSA vector."""
        # Simplified encoding — in production this uses the full
        # entity hierarchy and role-filler binding
        components = []
        for key, value in observation.items():
            if isinstance(value, (int, float)) and key not in ("timestamp",):
                # Encode numeric values as phase shifts
                phase = np.exp(1j * np.pi * value / 1000.0)
                role = self.algebra.get_or_create(f"obs_{key}")
                components.append(role * phase)

        if components:
            bundled = components[0]
            for c in components[1:]:
                bundled = bundled + c
            norm = np.abs(bundled)
            norm = np.where(norm < 1e-10, 1.0, norm)
            return bundled / norm

        return self.algebra.get_or_create("_obs_default")

    def batch_encode_observations(self, observations: list[dict]) -> np.ndarray:
        """Encode multiple observations as (N, D) matrix using Rust batch ops.

        When Rust matrix_compile_states is available, this compiles all
        observations in a single Rust call with Rayon parallelism.
        Falls back to sequential encoding otherwise.
        """
        if not observations:
            return np.array([])

        # Identify numeric keys across all observations
        numeric_keys = []
        for key in observations[0]:
            if key != "timestamp" and isinstance(observations[0][key], (int, float)):
                numeric_keys.append(key)

        if not numeric_keys:
            return np.array([self._encode_observation(obs) for obs in observations])

        # Get role vectors (R roles)
        role_vecs = np.array(
            [self.algebra.get_or_create(f"obs_{key}") for key in numeric_keys]
        )
        n_roles = len(numeric_keys)

        # Build filler matrix (N, R, D) — each filler is role * phase
        dim = role_vecs.shape[1]
        n_obs = len(observations)
        fillers = np.zeros((n_obs, n_roles, dim), dtype=complex)
        for i, obs in enumerate(observations):
            for r, key in enumerate(numeric_keys):
                val = obs.get(key, 0)
                if isinstance(val, (int, float)):
                    phase = np.exp(1j * np.pi * val / 1000.0)
                    fillers[i, r, :] = phase  # role * phase done in compile

        # Use Rust matrix_compile_states when available
        if hasattr(self.algebra, "matrix_compile_states") and n_obs >= 10:
            return self.algebra.matrix_compile_states(role_vecs, fillers)
        else:
            # Fallback: sequential encoding
            return np.array([self._encode_observation(obs) for obs in observations])

    def pipeline_status(self) -> dict:
        """Full status for the orchestrator's context window."""
        return {
            "feedback": self.feedback.feedback_summary(),
            "vendor_intel": self.vendor_intel.network_vendor_report(),
            "temporal": {
                "scales_active": sum(
                    1 for s in TimeScale if len(self.temporal.primitives.get(s, {})) > 0
                ),
                "total_patterns": sum(
                    len(p) for p in self.temporal.primitives.values()
                ),
                "cross_scale": len(self.temporal.cross_scale_patterns),
            },
            "predictions": {
                "active_interventions": len(self.predictive.active_interventions),
                "top_urgency": [
                    i.to_dict()
                    for i in self.predictive.prioritized_interventions(top_k=3)
                ],
            },
            "moat": self.moat.moat_report() if self.moat.snapshots else {},
        }
