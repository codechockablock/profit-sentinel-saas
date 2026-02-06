"""Predictive Inventory Alerts.

Uses velocity-based forecasting to predict stockouts and overstock
situations before they happen. No external ML dependencies — uses
deterministic velocity extrapolation and safety-stock math.

Algorithm:
    days_to_stockout = qty_on_hand / daily_sales_rate
    days_to_overstock = (qty_on_hand - reorder_point) / abs(daily_sales_rate)

Predictions are quantified in dollars:
    "SKU DMG-042 will stock out in 12 days. Estimated lost sales: $1,440."

Thresholds:
    Critical: stockout in <7 days
    Warning:  stockout in 7-14 days
    Watch:    stockout in 14-30 days
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from .models import Digest, Issue, IssueType, Sku

logger = logging.getLogger("sentinel.predictive")


# ---------------------------------------------------------------------------
# Alert severity
# ---------------------------------------------------------------------------


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    WATCH = "watch"
    INFO = "info"

    @property
    def days_threshold(self) -> int:
        return {
            AlertSeverity.CRITICAL: 7,
            AlertSeverity.WARNING: 14,
            AlertSeverity.WATCH: 30,
            AlertSeverity.INFO: 60,
        }[self]


class PredictionType(str, Enum):
    STOCKOUT = "stockout"
    OVERSTOCK = "overstock"
    VELOCITY_DROP = "velocity_drop"
    DEMAND_SURGE = "demand_surge"


# ---------------------------------------------------------------------------
# Prediction data classes
# ---------------------------------------------------------------------------


@dataclass
class InventoryPrediction:
    """A single predictive alert for an SKU."""

    sku_id: str
    store_id: str
    prediction_type: PredictionType
    severity: AlertSeverity
    days_until_event: float
    estimated_lost_revenue: float
    estimated_carrying_cost: float
    current_qty: float
    daily_velocity: float
    recommendation: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "sku_id": self.sku_id,
            "store_id": self.store_id,
            "prediction_type": self.prediction_type.value,
            "severity": self.severity.value,
            "days_until_event": round(self.days_until_event, 1),
            "estimated_lost_revenue": round(self.estimated_lost_revenue, 2),
            "estimated_carrying_cost": round(self.estimated_carrying_cost, 2),
            "current_qty": self.current_qty,
            "daily_velocity": round(self.daily_velocity, 2),
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class PredictiveReport:
    """Complete predictive inventory report."""

    store_id: str
    total_predictions: int
    critical_alerts: int
    warning_alerts: int
    total_revenue_at_risk: float
    total_carrying_cost_at_risk: float
    stockout_predictions: list[InventoryPrediction]
    overstock_predictions: list[InventoryPrediction]
    velocity_alerts: list[InventoryPrediction]
    top_recommendation: str

    def to_dict(self) -> dict:
        return {
            "store_id": self.store_id,
            "total_predictions": self.total_predictions,
            "critical_alerts": self.critical_alerts,
            "warning_alerts": self.warning_alerts,
            "total_revenue_at_risk": round(self.total_revenue_at_risk, 2),
            "total_carrying_cost_at_risk": round(self.total_carrying_cost_at_risk, 2),
            "stockout_predictions": [p.to_dict() for p in self.stockout_predictions],
            "overstock_predictions": [p.to_dict() for p in self.overstock_predictions],
            "velocity_alerts": [p.to_dict() for p in self.velocity_alerts],
            "top_recommendation": self.top_recommendation,
        }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Annualized carrying cost rate (NHPA typical: 20-25%)
CARRYING_COST_RATE = 0.22

# Overstock: >90 days supply at current velocity
OVERSTOCK_DAYS_THRESHOLD = 90

# Minimum daily velocity to compute stockout
MIN_VELOCITY = 0.01  # ~1 unit per 100 days

# Confidence decay per day projected
CONFIDENCE_DECAY_PER_DAY = 0.005  # 0.5% less confident per day ahead


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------


class PredictiveAlertEngine:
    """Generate predictive inventory alerts from pipeline data.

    Usage:
        engine = PredictiveAlertEngine()
        report = engine.predict_from_digest(digest, store_id="store-7")
        for alert in report.stockout_predictions:
            print(f"{alert.sku_id}: stockout in {alert.days_until_event:.0f} days")
    """

    def __init__(
        self,
        stockout_horizon_days: int = 30,
        overstock_threshold_days: int = OVERSTOCK_DAYS_THRESHOLD,
        carrying_cost_rate: float = CARRYING_COST_RATE,
    ):
        self.stockout_horizon = stockout_horizon_days
        self.overstock_threshold = overstock_threshold_days
        self.carrying_cost_rate = carrying_cost_rate

    def predict_from_digest(
        self,
        digest: Digest,
        store_id: str | None = None,
    ) -> PredictiveReport:
        """Generate predictions from a pipeline digest.

        Analyzes all SKUs across issues to predict stockouts
        and overstock situations.

        Args:
            digest: Pipeline digest with issues and SKUs.
            store_id: Optional store filter.

        Returns:
            PredictiveReport with categorized predictions.
        """
        issues = digest.issues
        if store_id:
            issues = [i for i in issues if i.store_id == store_id]

        # Collect unique SKUs (deduplicate across issues)
        sku_map: dict[str, tuple[Sku, str]] = {}  # sku_id -> (sku, store_id)
        issue_context: dict[str, list[Issue]] = defaultdict(list)

        for issue in issues:
            for sku in issue.skus:
                if sku.sku_id not in sku_map:
                    sku_map[sku.sku_id] = (sku, issue.store_id)
                issue_context[sku.sku_id].append(issue)

        stockouts: list[InventoryPrediction] = []
        overstocks: list[InventoryPrediction] = []
        velocity_alerts: list[InventoryPrediction] = []

        for sku_id, (sku, sku_store_id) in sku_map.items():
            related = issue_context.get(sku_id, [])

            # Stockout prediction
            stockout = self._predict_stockout(sku, sku_store_id, related)
            if stockout:
                stockouts.append(stockout)

            # Overstock prediction
            overstock = self._predict_overstock(sku, sku_store_id, related)
            if overstock:
                overstocks.append(overstock)

            # Velocity change detection
            velocity = self._detect_velocity_change(sku, sku_store_id, related)
            if velocity:
                velocity_alerts.append(velocity)

        # Sort by severity then days
        stockouts.sort(key=lambda p: (p.severity.value, p.days_until_event))
        overstocks.sort(
            key=lambda p: p.estimated_carrying_cost, reverse=True
        )
        velocity_alerts.sort(key=lambda p: p.severity.value)

        all_predictions = stockouts + overstocks + velocity_alerts
        critical = sum(1 for p in all_predictions if p.severity == AlertSeverity.CRITICAL)
        warning = sum(1 for p in all_predictions if p.severity == AlertSeverity.WARNING)
        revenue_risk = sum(p.estimated_lost_revenue for p in stockouts)
        carrying_risk = sum(p.estimated_carrying_cost for p in overstocks)

        top_rec = "Inventory levels are healthy." if not all_predictions else ""
        if stockouts:
            worst = stockouts[0]
            top_rec = (
                f"Priority: {worst.sku_id} will stock out in "
                f"{worst.days_until_event:.0f} days. "
                f"${worst.estimated_lost_revenue:,.0f} revenue at risk."
            )
        elif overstocks:
            worst = overstocks[0]
            top_rec = (
                f"Priority: {worst.sku_id} is overstocked with "
                f"{worst.current_qty:.0f} units "
                f"(${worst.estimated_carrying_cost:,.0f}/yr carrying cost)."
            )

        return PredictiveReport(
            store_id=store_id or "all",
            total_predictions=len(all_predictions),
            critical_alerts=critical,
            warning_alerts=warning,
            total_revenue_at_risk=revenue_risk,
            total_carrying_cost_at_risk=carrying_risk,
            stockout_predictions=stockouts,
            overstock_predictions=overstocks,
            velocity_alerts=velocity_alerts,
            top_recommendation=top_rec,
        )

    # -----------------------------------------------------------------
    # Individual predictors
    # -----------------------------------------------------------------

    def _predict_stockout(
        self,
        sku: Sku,
        store_id: str,
        issues: list[Issue],
    ) -> InventoryPrediction | None:
        """Predict stockout based on velocity extrapolation."""
        if sku.qty_on_hand <= 0:
            return None  # Already stocked out — not a prediction

        daily_velocity = sku.sales_last_30d / 30.0
        if daily_velocity < MIN_VELOCITY:
            return None  # No meaningful velocity

        days_to_stockout = sku.qty_on_hand / daily_velocity

        if days_to_stockout > self.stockout_horizon:
            return None  # Outside prediction window

        # Severity
        if days_to_stockout <= 7:
            severity = AlertSeverity.CRITICAL
        elif days_to_stockout <= 14:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.WATCH

        # Estimated lost revenue = daily_revenue × days_without_stock
        # Assume 7 days of lost sales after stockout
        daily_revenue = daily_velocity * sku.retail_price
        lost_revenue = daily_revenue * 7

        # Confidence decays further out
        confidence = max(
            0.3, 1.0 - (days_to_stockout * CONFIDENCE_DECAY_PER_DAY)
        )

        # Boost confidence if there are related low-stock issues
        has_low_stock_issue = any(
            i.issue_type == IssueType.RECEIVING_GAP for i in issues
        )
        if has_low_stock_issue:
            confidence = min(1.0, confidence + 0.10)

        recommendation = (
            f"Reorder {sku.sku_id} immediately. Current stock "
            f"({sku.qty_on_hand:.0f} units) will last ~{days_to_stockout:.0f} days "
            f"at current velocity ({daily_velocity:.1f}/day). "
            f"Lost revenue risk: ${lost_revenue:,.0f}."
        )

        return InventoryPrediction(
            sku_id=sku.sku_id,
            store_id=store_id,
            prediction_type=PredictionType.STOCKOUT,
            severity=severity,
            days_until_event=days_to_stockout,
            estimated_lost_revenue=lost_revenue,
            estimated_carrying_cost=0,
            current_qty=sku.qty_on_hand,
            daily_velocity=daily_velocity,
            recommendation=recommendation,
            confidence=confidence,
        )

    def _predict_overstock(
        self,
        sku: Sku,
        store_id: str,
        issues: list[Issue],
    ) -> InventoryPrediction | None:
        """Predict overstock based on days of supply."""
        if sku.qty_on_hand <= 0:
            return None

        daily_velocity = sku.sales_last_30d / 30.0

        # For items with no velocity, any positive stock is potential overstock
        if daily_velocity < MIN_VELOCITY:
            # Dead stock — flag if significant value
            inventory_value = sku.qty_on_hand * sku.unit_cost
            if inventory_value < 100:
                return None
            carrying_cost = inventory_value * self.carrying_cost_rate
            return InventoryPrediction(
                sku_id=sku.sku_id,
                store_id=store_id,
                prediction_type=PredictionType.OVERSTOCK,
                severity=AlertSeverity.WARNING,
                days_until_event=365,  # Will be overstocked indefinitely
                estimated_lost_revenue=0,
                estimated_carrying_cost=carrying_cost,
                current_qty=sku.qty_on_hand,
                daily_velocity=0,
                recommendation=(
                    f"Zero velocity on {sku.sku_id} with "
                    f"${inventory_value:,.0f} on hand. "
                    f"Annual carrying cost: ${carrying_cost:,.0f}. "
                    f"Consider markdown or return to vendor."
                ),
                confidence=0.90,
            )

        days_of_supply = sku.qty_on_hand / daily_velocity

        if days_of_supply <= self.overstock_threshold:
            return None  # Not overstocked

        inventory_value = sku.qty_on_hand * sku.unit_cost
        carrying_cost = inventory_value * self.carrying_cost_rate

        # Severity based on how much over threshold
        if days_of_supply > 365:
            severity = AlertSeverity.CRITICAL
        elif days_of_supply > 180:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.WATCH

        # Optimal qty = days_threshold × daily_velocity
        optimal_qty = self.overstock_threshold * daily_velocity
        excess_qty = sku.qty_on_hand - optimal_qty
        excess_value = excess_qty * sku.unit_cost

        recommendation = (
            f"Overstocked: {sku.sku_id} has {days_of_supply:.0f} days of supply "
            f"(target: {self.overstock_threshold}). "
            f"Excess: {excess_qty:.0f} units (${excess_value:,.0f}). "
            f"Consider promotion or reduced reorder."
        )

        return InventoryPrediction(
            sku_id=sku.sku_id,
            store_id=store_id,
            prediction_type=PredictionType.OVERSTOCK,
            severity=severity,
            days_until_event=days_of_supply,
            estimated_lost_revenue=0,
            estimated_carrying_cost=carrying_cost,
            current_qty=sku.qty_on_hand,
            daily_velocity=daily_velocity,
            recommendation=recommendation,
            confidence=0.80,
        )

    def _detect_velocity_change(
        self,
        sku: Sku,
        store_id: str,
        issues: list[Issue],
    ) -> InventoryPrediction | None:
        """Detect significant velocity changes suggesting demand shifts.

        Uses issue signals to infer velocity changes:
        - Overstock issue + low sales → demand drop
        - Receiving gap + high sales → demand surge
        """
        daily_velocity = sku.sales_last_30d / 30.0

        # Check for demand surge signals
        receiving_gap = any(
            i.issue_type == IssueType.RECEIVING_GAP for i in issues
        )
        if receiving_gap and daily_velocity > 1.0 and sku.qty_on_hand < 10:
            return InventoryPrediction(
                sku_id=sku.sku_id,
                store_id=store_id,
                prediction_type=PredictionType.DEMAND_SURGE,
                severity=AlertSeverity.WARNING,
                days_until_event=sku.qty_on_hand / max(daily_velocity, MIN_VELOCITY),
                estimated_lost_revenue=daily_velocity * sku.retail_price * 14,
                estimated_carrying_cost=0,
                current_qty=sku.qty_on_hand,
                daily_velocity=daily_velocity,
                recommendation=(
                    f"Demand surge detected on {sku.sku_id}. "
                    f"Selling {daily_velocity:.1f}/day with only "
                    f"{sku.qty_on_hand:.0f} units remaining. "
                    f"Expedite reorder."
                ),
                confidence=0.70,
            )

        # Check for demand drop signals
        overstock_issue = any(
            i.issue_type == IssueType.OVERSTOCK for i in issues
        )
        if overstock_issue and daily_velocity < 0.5 and sku.qty_on_hand > 50:
            inventory_value = sku.qty_on_hand * sku.unit_cost
            return InventoryPrediction(
                sku_id=sku.sku_id,
                store_id=store_id,
                prediction_type=PredictionType.VELOCITY_DROP,
                severity=AlertSeverity.WATCH,
                days_until_event=90,  # Projecting 90 days of continued slow movement
                estimated_lost_revenue=0,
                estimated_carrying_cost=inventory_value * self.carrying_cost_rate,
                current_qty=sku.qty_on_hand,
                daily_velocity=daily_velocity,
                recommendation=(
                    f"Velocity drop on {sku.sku_id}. Only {daily_velocity:.1f}/day "
                    f"with {sku.qty_on_hand:.0f} on hand "
                    f"(${inventory_value:,.0f}). Consider markdown."
                ),
                confidence=0.65,
            )

        return None


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def predict_inventory(
    digest: Digest,
    store_id: str | None = None,
    horizon_days: int = 30,
) -> PredictiveReport:
    """Generate predictive inventory alerts. Convenience wrapper.

    Args:
        digest: Pipeline digest.
        store_id: Optional store filter.
        horizon_days: Stockout prediction horizon in days.

    Returns:
        PredictiveReport with categorized predictions.
    """
    engine = PredictiveAlertEngine(stockout_horizon_days=horizon_days)
    return engine.predict_from_digest(digest, store_id=store_id)
