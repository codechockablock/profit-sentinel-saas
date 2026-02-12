"""Engine 3 — Counterfactual World Model.

Computes alternate timelines where the owner acted on findings.
Expresses the cost of inaction as concrete dollar amounts over time.

    "If you had adjusted retail when cost went up, you'd have $4,200
     more in the bank. That gap grows $93 every month you wait."

Architecture:
    Engine 1 (Rust) produces findings with SKU-level data.
    Engine 3 takes each finding, defines the intervention that would
    have resolved it, projects the alternate timeline forward, and
    computes the gap between what happened and what could have happened.

Design principles:
    - Pure functions: Finding → Counterfactual (no side effects)
    - Show your work: every output includes the formula and inputs
    - Honest uncertainty: outputs include confidence and range estimates
    - Domain-agnostic interface: the core computes timelines from
      generic parameters; retail-specific logic lives in strategies

Three counterfactual strategies for v1:
    1. MarginRestoration  — "adjust retail to restore benchmark margin"
    2. MarkdownRecovery   — "mark down dead stock to recover capital"
    3. ReorderCorrection  — "update reorder point to match actual velocity"

Usage:
    from sentinel_agent.counterfactual import CounterfactualEngine

    engine = CounterfactualEngine()
    results = engine.analyze(analysis_result)
    # Returns: list[Counterfactual] with cost_of_inaction, formula, etc.

    # Or enrich an existing analysis result dict in-place:
    engine.enrich(analysis_result)
    # Adds "counterfactual" key to each item in each leak category

Author: Joseph + Claude
Date: 2026-02-12
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("sentinel.engine3")

# ---------------------------------------------------------------------------
# Benchmarks — mirrored from Rust thresholds.rs
# ---------------------------------------------------------------------------

# Do It Best benchmark margin for hardware retail
DIB_BENCHMARK_MARGIN = 0.35

# Monthly carrying cost as % of inventory value (NHPA: 25-30% annual)
CARRYING_COST_MONTHLY = 0.02

# Assumed markdown discount to move dead stock (industry standard 20-40%)
CLEARANCE_DISCOUNT_RATE = 0.30

# Velocity multiplier when clearance-priced (conservative: 3-5x normal)
CLEARANCE_VELOCITY_MULTIPLIER = 3.0

# Months to look back for cost-of-inaction when detection date unknown
DEFAULT_MONTHS_ELAPSED = 1.0


# ---------------------------------------------------------------------------
# Core data structures (domain-agnostic)
# ---------------------------------------------------------------------------


class InterventionType(str, Enum):
    """What action should have been taken."""

    REPRICE = "reprice"
    MARKDOWN = "markdown"
    REORDER_ADJUST = "reorder_adjust"
    VENDOR_NEGOTIATE = "vendor_negotiate"
    DISCONTINUE = "discontinue"
    INVESTIGATE = "investigate"


@dataclass
class TimelinePoint:
    """A single point in time comparing actual vs counterfactual state."""

    month: int  # months from detection
    actual_cumulative: float  # cumulative actual outcome ($)
    counterfactual_cumulative: float  # cumulative if-acted outcome ($)
    gap: float  # counterfactual - actual (positive = money left on table)


@dataclass
class Counterfactual:
    """The output of Engine 3 for a single finding.

    Every field is designed to be directly renderable in a dashboard
    or PDF report with full transparency into the math.
    """

    # What was found
    sku_id: str
    issue_type: str
    finding_description: str

    # What should have been done
    intervention_type: InterventionType
    intervention_description: str

    # Time dimension
    assumed_detection_date: str  # ISO-8601 or "current_upload"
    months_elapsed: float
    still_accumulating: bool  # True if the finding is unresolved

    # The money
    actual_outcome: float  # What actually happened ($)
    counterfactual_outcome: float  # What would have happened ($)
    cost_of_inaction: float  # The gap (positive = lost money)
    cost_per_month: float  # Monthly bleed rate
    cost_per_day: float  # Daily bleed rate

    # Confidence
    confidence: float  # 0.0-1.0
    range_low: float  # Conservative estimate
    range_high: float  # Optimistic estimate

    # Show your work
    formula: str  # Human-readable formula
    inputs: dict[str, Any]  # Named inputs to the formula
    assumptions: list[str]  # What we assumed and why

    # Timeline projection
    timeline: list[TimelinePoint]  # Month-by-month if available

    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "sku_id": self.sku_id,
            "issue_type": self.issue_type,
            "finding_description": self.finding_description,
            "intervention": {
                "type": self.intervention_type.value,
                "description": self.intervention_description,
            },
            "time": {
                "assumed_detection_date": self.assumed_detection_date,
                "months_elapsed": round(self.months_elapsed, 1),
                "still_accumulating": self.still_accumulating,
            },
            "cost_of_inaction": {
                "total": round(self.cost_of_inaction, 2),
                "per_month": round(self.cost_per_month, 2),
                "per_day": round(self.cost_per_day, 2),
                "range_low": round(self.range_low, 2),
                "range_high": round(self.range_high, 2),
                "confidence": round(self.confidence, 2),
            },
            "actual_outcome": round(self.actual_outcome, 2),
            "counterfactual_outcome": round(self.counterfactual_outcome, 2),
            "formula": self.formula,
            "inputs": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.inputs.items()
            },
            "assumptions": self.assumptions,
            "timeline": [
                {
                    "month": tp.month,
                    "actual": round(tp.actual_cumulative, 2),
                    "counterfactual": round(tp.counterfactual_cumulative, 2),
                    "gap": round(tp.gap, 2),
                }
                for tp in self.timeline
            ],
        }


# ---------------------------------------------------------------------------
# Counterfactual Strategies (pluggable per issue type)
# ---------------------------------------------------------------------------


class CounterfactualStrategy(ABC):
    """Base class for counterfactual computation strategies.

    Each strategy knows how to compute the alternate timeline for
    a specific type of finding. The interface is domain-agnostic:
    it receives a dict of item data and returns a Counterfactual.

    To port Engine 3 to Parallax, implement new strategies for
    personal domains (SubscriptionLeak, PriceComparisonGap, etc.)
    without changing the engine core.
    """

    @abstractmethod
    def applies_to(self, issue_type: str) -> bool:
        """Whether this strategy handles the given issue type."""

    @abstractmethod
    def compute(
        self,
        item: dict,
        issue_type: str,
        months_elapsed: float,
        detection_date: str,
    ) -> Counterfactual | None:
        """Compute the counterfactual for one item.

        Args:
            item: SKU-level data dict from Engine 1 result adapter
            issue_type: The leak category (e.g., "margin_erosion")
            months_elapsed: Time since detection (or default)
            detection_date: ISO date string or "current_upload"

        Returns:
            Counterfactual if computable, None if insufficient data.
        """


class MarginRestoration(CounterfactualStrategy):
    """Counterfactual: "If you had adjusted retail to restore benchmark margin."

    Applies to: margin_erosion, price_discrepancy

    Math:
        current_profit_per_unit = retail - cost
        target_retail = cost / (1 - benchmark_margin)
        target_profit_per_unit = target_retail - cost
        profit_gap_per_unit = target_profit_per_unit - current_profit_per_unit
        monthly_cost = profit_gap_per_unit × monthly_velocity
        total_cost = monthly_cost × months_elapsed
    """

    def __init__(self, benchmark_margin: float = DIB_BENCHMARK_MARGIN):
        self.benchmark_margin = benchmark_margin

    def applies_to(self, issue_type: str) -> bool:
        return issue_type in (
            "margin_erosion",
            "MarginErosion",
            "price_discrepancy",
            "PriceDiscrepancy",
        )

    def compute(
        self,
        item: dict,
        issue_type: str,
        months_elapsed: float,
        detection_date: str,
    ) -> Counterfactual | None:
        cost = _safe_float(item.get("cost", item.get("unit_cost", 0)))
        retail = _safe_float(item.get("revenue", item.get("retail_price", 0)))
        sold = _safe_float(item.get("sold", item.get("sales_last_30d", 0)))
        margin = _safe_float(item.get("margin", item.get("margin_pct", 0)))
        sku_id = item.get("sku", item.get("sku_id", "unknown"))

        # Normalize margin to decimal if passed as percentage
        if margin > 1.0:
            margin = margin / 100.0

        if cost <= 0 or retail <= 0 or sold <= 0:
            return None

        # Current state
        current_profit_per_unit = retail - cost
        monthly_velocity = sold  # sold field is last-30-days

        # Counterfactual: what if retail adjusted to benchmark?
        target_retail = cost / (1.0 - self.benchmark_margin)
        target_profit_per_unit = target_retail - cost
        profit_gap_per_unit = target_profit_per_unit - current_profit_per_unit

        if profit_gap_per_unit <= 0:
            return None  # Already at or above benchmark

        # Project forward
        monthly_cost = profit_gap_per_unit * monthly_velocity
        total_cost = monthly_cost * months_elapsed

        # Confidence based on velocity stability
        # Higher velocity = more confident in projection
        confidence = min(0.95, 0.5 + (monthly_velocity / 20.0) * 0.3)

        # Range: ±20% based on velocity variance
        range_low = total_cost * 0.8
        range_high = total_cost * 1.2

        # Build timeline
        timeline = []
        actual_cum = 0.0
        cf_cum = 0.0
        for m in range(1, int(months_elapsed) + 2):
            actual_cum += current_profit_per_unit * monthly_velocity
            cf_cum += target_profit_per_unit * monthly_velocity
            timeline.append(
                TimelinePoint(
                    month=m,
                    actual_cumulative=actual_cum,
                    counterfactual_cumulative=cf_cum,
                    gap=cf_cum - actual_cum,
                )
            )

        return Counterfactual(
            sku_id=sku_id,
            issue_type=issue_type,
            finding_description=(
                f"Margin at {margin * 100:.0f}% vs {self.benchmark_margin * 100:.0f}% "
                f"benchmark ({profit_gap_per_unit:.2f}/unit gap)"
            ),
            intervention_type=InterventionType.REPRICE,
            intervention_description=(
                f"Adjust retail from ${retail:.2f} to ${target_retail:.2f} "
                f"to restore {self.benchmark_margin * 100:.0f}% margin"
            ),
            assumed_detection_date=detection_date,
            months_elapsed=months_elapsed,
            still_accumulating=True,
            actual_outcome=current_profit_per_unit * monthly_velocity * months_elapsed,
            counterfactual_outcome=target_profit_per_unit
            * monthly_velocity
            * months_elapsed,
            cost_of_inaction=total_cost,
            cost_per_month=monthly_cost,
            cost_per_day=monthly_cost / 30.0,
            confidence=confidence,
            range_low=range_low,
            range_high=range_high,
            formula=(
                f"(${target_retail:.2f} - ${retail:.2f}) "
                f"× {monthly_velocity:.0f} units/mo "
                f"× {months_elapsed:.1f} months"
            ),
            inputs={
                "current_retail": retail,
                "target_retail": target_retail,
                "unit_cost": cost,
                "current_margin": margin,
                "benchmark_margin": self.benchmark_margin,
                "monthly_velocity": monthly_velocity,
                "months_elapsed": months_elapsed,
                "profit_gap_per_unit": profit_gap_per_unit,
            },
            assumptions=[
                f"Benchmark margin: {self.benchmark_margin * 100:.0f}% (Do It Best hardware avg)",
                f"Velocity stable at {monthly_velocity:.0f} units/month (last 30 days)",
                "Price adjustment would not materially affect demand",
                "Cost remains stable over projection period",
            ],
            timeline=timeline,
        )


class MarkdownRecovery(CounterfactualStrategy):
    """Counterfactual: "If you had marked this down to recover capital."

    Applies to: dead_stock, overstock

    Math:
        current_state = qty × cost (capital tied up, earning nothing)
        carrying_cost = qty × cost × carrying_rate × months
        clearance_retail = retail × (1 - discount_rate)
        clearance_velocity = base_velocity × velocity_multiplier (or min 1/mo)
        months_to_clear = qty / clearance_velocity
        capital_recovered = min(qty, clearance_velocity × months) × clearance_retail
        cost_of_inaction = carrying_cost + opportunity_cost_of_tied_capital
    """

    def __init__(
        self,
        carrying_cost_monthly: float = CARRYING_COST_MONTHLY,
        clearance_discount: float = CLEARANCE_DISCOUNT_RATE,
        clearance_velocity_mult: float = CLEARANCE_VELOCITY_MULTIPLIER,
    ):
        self.carrying_cost_monthly = carrying_cost_monthly
        self.clearance_discount = clearance_discount
        self.clearance_velocity_mult = clearance_velocity_mult

    def applies_to(self, issue_type: str) -> bool:
        return issue_type in (
            "dead_stock",
            "DeadStock",
            "overstock",
            "Overstock",
            "patronage_miss",
            "PatronageMiss",
        )

    def compute(
        self,
        item: dict,
        issue_type: str,
        months_elapsed: float,
        detection_date: str,
    ) -> Counterfactual | None:
        qty = _safe_float(item.get("quantity", item.get("qty_on_hand", 0)))
        cost = _safe_float(item.get("cost", item.get("unit_cost", 0)))
        retail = _safe_float(item.get("revenue", item.get("retail_price", 0)))
        sold = _safe_float(item.get("sold", item.get("sales_last_30d", 0)))
        sku_id = item.get("sku", item.get("sku_id", "unknown"))

        if qty <= 0 or cost <= 0:
            return None

        # If retail is 0 or below cost, use cost as base
        if retail <= 0:
            retail = cost * 1.3  # Assume minimal markup

        inventory_value = qty * cost

        # Actual: capital sits there, accumulating carrying cost
        carrying_cost = inventory_value * self.carrying_cost_monthly * months_elapsed

        # Counterfactual: markdown and clear
        clearance_price = retail * (1.0 - self.clearance_discount)
        clearance_velocity = max(1.0, sold * self.clearance_velocity_mult)
        if sold == 0:
            clearance_velocity = max(1.0, qty / 6.0)  # Assume 6 months to clear

        months_to_clear = qty / clearance_velocity
        units_cleared = min(qty, clearance_velocity * months_elapsed)
        capital_recovered = units_cleared * clearance_price
        remaining_carrying = (
            max(0, qty - units_cleared) * cost * self.carrying_cost_monthly
        )

        # Cost of inaction = carrying cost incurred + capital still tied up
        # vs counterfactual where you recovered capital and stopped the bleed
        cost_of_inaction = carrying_cost
        if months_elapsed >= months_to_clear:
            # Would have fully cleared by now
            cost_of_inaction += (inventory_value - capital_recovered) * 0  # sunk cost
            # But the big number is: they'd have capital_recovered back to deploy
            cost_of_inaction = carrying_cost  # Pure carrying cost saved
        else:
            # Partial clearing
            cost_of_inaction = carrying_cost

        # The real cost: carrying cost + opportunity cost of frozen capital
        # Using carrying rate as proxy for opportunity cost (conservative)
        total_cost = carrying_cost  # Already includes the carrying

        monthly_cost = inventory_value * self.carrying_cost_monthly
        if monthly_cost <= 0.01:
            return None  # Trivial

        # Confidence: lower for items with zero velocity (uncertain clearance rate)
        confidence = 0.75 if sold > 0 else 0.55
        range_low = total_cost * 0.7
        range_high = total_cost * 1.4

        # Timeline
        timeline = []
        for m in range(1, int(months_elapsed) + 2):
            # Actual: just carrying cost accumulating
            actual = inventory_value * self.carrying_cost_monthly * m
            # Counterfactual: clearing inventory, carrying cost on remainder
            cleared = min(qty, clearance_velocity * m)
            remaining = max(0, qty - cleared)
            cf_carrying = remaining * cost * self.carrying_cost_monthly * m
            # Net counterfactual cost (carrying on remaining - revenue from cleared)
            cf_cost = cf_carrying  # Less carrying because less inventory
            timeline.append(
                TimelinePoint(
                    month=m,
                    actual_cumulative=actual,
                    counterfactual_cumulative=cf_cost,
                    gap=actual - cf_cost,
                )
            )

        return Counterfactual(
            sku_id=sku_id,
            issue_type=issue_type,
            finding_description=(
                f"{int(qty)} units on hand, selling {sold:.0f}/month. "
                f"${inventory_value:,.0f} in capital tied up."
            ),
            intervention_type=InterventionType.MARKDOWN,
            intervention_description=(
                f"Mark down {self.clearance_discount * 100:.0f}% "
                f"(${retail:.2f} → ${clearance_price:.2f}) to accelerate sell-through. "
                f"Estimated {months_to_clear:.0f} months to clear at "
                f"{clearance_velocity:.0f} units/month."
            ),
            assumed_detection_date=detection_date,
            months_elapsed=months_elapsed,
            still_accumulating=qty > 0,
            actual_outcome=-carrying_cost,  # Negative = pure loss
            counterfactual_outcome=capital_recovered
            - (remaining_carrying * months_elapsed),
            cost_of_inaction=total_cost,
            cost_per_month=monthly_cost,
            cost_per_day=monthly_cost / 30.0,
            confidence=confidence,
            range_low=range_low,
            range_high=range_high,
            formula=(
                f"{int(qty)} units × ${cost:.2f} cost "
                f"× {self.carrying_cost_monthly * 100:.0f}% monthly carrying "
                f"× {months_elapsed:.1f} months"
            ),
            inputs={
                "quantity": qty,
                "unit_cost": cost,
                "current_retail": retail,
                "clearance_price": clearance_price,
                "monthly_velocity_current": sold,
                "monthly_velocity_clearance": clearance_velocity,
                "months_to_clear": months_to_clear,
                "carrying_rate_monthly": self.carrying_cost_monthly,
                "inventory_value": inventory_value,
                "months_elapsed": months_elapsed,
            },
            assumptions=[
                f"Carrying cost: {self.carrying_cost_monthly * 100:.0f}%/month "
                f"({self.carrying_cost_monthly * 12 * 100:.0f}%/year, NHPA benchmark)",
                f"Clearance discount: {self.clearance_discount * 100:.0f}% off retail",
                f"Clearance velocity: {clearance_velocity:.0f} units/month "
                f"({'estimated' if sold == 0 else f'{self.clearance_velocity_mult:.0f}x current'})",
                "Carrying cost includes storage, insurance, obsolescence risk",
            ],
            timeline=timeline,
        )


class ReorderCorrection(CounterfactualStrategy):
    """Counterfactual: "If you had updated the reorder point when velocity dropped."

    Applies to: overstock (specifically items with active on-order qty)

    Math:
        optimal_stock = monthly_velocity × lead_time_months × safety_factor
        excess_qty = qty_on_hand + on_order - optimal_stock
        excess_cost = excess_qty × unit_cost
        carrying_waste = excess_cost × carrying_rate × months
        cost_of_inaction = carrying_waste (money spent warehousing excess)
    """

    LEAD_TIME_MONTHS = 1.0  # Typical Do It Best warehouse lead time
    SAFETY_FACTOR = 1.5  # 50% safety stock buffer

    def __init__(self, carrying_cost_monthly: float = CARRYING_COST_MONTHLY):
        self.carrying_cost_monthly = carrying_cost_monthly

    def applies_to(self, issue_type: str) -> bool:
        return issue_type in (
            "overstock",
            "Overstock",
        )

    def compute(
        self,
        item: dict,
        issue_type: str,
        months_elapsed: float,
        detection_date: str,
    ) -> Counterfactual | None:
        qty = _safe_float(item.get("quantity", item.get("qty_on_hand", 0)))
        cost = _safe_float(item.get("cost", item.get("unit_cost", 0)))
        sold = _safe_float(item.get("sold", item.get("sales_last_30d", 0)))
        on_order = _safe_float(item.get("on_order", item.get("on_order_qty", 0)))
        sku_id = item.get("sku", item.get("sku_id", "unknown"))

        if qty <= 0 or cost <= 0:
            return None

        monthly_velocity = max(sold, 0.5)  # Floor at 0.5 for very slow movers
        optimal_stock = monthly_velocity * self.LEAD_TIME_MONTHS * self.SAFETY_FACTOR
        total_position = qty + on_order
        excess_qty = max(0, total_position - optimal_stock)

        if excess_qty <= 0:
            return None  # Not actually overstocked

        excess_value = excess_qty * cost
        carrying_waste = excess_value * self.carrying_cost_monthly * months_elapsed
        monthly_waste = excess_value * self.carrying_cost_monthly

        if monthly_waste <= 0.01:
            return None

        # Confidence: higher if we have real velocity data
        confidence = 0.70 if sold > 0 else 0.50
        range_low = carrying_waste * 0.7
        range_high = carrying_waste * 1.3

        # Timeline
        timeline = []
        for m in range(1, int(months_elapsed) + 2):
            actual = excess_value * self.carrying_cost_monthly * m
            # Counterfactual: no excess, no carrying cost on it
            timeline.append(
                TimelinePoint(
                    month=m,
                    actual_cumulative=actual,
                    counterfactual_cumulative=0.0,
                    gap=actual,
                )
            )

        on_order_note = (
            f" (plus {int(on_order)} more on order!)" if on_order > 0 else ""
        )

        return Counterfactual(
            sku_id=sku_id,
            issue_type=issue_type,
            finding_description=(
                f"{int(qty)} on hand{on_order_note}, "
                f"selling {sold:.0f}/month. "
                f"Optimal stock: {optimal_stock:.0f} units. "
                f"Excess: {int(excess_qty)} units (${excess_value:,.0f})."
            ),
            intervention_type=InterventionType.REORDER_ADJUST,
            intervention_description=(
                f"Set reorder point to {optimal_stock:.0f} units "
                f"({monthly_velocity:.0f}/mo × {self.LEAD_TIME_MONTHS:.0f}mo lead × "
                f"{self.SAFETY_FACTOR:.1f} safety). "
                f"{'Cancel pending PO for ' + str(int(on_order)) + ' units. ' if on_order > 0 else ''}"
                f"Stop carrying {int(excess_qty)} excess units."
            ),
            assumed_detection_date=detection_date,
            months_elapsed=months_elapsed,
            still_accumulating=True,
            actual_outcome=-carrying_waste,
            counterfactual_outcome=0.0,
            cost_of_inaction=carrying_waste,
            cost_per_month=monthly_waste,
            cost_per_day=monthly_waste / 30.0,
            confidence=confidence,
            range_low=range_low,
            range_high=range_high,
            formula=(
                f"{int(excess_qty)} excess units × ${cost:.2f} cost "
                f"× {self.carrying_cost_monthly * 100:.0f}% monthly carrying "
                f"× {months_elapsed:.1f} months"
            ),
            inputs={
                "quantity_on_hand": qty,
                "on_order_qty": on_order,
                "total_position": total_position,
                "optimal_stock": optimal_stock,
                "excess_qty": excess_qty,
                "unit_cost": cost,
                "monthly_velocity": monthly_velocity,
                "lead_time_months": self.LEAD_TIME_MONTHS,
                "safety_factor": self.SAFETY_FACTOR,
                "carrying_rate_monthly": self.carrying_cost_monthly,
                "excess_value": excess_value,
                "months_elapsed": months_elapsed,
            },
            assumptions=[
                f"Optimal stock = velocity × {self.LEAD_TIME_MONTHS:.0f}mo lead × "
                f"{self.SAFETY_FACTOR:.1f} safety factor",
                f"Carrying cost: {self.carrying_cost_monthly * 100:.0f}%/month (NHPA benchmark)",
                f"Monthly velocity: {monthly_velocity:.0f} units "
                f"({'actual' if sold > 0 else 'estimated floor of 0.5'})",
                f"Lead time: {self.LEAD_TIME_MONTHS:.0f} month (Do It Best warehouse avg)",
            ],
            timeline=timeline,
        )


# ---------------------------------------------------------------------------
# Engine 3 — the orchestrator
# ---------------------------------------------------------------------------


# Map leak category keys (from result_adapter) to issue type strings
_LEAK_KEY_MAP: dict[str, str] = {
    "margin_erosion": "MarginErosion",
    "dead_stock": "DeadStock",
    "overstock": "Overstock",
    "price_discrepancy": "PriceDiscrepancy",
    "patronage_miss": "PatronageMiss",
    "negative_inventory": "NegativeInventory",
    "receiving_gap": "ReceivingGap",
    "vendor_short_ship": "VendorShortShip",
    "shrinkage_pattern": "ShrinkagePattern",
    "zero_cost_anomaly": "ZeroCostAnomaly",
    "purchasing_leakage": "PurchasingLeakage",
}


class CounterfactualEngine:
    """Engine 3: Counterfactual World Model.

    Takes Engine 1 analysis results and computes alternate timelines
    for every actionable finding.

    Usage:
        engine = CounterfactualEngine()

        # Analyze a full result set
        counterfactuals = engine.analyze(analysis_result)

        # Or enrich the result dict in-place (adds "counterfactual" to items)
        engine.enrich(analysis_result)

        # Or compute for a single item
        cf = engine.compute_one(item_dict, "margin_erosion", months_elapsed=2.0)

    The engine is stateless — all context comes from the finding data.
    Temporal tracking (comparing this upload to previous uploads) is
    handled by the caller, which provides months_elapsed.
    """

    def __init__(
        self,
        strategies: list[CounterfactualStrategy] | None = None,
        default_months_elapsed: float = DEFAULT_MONTHS_ELAPSED,
    ):
        self.strategies = strategies or [
            MarginRestoration(),
            MarkdownRecovery(),
            ReorderCorrection(),
        ]
        self.default_months_elapsed = default_months_elapsed

    def analyze(
        self,
        analysis_result: dict,
        months_elapsed: float | None = None,
        detection_date: str = "current_upload",
    ) -> list[Counterfactual]:
        """Compute counterfactuals for all findings in an analysis result.

        Args:
            analysis_result: Full result dict from Engine 1 (result_adapter output).
                Expected shape: {"leaks": {"margin_erosion": {"items": [...]}}}
            months_elapsed: Override time since detection. If None, uses default.
            detection_date: When the finding was first detected.

        Returns:
            List of Counterfactual objects, sorted by cost_of_inaction descending.
        """
        start = time.monotonic()
        elapsed = months_elapsed or self.default_months_elapsed
        results: list[Counterfactual] = []

        leaks = analysis_result.get("leaks", {})

        for leak_key, leak_data in leaks.items():
            if not isinstance(leak_data, dict):
                continue

            items = leak_data.get("items", [])
            issue_type = _LEAK_KEY_MAP.get(leak_key, leak_key)

            for item in items:
                cf = self._compute_best(item, issue_type, elapsed, detection_date)
                if cf is not None:
                    results.append(cf)

        # Sort by cost of inaction (biggest losses first)
        results.sort(key=lambda c: c.cost_of_inaction, reverse=True)

        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Engine 3: computed %d counterfactuals from %d leak categories "
            "in %.1fms (months_elapsed=%.1f)",
            len(results),
            len(leaks),
            duration_ms,
            elapsed,
        )

        return results

    def enrich(
        self,
        analysis_result: dict,
        months_elapsed: float | None = None,
        detection_date: str = "current_upload",
    ) -> dict:
        """Enrich an analysis result dict with counterfactual data in-place.

        Adds a "counterfactual" key to each item that has a computable
        counterfactual. Also adds a top-level "engine3_summary" key.

        Returns the modified analysis_result for chaining.
        """
        elapsed = months_elapsed or self.default_months_elapsed
        total_cost = 0.0
        total_monthly = 0.0
        count = 0

        leaks = analysis_result.get("leaks", {})

        for leak_key, leak_data in leaks.items():
            if not isinstance(leak_data, dict):
                continue

            items = leak_data.get("items", [])
            issue_type = _LEAK_KEY_MAP.get(leak_key, leak_key)

            for item in items:
                cf = self._compute_best(item, issue_type, elapsed, detection_date)
                if cf is not None:
                    item["counterfactual"] = cf.to_dict()
                    total_cost += cf.cost_of_inaction
                    total_monthly += cf.cost_per_month
                    count += 1

        # Top-level summary
        analysis_result["engine3_summary"] = {
            "total_cost_of_inaction": round(total_cost, 2),
            "monthly_bleed_rate": round(total_monthly, 2),
            "daily_bleed_rate": round(total_monthly / 30.0, 2),
            "findings_with_counterfactuals": count,
            "months_elapsed": elapsed,
            "message": (
                f"Unresolved findings are costing an estimated "
                f"${total_monthly:,.0f}/month (${total_monthly / 30:,.0f}/day). "
                f"Total cost of inaction over {elapsed:.1f} months: "
                f"${total_cost:,.0f}."
            ),
        }

        return analysis_result

    def compute_one(
        self,
        item: dict,
        issue_type: str,
        months_elapsed: float | None = None,
        detection_date: str = "current_upload",
    ) -> Counterfactual | None:
        """Compute counterfactual for a single item.

        Convenience method for testing and per-item API endpoints.
        """
        elapsed = months_elapsed or self.default_months_elapsed
        return self._compute_best(item, issue_type, elapsed, detection_date)

    def _compute_best(
        self,
        item: dict,
        issue_type: str,
        months_elapsed: float,
        detection_date: str,
    ) -> Counterfactual | None:
        """Find the best applicable strategy and compute.

        For overstock items with active on-order quantities,
        ReorderCorrection takes priority over MarkdownRecovery
        because the most impactful intervention is stopping the
        bleeding (cancel PO) rather than clearing existing stock.
        """
        on_order = _safe_float(item.get("on_order", item.get("on_order_qty", 0)))

        # Priority override: overstock with active POs → reorder correction first
        if on_order > 0 and issue_type in ("overstock", "Overstock"):
            for strategy in self.strategies:
                if isinstance(strategy, ReorderCorrection):
                    try:
                        result = strategy.compute(
                            item, issue_type, months_elapsed, detection_date
                        )
                        if result is not None:
                            return result
                    except Exception as e:
                        logger.warning(
                            "Engine 3 ReorderCorrection failed for %s: %s",
                            item.get("sku", "unknown"),
                            e,
                        )

        # Default: first matching strategy wins
        for strategy in self.strategies:
            if strategy.applies_to(issue_type):
                try:
                    return strategy.compute(
                        item, issue_type, months_elapsed, detection_date
                    )
                except Exception as e:
                    logger.warning(
                        "Engine 3 strategy %s failed for %s: %s",
                        type(strategy).__name__,
                        item.get("sku", "unknown"),
                        e,
                    )
        return None

    def get_aggregate_summary(self, counterfactuals: list[Counterfactual]) -> dict:
        """Compute aggregate summary for a set of counterfactuals.

        Useful for the Morning Digest one-liner:
            "Total cost of unresolved findings: $X/month and growing"
        """
        if not counterfactuals:
            return {
                "total_cost_of_inaction": 0.0,
                "monthly_bleed_rate": 0.0,
                "daily_bleed_rate": 0.0,
                "count": 0,
                "by_intervention_type": {},
                "top_items": [],
                "message": "No actionable counterfactuals computed.",
            }

        total_cost = sum(c.cost_of_inaction for c in counterfactuals)
        monthly = sum(c.cost_per_month for c in counterfactuals)

        # Group by intervention type
        by_type: dict[str, dict] = {}
        for cf in counterfactuals:
            key = cf.intervention_type.value
            if key not in by_type:
                by_type[key] = {"count": 0, "total_cost": 0.0, "monthly": 0.0}
            by_type[key]["count"] += 1
            by_type[key]["total_cost"] += cf.cost_of_inaction
            by_type[key]["monthly"] += cf.cost_per_month

        # Round the grouped values
        for v in by_type.values():
            v["total_cost"] = round(v["total_cost"], 2)
            v["monthly"] = round(v["monthly"], 2)

        # Top 5 by cost
        top = sorted(counterfactuals, key=lambda c: c.cost_of_inaction, reverse=True)[
            :5
        ]

        return {
            "total_cost_of_inaction": round(total_cost, 2),
            "monthly_bleed_rate": round(monthly, 2),
            "daily_bleed_rate": round(monthly / 30.0, 2),
            "count": len(counterfactuals),
            "by_intervention_type": by_type,
            "top_items": [
                {
                    "sku_id": c.sku_id,
                    "cost_of_inaction": round(c.cost_of_inaction, 2),
                    "cost_per_month": round(c.cost_per_month, 2),
                    "intervention": c.intervention_description,
                    "formula": c.formula,
                }
                for c in top
            ],
            "message": (
                f"Unresolved findings are costing ~${monthly:,.0f}/month "
                f"(${monthly / 30:,.0f}/day). "
                f"{len(counterfactuals)} items have actionable counterfactuals."
            ),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> float:
    """Safely convert to float, returning 0.0 on failure."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
