"""
Dead Stock Configuration & Inventory Lifecycle
=================================================

User-configurable dead stock thresholds with tiered severity,
per-category overrides, and integration points for every
component in the pipeline.

The customer knows their inventory better than we do.
A hardware store owner knows commercial deadbolts sit for months.
A garden center knows everything should move within a season.
This module lets them tell the system what "dead" means for
their business.

Four tiers, each with different severity and recommended actions:

- WATCHLIST (default 60 days): "Worth keeping an eye on"
- ATTENTION (default 120 days): "Consider taking action"
- ACTION_REQUIRED (default 180 days): "This is costing you money"
- WRITEOFF (default 360 days): "Likely unsellable at any price"

All thresholds are user-configurable per store, per category,
with sensible defaults that work for most independent hardware
stores.

Author: Joseph + Claude
Date: 2026-02-10
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# =============================================================================
# TIERS
# =============================================================================


class DeadStockTier(Enum):
    """Severity tiers for dead stock classification."""

    ACTIVE = "active"  # Selling normally
    WATCHLIST = "watchlist"  # Flagged, monitoring
    ATTENTION = "attention"  # Needs review
    ACTION_REQUIRED = "action_required"  # Actively losing money
    WRITEOFF = "writeoff"  # Consider disposal

    @property
    def severity(self) -> int:
        """Numeric severity for sorting. Higher = worse."""
        return {
            DeadStockTier.ACTIVE: 0,
            DeadStockTier.WATCHLIST: 1,
            DeadStockTier.ATTENTION: 2,
            DeadStockTier.ACTION_REQUIRED: 3,
            DeadStockTier.WRITEOFF: 4,
        }[self]

    @property
    def dashboard_color(self) -> str:
        """Color for dashboard display."""
        return {
            DeadStockTier.ACTIVE: "green",
            DeadStockTier.WATCHLIST: "yellow",
            DeadStockTier.ATTENTION: "orange",
            DeadStockTier.ACTION_REQUIRED: "red",
            DeadStockTier.WRITEOFF: "darkred",
        }[self]

    @property
    def confidence_language(self) -> str:
        """
        What language the LLM should use for this tier.

        The response validator checks that findings at each tier
        use the appropriate severity of language. Watchlist items
        get hedging language. Action required items get definitive
        language.
        """
        return {
            DeadStockTier.ACTIVE: "",
            DeadStockTier.WATCHLIST: (
                "may,might,worth monitoring,keep an eye on,"
                "could become,appears to be slowing"
            ),
            DeadStockTier.ATTENTION: (
                "suggests,pattern indicates,recommend reviewing,"
                "consider,trending toward"
            ),
            DeadStockTier.ACTION_REQUIRED: (
                "clearly shows,costing you,recommend action,"
                "losing money,tied up,should"
            ),
            DeadStockTier.WRITEOFF: (
                "has not sold,unsellable,write off,"
                "carrying cost,dispose,donate,total loss"
            ),
        }[self]

    @property
    def recommended_actions(self) -> list[str]:
        """Default recommended actions per tier."""
        return {
            DeadStockTier.ACTIVE: [],
            DeadStockTier.WATCHLIST: [
                "Monitor velocity trend over next 30 days",
                "Check if seasonal pattern explains the slowdown",
            ],
            DeadStockTier.ATTENTION: [
                "Review pricing — is it competitive?",
                "Check if another store has demand (transfer opportunity)",
                "Consider a limited promotion to test price sensitivity",
            ],
            DeadStockTier.ACTION_REQUIRED: [
                "Transfer to a store with active demand",
                "Run clearance at 30-50% markdown",
                "Bundle with complementary selling items",
                "Contact vendor about return or credit options",
            ],
            DeadStockTier.WRITEOFF: [
                "Donate for tax write-off",
                "Liquidate through bulk disposal channel",
                "Return to vendor if agreement allows",
                "Remove from active inventory count",
            ],
        }[self]


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class DeadStockThresholds:
    """
    Threshold configuration for a single scope (global or per-category).

    All values are in days since last sale.
    """

    watchlist_days: int = 60
    attention_days: int = 120
    action_days: int = 180
    writeoff_days: int = 360

    def validate(self) -> list[str]:
        """Validate that thresholds are in ascending order."""
        errors = []
        if self.watchlist_days <= 0:
            errors.append("Watchlist threshold must be positive")
        if self.attention_days <= self.watchlist_days:
            errors.append(
                f"Attention ({self.attention_days}) must be greater "
                f"than Watchlist ({self.watchlist_days})"
            )
        if self.action_days <= self.attention_days:
            errors.append(
                f"Action Required ({self.action_days}) must be greater "
                f"than Attention ({self.attention_days})"
            )
        if self.writeoff_days <= self.action_days:
            errors.append(
                f"Write-off ({self.writeoff_days}) must be greater "
                f"than Action Required ({self.action_days})"
            )
        return errors

    def to_dict(self) -> dict:
        return {
            "watchlist_days": self.watchlist_days,
            "attention_days": self.attention_days,
            "action_days": self.action_days,
            "writeoff_days": self.writeoff_days,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeadStockThresholds":
        return cls(
            watchlist_days=data.get("watchlist_days", 60),
            attention_days=data.get("attention_days", 120),
            action_days=data.get("action_days", 180),
            writeoff_days=data.get("writeoff_days", 360),
        )


@dataclass
class DeadStockConfig:
    """
    Complete dead stock configuration for a customer.

    Includes global defaults and per-category overrides.
    Stored per customer and loaded when the pipeline runs.

    Every component reads from this config:
    - Transfer matcher: uses thresholds to determine urgency
    - Predictive engine: uses thresholds to set warning horizons
    - Finding classifier: uses thresholds to set tier
    - Dashboard: uses thresholds for color coding
    - Response validator: uses tier language rules
    """

    # Global defaults
    global_thresholds: DeadStockThresholds = field(default_factory=DeadStockThresholds)

    # Per-category overrides (category name → thresholds)
    category_overrides: dict[str, DeadStockThresholds] = field(default_factory=dict)

    # Minimum capital at risk to trigger any alert
    min_capital_threshold: float = 50.0

    # Minimum velocity to NOT be considered slowing
    # Items below this AND within a tier get flagged
    min_healthy_velocity: float = 0.5  # units/week

    # Customer metadata
    customer_id: str = ""
    store_id: str = ""
    last_modified: float = 0.0
    modified_by: str = ""

    def get_thresholds(self, category: str = None) -> DeadStockThresholds:
        """Get thresholds for a category, falling back to global."""
        if category and category in self.category_overrides:
            return self.category_overrides[category]
        return self.global_thresholds

    def classify(
        self,
        days_since_last_sale: int,
        current_stock: int = 0,
        unit_cost: float = 0.0,
        velocity: float = 0.0,
        category: str = None,
    ) -> "ClassificationResult":
        """
        Classify an item's dead stock tier.

        Returns the tier plus supporting information for
        the finding, dashboard, and response validator.
        """
        thresholds = self.get_thresholds(category)
        capital_at_risk = current_stock * unit_cost

        # Determine tier from days
        if days_since_last_sale >= thresholds.writeoff_days:
            tier = DeadStockTier.WRITEOFF
        elif days_since_last_sale >= thresholds.action_days:
            tier = DeadStockTier.ACTION_REQUIRED
        elif days_since_last_sale >= thresholds.attention_days:
            tier = DeadStockTier.ATTENTION
        elif days_since_last_sale >= thresholds.watchlist_days:
            tier = DeadStockTier.WATCHLIST
        else:
            tier = DeadStockTier.ACTIVE

        # Velocity can bump tier up (but not down)
        # An item selling 0.1/week for 50 days is worse than
        # an item that sold 5 units on day 55 and nothing since
        if (
            tier == DeadStockTier.ACTIVE
            and velocity < self.min_healthy_velocity
            and velocity > 0
            and days_since_last_sale > thresholds.watchlist_days * 0.7
        ):
            tier = DeadStockTier.WATCHLIST

        # Check capital threshold — don't alert on penny items
        should_alert = (
            tier != DeadStockTier.ACTIVE
            and capital_at_risk >= self.min_capital_threshold
        )

        # Days until next tier
        days_to_next = self._days_to_next_tier(days_since_last_sale, thresholds, tier)

        # Carrying cost estimate (opportunity cost of capital)
        # Using 8% annual cost of capital as default
        annual_carry_rate = 0.08
        daily_carry = capital_at_risk * annual_carry_rate / 365
        carrying_cost_to_date = daily_carry * days_since_last_sale

        return ClassificationResult(
            tier=tier,
            days_since_last_sale=days_since_last_sale,
            capital_at_risk=capital_at_risk,
            carrying_cost=carrying_cost_to_date,
            should_alert=should_alert,
            days_to_next_tier=days_to_next,
            thresholds_used=thresholds,
            category=category,
            recommended_actions=tier.recommended_actions,
            dashboard_color=tier.dashboard_color,
            confidence_language=tier.confidence_language,
        )

    def _days_to_next_tier(
        self,
        current_days: int,
        thresholds: DeadStockThresholds,
        current_tier: DeadStockTier,
    ) -> int | None:
        """How many days until this item escalates to the next tier."""
        tier_boundaries = [
            (DeadStockTier.WATCHLIST, thresholds.watchlist_days),
            (DeadStockTier.ATTENTION, thresholds.attention_days),
            (DeadStockTier.ACTION_REQUIRED, thresholds.action_days),
            (DeadStockTier.WRITEOFF, thresholds.writeoff_days),
        ]

        for tier, boundary in tier_boundaries:
            if boundary > current_days:
                return boundary - current_days

        return None  # Already at highest tier

    def validate(self) -> list[str]:
        """Validate entire configuration."""
        errors = self.global_thresholds.validate()
        for category, override in self.category_overrides.items():
            cat_errors = override.validate()
            for err in cat_errors:
                errors.append(f"Category '{category}': {err}")
        if self.min_capital_threshold < 0:
            errors.append("Minimum capital threshold cannot be negative")
        return errors

    def to_dict(self) -> dict:
        """Serialize for storage/API."""
        return {
            "global_thresholds": self.global_thresholds.to_dict(),
            "category_overrides": {
                cat: thresh.to_dict() for cat, thresh in self.category_overrides.items()
            },
            "min_capital_threshold": self.min_capital_threshold,
            "min_healthy_velocity": self.min_healthy_velocity,
            "customer_id": self.customer_id,
            "store_id": self.store_id,
            "last_modified": self.last_modified,
            "modified_by": self.modified_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeadStockConfig":
        """Deserialize from storage/API."""
        config = cls(
            global_thresholds=DeadStockThresholds.from_dict(
                data.get("global_thresholds", {})
            ),
            min_capital_threshold=data.get("min_capital_threshold", 50.0),
            min_healthy_velocity=data.get("min_healthy_velocity", 0.5),
            customer_id=data.get("customer_id", ""),
            store_id=data.get("store_id", ""),
            last_modified=data.get("last_modified", 0.0),
            modified_by=data.get("modified_by", ""),
        )
        for cat, thresh_data in data.get("category_overrides", {}).items():
            config.category_overrides[cat] = DeadStockThresholds.from_dict(thresh_data)
        return config


# =============================================================================
# CLASSIFICATION RESULT
# =============================================================================


@dataclass
class ClassificationResult:
    """
    Complete classification of an item's dead stock status.

    Contains everything every downstream component needs:
    - tier: for the finding classifier and dashboard
    - capital_at_risk: for dollar amount in findings
    - carrying_cost: for the "real cost" calculation
    - should_alert: for the notification system
    - days_to_next_tier: for the predictive engine
    - recommended_actions: for the finding's action field
    - dashboard_color: for the UI
    - confidence_language: for the response validator
    """

    tier: DeadStockTier
    days_since_last_sale: int
    capital_at_risk: float
    carrying_cost: float
    should_alert: bool
    days_to_next_tier: int | None
    thresholds_used: DeadStockThresholds
    category: str | None
    recommended_actions: list[str]
    dashboard_color: str
    confidence_language: str

    def to_finding_description(self, item_description: str = "", units: int = 0) -> str:
        """
        Generate the finding description using the appropriate
        language for this tier.

        The response validator checks that findings at each tier
        use language consistent with the confidence_language field.
        """
        if self.tier == DeadStockTier.WATCHLIST:
            return (
                f"{item_description} may be slowing down. "
                f"No sales in {self.days_since_last_sale} days. "
                f"{units} units (${self.capital_at_risk:,.0f}) worth "
                f"monitoring over the next "
                f"{self.days_to_next_tier or 30} days."
            )

        elif self.tier == DeadStockTier.ATTENTION:
            return (
                f"{item_description} hasn't sold in "
                f"{self.days_since_last_sale} days. Pattern suggests "
                f"declining demand. {units} units tying up "
                f"${self.capital_at_risk:,.0f} in capital. "
                f"Recommend reviewing pricing or transfer options."
            )

        elif self.tier == DeadStockTier.ACTION_REQUIRED:
            return (
                f"{item_description} hasn't sold in "
                f"{self.days_since_last_sale} days. "
                f"{units} units clearly show no active demand, "
                f"costing you ${self.capital_at_risk:,.0f} in tied-up "
                f"capital plus ${self.carrying_cost:,.0f} in carrying costs."
            )

        elif self.tier == DeadStockTier.WRITEOFF:
            return (
                f"{item_description} has not sold in "
                f"{self.days_since_last_sale} days ({self.days_since_last_sale // 30} months). "
                f"{units} units with total loss exposure of "
                f"${self.capital_at_risk + self.carrying_cost:,.0f} "
                f"(${self.capital_at_risk:,.0f} capital + "
                f"${self.carrying_cost:,.0f} carrying costs). "
                f"Consider write-off or disposal."
            )

        return ""

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "severity": self.tier.severity,
            "days_since_last_sale": self.days_since_last_sale,
            "capital_at_risk": round(self.capital_at_risk, 2),
            "carrying_cost": round(self.carrying_cost, 2),
            "total_cost": round(self.capital_at_risk + self.carrying_cost, 2),
            "should_alert": self.should_alert,
            "days_to_next_tier": self.days_to_next_tier,
            "dashboard_color": self.dashboard_color,
            "recommended_actions": self.recommended_actions,
            "category": self.category,
        }


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================


class ConfigPresets:
    """
    Industry-specific preset configurations.

    Shown to the customer during onboarding:
    "What type of store do you run?"

    The customer picks a preset, then adjusts from there.
    This gets them 80% of the way to good thresholds immediately.
    """

    @staticmethod
    def hardware_store() -> DeadStockConfig:
        """Independent hardware store defaults."""
        config = DeadStockConfig(
            global_thresholds=DeadStockThresholds(
                watchlist_days=60,
                attention_days=120,
                action_days=180,
                writeoff_days=360,
            ),
            min_capital_threshold=50.0,
        )
        # Seasonal items should be flagged faster
        config.category_overrides["Seasonal"] = DeadStockThresholds(
            watchlist_days=30,
            attention_days=60,
            action_days=90,
            writeoff_days=180,
        )
        # Paint is perishable-ish (can settle, separate)
        config.category_overrides["Paint"] = DeadStockThresholds(
            watchlist_days=45,
            attention_days=90,
            action_days=150,
            writeoff_days=300,
        )
        # Specialty/contractor items move slowly
        config.category_overrides["Commercial Hardware"] = DeadStockThresholds(
            watchlist_days=90,
            attention_days=180,
            action_days=270,
            writeoff_days=540,
        )
        return config

    @staticmethod
    def garden_center() -> DeadStockConfig:
        """Garden center — everything is seasonal."""
        config = DeadStockConfig(
            global_thresholds=DeadStockThresholds(
                watchlist_days=30,
                attention_days=60,
                action_days=90,
                writeoff_days=180,
            ),
            min_capital_threshold=25.0,
        )
        # Hard goods (tools, pots) can sit longer
        config.category_overrides["Hard Goods"] = DeadStockThresholds(
            watchlist_days=60,
            attention_days=120,
            action_days=180,
            writeoff_days=360,
        )
        # Live plants — very short window
        config.category_overrides["Live Plants"] = DeadStockThresholds(
            watchlist_days=14,
            attention_days=21,
            action_days=30,
            writeoff_days=45,
        )
        return config

    @staticmethod
    def lumber_yard() -> DeadStockConfig:
        """Lumber/building materials — high value, slow turns."""
        config = DeadStockConfig(
            global_thresholds=DeadStockThresholds(
                watchlist_days=90,
                attention_days=180,
                action_days=270,
                writeoff_days=540,
            ),
            min_capital_threshold=100.0,
            min_healthy_velocity=0.3,
        )
        # Fasteners and small items should turn faster
        config.category_overrides["Fasteners"] = DeadStockThresholds(
            watchlist_days=60,
            attention_days=120,
            action_days=180,
            writeoff_days=360,
        )
        return config

    @staticmethod
    def convenience_store() -> DeadStockConfig:
        """Convenience/general store — fast turns expected."""
        config = DeadStockConfig(
            global_thresholds=DeadStockThresholds(
                watchlist_days=14,
                attention_days=30,
                action_days=60,
                writeoff_days=120,
            ),
            min_capital_threshold=10.0,
            min_healthy_velocity=1.0,
        )
        return config

    @staticmethod
    def all_presets() -> dict[str, DeadStockConfig]:
        """All available presets."""
        return {
            "hardware_store": ConfigPresets.hardware_store(),
            "garden_center": ConfigPresets.garden_center(),
            "lumber_yard": ConfigPresets.lumber_yard(),
            "convenience_store": ConfigPresets.convenience_store(),
        }


# =============================================================================
# LIFECYCLE TRACKER
# =============================================================================


class InventoryLifecycleTracker:
    """
    Tracks items across their entire lifecycle through the
    dead stock tiers.

    Records when items enter and exit each tier, enabling:
    - "How long do items typically sit in Watchlist before
       they either recover or escalate?"
    - "Which categories have the highest escalation rate?"
    - "What percentage of Attention items are saved by transfers?"

    This data feeds the moat metrics — it shows the system
    getting better at catching items earlier and preventing
    escalation.
    """

    def __init__(self, config: DeadStockConfig):
        self.config = config

        # Track tier transitions per item
        # entity_key → list of {tier, entered_at, exited_at}
        self.lifecycle_history: dict[str, list[dict]] = defaultdict(list)

        # Current tier per item
        self.current_tiers: dict[str, DeadStockTier] = {}

        # Aggregate stats
        self.tier_transitions: dict[str, int] = defaultdict(int)
        self.recoveries: int = 0  # Items that went back to ACTIVE
        self.escalations: int = 0  # Items that moved to higher tier
        self.interventions: int = 0  # Items acted on (transfer/clearance)
        self.intervention_successes: int = 0

    def update_item(
        self,
        entity_key: str,
        classification: ClassificationResult,
        timestamp: float = None,
    ):
        """
        Update an item's lifecycle status.

        Detects tier transitions and records them.
        """
        timestamp = timestamp or time.time()
        new_tier = classification.tier
        old_tier = self.current_tiers.get(entity_key, DeadStockTier.ACTIVE)

        if new_tier != old_tier:
            # Record the transition
            transition_key = f"{old_tier.value}->{new_tier.value}"
            self.tier_transitions[transition_key] += 1

            # Close the old tier record
            history = self.lifecycle_history[entity_key]
            if history and history[-1].get("exited_at") is None:
                history[-1]["exited_at"] = timestamp
                history[-1]["duration_days"] = (
                    timestamp - history[-1]["entered_at"]
                ) / 86400

            # Open new tier record
            history.append(
                {
                    "tier": new_tier.value,
                    "entered_at": timestamp,
                    "exited_at": None,
                    "category": classification.category,
                    "capital_at_risk": classification.capital_at_risk,
                }
            )

            # Track recoveries vs escalations
            if new_tier.severity < old_tier.severity:
                self.recoveries += 1
            elif new_tier.severity > old_tier.severity:
                self.escalations += 1

            self.current_tiers[entity_key] = new_tier

    def record_intervention(
        self, entity_key: str, intervention_type: str, success: bool
    ):
        """Record that action was taken on an item."""
        self.interventions += 1
        if success:
            self.intervention_successes += 1

    def tier_distribution(self) -> dict[str, int]:
        """Current distribution of items across tiers."""
        dist = defaultdict(int)
        for tier in self.current_tiers.values():
            dist[tier.value] += 1
        return dict(dist)

    def avg_time_in_tier(self, tier: DeadStockTier) -> float | None:
        """Average days items spend in a specific tier."""
        durations = []
        for entity_key, history in self.lifecycle_history.items():
            for record in history:
                if record["tier"] == tier.value and "duration_days" in record:
                    durations.append(record["duration_days"])
        if durations:
            return sum(durations) / len(durations)
        return None

    def escalation_rate(self) -> float:
        """What percentage of flagged items escalate vs recover."""
        total = self.recoveries + self.escalations
        if total == 0:
            return 0.0
        return self.escalations / total

    def intervention_success_rate(self) -> float:
        """What percentage of interventions succeeded."""
        if self.interventions == 0:
            return 0.0
        return self.intervention_successes / self.interventions

    def lifecycle_report(self) -> dict:
        """Full lifecycle report for the moat metrics."""
        return {
            "tier_distribution": self.tier_distribution(),
            "total_tracked": len(self.current_tiers),
            "recoveries": self.recoveries,
            "escalations": self.escalations,
            "escalation_rate": round(self.escalation_rate(), 3),
            "interventions": self.interventions,
            "intervention_success_rate": round(self.intervention_success_rate(), 3),
            "avg_time_in_tiers": {
                tier.value: round(self.avg_time_in_tier(tier) or 0, 1)
                for tier in DeadStockTier
                if tier != DeadStockTier.ACTIVE
            },
            "transition_counts": dict(self.tier_transitions),
        }


# =============================================================================
# TEST
# =============================================================================


def run_dead_stock_config_test():
    """
    Test the dead stock configuration system with realistic
    hardware store scenarios.
    """
    print("=" * 70)
    print("DEAD STOCK CONFIGURATION TEST")
    print("=" * 70)
    print()

    # Use hardware store preset
    config = ConfigPresets.hardware_store()
    errors = config.validate()
    assert len(errors) == 0, f"Validation errors: {errors}"
    print("Hardware store preset loaded and validated.")
    print(f"  Global: {config.global_thresholds.to_dict()}")
    print(f"  Overrides: {list(config.category_overrides.keys())}")
    print()

    # Test classification at each tier
    test_items = [
        ("Active paint brush", 15, 30, 6.50, 2.5, "Paint Supplies"),
        ("Slow cabinet pulls", 50, 47, 12.50, 0.3, "Hardware"),
        ("Watchlist deadbolt", 65, 23, 34.00, 0.0, "Hardware"),
        ("Attention copper pipe", 125, 85, 8.75, 0.0, "Plumbing"),
        ("Action needed anchors", 185, 30, 18.00, 0.0, "Fasteners"),
        ("Writeoff smart home", 400, 12, 85.00, 0.0, "Electrical"),
        # Seasonal override test
        ("Dead xmas lights", 45, 50, 3.00, 0.0, "Seasonal"),
        # Commercial override test
        ("Slow commercial lock", 100, 5, 250.00, 0.1, "Commercial Hardware"),
        # Below capital threshold
        ("Cheap washers", 200, 10, 0.25, 0.0, "Fasteners"),
    ]

    print("Item Classifications:")
    print("-" * 70)

    tracker = InventoryLifecycleTracker(config)
    base_time = time.time()

    for desc, days, stock, cost, velocity, category in test_items:
        result = config.classify(
            days_since_last_sale=days,
            current_stock=stock,
            unit_cost=cost,
            velocity=velocity,
            category=category,
        )

        tier_str = result.tier.value.upper()
        alert_str = "ALERT" if result.should_alert else "silent"
        color = result.dashboard_color

        print(f"  {desc}")
        print(f"    Tier: {tier_str} ({color}) | {alert_str}")
        print(
            f"    Capital: ${result.capital_at_risk:,.2f} | "
            f"Carrying cost: ${result.carrying_cost:,.2f}"
        )
        if result.days_to_next_tier:
            print(f"    Escalates in: {result.days_to_next_tier} days")
        if result.recommended_actions:
            print(f"    Top action: {result.recommended_actions[0]}")

        # Test finding description
        finding = result.to_finding_description(desc, stock)
        if finding:
            print(f'    Finding: "{finding[:80]}..."')

        print()

        # Track in lifecycle
        entity_key = desc.replace(" ", "_").lower()
        tracker.update_item(entity_key, result, base_time)

    # Test specific assertions
    # Seasonal items should flag earlier (30/60/90/180 thresholds)
    # At 45 days: past watchlist (30), not yet attention (60) = WATCHLIST
    seasonal_result = config.classify(45, 50, 3.00, 0.0, "Seasonal")
    assert (
        seasonal_result.tier == DeadStockTier.WATCHLIST
    ), f"Seasonal at 45 days should be WATCHLIST, got {seasonal_result.tier}"

    # Seasonal at 65 days should be ATTENTION (past 60 threshold)
    seasonal_65 = config.classify(65, 50, 3.00, 0.0, "Seasonal")
    assert (
        seasonal_65.tier == DeadStockTier.ATTENTION
    ), f"Seasonal at 65 days should be ATTENTION, got {seasonal_65.tier}"

    # Regular items at 45 days should still be ACTIVE (global watchlist = 60)
    regular_result = config.classify(45, 50, 3.00, 0.0, "Hardware")
    assert (
        regular_result.tier == DeadStockTier.ACTIVE
    ), f"Regular at 45 days should be ACTIVE, got {regular_result.tier}"

    # Commercial hardware at 100 days should still be just WATCHLIST
    commercial_result = config.classify(100, 5, 250.00, 0.1, "Commercial Hardware")
    assert (
        commercial_result.tier == DeadStockTier.WATCHLIST
    ), f"Commercial at 100 days should be WATCHLIST, got {commercial_result.tier}"

    # Below capital threshold should not alert
    cheap_result = config.classify(200, 10, 0.25, 0.0, "Fasteners")
    assert (
        not cheap_result.should_alert
    ), "Items below capital threshold should not alert"

    print("=" * 70)
    print("ASSERTIONS PASSED:")
    print("  - Seasonal items flag at 45 days (WATCHLIST — earlier than global)")
    print("  - Seasonal items escalate at 65 days (ATTENTION)")
    print("  - Regular items OK at 45 days (ACTIVE)")
    print("  - Commercial hardware relaxed at 100 days (WATCHLIST)")
    print("  - Below capital threshold = silent (no alert)")
    print()

    # Lifecycle report
    print("Lifecycle Report:")
    report = tracker.lifecycle_report()
    print(f"  Tracked: {report['total_tracked']} items")
    print(f"  Tier distribution: {report['tier_distribution']}")
    print()

    # Serialization round-trip
    serialized = config.to_dict()
    restored = DeadStockConfig.from_dict(serialized)
    assert restored.global_thresholds.watchlist_days == 60
    assert "Seasonal" in restored.category_overrides
    assert restored.category_overrides["Seasonal"].watchlist_days == 30
    print("Serialization round-trip: PASSED")
    print()

    # Show all presets
    print("Available presets:")
    for name, preset in ConfigPresets.all_presets().items():
        g = preset.global_thresholds
        print(
            f"  {name}: {g.watchlist_days}/{g.attention_days}/"
            f"{g.action_days}/{g.writeoff_days} days | "
            f"min capital ${preset.min_capital_threshold}"
        )

    return config, tracker


if __name__ == "__main__":
    config, tracker = run_dead_stock_config_test()
