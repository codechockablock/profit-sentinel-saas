"""Pydantic models matching the Rust sentinel-server JSON contract.

These models are the single source of truth for the Python-Rust bridge.
The Rust server outputs JSON with --json flag; these models parse it
via model_validate_json() for zero-copy deserialization.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class IssueType(str, Enum):
    """Maps 1:1 to Rust IssueType enum variants."""

    RECEIVING_GAP = "ReceivingGap"
    DEAD_STOCK = "DeadStock"
    MARGIN_EROSION = "MarginErosion"
    NEGATIVE_INVENTORY = "NegativeInventory"
    VENDOR_SHORT_SHIP = "VendorShortShip"
    PURCHASING_LEAKAGE = "PurchasingLeakage"
    PATRONAGE_MISS = "PatronageMiss"
    SHRINKAGE_PATTERN = "ShrinkagePattern"
    ZERO_COST_ANOMALY = "ZeroCostAnomaly"
    PRICE_DISCREPANCY = "PriceDiscrepancy"
    OVERSTOCK = "Overstock"

    @property
    def display_name(self) -> str:
        """Human-readable name for display."""
        return {
            IssueType.RECEIVING_GAP: "Receiving Gap",
            IssueType.DEAD_STOCK: "Dead Stock",
            IssueType.MARGIN_EROSION: "Margin Erosion",
            IssueType.NEGATIVE_INVENTORY: "Negative Inventory",
            IssueType.VENDOR_SHORT_SHIP: "Vendor Short Ship",
            IssueType.PURCHASING_LEAKAGE: "Purchasing Leakage",
            IssueType.PATRONAGE_MISS: "Patronage Miss",
            IssueType.SHRINKAGE_PATTERN: "Shrinkage Pattern",
            IssueType.ZERO_COST_ANOMALY: "Zero Cost Anomaly",
            IssueType.PRICE_DISCREPANCY: "Price Discrepancy",
            IssueType.OVERSTOCK: "Overstock",
        }[self]

    @property
    def action_verb(self) -> str:
        """Short action description for delegation."""
        return {
            IssueType.RECEIVING_GAP: "Investigate receiving discrepancy",
            IssueType.DEAD_STOCK: "Review and markdown dead inventory",
            IssueType.MARGIN_EROSION: "Review pricing and vendor terms",
            IssueType.NEGATIVE_INVENTORY: "Investigate negative on-hand",
            IssueType.VENDOR_SHORT_SHIP: "Contact vendor about damaged shipment",
            IssueType.PURCHASING_LEAKAGE: "Renegotiate vendor pricing",
            IssueType.PATRONAGE_MISS: "Markdown seasonal overstock",
            IssueType.SHRINKAGE_PATTERN: "Investigate potential shrinkage",
            IssueType.ZERO_COST_ANOMALY: "Correct missing cost data",
            IssueType.PRICE_DISCREPANCY: "Correct below-cost pricing",
            IssueType.OVERSTOCK: "Reduce excess inventory",
        }[self]


class RootCause(str, Enum):
    """Maps 1:1 to Rust RootCause enum variants.

    Evidence-based root cause attribution determined by positive-similarity
    scoring against observed inventory signals.
    """

    THEFT = "Theft"
    VENDOR_INCREASE = "VendorIncrease"
    REBATE_TIMING = "RebateTiming"
    MARGIN_LEAK = "MarginLeak"
    DEMAND_SHIFT = "DemandShift"
    QUALITY_ISSUE = "QualityIssue"
    PRICING_ERROR = "PricingError"
    INVENTORY_DRIFT = "InventoryDrift"

    @property
    def display_name(self) -> str:
        """Human-readable name for display."""
        return {
            RootCause.THEFT: "Theft / Shrinkage",
            RootCause.VENDOR_INCREASE: "Vendor Price Increase",
            RootCause.REBATE_TIMING: "Rebate Timing Mismatch",
            RootCause.MARGIN_LEAK: "Margin Leak",
            RootCause.DEMAND_SHIFT: "Demand Shift",
            RootCause.QUALITY_ISSUE: "Quality Issue",
            RootCause.PRICING_ERROR: "Pricing Error",
            RootCause.INVENTORY_DRIFT: "Inventory Drift",
        }[self]

    @property
    def severity(self) -> str:
        """Severity category for routing decisions."""
        return {
            RootCause.THEFT: "critical",
            RootCause.VENDOR_INCREASE: "high",
            RootCause.REBATE_TIMING: "medium",
            RootCause.MARGIN_LEAK: "high",
            RootCause.DEMAND_SHIFT: "medium",
            RootCause.QUALITY_ISSUE: "high",
            RootCause.PRICING_ERROR: "high",
            RootCause.INVENTORY_DRIFT: "medium",
        }[self]

    @property
    def recommendations(self) -> list[str]:
        """Actionable recommendations for this root cause."""
        return {
            RootCause.THEFT: [
                "Review security footage for high-value items",
                "Analyze void/return patterns by employee",
                "Conduct surprise cycle counts",
                "Check receiving accuracy against POs",
            ],
            RootCause.VENDOR_INCREASE: [
                "Review recent vendor invoices for price changes",
                "Compare to co-op contract pricing",
                "Negotiate volume discounts",
                "Evaluate alternative vendors",
            ],
            RootCause.REBATE_TIMING: [
                "Verify rebate accrual timing with co-op",
                "Check payment terms on affected POs",
                "Review dating program utilization",
                "Reconcile pending rebates with vendor",
            ],
            RootCause.MARGIN_LEAK: [
                "Audit promotional pricing for stuck discounts",
                "Review markdown cadence",
                "Compare retail to MSRP/suggested retail",
                "Check for unauthorized price overrides",
            ],
            RootCause.DEMAND_SHIFT: [
                "Review category sales trends",
                "Adjust reorder points based on velocity",
                "Consider markdown or transfer",
                "Evaluate seasonal timing",
            ],
            RootCause.QUALITY_ISSUE: [
                "Check return rates by vendor/SKU",
                "File vendor claims for defective product",
                "Review product condition on shelf",
                "Contact vendor quality department",
            ],
            RootCause.PRICING_ERROR: [
                "Verify POS retail price vs cost",
                "Check for data entry errors",
                "Review recent price file imports",
                "Correct pricing in POS system",
            ],
            RootCause.INVENTORY_DRIFT: [
                "Schedule cycle count for affected items",
                "Review receiving process accuracy",
                "Check for transfer errors between locations",
                "Audit bin locations and quantities",
            ],
        }[self]


class TrendDirection(str, Enum):
    """Maps 1:1 to Rust TrendDirection enum variants."""

    WORSENING = "Worsening"
    STABLE = "Stable"
    IMPROVING = "Improving"

    @property
    def arrow(self) -> str:
        return {
            TrendDirection.WORSENING: "\u2191",
            TrendDirection.STABLE: "\u2192",
            TrendDirection.IMPROVING: "\u2193",
        }[self]

    @property
    def description(self) -> str:
        return {
            TrendDirection.WORSENING: "worsening",
            TrendDirection.STABLE: "stable",
            TrendDirection.IMPROVING: "improving",
        }[self]


class Sku(BaseModel):
    """Detail record for a single SKU within an issue."""

    sku_id: str
    qty_on_hand: float
    unit_cost: float
    retail_price: float
    margin_pct: float
    sales_last_30d: float
    days_since_receipt: float
    is_damaged: bool
    on_order_qty: float
    is_seasonal: bool

    @property
    def total_value(self) -> float:
        """Absolute inventory value at cost."""
        return abs(self.qty_on_hand) * self.unit_cost

    @property
    def margin_display(self) -> str:
        return f"{self.margin_pct * 100:.0f}%"


class CauseScoreDetail(BaseModel):
    """Score for a single cause hypothesis from the VSA evidence scorer.

    All 8 root cause hypotheses are scored; the list is ranked by score
    (highest first). Phase 13 uses these to build transparent proof trees.
    """

    cause: str
    score: float
    evidence_count: int


class Issue(BaseModel):
    """A prioritized issue from the Rust pipeline."""

    id: str
    issue_type: IssueType
    store_id: str
    dollar_impact: float
    confidence: float
    trend_direction: TrendDirection
    priority_score: float
    urgency_score: float
    detection_timestamp: str
    skus: list[Sku]
    context: str
    root_cause: RootCause | None = None
    root_cause_confidence: float | None = None
    # Phase 13: Detailed evidence for symbolic bridge
    cause_scores: list[CauseScoreDetail] = Field(default_factory=list)
    root_cause_ambiguity: float | None = None
    active_signals: list[str] = Field(default_factory=list)

    @property
    def sku_count(self) -> int:
        return len(self.skus)

    @property
    def is_urgent(self) -> bool:
        """Priority >= 8.0 is urgent."""
        return self.priority_score >= 8.0

    @property
    def dollar_display(self) -> str:
        """Formatted dollar amount with commas."""
        return f"${self.dollar_impact:,.0f}"

    @property
    def root_cause_display(self) -> str:
        """Human-readable root cause with confidence."""
        if self.root_cause is None:
            return "Unknown"
        conf = self.root_cause_confidence or 0.0
        return f"{self.root_cause.display_name} ({conf:.0%})"


class Summary(BaseModel):
    """Aggregate summary of the digest."""

    total_issues: int
    total_dollar_impact: float
    stores_affected: int
    records_processed: int
    issues_detected: int
    issues_filtered_out: int

    @property
    def total_dollar_display(self) -> str:
        return f"${self.total_dollar_impact:,.0f}"


class Digest(BaseModel):
    """Top-level digest from the Rust pipeline.

    Parse from JSON:
        digest = Digest.model_validate_json(json_bytes)
    """

    generated_at: str
    store_filter: list[str]
    pipeline_ms: int
    issues: list[Issue]
    summary: Summary

    @property
    def generated_datetime(self) -> datetime:
        """Parse the ISO-8601 timestamp."""
        return datetime.fromisoformat(self.generated_at)

    @property
    def has_urgent_issues(self) -> bool:
        return any(issue.is_urgent for issue in self.issues)

    def issues_by_store(self) -> dict[str, list[Issue]]:
        """Group issues by store ID."""
        result: dict[str, list[Issue]] = {}
        for issue in self.issues:
            result.setdefault(issue.store_id, []).append(issue)
        return result

    def issues_by_type(self) -> dict[IssueType, list[Issue]]:
        """Group issues by type."""
        result: dict[IssueType, list[Issue]] = {}
        for issue in self.issues:
            result.setdefault(issue.issue_type, []).append(issue)
        return result


# ---------------------------------------------------------------------------
# Delegation / Task models
# ---------------------------------------------------------------------------


class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Task(BaseModel):
    """A delegated task created from an issue."""

    task_id: str
    issue_id: str
    issue_type: IssueType
    store_id: str
    assignee: str
    deadline: datetime
    priority: TaskPriority
    title: str
    description: str
    action_items: list[str]
    dollar_impact: float
    skus: list[Sku]
    created_at: datetime = Field(default_factory=datetime.now)


class CallPrep(BaseModel):
    """Vendor call preparation package."""

    issue_id: str
    store_id: str
    vendor_name: str
    issue_summary: str
    affected_skus: list[Sku]
    total_dollar_impact: float
    talking_points: list[str]
    questions_to_ask: list[str]
    historical_context: str
