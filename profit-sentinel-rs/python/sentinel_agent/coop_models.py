"""Pydantic models for the Co-op Intelligence module.

Covers co-op affiliation, patronage programs, rebate tiers,
inventory health metrics, and category mix analysis. All dollar
amounts are explicit — every model quantifies its impact.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Co-op Affiliation
# ---------------------------------------------------------------------------


class CoopType(str, Enum):
    """The big three co-ops / distributors in independent hardware."""

    DO_IT_BEST = "DoItBest"
    ACE = "Ace"
    ORGILL = "Orgill"


class PatronageCategory(str, Enum):
    """Do It Best patronage rate categories."""

    REGULAR_WAREHOUSE = "RegularWarehouse"
    PROMOTIONAL_WAREHOUSE = "PromotionalWarehouse"
    DIRECT_SHIP = "DirectShip"
    LUMBER = "Lumber"


# Do It Best Classic patronage rates (FY2024)
PATRONAGE_RATES: dict[PatronageCategory, float] = {
    PatronageCategory.REGULAR_WAREHOUSE: 0.1111,  # 11.11%
    PatronageCategory.PROMOTIONAL_WAREHOUSE: 0.0584,  # 5.84%
    PatronageCategory.DIRECT_SHIP: 0.0106,  # 1.06%
    PatronageCategory.LUMBER: 0.0074,  # 0.74%
}

# Additional 2% cash discount on warehouse invoices
WAREHOUSE_CASH_DISCOUNT = 0.02


class CoopAffiliation(BaseModel):
    """A store's co-op membership."""

    store_id: str
    coop_type: CoopType
    member_number: str = ""
    annual_purchases: float = 0.0
    patronage_earned_ytd: float = 0.0


class PatronageProgram(BaseModel):
    """A specific patronage / rebate program from a co-op."""

    coop_type: CoopType
    category: PatronageCategory
    rate: float
    description: str


# ---------------------------------------------------------------------------
# Patronage Leakage & Consolidation
# ---------------------------------------------------------------------------


class VendorPurchase(BaseModel):
    """A purchase record from a specific vendor."""

    vendor_id: str
    vendor_name: str
    sku_id: str
    category: str
    quantity: float
    unit_cost: float
    total_cost: float
    is_coop_available: bool = False
    purchase_date: date | None = None

    @property
    def annualized_cost(self) -> float:
        return self.total_cost


class PatronageLeakage(BaseModel):
    """A detected patronage leakage opportunity.

    Purchases from non-co-op vendors where co-op equivalents exist.
    """

    store_id: str
    vendor_name: str
    category: str
    non_coop_spend: float
    coop_equivalent_available: bool = True
    current_rebate_rate: float = 0.0
    coop_rebate_rate: float = 0.1111  # warehouse default
    annual_leakage: float = 0.0
    affected_skus: list[str] = Field(default_factory=list)

    @property
    def rebate_differential(self) -> float:
        return self.coop_rebate_rate - self.current_rebate_rate

    @property
    def dollar_display(self) -> str:
        return f"${self.annual_leakage:,.0f}"


class ConsolidationOpportunity(BaseModel):
    """A vendor consolidation opportunity with quantified benefit."""

    store_id: str
    category: str
    current_vendor_count: int
    vendors: list[str]
    total_category_spend: float
    shiftable_spend: float
    coop_rebate_rate: float
    cash_discount_rate: float = WAREHOUSE_CASH_DISCOUNT
    annual_benefit: float = 0.0
    recommendation: str = ""

    @property
    def benefit_display(self) -> str:
        return f"${self.annual_benefit:,.0f}"


# ---------------------------------------------------------------------------
# Inventory Health
# ---------------------------------------------------------------------------


class TurnClassification(str, Enum):
    """Turn rate classification per NHPA benchmarks."""

    FAST_MOVER = "FastMover"  # >4.0x
    HEALTHY = "Healthy"  # 2.5-4.0x
    SLOW_MOVER = "SlowMover"  # 1.5-2.5x
    WEAK = "Weak"  # 0.5-1.5x
    DEAD = "Dead"  # <0.5x

    @property
    def action(self) -> str:
        return {
            TurnClassification.FAST_MOVER: "Monitor stockouts, consider adding depth",
            TurnClassification.HEALTHY: "Maintain current levels",
            TurnClassification.SLOW_MOVER: "Reduce depth, consider promotion",
            TurnClassification.WEAK: "Clearance candidate",
            TurnClassification.DEAD: "Liquidate immediately",
        }[self]

    @property
    def label(self) -> str:
        return {
            TurnClassification.FAST_MOVER: "Fast Mover",
            TurnClassification.HEALTHY: "Healthy",
            TurnClassification.SLOW_MOVER: "Slow Mover",
            TurnClassification.WEAK: "Weak",
            TurnClassification.DEAD: "Dead Stock",
        }[self]


class SkuHealth(BaseModel):
    """Health metrics for a single SKU."""

    sku_id: str
    store_id: str
    category: str
    qty_on_hand: float
    unit_cost: float
    retail_price: float
    margin_pct: float
    annual_sales_units: float
    annual_cogs: float
    avg_inventory_cost: float
    turn_rate: float
    gmroi: float
    classification: TurnClassification
    carrying_cost_annual: float
    days_of_supply: float

    @property
    def inventory_value(self) -> float:
        return abs(self.qty_on_hand) * self.unit_cost


class GMROIAnalysis(BaseModel):
    """GMROI analysis for a category or store."""

    store_id: str
    category: str
    total_inventory_cost: float
    total_annual_cogs: float
    total_annual_sales: float
    gross_margin_pct: float
    turn_rate: float
    gmroi: float
    sku_count: int
    fast_mover_count: int
    healthy_count: int
    slow_mover_count: int
    weak_count: int
    dead_count: int

    @property
    def is_profitable(self) -> bool:
        """GMROI > 1.0 means money earned on inventory investment."""
        return self.gmroi > 1.0

    @property
    def performance_label(self) -> str:
        if self.gmroi >= 2.70:
            return "High-Profit"
        if self.gmroi >= 1.77:
            return "Average"
        return "Below Average"


class InventoryHealthReport(BaseModel):
    """Full inventory health report for a store."""

    store_id: str
    generated_at: datetime = Field(default_factory=datetime.now)
    total_inventory_value: float
    total_dead_stock_value: float
    dead_stock_pct: float
    annual_carrying_cost: float
    overall_turn_rate: float
    overall_gmroi: float
    category_analyses: list[GMROIAnalysis] = Field(default_factory=list)
    sku_details: list[SkuHealth] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)

    @property
    def dead_stock_display(self) -> str:
        return f"${self.total_dead_stock_value:,.0f}"

    @property
    def carrying_cost_display(self) -> str:
        return f"${self.annual_carrying_cost:,.0f}"


# ---------------------------------------------------------------------------
# Vendor Rebates
# ---------------------------------------------------------------------------


class RebateTier(BaseModel):
    """A single tier in a vendor rebate program."""

    tier_name: str
    threshold: float
    rebate_pct: float
    description: str = ""


class VendorRebateProgram(BaseModel):
    """A vendor's rebate program with tiers."""

    vendor_id: str
    vendor_name: str
    program_name: str
    program_type: str  # volume, growth, mix, time
    tiers: list[RebateTier]
    period_start: date
    period_end: date
    category: str = ""


class VendorRebateStatus(BaseModel):
    """Current progress toward a vendor rebate threshold."""

    program: VendorRebateProgram
    store_id: str
    ytd_purchases: float
    current_tier: RebateTier | None
    next_tier: RebateTier | None
    shortfall: float
    days_remaining: int
    daily_run_rate: float
    projected_total: float
    on_track: bool
    current_rebate_value: float
    next_tier_rebate_value: float
    incremental_value: float
    recommendation: str = ""

    @property
    def is_at_risk(self) -> bool:
        """At risk of missing next tier."""
        return not self.on_track and self.next_tier is not None

    @property
    def shortfall_display(self) -> str:
        return f"${self.shortfall:,.0f}"

    @property
    def incremental_display(self) -> str:
        return f"${self.incremental_value:,.0f}"


# ---------------------------------------------------------------------------
# Category Mix
# ---------------------------------------------------------------------------

# NHPA High-Profit store category mix targets (% of revenue)
CATEGORY_BENCHMARKS: dict[str, dict] = {
    "Paint": {"target_pct": 0.12, "target_margin": 0.45, "priority": "High"},
    "Electrical": {"target_pct": 0.11, "target_margin": 0.40, "priority": "High"},
    "Plumbing": {"target_pct": 0.09, "target_margin": 0.37, "priority": "Medium"},
    "Hand Tools": {"target_pct": 0.08, "target_margin": 0.42, "priority": "Medium"},
    "Fasteners": {"target_pct": 0.06, "target_margin": 0.50, "priority": "Low"},
    "Lumber": {"target_pct": 0.15, "target_margin": 0.25, "priority": "Volume"},
    "Seasonal": {"target_pct": 0.10, "target_margin": 0.42, "priority": "Timing"},
    "Services": {"target_pct": 0.05, "target_margin": 0.65, "priority": "High"},
    "General Hardware": {
        "target_pct": 0.10,
        "target_margin": 0.38,
        "priority": "Medium",
    },
    "Other": {"target_pct": 0.14, "target_margin": 0.35, "priority": "Low"},
}


class BenchmarkComparison(BaseModel):
    """Comparison of a single category against NHPA benchmarks."""

    category: str
    store_pct: float
    benchmark_pct: float
    store_margin: float
    benchmark_margin: float
    store_revenue: float
    gap_pct: float  # positive = over-indexed, negative = under-indexed
    margin_gap: float  # positive = beating benchmark
    dollar_opportunity: float  # annual $ impact of closing the gap
    recommendation: str

    @property
    def is_under_indexed(self) -> bool:
        return self.gap_pct < -0.01  # >1% under

    @property
    def opportunity_display(self) -> str:
        return f"${self.dollar_opportunity:,.0f}"


class CategoryMixAnalysis(BaseModel):
    """Full category mix analysis for a store."""

    store_id: str
    total_revenue: float
    total_margin_pct: float
    comparisons: list[BenchmarkComparison]
    total_opportunity: float
    top_expansion_categories: list[str]
    top_contraction_categories: list[str]

    @property
    def opportunity_display(self) -> str:
        return f"${self.total_opportunity:,.0f}"


# ---------------------------------------------------------------------------
# Co-op Intelligence Alert (unified output)
# ---------------------------------------------------------------------------


class CoopAlertType(str, Enum):
    """Types of co-op intelligence alerts."""

    PATRONAGE_LEAKAGE = "PatronageLeakage"
    DEAD_STOCK_ALERT = "DeadStockAlert"
    REBATE_THRESHOLD_RISK = "RebateThresholdRisk"
    MIX_IMBALANCE = "MixImbalance"
    CONSOLIDATION_OPPORTUNITY = "ConsolidationOpportunity"
    GMROI_WARNING = "GMROIWarning"

    @property
    def display_name(self) -> str:
        return {
            CoopAlertType.PATRONAGE_LEAKAGE: "Patronage Leakage",
            CoopAlertType.DEAD_STOCK_ALERT: "Dead Stock Alert",
            CoopAlertType.REBATE_THRESHOLD_RISK: "Rebate Threshold Risk",
            CoopAlertType.MIX_IMBALANCE: "Category Mix Imbalance",
            CoopAlertType.CONSOLIDATION_OPPORTUNITY: "Consolidation Opportunity",
            CoopAlertType.GMROI_WARNING: "GMROI Warning",
        }[self]

    @property
    def icon(self) -> str:
        return {
            CoopAlertType.PATRONAGE_LEAKAGE: "!!",
            CoopAlertType.DEAD_STOCK_ALERT: "!!",
            CoopAlertType.REBATE_THRESHOLD_RISK: "! ",
            CoopAlertType.MIX_IMBALANCE: "  ",
            CoopAlertType.CONSOLIDATION_OPPORTUNITY: "  ",
            CoopAlertType.GMROI_WARNING: "! ",
        }[self]


class CoopAlert(BaseModel):
    """A single co-op intelligence alert."""

    alert_type: CoopAlertType
    store_id: str
    title: str
    dollar_impact: float
    detail: str
    recommendation: str
    confidence: float = 0.8

    @property
    def dollar_display(self) -> str:
        return f"${self.dollar_impact:,.0f}"


class CoopIntelligenceReport(BaseModel):
    """Combined co-op intelligence report for a store."""

    store_id: str
    generated_at: datetime = Field(default_factory=datetime.now)
    affiliation: CoopAffiliation | None = None
    alerts: list[CoopAlert] = Field(default_factory=list)
    health_report: InventoryHealthReport | None = None
    rebate_statuses: list[VendorRebateStatus] = Field(default_factory=list)
    category_analysis: CategoryMixAnalysis | None = None
    total_opportunity: float = 0.0

    @property
    def opportunity_display(self) -> str:
        return f"${self.total_opportunity:,.0f}"

    @property
    def alert_count(self) -> int:
        return len(self.alerts)
