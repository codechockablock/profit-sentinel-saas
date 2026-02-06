"""Inventory Health Scorer.

Calculates GMROI, turn rates, carrying costs, and dead stock metrics
per NHPA benchmarks. Every SKU gets a health classification and every
category gets a dollar-quantified performance score.

Key metrics:
- GMROI: Gross Margin Return on Inventory Investment
  GMROI = (Annual Sales × Gross Margin %) / Avg Inventory Cost
  Industry average: $1.77, High-profit target: $2.70+

- Turn Rate: Annual COGS / Avg Inventory Cost
  >4.0x = Fast Mover, 2.5-4.0x = Healthy, <0.5x = Dead Stock

- Carrying Cost: 25-30% of inventory value annually
  (storage + insurance + obsolescence + opportunity + handling)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from .coop_models import (
    CoopAlert,
    CoopAlertType,
    GMROIAnalysis,
    InventoryHealthReport,
    SkuHealth,
    TurnClassification,
)
from .models import Digest, Issue, IssueType, Sku

# Annual carrying cost as % of inventory value (NHPA benchmark: 25-30%)
CARRYING_COST_RATE = 0.27

# GMROI thresholds
GMROI_HIGH_PROFIT = 2.70
GMROI_AVERAGE = 1.77
GMROI_MINIMUM = 1.0

# Default margin benchmark
DEFAULT_MARGIN_BENCHMARK = 0.35


class InventoryHealthScorer:
    """Score inventory health at the SKU, category, and store level.

    Usage:
        scorer = InventoryHealthScorer()
        report = scorer.score_from_digest(digest, store_id="store-7")
        print(f"Dead stock: {report.dead_stock_display}")
        print(f"GMROI: {report.overall_gmroi:.2f}")
    """

    def __init__(self, carrying_cost_rate: float = CARRYING_COST_RATE):
        self.carrying_cost_rate = carrying_cost_rate

    # -----------------------------------------------------------------
    # SKU-Level Health
    # -----------------------------------------------------------------

    def score_sku(
        self,
        sku: Sku,
        store_id: str,
        category: str = "",
        annual_sales_units: float | None = None,
    ) -> SkuHealth:
        """Calculate health metrics for a single SKU.

        Args:
            sku: SKU data from the pipeline.
            store_id: Store ID for this SKU.
            category: Product category.
            annual_sales_units: Annual unit sales. If None, estimates
                from sales_last_30d × 12.

        Returns:
            SkuHealth with GMROI, turn rate, classification, etc.
        """
        # Estimate annual sales from 30-day data
        if annual_sales_units is None:
            annual_sales_units = sku.sales_last_30d * 12

        annual_cogs = annual_sales_units * sku.unit_cost
        annual_sales_revenue = annual_sales_units * sku.retail_price

        # Average inventory cost (use current on-hand as proxy)
        avg_inventory_cost = max(abs(sku.qty_on_hand) * sku.unit_cost, 0.01)

        # Turn rate = Annual COGS / Avg Inventory Cost
        turn_rate = annual_cogs / avg_inventory_cost if avg_inventory_cost > 0 else 0.0

        # GMROI = (Annual Sales × Gross Margin %) / Avg Inventory Cost
        gmroi = (
            (annual_sales_revenue * sku.margin_pct) / avg_inventory_cost
            if avg_inventory_cost > 0
            else 0.0
        )

        # Classify
        classification = self._classify_turn_rate(turn_rate)

        # Carrying cost
        carrying_cost = abs(sku.qty_on_hand) * sku.unit_cost * self.carrying_cost_rate

        # Days of supply
        daily_sales = annual_sales_units / 365.0 if annual_sales_units > 0 else 0.0
        days_of_supply = (
            abs(sku.qty_on_hand) / daily_sales if daily_sales > 0 else 999.0
        )

        return SkuHealth(
            sku_id=sku.sku_id,
            store_id=store_id,
            category=category or self._guess_category(sku.sku_id),
            qty_on_hand=sku.qty_on_hand,
            unit_cost=sku.unit_cost,
            retail_price=sku.retail_price,
            margin_pct=sku.margin_pct,
            annual_sales_units=annual_sales_units,
            annual_cogs=annual_cogs,
            avg_inventory_cost=avg_inventory_cost,
            turn_rate=turn_rate,
            gmroi=gmroi,
            classification=classification,
            carrying_cost_annual=carrying_cost,
            days_of_supply=days_of_supply,
        )

    def _classify_turn_rate(self, turn_rate: float) -> TurnClassification:
        """Classify turn rate per NHPA benchmarks."""
        if turn_rate > 4.0:
            return TurnClassification.FAST_MOVER
        if turn_rate >= 2.5:
            return TurnClassification.HEALTHY
        if turn_rate >= 1.5:
            return TurnClassification.SLOW_MOVER
        if turn_rate >= 0.5:
            return TurnClassification.WEAK
        return TurnClassification.DEAD

    def _guess_category(self, sku_id: str) -> str:
        """Guess category from SKU prefix."""
        prefix = sku_id.split("-")[0] if "-" in sku_id else sku_id[:3]
        return {
            "PNT": "Paint",
            "ELC": "Electrical",
            "PLB": "Plumbing",
            "HRD": "Hand Tools",
            "FST": "Fasteners",
            "LBR": "Lumber",
            "SEA": "Seasonal",
            "FLR": "General Hardware",
            "TLS": "Hand Tools",
            "SSN": "Seasonal",
            "DMG": "General Hardware",
        }.get(prefix, "Other")

    # -----------------------------------------------------------------
    # Category-Level GMROI
    # -----------------------------------------------------------------

    def analyze_category(
        self,
        sku_healths: list[SkuHealth],
        store_id: str,
        category: str,
    ) -> GMROIAnalysis:
        """Aggregate SKU health data into a category analysis.

        Args:
            sku_healths: List of SkuHealth for this category.
            store_id: Store ID.
            category: Category name.

        Returns:
            GMROIAnalysis with turn rate, GMROI, and classification counts.
        """
        if not sku_healths:
            return GMROIAnalysis(
                store_id=store_id,
                category=category,
                total_inventory_cost=0.0,
                total_annual_cogs=0.0,
                total_annual_sales=0.0,
                gross_margin_pct=0.0,
                turn_rate=0.0,
                gmroi=0.0,
                sku_count=0,
                fast_mover_count=0,
                healthy_count=0,
                slow_mover_count=0,
                weak_count=0,
                dead_count=0,
            )

        total_inventory = sum(s.avg_inventory_cost for s in sku_healths)
        total_cogs = sum(s.annual_cogs for s in sku_healths)
        total_sales = sum(s.annual_sales_units * s.retail_price for s in sku_healths)

        gross_margin_pct = (
            (total_sales - total_cogs) / total_sales if total_sales > 0 else 0.0
        )

        turn_rate = total_cogs / total_inventory if total_inventory > 0 else 0.0
        gmroi = (
            (total_sales * gross_margin_pct) / total_inventory
            if total_inventory > 0
            else 0.0
        )

        # Count classifications
        counts = defaultdict(int)
        for s in sku_healths:
            counts[s.classification] += 1

        return GMROIAnalysis(
            store_id=store_id,
            category=category,
            total_inventory_cost=total_inventory,
            total_annual_cogs=total_cogs,
            total_annual_sales=total_sales,
            gross_margin_pct=gross_margin_pct,
            turn_rate=turn_rate,
            gmroi=gmroi,
            sku_count=len(sku_healths),
            fast_mover_count=counts.get(TurnClassification.FAST_MOVER, 0),
            healthy_count=counts.get(TurnClassification.HEALTHY, 0),
            slow_mover_count=counts.get(TurnClassification.SLOW_MOVER, 0),
            weak_count=counts.get(TurnClassification.WEAK, 0),
            dead_count=counts.get(TurnClassification.DEAD, 0),
        )

    # -----------------------------------------------------------------
    # Full Store Report
    # -----------------------------------------------------------------

    def score_from_digest(
        self,
        digest: Digest,
        store_id: str | None = None,
    ) -> InventoryHealthReport:
        """Generate a full health report from pipeline digest.

        Scores every SKU in the digest's issues, aggregates by category,
        and produces a store-level report.

        Args:
            digest: Pipeline digest with issues and SKU details.
            store_id: Filter to a specific store. If None, uses first store.

        Returns:
            InventoryHealthReport with per-SKU and per-category analysis.
        """
        # Filter issues by store if specified
        issues = digest.issues
        if store_id:
            issues = [i for i in issues if i.store_id == store_id]
        elif issues:
            store_id = issues[0].store_id
        else:
            store_id = store_id or "unknown"

        # Score each SKU from all issues
        sku_healths: list[SkuHealth] = []
        seen_skus: set[str] = set()

        for issue in issues:
            category = self._issue_type_to_category(issue)
            for sku in issue.skus:
                if sku.sku_id not in seen_skus:
                    seen_skus.add(sku.sku_id)
                    health = self.score_sku(sku, store_id, category)
                    sku_healths.append(health)

        # Group by category for analysis
        by_category: dict[str, list[SkuHealth]] = defaultdict(list)
        for sh in sku_healths:
            by_category[sh.category].append(sh)

        category_analyses = [
            self.analyze_category(skus, store_id, cat)
            for cat, skus in sorted(by_category.items())
        ]

        # Store-level aggregates
        total_inventory = sum(sh.avg_inventory_cost for sh in sku_healths)
        dead_stock = [
            sh for sh in sku_healths if sh.classification == TurnClassification.DEAD
        ]
        total_dead_value = sum(sh.inventory_value for sh in dead_stock)

        dead_stock_pct = (
            total_dead_value / total_inventory if total_inventory > 0 else 0.0
        )

        total_carrying_cost = sum(sh.carrying_cost_annual for sh in sku_healths)
        total_cogs = sum(sh.annual_cogs for sh in sku_healths)
        total_sales = sum(sh.annual_sales_units * sh.retail_price for sh in sku_healths)
        overall_turn = total_cogs / total_inventory if total_inventory > 0 else 0.0
        overall_margin = (
            (total_sales - total_cogs) / total_sales if total_sales > 0 else 0.0
        )
        overall_gmroi = (
            (total_sales * overall_margin) / total_inventory
            if total_inventory > 0
            else 0.0
        )

        # Generate alerts
        alerts = self._generate_health_alerts(
            sku_healths,
            category_analyses,
            total_dead_value,
            overall_gmroi,
        )

        return InventoryHealthReport(
            store_id=store_id,
            total_inventory_value=total_inventory,
            total_dead_stock_value=total_dead_value,
            dead_stock_pct=dead_stock_pct,
            annual_carrying_cost=total_carrying_cost,
            overall_turn_rate=overall_turn,
            overall_gmroi=overall_gmroi,
            category_analyses=category_analyses,
            sku_details=sku_healths,
            alerts=[a.title for a in alerts],
        )

    def _issue_type_to_category(self, issue: Issue) -> str:
        """Infer category from issue type and SKU prefix."""
        if issue.skus:
            return self._guess_category(issue.skus[0].sku_id)
        return "Other"

    # -----------------------------------------------------------------
    # Health Alert Generation
    # -----------------------------------------------------------------

    def generate_alerts(
        self,
        report: InventoryHealthReport,
    ) -> list[CoopAlert]:
        """Generate alerts from an inventory health report.

        Args:
            report: Completed InventoryHealthReport.

        Returns:
            List of CoopAlert for dead stock, GMROI warnings, etc.
        """
        return self._generate_health_alerts(
            report.sku_details,
            report.category_analyses,
            report.total_dead_stock_value,
            report.overall_gmroi,
        )

    def _generate_health_alerts(
        self,
        sku_healths: list[SkuHealth],
        category_analyses: list[GMROIAnalysis],
        total_dead_value: float,
        overall_gmroi: float,
    ) -> list[CoopAlert]:
        """Generate health-related co-op alerts."""
        alerts: list[CoopAlert] = []
        store_id = sku_healths[0].store_id if sku_healths else "unknown"

        # Dead stock alert
        if total_dead_value >= 500:
            dead_skus = [
                s for s in sku_healths if s.classification == TurnClassification.DEAD
            ]
            carrying_cost = total_dead_value * self.carrying_cost_rate
            alerts.append(
                CoopAlert(
                    alert_type=CoopAlertType.DEAD_STOCK_ALERT,
                    store_id=store_id,
                    title=f"Dead Stock: ${total_dead_value:,.0f} in {len(dead_skus)} SKUs",
                    dollar_impact=carrying_cost,
                    detail=(
                        f"{len(dead_skus)} SKUs with turn rate below 0.5x "
                        f"holding ${total_dead_value:,.0f} in inventory. "
                        f"Annual carrying cost: ${carrying_cost:,.0f}."
                    ),
                    recommendation=(
                        f"Liquidate dead stock to recover ${total_dead_value:,.0f}. "
                        f"Priority: items with highest carrying cost first. "
                        f"Consider markdown, vendor return, or donation for tax credit."
                    ),
                    confidence=0.90,
                )
            )

        # GMROI warning for underperforming categories
        for analysis in category_analyses:
            if analysis.gmroi < GMROI_MINIMUM and analysis.total_inventory_cost > 500:
                opportunity = analysis.total_inventory_cost * (
                    GMROI_AVERAGE - analysis.gmroi
                )
                alerts.append(
                    CoopAlert(
                        alert_type=CoopAlertType.GMROI_WARNING,
                        store_id=store_id,
                        title=(
                            f"Low GMROI: {analysis.category} "
                            f"at ${analysis.gmroi:.2f} (target: ${GMROI_AVERAGE:.2f})"
                        ),
                        dollar_impact=max(opportunity, 0),
                        detail=(
                            f"{analysis.category}: GMROI of ${analysis.gmroi:.2f} "
                            f"vs industry average ${GMROI_AVERAGE:.2f}. "
                            f"{analysis.dead_count} dead + {analysis.weak_count} weak "
                            f"out of {analysis.sku_count} SKUs."
                        ),
                        recommendation=(
                            f"Review {analysis.category} assortment. "
                            f"Reduce depth on {analysis.dead_count + analysis.weak_count} "
                            f"underperforming SKUs. Target GMROI of ${GMROI_AVERAGE:.2f}+."
                        ),
                        confidence=0.80,
                    )
                )

        # Sort by dollar impact descending
        alerts.sort(key=lambda a: a.dollar_impact, reverse=True)
        return alerts
