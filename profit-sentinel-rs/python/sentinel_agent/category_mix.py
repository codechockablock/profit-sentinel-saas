"""Category Mix Optimizer.

Compares a store's category revenue mix against NHPA high-profit
benchmarks and quantifies the dollar opportunity of rebalancing.

Example output:
  "Paint is 7% of revenue vs 12% benchmark. Expanding Paint by
   $50,000 at 45% margin adds $22,500 in gross profit. Priority: High."

Uses NHPA 2025 Cost of Doing Business benchmarks:
- Typical store: 2.9% pre-tax profit
- High-profit store: 7.1% pre-tax profit
- Key difference: high-profit stores over-index on high-margin categories
  (Paint, Electrical, Services) and manage Lumber for volume, not margin.
"""

from __future__ import annotations

from .coop_models import (
    CATEGORY_BENCHMARKS,
    BenchmarkComparison,
    CategoryMixAnalysis,
    CoopAlert,
    CoopAlertType,
)


class CategoryMixOptimizer:
    """Compare store category mix to NHPA high-profit benchmarks.

    Usage:
        optimizer = CategoryMixOptimizer()
        analysis = optimizer.analyze(
            store_id="store-7",
            category_revenue={"Paint": 84000, "Electrical": 77000, ...},
            category_margins={"Paint": 0.42, "Electrical": 0.38, ...},
        )
        print(f"Total opportunity: {analysis.opportunity_display}")
    """

    def __init__(
        self,
        benchmarks: dict[str, dict] | None = None,
    ):
        self.benchmarks = benchmarks or CATEGORY_BENCHMARKS

    def analyze(
        self,
        store_id: str,
        category_revenue: dict[str, float],
        category_margins: dict[str, float] | None = None,
    ) -> CategoryMixAnalysis:
        """Analyze category mix vs NHPA benchmarks.

        Args:
            store_id: Store ID.
            category_revenue: Dict of category → annual revenue.
            category_margins: Optional dict of category → gross margin %.
                If not provided, uses benchmark margins for comparison only.

        Returns:
            CategoryMixAnalysis with per-category comparisons and total opportunity.
        """
        if category_margins is None:
            category_margins = {}

        total_revenue = sum(category_revenue.values())
        if total_revenue == 0:
            return CategoryMixAnalysis(
                store_id=store_id,
                total_revenue=0,
                total_margin_pct=0,
                comparisons=[],
                total_opportunity=0,
                top_expansion_categories=[],
                top_contraction_categories=[],
            )

        # Build comparison for each benchmark category
        comparisons: list[BenchmarkComparison] = []
        total_margin_dollars = 0.0

        for cat_name, benchmark in self.benchmarks.items():
            store_revenue = category_revenue.get(cat_name, 0.0)
            store_pct = store_revenue / total_revenue if total_revenue > 0 else 0.0
            benchmark_pct = benchmark["target_pct"]
            benchmark_margin = benchmark["target_margin"]
            store_margin = category_margins.get(cat_name, benchmark_margin)

            gap_pct = store_pct - benchmark_pct
            margin_gap = store_margin - benchmark_margin

            # Dollar opportunity: revenue shift needed × margin impact
            dollar_opportunity = self._calculate_opportunity(
                store_pct,
                benchmark_pct,
                store_margin,
                benchmark_margin,
                total_revenue,
                cat_name,
            )

            total_margin_dollars += store_revenue * store_margin

            recommendation = self._build_recommendation(
                cat_name,
                gap_pct,
                margin_gap,
                dollar_opportunity,
                benchmark["priority"],
                store_revenue,
                total_revenue,
            )

            comparisons.append(
                BenchmarkComparison(
                    category=cat_name,
                    store_pct=store_pct,
                    benchmark_pct=benchmark_pct,
                    store_margin=store_margin,
                    benchmark_margin=benchmark_margin,
                    store_revenue=store_revenue,
                    gap_pct=gap_pct,
                    margin_gap=margin_gap,
                    dollar_opportunity=dollar_opportunity,
                    recommendation=recommendation,
                )
            )

        # Sort by absolute opportunity descending
        comparisons.sort(key=lambda c: abs(c.dollar_opportunity), reverse=True)

        total_margin_pct = (
            total_margin_dollars / total_revenue if total_revenue > 0 else 0.0
        )

        total_opportunity = sum(
            c.dollar_opportunity for c in comparisons if c.dollar_opportunity > 0
        )

        # Identify top expansion and contraction categories
        expansion = [
            c.category
            for c in comparisons
            if c.gap_pct < -0.01 and c.dollar_opportunity > 0
        ][:3]

        contraction = [
            c.category for c in comparisons if c.gap_pct > 0.02  # >2% over-indexed
        ][:3]

        return CategoryMixAnalysis(
            store_id=store_id,
            total_revenue=total_revenue,
            total_margin_pct=total_margin_pct,
            comparisons=comparisons,
            total_opportunity=total_opportunity,
            top_expansion_categories=expansion,
            top_contraction_categories=contraction,
        )

    def _calculate_opportunity(
        self,
        store_pct: float,
        benchmark_pct: float,
        store_margin: float,
        benchmark_margin: float,
        total_revenue: float,
        category: str,
    ) -> float:
        """Calculate the dollar opportunity of closing the gap.

        For under-indexed high-margin categories:
          opportunity = revenue_gap × target_margin

        For over-indexed low-margin categories:
          opportunity = excess_revenue × margin_improvement_if_shifted
        """
        gap_pct = store_pct - benchmark_pct

        if gap_pct < -0.01:
            # Under-indexed: opportunity to expand
            revenue_gap = abs(gap_pct) * total_revenue
            # Use the higher of store or benchmark margin
            margin = max(store_margin, benchmark_margin)
            return revenue_gap * margin

        elif gap_pct > 0.02:
            # Over-indexed: could shift revenue to higher-margin categories
            # Only flag if this category has below-average margins
            if store_margin < 0.35:  # below average
                excess_revenue = (gap_pct - 0.01) * total_revenue
                margin_improvement = 0.35 - store_margin  # shift to avg margin
                return excess_revenue * margin_improvement

        return 0.0

    def _build_recommendation(
        self,
        category: str,
        gap_pct: float,
        margin_gap: float,
        dollar_opportunity: float,
        priority: str,
        store_revenue: float,
        total_revenue: float,
    ) -> str:
        """Build a specific recommendation for a category."""
        if abs(gap_pct) < 0.01:
            # Within 1% of benchmark
            if margin_gap < -0.03:
                return (
                    f"{category} mix is on target but margin is "
                    f"{abs(margin_gap) * 100:.0f}pp below benchmark. "
                    f"Focus on vendor negotiations and retail pricing."
                )
            return f"{category} is well-positioned. Maintain current approach."

        if gap_pct < -0.01:
            # Under-indexed
            revenue_needed = abs(gap_pct) * total_revenue
            return (
                f"Expand {category} by ${revenue_needed:,.0f} to reach "
                f"{self.benchmarks[category]['target_pct'] * 100:.0f}% "
                f"benchmark. At {self.benchmarks[category]['target_margin'] * 100:.0f}% "
                f"margin, adds ${dollar_opportunity:,.0f} gross profit. "
                f"Priority: {priority}."
            )

        if gap_pct > 0.02:
            # Over-indexed
            excess = gap_pct * total_revenue
            if dollar_opportunity > 0:
                return (
                    f"{category} is over-indexed by "
                    f"${excess:,.0f}. Consider shifting "
                    f"some volume to higher-margin categories. "
                    f"Potential margin improvement: ${dollar_opportunity:,.0f}."
                )
            return (
                f"{category} is over-indexed at "
                f"{(gap_pct + self.benchmarks[category]['target_pct']) * 100:.0f}% "
                f"vs {self.benchmarks[category]['target_pct'] * 100:.0f}% target. "
                f"Margin is acceptable. Monitor for balance."
            )

        return f"{category}: minor variance from benchmark. No action needed."

    # -----------------------------------------------------------------
    # Alert Generation
    # -----------------------------------------------------------------

    def generate_alerts(
        self,
        analysis: CategoryMixAnalysis,
    ) -> list[CoopAlert]:
        """Generate mix imbalance alerts.

        Args:
            analysis: CategoryMixAnalysis from analyze().

        Returns:
            List of CoopAlert for significant mix imbalances.
        """
        alerts: list[CoopAlert] = []

        for comparison in analysis.comparisons:
            # Only alert on significant under-indexed high-priority categories
            if comparison.is_under_indexed and comparison.dollar_opportunity >= 500:
                priority = self.benchmarks.get(
                    comparison.category,
                    {},
                ).get("priority", "Low")

                # Only alert on High or Medium priority categories
                if priority not in ("High", "Medium"):
                    continue

                alerts.append(
                    CoopAlert(
                        alert_type=CoopAlertType.MIX_IMBALANCE,
                        store_id=analysis.store_id,
                        title=(
                            f"Under-indexed: {comparison.category} "
                            f"at {comparison.store_pct * 100:.0f}% "
                            f"vs {comparison.benchmark_pct * 100:.0f}% target"
                        ),
                        dollar_impact=comparison.dollar_opportunity,
                        detail=(
                            f"{comparison.category}: "
                            f"{comparison.store_pct * 100:.1f}% of revenue vs "
                            f"{comparison.benchmark_pct * 100:.0f}% NHPA target. "
                            f"Gap: ${abs(comparison.gap_pct) * analysis.total_revenue:,.0f} "
                            f"in revenue."
                        ),
                        recommendation=comparison.recommendation,
                        confidence=0.70,
                    )
                )

        # Sort by dollar impact descending
        alerts.sort(key=lambda a: a.dollar_impact, reverse=True)
        return alerts
