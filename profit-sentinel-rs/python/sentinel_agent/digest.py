"""Morning Digest Generator.

Orchestrates the Rust pipeline call and transforms structured output
into a natural language digest suitable for an executive's morning review.

Now includes Co-op Intelligence: patronage leakage, inventory health,
vendor rebate tracking, and category mix analysis — all quantified
in dollars and integrated into the morning briefing.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .category_mix import CategoryMixOptimizer
from .coop_intelligence import CoopIntelligence
from .coop_models import (
    CoopAffiliation,
    CoopAlert,
    CoopIntelligenceReport,
    CoopType,
    VendorPurchase,
)
from .engine import SentinelEngine
from .inventory_health import InventoryHealthScorer
from .llm_layer import (
    render_coop_report,
    render_digest,
    render_inventory_health_summary,
)
from .models import Digest
from .vendor_rebates import VendorRebateTracker


class MorningDigestGenerator:
    """Generate a morning digest from inventory data.

    Usage:
        gen = MorningDigestGenerator()
        digest = gen.generate("fixtures/sample_inventory.csv", stores=["store-7"])
        print(gen.render(digest))

    With Co-op Intelligence:
        gen = MorningDigestGenerator()
        digest = gen.generate("inventory.csv", stores=["store-7"])
        coop_report = gen.generate_coop_report(
            digest, store_id="store-7",
            purchases=[...],  # vendor purchase records
            vendor_ytd={"DMG": 18500, "PNT": 12000},
        )
        print(gen.render_full(digest, coop_report))
    """

    def __init__(self, engine: SentinelEngine | None = None):
        self._engine = engine or SentinelEngine()
        self._health_scorer = InventoryHealthScorer()
        self._rebate_tracker = VendorRebateTracker()
        self._mix_optimizer = CategoryMixOptimizer()

    def generate(
        self,
        csv_path: str | Path,
        stores: list[str] | None = None,
        top_k: int = 5,
    ) -> Digest:
        """Run the pipeline and return a typed Digest.

        Args:
            csv_path: Path to inventory CSV file.
            stores: Optional store ID filter.
            top_k: Number of top issues to surface.

        Returns:
            Digest model with issues, SKU details, and summary.
        """
        return self._engine.run(csv_path, stores=stores, top_k=top_k)

    def render(self, digest: Digest) -> str:
        """Render a Digest into natural language text.

        Args:
            digest: Digest model from generate().

        Returns:
            Formatted string suitable for display or messaging.
        """
        return render_digest(digest)

    def generate_and_render(
        self,
        csv_path: str | Path,
        stores: list[str] | None = None,
        top_k: int = 5,
    ) -> str:
        """Run pipeline and render in one step.

        Convenience method for simple use cases.
        """
        digest = self.generate(csv_path, stores=stores, top_k=top_k)
        return self.render(digest)

    # -----------------------------------------------------------------
    # Co-op Intelligence Integration
    # -----------------------------------------------------------------

    def generate_coop_report(
        self,
        digest: Digest,
        store_id: str,
        affiliation: CoopAffiliation | None = None,
        purchases: list[VendorPurchase] | None = None,
        vendor_ytd: dict[str, float] | None = None,
        category_revenue: dict[str, float] | None = None,
        category_margins: dict[str, float] | None = None,
        as_of: date | None = None,
    ) -> CoopIntelligenceReport:
        """Generate a full co-op intelligence report for a store.

        Combines all four analysis engines:
        1. Inventory health (GMROI, turn rates, dead stock)
        2. Patronage leakage (non-co-op vendor detection)
        3. Vendor rebate tracking (threshold risk alerts)
        4. Category mix optimization (NHPA benchmark comparison)

        Args:
            digest: Pipeline digest (for inventory health scoring).
            store_id: Store ID to analyze.
            affiliation: Co-op affiliation. Defaults to Do It Best.
            purchases: Vendor purchase records for leakage detection.
            vendor_ytd: Dict of vendor_id → YTD purchases for rebates.
            category_revenue: Dict of category → annual revenue for mix.
            category_margins: Dict of category → margin % for mix.
            as_of: Date for rebate projections. Defaults to today.

        Returns:
            CoopIntelligenceReport with all alerts and analyses.
        """
        if affiliation is None:
            affiliation = CoopAffiliation(
                store_id=store_id,
                coop_type=CoopType.DO_IT_BEST,
            )

        all_alerts: list[CoopAlert] = []

        # 1. Inventory health
        health_report = self._health_scorer.score_from_digest(digest, store_id)
        health_alerts = self._health_scorer.generate_alerts(health_report)
        all_alerts.extend(health_alerts)

        # 2. Patronage leakage
        if purchases:
            coop = CoopIntelligence(affiliation)
            coop_alerts = coop.generate_alerts(purchases)
            all_alerts.extend(coop_alerts)

        # 3. Vendor rebate tracking
        rebate_statuses = []
        if vendor_ytd:
            rebate_statuses = self._rebate_tracker.evaluate_all(
                store_id,
                vendor_ytd,
                as_of,
            )
            rebate_alerts = self._rebate_tracker.generate_alerts(rebate_statuses)
            all_alerts.extend(rebate_alerts)

        # 4. Category mix
        category_analysis = None
        if category_revenue:
            category_analysis = self._mix_optimizer.analyze(
                store_id,
                category_revenue,
                category_margins,
            )
            mix_alerts = self._mix_optimizer.generate_alerts(category_analysis)
            all_alerts.extend(mix_alerts)

        # Sort all alerts by dollar impact
        all_alerts.sort(key=lambda a: a.dollar_impact, reverse=True)

        total_opportunity = sum(a.dollar_impact for a in all_alerts)

        return CoopIntelligenceReport(
            store_id=store_id,
            affiliation=affiliation,
            alerts=all_alerts,
            health_report=health_report,
            rebate_statuses=rebate_statuses,
            category_analysis=category_analysis,
            total_opportunity=total_opportunity,
        )

    def render_coop(self, report: CoopIntelligenceReport) -> str:
        """Render a co-op intelligence report as natural language text."""
        return render_coop_report(report)

    def render_full(
        self,
        digest: Digest,
        coop_report: CoopIntelligenceReport | None = None,
    ) -> str:
        """Render both pipeline digest and co-op intelligence.

        The combined view an executive sees at 6 AM:
        1. Pipeline issues (anomalies, problems)
        2. Co-op optimization opportunities (money on the table)
        """
        parts: list[str] = []

        # Pipeline digest
        parts.append(render_digest(digest))

        # Co-op intelligence section
        if coop_report and coop_report.alerts:
            parts.append("")
            parts.append("=" * 50)
            parts.append("")
            parts.append(render_coop_report(coop_report))

            # Inventory health summary
            if coop_report.health_report:
                parts.append(render_inventory_health_summary(coop_report.health_report))

        return "\n".join(parts)
