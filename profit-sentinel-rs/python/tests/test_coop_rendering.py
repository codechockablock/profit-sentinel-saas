"""Tests for co-op intelligence rendering in llm_layer."""

from datetime import date

import pytest
from sentinel_agent.coop_models import (
    CATEGORY_BENCHMARKS,
    BenchmarkComparison,
    CategoryMixAnalysis,
    CoopAffiliation,
    CoopAlert,
    CoopAlertType,
    CoopIntelligenceReport,
    CoopType,
    GMROIAnalysis,
    InventoryHealthReport,
    RebateTier,
    VendorRebateProgram,
    VendorRebateStatus,
)
from sentinel_agent.llm_layer import (
    render_category_mix_summary,
    render_coop_alert,
    render_coop_report,
    render_inventory_health_summary,
    render_rebate_status,
)


class TestRenderCoopAlert:
    def test_patronage_leakage_alert(self):
        alert = CoopAlert(
            alert_type=CoopAlertType.PATRONAGE_LEAKAGE,
            store_id="store-7",
            title="Patronage Leakage: $18,400 in Paint to ColorMax",
            dollar_impact=2044.0,
            detail="Buying Paint from ColorMax instead of co-op warehouse.",
            recommendation="Shift $18,400 to co-op warehouse. Annual gain: $2,044.",
        )
        text = render_coop_alert(alert)
        assert "!!" in text  # Patronage leakage icon
        assert "$2,044" in text
        assert "Shift" in text
        assert ">>" in text  # Recommendation prefix

    def test_dead_stock_alert(self):
        alert = CoopAlert(
            alert_type=CoopAlertType.DEAD_STOCK_ALERT,
            store_id="store-7",
            title="Dead Stock: $15,000 in 12 SKUs",
            dollar_impact=4050.0,
            detail="12 SKUs with turn rate below 0.5x.",
            recommendation="Liquidate dead stock.",
        )
        text = render_coop_alert(alert)
        assert "!!" in text
        assert "$4,050" in text
        assert "Dead Stock" in text

    def test_gmroi_warning(self):
        alert = CoopAlert(
            alert_type=CoopAlertType.GMROI_WARNING,
            store_id="store-7",
            title="Low GMROI: Paint at $0.80",
            dollar_impact=5000.0,
            detail="Paint: GMROI below target.",
            recommendation="Review assortment.",
        )
        text = render_coop_alert(alert)
        assert "! " in text  # GMROI warning icon
        assert "GMROI" in text


class TestRenderCoopReport:
    def test_full_report(self):
        alerts = [
            CoopAlert(
                alert_type=CoopAlertType.PATRONAGE_LEAKAGE,
                store_id="store-7",
                title="Leakage Alert",
                dollar_impact=2000,
                detail="Detail text",
                recommendation="Action item",
            ),
            CoopAlert(
                alert_type=CoopAlertType.DEAD_STOCK_ALERT,
                store_id="store-7",
                title="Dead Stock Alert",
                dollar_impact=1000,
                detail="Detail text",
                recommendation="Action item",
            ),
        ]
        report = CoopIntelligenceReport(
            store_id="store-7",
            alerts=alerts,
            total_opportunity=3000,
        )
        text = render_coop_report(report)
        assert "CO-OP INTELLIGENCE REPORT" in text
        assert "Store 7" in text
        assert "$3,000" in text
        assert "Leakage Alert" in text
        assert "Dead Stock Alert" in text

    def test_empty_report(self):
        report = CoopIntelligenceReport(
            store_id="store-7",
            total_opportunity=0,
        )
        text = render_coop_report(report)
        assert "No co-op optimization alerts" in text


class TestRenderInventoryHealth:
    def test_health_summary(self):
        report = InventoryHealthReport(
            store_id="store-7",
            total_inventory_value=100000,
            total_dead_stock_value=15000,
            dead_stock_pct=0.15,
            annual_carrying_cost=4050,
            overall_turn_rate=2.5,
            overall_gmroi=1.80,
            category_analyses=[
                GMROIAnalysis(
                    store_id="store-7",
                    category="Paint",
                    total_inventory_cost=20000,
                    total_annual_cogs=60000,
                    total_annual_sales=100000,
                    gross_margin_pct=0.40,
                    turn_rate=3.0,
                    gmroi=2.0,
                    sku_count=15,
                    fast_mover_count=5,
                    healthy_count=5,
                    slow_mover_count=3,
                    weak_count=1,
                    dead_count=1,
                ),
            ],
        )
        text = render_inventory_health_summary(report)
        assert "INVENTORY HEALTH" in text
        assert "Store 7" in text
        assert "$100,000" in text
        assert "GMROI: 1.80" in text
        assert "2.5x" in text
        assert "$15,000" in text
        assert "15%" in text
        assert "Paint" in text

    def test_no_problem_categories(self):
        report = InventoryHealthReport(
            store_id="store-7",
            total_inventory_value=50000,
            total_dead_stock_value=0,
            dead_stock_pct=0,
            annual_carrying_cost=0,
            overall_turn_rate=3.0,
            overall_gmroi=2.5,
        )
        text = render_inventory_health_summary(report)
        assert "INVENTORY HEALTH" in text
        # No "Categories needing attention" since no analyses with issues
        assert "Categories needing attention" not in text


class TestRenderRebateStatus:
    def _make_status(
        self,
        current_tier_name: str = "Bronze",
        current_rate: float = 0.02,
        next_tier_name: str = "Silver",
        next_threshold: float = 25000,
        next_rate: float = 0.035,
        ytd: float = 15000,
        on_track: bool = True,
    ) -> VendorRebateStatus:
        program = VendorRebateProgram(
            vendor_id="DMG",
            vendor_name="Martin's Supply Co.",
            program_name="Volume Incentive",
            program_type="volume",
            tiers=[
                RebateTier(
                    tier_name=current_tier_name,
                    threshold=10000,
                    rebate_pct=current_rate,
                ),
                RebateTier(
                    tier_name=next_tier_name,
                    threshold=next_threshold,
                    rebate_pct=next_rate,
                ),
            ],
            period_start=date(2026, 1, 1),
            period_end=date(2026, 12, 31),
        )
        return VendorRebateStatus(
            program=program,
            store_id="store-7",
            ytd_purchases=ytd,
            current_tier=program.tiers[0],
            next_tier=program.tiers[1],
            shortfall=next_threshold - ytd,
            days_remaining=200,
            daily_run_rate=ytd / 165,
            projected_total=ytd + (ytd / 165 * 200),
            on_track=on_track,
            current_rebate_value=ytd * current_rate,
            next_tier_rebate_value=next_threshold * next_rate,
            incremental_value=next_threshold * next_rate - ytd * current_rate,
            recommendation="On track to hit Silver tier." if on_track else "At risk.",
        )

    def test_renders_vendor_name(self):
        status = self._make_status()
        text = render_rebate_status(status)
        assert "Martin's Supply" in text

    def test_renders_current_tier(self):
        status = self._make_status()
        text = render_rebate_status(status)
        assert "Bronze" in text
        assert "2.0%" in text

    def test_renders_progress(self):
        status = self._make_status(ytd=15000, next_threshold=25000)
        text = render_rebate_status(status)
        assert "$15,000" in text
        assert "$25,000" in text
        assert "60%" in text  # 15000/25000

    def test_renders_recommendation(self):
        status = self._make_status(on_track=True)
        text = render_rebate_status(status)
        assert "On track" in text


class TestRenderCategoryMix:
    def test_renders_summary(self):
        comparisons = [
            BenchmarkComparison(
                category="Paint",
                store_pct=0.05,
                benchmark_pct=0.12,
                store_margin=0.42,
                benchmark_margin=0.45,
                store_revenue=50000,
                gap_pct=-0.07,
                margin_gap=-0.03,
                dollar_opportunity=31500,
                recommendation="Expand Paint",
            ),
        ]
        analysis = CategoryMixAnalysis(
            store_id="store-7",
            total_revenue=1000000,
            total_margin_pct=0.35,
            comparisons=comparisons,
            total_opportunity=31500,
            top_expansion_categories=["Paint"],
            top_contraction_categories=[],
        )
        text = render_category_mix_summary(analysis)
        assert "CATEGORY MIX ANALYSIS" in text
        assert "Store 7" in text
        assert "$1,000,000" in text
        assert "35%" in text
        assert "Paint" in text
        assert "Expand" in text

    def test_shows_contraction(self):
        comparisons = [
            BenchmarkComparison(
                category="Lumber",
                store_pct=0.25,
                benchmark_pct=0.15,
                store_margin=0.22,
                benchmark_margin=0.25,
                store_revenue=250000,
                gap_pct=0.10,
                margin_gap=-0.03,
                dollar_opportunity=0,
                recommendation="Monitor Lumber",
            ),
        ]
        analysis = CategoryMixAnalysis(
            store_id="store-7",
            total_revenue=1000000,
            total_margin_pct=0.30,
            comparisons=comparisons,
            total_opportunity=0,
            top_expansion_categories=[],
            top_contraction_categories=["Lumber"],
        )
        text = render_category_mix_summary(analysis)
        assert "Monitor" in text
        assert "Lumber" in text
