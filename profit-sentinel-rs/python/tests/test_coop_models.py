"""Tests for Co-op Intelligence Pydantic models."""

import pytest
from sentinel_agent.coop_models import (
    CATEGORY_BENCHMARKS,
    PATRONAGE_RATES,
    WAREHOUSE_CASH_DISCOUNT,
    BenchmarkComparison,
    CategoryMixAnalysis,
    ConsolidationOpportunity,
    CoopAffiliation,
    CoopAlert,
    CoopAlertType,
    CoopIntelligenceReport,
    CoopType,
    GMROIAnalysis,
    InventoryHealthReport,
    PatronageCategory,
    PatronageLeakage,
    PatronageProgram,
    RebateTier,
    SkuHealth,
    TurnClassification,
    VendorPurchase,
    VendorRebateProgram,
    VendorRebateStatus,
)


class TestCoopType:
    def test_all_three_coops(self):
        assert CoopType.DO_IT_BEST.value == "DoItBest"
        assert CoopType.ACE.value == "Ace"
        assert CoopType.ORGILL.value == "Orgill"


class TestPatronageRates:
    def test_warehouse_rate(self):
        assert PATRONAGE_RATES[PatronageCategory.REGULAR_WAREHOUSE] == 0.1111

    def test_promotional_rate(self):
        assert PATRONAGE_RATES[PatronageCategory.PROMOTIONAL_WAREHOUSE] == 0.0584

    def test_direct_ship_rate(self):
        assert PATRONAGE_RATES[PatronageCategory.DIRECT_SHIP] == 0.0106

    def test_lumber_rate(self):
        assert PATRONAGE_RATES[PatronageCategory.LUMBER] == 0.0074

    def test_cash_discount(self):
        assert WAREHOUSE_CASH_DISCOUNT == 0.02

    def test_warehouse_is_highest(self):
        rates = list(PATRONAGE_RATES.values())
        assert max(rates) == 0.1111


class TestPatronageLeakage:
    def test_rebate_differential(self):
        leakage = PatronageLeakage(
            store_id="store-7",
            vendor_name="Test Vendor",
            category="Paint",
            non_coop_spend=10000,
            current_rebate_rate=0.02,
            coop_rebate_rate=0.1111,
            annual_leakage=911.0,
        )
        assert abs(leakage.rebate_differential - 0.0911) < 0.001

    def test_dollar_display(self):
        leakage = PatronageLeakage(
            store_id="store-7",
            vendor_name="Test",
            category="Paint",
            non_coop_spend=5000,
            annual_leakage=1234.0,
        )
        assert leakage.dollar_display == "$1,234"


class TestConsolidationOpportunity:
    def test_benefit_display(self):
        opp = ConsolidationOpportunity(
            store_id="store-7",
            category="Paint",
            current_vendor_count=3,
            vendors=["A", "B", "C"],
            total_category_spend=50000,
            shiftable_spend=20000,
            coop_rebate_rate=0.1111,
            annual_benefit=2622.0,
        )
        assert opp.benefit_display == "$2,622"


class TestTurnClassification:
    def test_fast_mover_action(self):
        assert "stockout" in TurnClassification.FAST_MOVER.action.lower()

    def test_dead_action(self):
        assert "liquidate" in TurnClassification.DEAD.action.lower()

    def test_all_have_labels(self):
        for tc in TurnClassification:
            assert tc.label
            assert tc.action

    def test_labels(self):
        assert TurnClassification.FAST_MOVER.label == "Fast Mover"
        assert TurnClassification.DEAD.label == "Dead Stock"


class TestGMROIAnalysis:
    def test_is_profitable(self):
        analysis = GMROIAnalysis(
            store_id="store-7",
            category="Paint",
            total_inventory_cost=10000,
            total_annual_cogs=30000,
            total_annual_sales=50000,
            gross_margin_pct=0.40,
            turn_rate=3.0,
            gmroi=2.0,
            sku_count=10,
            fast_mover_count=3,
            healthy_count=4,
            slow_mover_count=2,
            weak_count=1,
            dead_count=0,
        )
        assert analysis.is_profitable

    def test_not_profitable(self):
        analysis = GMROIAnalysis(
            store_id="store-7",
            category="Other",
            total_inventory_cost=10000,
            total_annual_cogs=5000,
            total_annual_sales=6000,
            gross_margin_pct=0.17,
            turn_rate=0.5,
            gmroi=0.6,
            sku_count=5,
            fast_mover_count=0,
            healthy_count=0,
            slow_mover_count=1,
            weak_count=2,
            dead_count=2,
        )
        assert not analysis.is_profitable

    def test_performance_labels(self):
        # High profit
        a = GMROIAnalysis(
            store_id="s",
            category="c",
            total_inventory_cost=0,
            total_annual_cogs=0,
            total_annual_sales=0,
            gross_margin_pct=0,
            turn_rate=0,
            gmroi=3.0,
            sku_count=0,
            fast_mover_count=0,
            healthy_count=0,
            slow_mover_count=0,
            weak_count=0,
            dead_count=0,
        )
        assert a.performance_label == "High-Profit"

        # Average
        a.gmroi = 2.0
        # Need to create new instance since Pydantic models are immutable by default
        a2 = a.model_copy(update={"gmroi": 2.0})
        assert a2.performance_label == "Average"

        # Below average
        a3 = a.model_copy(update={"gmroi": 1.0})
        assert a3.performance_label == "Below Average"


class TestInventoryHealthReport:
    def test_dead_stock_display(self):
        report = InventoryHealthReport(
            store_id="store-7",
            total_inventory_value=100000,
            total_dead_stock_value=15000,
            dead_stock_pct=0.15,
            annual_carrying_cost=4050,
            overall_turn_rate=2.5,
            overall_gmroi=1.8,
        )
        assert report.dead_stock_display == "$15,000"
        assert report.carrying_cost_display == "$4,050"


class TestCoopAlertType:
    def test_all_have_display_names(self):
        for at in CoopAlertType:
            assert at.display_name
            assert at.icon is not None

    def test_specific_display_names(self):
        assert CoopAlertType.PATRONAGE_LEAKAGE.display_name == "Patronage Leakage"
        assert CoopAlertType.GMROI_WARNING.display_name == "GMROI Warning"


class TestCoopAlert:
    def test_dollar_display(self):
        alert = CoopAlert(
            alert_type=CoopAlertType.PATRONAGE_LEAKAGE,
            store_id="store-7",
            title="Test Alert",
            dollar_impact=2500.0,
            detail="Detail text",
            recommendation="Do something",
        )
        assert alert.dollar_display == "$2,500"


class TestCoopIntelligenceReport:
    def test_opportunity_display(self):
        report = CoopIntelligenceReport(
            store_id="store-7",
            total_opportunity=12345.0,
        )
        assert report.opportunity_display == "$12,345"
        assert report.alert_count == 0

    def test_alert_count(self):
        alert = CoopAlert(
            alert_type=CoopAlertType.DEAD_STOCK_ALERT,
            store_id="store-7",
            title="Test",
            dollar_impact=1000,
            detail="Detail",
            recommendation="Action",
        )
        report = CoopIntelligenceReport(
            store_id="store-7",
            alerts=[alert, alert],
            total_opportunity=2000,
        )
        assert report.alert_count == 2


class TestCategoryBenchmarks:
    def test_paint_benchmark(self):
        assert CATEGORY_BENCHMARKS["Paint"]["target_pct"] == 0.12
        assert CATEGORY_BENCHMARKS["Paint"]["target_margin"] == 0.45

    def test_services_is_high_priority(self):
        assert CATEGORY_BENCHMARKS["Services"]["priority"] == "High"

    def test_all_categories_sum_to_one(self):
        total = sum(b["target_pct"] for b in CATEGORY_BENCHMARKS.values())
        assert abs(total - 1.0) < 0.01

    def test_benchmark_count(self):
        assert len(CATEGORY_BENCHMARKS) == 10


class TestVendorRebateStatus:
    def test_is_at_risk(self):
        program = VendorRebateProgram(
            vendor_id="DMG",
            vendor_name="Martin's",
            program_name="Test",
            program_type="volume",
            tiers=[RebateTier(tier_name="Gold", threshold=50000, rebate_pct=0.05)],
            period_start="2026-01-01",
            period_end="2026-12-31",
        )
        status = VendorRebateStatus(
            program=program,
            store_id="store-7",
            ytd_purchases=20000,
            current_tier=None,
            next_tier=program.tiers[0],
            shortfall=30000,
            days_remaining=180,
            daily_run_rate=100,
            projected_total=38000,
            on_track=False,
            current_rebate_value=0,
            next_tier_rebate_value=2500,
            incremental_value=2500,
        )
        assert status.is_at_risk
        assert status.shortfall_display == "$30,000"
        assert status.incremental_display == "$2,500"


class TestBenchmarkComparison:
    def test_is_under_indexed(self):
        comp = BenchmarkComparison(
            category="Paint",
            store_pct=0.07,
            benchmark_pct=0.12,
            store_margin=0.42,
            benchmark_margin=0.45,
            store_revenue=70000,
            gap_pct=-0.05,
            margin_gap=-0.03,
            dollar_opportunity=22500,
            recommendation="Expand Paint",
        )
        assert comp.is_under_indexed
        assert comp.opportunity_display == "$22,500"

    def test_not_under_indexed(self):
        comp = BenchmarkComparison(
            category="Lumber",
            store_pct=0.16,
            benchmark_pct=0.15,
            store_margin=0.25,
            benchmark_margin=0.25,
            store_revenue=160000,
            gap_pct=0.01,
            margin_gap=0.0,
            dollar_opportunity=0,
            recommendation="On target",
        )
        assert not comp.is_under_indexed
