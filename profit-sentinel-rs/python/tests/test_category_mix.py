"""Tests for the CategoryMixOptimizer."""

import pytest
from sentinel_agent.category_mix import CategoryMixOptimizer
from sentinel_agent.coop_models import CATEGORY_BENCHMARKS, CoopAlertType


class TestCategoryAnalysis:
    def setup_method(self):
        self.optimizer = CategoryMixOptimizer()

    def test_empty_revenue(self):
        analysis = self.optimizer.analyze("store-7", {})
        assert analysis.total_revenue == 0
        assert analysis.total_opportunity == 0
        assert len(analysis.comparisons) == 0

    def test_on_target_store(self):
        """A store that matches benchmarks exactly."""
        total = 1_000_000
        revenue = {
            cat: benchmarks["target_pct"] * total
            for cat, benchmarks in CATEGORY_BENCHMARKS.items()
        }
        analysis = self.optimizer.analyze("store-7", revenue)
        assert analysis.total_revenue == total
        # Should have minimal opportunities since on target
        # (some float precision issues might create tiny opportunities)
        for comp in analysis.comparisons:
            assert abs(comp.gap_pct) < 0.01

    def test_detects_under_indexed_category(self):
        """Paint at 5% vs 12% benchmark."""
        revenue = {
            "Paint": 50000,
            "Electrical": 110000,
            "Plumbing": 90000,
            "Hand Tools": 80000,
            "Fasteners": 60000,
            "Lumber": 150000,
            "Seasonal": 100000,
            "Services": 50000,
            "General Hardware": 100000,
            "Other": 210000,
        }
        total = sum(revenue.values())
        assert total == 1_000_000

        analysis = self.optimizer.analyze("store-7", revenue)

        # Find Paint comparison
        paint = next(c for c in analysis.comparisons if c.category == "Paint")
        assert paint.store_pct == 0.05  # 50000/1000000
        assert paint.benchmark_pct == 0.12
        assert paint.gap_pct < 0  # Under-indexed
        assert paint.is_under_indexed
        assert paint.dollar_opportunity > 0

    def test_detects_over_indexed_category(self):
        """Other at 30% vs 14% benchmark."""
        revenue = {
            "Paint": 120000,
            "Electrical": 110000,
            "Plumbing": 90000,
            "Hand Tools": 80000,
            "Fasteners": 60000,
            "Lumber": 150000,
            "Seasonal": 100000,
            "Services": 50000,
            "General Hardware": 100000,
            "Other": 300000,
        }
        sum(revenue.values())

        # Give Other a low margin to trigger opportunity
        margins = {"Other": 0.20}  # Below 0.35 threshold

        analysis = self.optimizer.analyze("store-7", revenue, margins)

        other_comp = next(c for c in analysis.comparisons if c.category == "Other")
        assert other_comp.gap_pct > 0.02  # Over-indexed

    def test_expansion_categories_identified(self):
        revenue = {
            "Paint": 30000,  # 3% vs 12% — big gap
            "Electrical": 50000,  # 5% vs 11% — big gap
            "Plumbing": 90000,
            "Hand Tools": 80000,
            "Fasteners": 60000,
            "Lumber": 200000,
            "Seasonal": 100000,
            "Services": 20000,  # 2% vs 5% — big gap
            "General Hardware": 150000,
            "Other": 220000,
        }
        analysis = self.optimizer.analyze("store-7", revenue)

        assert len(analysis.top_expansion_categories) >= 2
        assert "Paint" in analysis.top_expansion_categories

    def test_total_opportunity_positive(self):
        revenue = {
            "Paint": 30000,
            "Electrical": 30000,
            "Plumbing": 30000,
            "Hand Tools": 30000,
            "Fasteners": 30000,
            "Lumber": 300000,
            "Seasonal": 100000,
            "Services": 10000,
            "General Hardware": 100000,
            "Other": 340000,
        }
        analysis = self.optimizer.analyze("store-7", revenue)
        assert analysis.total_opportunity > 0
        assert analysis.opportunity_display.startswith("$")

    def test_margin_comparison(self):
        revenue = {"Paint": 120000, "Other": 880000}
        margins = {"Paint": 0.50, "Other": 0.30}

        analysis = self.optimizer.analyze("store-7", revenue, margins)

        paint = next(c for c in analysis.comparisons if c.category == "Paint")
        assert paint.store_margin == 0.50
        assert paint.benchmark_margin == 0.45
        assert abs(paint.margin_gap - 0.05) < 0.001  # Beating benchmark


class TestRecommendations:
    def setup_method(self):
        self.optimizer = CategoryMixOptimizer()

    def test_under_indexed_recommendation(self):
        revenue = {
            "Paint": 50000,
            "Other": 950000,
        }
        analysis = self.optimizer.analyze("store-7", revenue)
        paint = next(c for c in analysis.comparisons if c.category == "Paint")
        assert "Expand" in paint.recommendation or "expand" in paint.recommendation
        assert "$" in paint.recommendation

    def test_on_target_recommendation(self):
        revenue = {
            "Paint": 120000,
            "Other": 880000,
        }
        analysis = self.optimizer.analyze("store-7", revenue)
        paint = next(c for c in analysis.comparisons if c.category == "Paint")
        assert (
            "well-positioned" in paint.recommendation.lower()
            or "maintain" in paint.recommendation.lower()
            or "on target" in paint.recommendation.lower()
        )


class TestAlertGeneration:
    def setup_method(self):
        self.optimizer = CategoryMixOptimizer()

    def test_generates_mix_alerts(self):
        revenue = {
            "Paint": 30000,  # 3% vs 12% — big gap, High priority
            "Electrical": 30000,  # 3% vs 11% — big gap, High priority
            "Plumbing": 30000,  # 3% vs 9% — big gap, Medium priority
            "Hand Tools": 30000,
            "Fasteners": 30000,
            "Lumber": 250000,
            "Seasonal": 100000,
            "Services": 10000,
            "General Hardware": 100000,
            "Other": 390000,
        }
        analysis = self.optimizer.analyze("store-7", revenue)
        alerts = self.optimizer.generate_alerts(analysis)

        assert len(alerts) >= 1
        assert all(a.alert_type == CoopAlertType.MIX_IMBALANCE for a in alerts)

    def test_only_high_medium_priority(self):
        """Should only alert on High/Medium priority categories."""
        revenue = {
            "Fasteners": 10000,  # Low priority
            "Other": 990000,  # Low priority
        }
        analysis = self.optimizer.analyze("store-7", revenue)
        alerts = self.optimizer.generate_alerts(analysis)

        # Fasteners and Other are Low priority, shouldn't alert
        alert_categories = [a.title for a in alerts]
        assert not any("Fasteners" in t for t in alert_categories)

    def test_alerts_have_recommendations(self):
        revenue = {
            "Paint": 20000,
            "Electrical": 20000,
            "Other": 960000,
        }
        analysis = self.optimizer.analyze("store-7", revenue)
        alerts = self.optimizer.generate_alerts(analysis)

        for alert in alerts:
            assert alert.recommendation
            assert "$" in alert.recommendation or "%" in alert.recommendation

    def test_alerts_sorted_by_impact(self):
        revenue = {
            "Paint": 20000,
            "Electrical": 30000,
            "Plumbing": 20000,
            "Services": 5000,
            "Other": 925000,
        }
        analysis = self.optimizer.analyze("store-7", revenue)
        alerts = self.optimizer.generate_alerts(analysis)

        for i in range(len(alerts) - 1):
            assert alerts[i].dollar_impact >= alerts[i + 1].dollar_impact

    def test_no_alerts_when_on_target(self):
        total = 1_000_000
        revenue = {
            cat: benchmarks["target_pct"] * total
            for cat, benchmarks in CATEGORY_BENCHMARKS.items()
        }
        analysis = self.optimizer.analyze("store-7", revenue)
        alerts = self.optimizer.generate_alerts(analysis)
        assert len(alerts) == 0


class TestCustomBenchmarks:
    def test_custom_benchmarks(self):
        custom = {
            "Widgets": {"target_pct": 0.50, "target_margin": 0.40, "priority": "High"},
            "Gadgets": {
                "target_pct": 0.50,
                "target_margin": 0.30,
                "priority": "Medium",
            },
        }
        optimizer = CategoryMixOptimizer(benchmarks=custom)
        revenue = {"Widgets": 200000, "Gadgets": 800000}
        analysis = optimizer.analyze("store-7", revenue)

        widgets = next(c for c in analysis.comparisons if c.category == "Widgets")
        assert widgets.store_pct == 0.20  # 200k/1M
        assert widgets.benchmark_pct == 0.50
        assert widgets.is_under_indexed
