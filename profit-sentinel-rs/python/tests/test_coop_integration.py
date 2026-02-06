"""Integration tests for co-op intelligence pipeline.

Tests the full flow: Digest → InventoryHealth → CoopIntelligence →
VendorRebates → CategoryMix → CoopIntelligenceReport → Rendered text.

These tests don't need the Rust binary — they use pre-built Digest
objects to test the Python co-op intelligence layer end-to-end.
"""

import json
from datetime import date

import pytest
from sentinel_agent.category_mix import CategoryMixOptimizer
from sentinel_agent.coop_intelligence import CoopIntelligence
from sentinel_agent.coop_models import (
    CoopAffiliation,
    CoopAlert,
    CoopAlertType,
    CoopIntelligenceReport,
    CoopType,
    VendorPurchase,
)
from sentinel_agent.inventory_health import InventoryHealthScorer
from sentinel_agent.llm_layer import (
    render_coop_report,
    render_inventory_health_summary,
)
from sentinel_agent.models import Digest
from sentinel_agent.vendor_rebates import VendorRebateTracker

# A realistic digest covering multiple issue types
DIGEST_JSON = json.dumps(
    {
        "generated_at": "2026-02-05T06:00:00+00:00",
        "store_filter": ["store-7"],
        "pipeline_ms": 12,
        "issues": [
            {
                "id": "store-7-DeadStock-001",
                "issue_type": "DeadStock",
                "store_id": "store-7",
                "dollar_impact": 5000.0,
                "confidence": 0.89,
                "trend_direction": "Worsening",
                "priority_score": 11.37,
                "urgency_score": 0.9,
                "detection_timestamp": "2025-01-02T00:00:00Z",
                "skus": [
                    {
                        "sku_id": "SEA-1201",
                        "qty_on_hand": 100.0,
                        "unit_cost": 50.0,
                        "retail_price": 67.5,
                        "margin_pct": 0.35,
                        "sales_last_30d": 0.0,
                        "days_since_receipt": 180.0,
                        "is_damaged": False,
                        "on_order_qty": 0.0,
                        "is_seasonal": False,
                    },
                    {
                        "sku_id": "SEA-1202",
                        "qty_on_hand": 40.0,
                        "unit_cost": 25.0,
                        "retail_price": 35.0,
                        "margin_pct": 0.28,
                        "sales_last_30d": 0.0,
                        "days_since_receipt": 200.0,
                        "is_damaged": False,
                        "on_order_qty": 0.0,
                        "is_seasonal": True,
                    },
                ],
                "context": "2 SKUs with zero sales for 180+ days.",
            },
            {
                "id": "store-7-MarginErosion-001",
                "issue_type": "MarginErosion",
                "store_id": "store-7",
                "dollar_impact": 2422.0,
                "confidence": 0.87,
                "trend_direction": "Worsening",
                "priority_score": 10.17,
                "urgency_score": 0.9,
                "detection_timestamp": "2025-01-02T00:00:00Z",
                "skus": [
                    {
                        "sku_id": "PNT-1001",
                        "qty_on_hand": 50.0,
                        "unit_cost": 100.0,
                        "retail_price": 105.0,
                        "margin_pct": 0.05,
                        "sales_last_30d": 10.0,
                        "days_since_receipt": 30.0,
                        "is_damaged": False,
                        "on_order_qty": 0.0,
                        "is_seasonal": False,
                    }
                ],
                "context": "Low margin.",
            },
            {
                "id": "store-7-NegativeInventory-002",
                "issue_type": "NegativeInventory",
                "store_id": "store-7",
                "dollar_impact": 1104.5,
                "confidence": 0.87,
                "trend_direction": "Stable",
                "priority_score": 4.45,
                "urgency_score": 0.5,
                "detection_timestamp": "2025-01-02T00:00:00Z",
                "skus": [
                    {
                        "sku_id": "ELC-4401",
                        "qty_on_hand": -47.0,
                        "unit_cost": 23.5,
                        "retail_price": 31.73,
                        "margin_pct": 0.35,
                        "sales_last_30d": 10.0,
                        "days_since_receipt": 30.0,
                        "is_damaged": False,
                        "on_order_qty": 0.0,
                        "is_seasonal": False,
                    }
                ],
                "context": "47 units short at $23.50/unit.",
            },
        ],
        "summary": {
            "total_issues": 3,
            "total_dollar_impact": 8526.5,
            "stores_affected": 1,
            "records_processed": 20,
            "issues_detected": 3,
            "issues_filtered_out": 0,
        },
    }
)


# Vendor purchase records simulating non-co-op purchases
VENDOR_PURCHASES = [
    VendorPurchase(
        vendor_id="DMG",
        vendor_name="Martin's Supply Co.",
        sku_id="DMG-0101",
        category="General Hardware",
        quantity=50,
        unit_cost=30.0,
        total_cost=1500.0,
        is_coop_available=False,
    ),
    VendorPurchase(
        vendor_id="DMG",
        vendor_name="Martin's Supply Co.",
        sku_id="DMG-0102",
        category="General Hardware",
        quantity=100,
        unit_cost=45.0,
        total_cost=4500.0,
        is_coop_available=False,
    ),
    VendorPurchase(
        vendor_id="EXT",
        vendor_name="External Paint Co.",
        sku_id="EXT-0001",
        category="Paint",
        quantity=200,
        unit_cost=60.0,
        total_cost=12000.0,
        is_coop_available=False,
    ),
    VendorPurchase(
        vendor_id="ELC",
        vendor_name="National Electrical",
        sku_id="ELC-4401",
        category="Electrical",
        quantity=100,
        unit_cost=23.5,
        total_cost=2350.0,
        is_coop_available=True,  # This IS a co-op purchase
    ),
    VendorPurchase(
        vendor_id="PNT",
        vendor_name="ColorMax Paint Supply",
        sku_id="PNT-1001",
        category="Paint",
        quantity=50,
        unit_cost=100.0,
        total_cost=5000.0,
        is_coop_available=True,  # Co-op purchase
    ),
]


class TestFullCoopPipeline:
    """End-to-end test: Digest → all analyzers → report → text."""

    def setup_method(self):
        self.digest = Digest.model_validate_json(DIGEST_JSON)
        self.affiliation = CoopAffiliation(
            store_id="store-7",
            coop_type=CoopType.DO_IT_BEST,
        )

    def test_inventory_health_from_digest(self):
        scorer = InventoryHealthScorer()
        report = scorer.score_from_digest(self.digest, "store-7")

        assert report.store_id == "store-7"
        assert report.total_inventory_value > 0
        assert report.total_dead_stock_value > 0
        assert len(report.sku_details) == 4  # 2 dead + 1 margin + 1 negative
        assert len(report.category_analyses) >= 1

    def test_patronage_leakage_detection(self):
        coop = CoopIntelligence(self.affiliation)
        leakages = coop.detect_patronage_leakage(VENDOR_PURCHASES)

        # Should detect Martin's Supply and External Paint as non-co-op
        non_coop_vendors = {l.vendor_name for l in leakages}
        assert "Martin's Supply Co." in non_coop_vendors
        assert "External Paint Co." in non_coop_vendors

        # Should NOT flag co-op purchases
        assert "National Electrical" not in non_coop_vendors

    def test_leakage_dollar_calculation(self):
        coop = CoopIntelligence(self.affiliation)
        leakages = coop.detect_patronage_leakage(VENDOR_PURCHASES)

        # Martin's: $6,000 × 11.11% = $666.60
        martins = next(l for l in leakages if "Martin" in l.vendor_name)
        assert abs(martins.annual_leakage - 6000 * 0.1111) < 1

        # External Paint: $12,000 × 11.11% = $1,333.20
        ext_paint = next(l for l in leakages if "External" in l.vendor_name)
        assert abs(ext_paint.annual_leakage - 12000 * 0.1111) < 1

    def test_consolidation_opportunities(self):
        coop = CoopIntelligence(self.affiliation)
        opps = coop.find_consolidation_opportunities(VENDOR_PURCHASES)

        # Paint has 2 vendors (External + ColorMax), should flag
        paint_opps = [o for o in opps if o.category == "Paint"]
        assert len(paint_opps) >= 1 or len(opps) >= 1

    def test_vendor_rebate_tracking(self):
        tracker = VendorRebateTracker()
        vendor_ytd = {"DMG": 8000, "PNT": 12000, "ELC": 5000}
        statuses = tracker.evaluate_all(
            "store-7",
            vendor_ytd,
            as_of=date(2026, 11, 1),
        )

        assert len(statuses) >= 2  # DMG and PNT have programs

        # Late in year with low purchases => at risk
        at_risk = [s for s in statuses if s.is_at_risk]
        assert len(at_risk) >= 1

    def test_category_mix_analysis(self):
        optimizer = CategoryMixOptimizer()
        revenue = {
            "Paint": 50000,
            "Electrical": 80000,
            "General Hardware": 120000,
            "Seasonal": 150000,
            "Other": 600000,
        }
        analysis = optimizer.analyze("store-7", revenue)

        assert analysis.total_revenue == 1000000
        assert analysis.total_opportunity > 0
        assert len(analysis.top_expansion_categories) >= 1

    def test_combined_alert_generation(self):
        """All analyzers produce alerts that can be combined."""
        # Health alerts
        scorer = InventoryHealthScorer()
        report = scorer.score_from_digest(self.digest, "store-7")
        health_alerts = scorer.generate_alerts(report)

        # Leakage alerts
        coop = CoopIntelligence(self.affiliation)
        coop_alerts = coop.generate_alerts(VENDOR_PURCHASES)

        # Rebate alerts
        tracker = VendorRebateTracker()
        vendor_ytd = {"DMG": 8000, "PNT": 12000}
        statuses = tracker.evaluate_all(
            "store-7",
            vendor_ytd,
            as_of=date(2026, 11, 1),
        )
        rebate_alerts = tracker.generate_alerts(statuses)

        # Mix alerts
        optimizer = CategoryMixOptimizer()
        revenue = {"Paint": 50000, "Electrical": 30000, "Other": 920000}
        analysis = optimizer.analyze("store-7", revenue)
        mix_alerts = optimizer.generate_alerts(analysis)

        # Combine all
        all_alerts = health_alerts + coop_alerts + rebate_alerts + mix_alerts
        all_alerts.sort(key=lambda a: a.dollar_impact, reverse=True)

        assert len(all_alerts) >= 3  # Should have alerts from multiple sources

        # Verify alert types are diverse
        alert_types = {a.alert_type for a in all_alerts}
        assert len(alert_types) >= 2

    def test_report_rendering(self):
        """Report renders to readable text."""
        alerts = [
            CoopAlert(
                alert_type=CoopAlertType.PATRONAGE_LEAKAGE,
                store_id="store-7",
                title="Patronage Leakage: $6,000 to Martin's Supply",
                dollar_impact=666.0,
                detail="Buying General Hardware outside co-op.",
                recommendation="Shift $6,000 to co-op warehouse.",
            ),
            CoopAlert(
                alert_type=CoopAlertType.DEAD_STOCK_ALERT,
                store_id="store-7",
                title="Dead Stock: $6,000 in 2 SKUs",
                dollar_impact=1620.0,
                detail="2 SKUs with zero sales.",
                recommendation="Liquidate immediately.",
            ),
        ]
        scorer = InventoryHealthScorer()
        health_report = scorer.score_from_digest(self.digest, "store-7")

        report = CoopIntelligenceReport(
            store_id="store-7",
            affiliation=self.affiliation,
            alerts=alerts,
            health_report=health_report,
            total_opportunity=2286.0,
        )

        text = render_coop_report(report)
        assert "CO-OP INTELLIGENCE REPORT" in text
        assert "Store 7" in text
        assert "$2,286" in text
        assert "Patronage Leakage" in text
        assert "Dead Stock" in text

    def test_health_summary_rendering(self):
        scorer = InventoryHealthScorer()
        report = scorer.score_from_digest(self.digest, "store-7")

        text = render_inventory_health_summary(report)
        assert "INVENTORY HEALTH" in text
        assert "Store 7" in text
        assert "GMROI" in text
        assert "$" in text


class TestAlertTypeDistribution:
    """Verify each analyzer produces the correct alert types."""

    def test_health_alert_types(self):
        scorer = InventoryHealthScorer()
        digest = Digest.model_validate_json(DIGEST_JSON)
        report = scorer.score_from_digest(digest, "store-7")
        alerts = scorer.generate_alerts(report)
        for alert in alerts:
            assert alert.alert_type in (
                CoopAlertType.DEAD_STOCK_ALERT,
                CoopAlertType.GMROI_WARNING,
            )

    def test_leakage_alert_types(self):
        coop = CoopIntelligence(
            CoopAffiliation(
                store_id="store-7",
                coop_type=CoopType.DO_IT_BEST,
            )
        )
        alerts = coop.generate_alerts(VENDOR_PURCHASES)
        for alert in alerts:
            assert alert.alert_type in (
                CoopAlertType.PATRONAGE_LEAKAGE,
                CoopAlertType.CONSOLIDATION_OPPORTUNITY,
            )

    def test_rebate_alert_types(self):
        tracker = VendorRebateTracker()
        statuses = tracker.evaluate_all(
            "store-7",
            {"DMG": 5000},
            as_of=date(2026, 11, 1),
        )
        alerts = tracker.generate_alerts(statuses)
        for alert in alerts:
            assert alert.alert_type == CoopAlertType.REBATE_THRESHOLD_RISK

    def test_mix_alert_types(self):
        optimizer = CategoryMixOptimizer()
        revenue = {"Paint": 20000, "Other": 980000}
        analysis = optimizer.analyze("store-7", revenue)
        alerts = optimizer.generate_alerts(analysis)
        for alert in alerts:
            assert alert.alert_type == CoopAlertType.MIX_IMBALANCE


class TestAllAlertsHaveDollars:
    """Every alert must quantify its impact in dollars."""

    def test_all_alerts_have_dollar_impact(self):
        # Health
        scorer = InventoryHealthScorer()
        digest = Digest.model_validate_json(DIGEST_JSON)
        report = scorer.score_from_digest(digest, "store-7")
        for alert in scorer.generate_alerts(report):
            assert alert.dollar_impact > 0
            assert "$" in alert.recommendation

        # Leakage
        coop = CoopIntelligence(
            CoopAffiliation(
                store_id="store-7",
                coop_type=CoopType.DO_IT_BEST,
            )
        )
        for alert in coop.generate_alerts(VENDOR_PURCHASES):
            assert alert.dollar_impact > 0
            assert "$" in alert.recommendation

        # Rebates
        tracker = VendorRebateTracker()
        statuses = tracker.evaluate_all(
            "store-7",
            {"DMG": 5000},
            as_of=date(2026, 11, 1),
        )
        for alert in tracker.generate_alerts(statuses):
            assert alert.dollar_impact > 0
