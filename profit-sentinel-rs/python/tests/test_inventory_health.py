"""Tests for the InventoryHealthScorer."""

import json

import pytest
from sentinel_agent.coop_models import CoopAlertType, TurnClassification
from sentinel_agent.inventory_health import (
    CARRYING_COST_RATE,
    GMROI_AVERAGE,
    InventoryHealthScorer,
)
from sentinel_agent.models import Digest, Sku


def _make_sku(
    sku_id: str = "PNT-1001",
    qty_on_hand: float = 50.0,
    unit_cost: float = 100.0,
    retail_price: float = 150.0,
    margin_pct: float = 0.33,
    sales_last_30d: float = 10.0,
    days_since_receipt: float = 30.0,
) -> Sku:
    return Sku(
        sku_id=sku_id,
        qty_on_hand=qty_on_hand,
        unit_cost=unit_cost,
        retail_price=retail_price,
        margin_pct=margin_pct,
        sales_last_30d=sales_last_30d,
        days_since_receipt=days_since_receipt,
        is_damaged=False,
        on_order_qty=0,
        is_seasonal=False,
    )


SAMPLE_JSON = json.dumps(
    {
        "generated_at": "2026-02-05T06:00:00+00:00",
        "store_filter": ["store-7"],
        "pipeline_ms": 10,
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
                    }
                ],
                "context": "Dead stock.",
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
        ],
        "summary": {
            "total_issues": 2,
            "total_dollar_impact": 7422.0,
            "stores_affected": 1,
            "records_processed": 10,
            "issues_detected": 2,
            "issues_filtered_out": 0,
        },
    }
)


class TestSkuScoring:
    def setup_method(self):
        self.scorer = InventoryHealthScorer()

    def test_fast_mover_classification(self):
        sku = _make_sku(
            qty_on_hand=10,
            unit_cost=50,
            sales_last_30d=20,
        )
        health = self.scorer.score_sku(sku, "store-7", "Paint")
        # Annual sales = 20 × 12 = 240 units
        # Annual COGS = 240 × 50 = 12,000
        # Avg inventory = 10 × 50 = 500
        # Turn rate = 12,000 / 500 = 24.0 => Fast Mover
        assert health.classification == TurnClassification.FAST_MOVER

    def test_dead_stock_classification(self):
        sku = _make_sku(
            qty_on_hand=100,
            unit_cost=50,
            sales_last_30d=0,
        )
        health = self.scorer.score_sku(sku, "store-7", "Seasonal")
        # Annual sales = 0 => turn rate = 0 => Dead
        assert health.classification == TurnClassification.DEAD

    def test_healthy_classification(self):
        sku = _make_sku(
            qty_on_hand=40,
            unit_cost=100,
            sales_last_30d=10,
        )
        health = self.scorer.score_sku(sku, "store-7", "Paint")
        # Annual sales = 120 units, COGS = 12,000
        # Avg inv = 4,000, Turn = 3.0 => Healthy
        assert health.classification == TurnClassification.HEALTHY

    def test_gmroi_calculation(self):
        sku = _make_sku(
            qty_on_hand=40,
            unit_cost=100,
            retail_price=150,
            margin_pct=0.33,
            sales_last_30d=10,
        )
        health = self.scorer.score_sku(sku, "store-7", "Paint")
        # Annual sales revenue = 120 × 150 = 18,000
        # GMROI = (18,000 × 0.33) / 4,000 = 1.485
        expected_gmroi = (120 * 150 * 0.33) / (40 * 100)
        assert abs(health.gmroi - expected_gmroi) < 0.01

    def test_carrying_cost(self):
        sku = _make_sku(qty_on_hand=100, unit_cost=50)
        health = self.scorer.score_sku(sku, "store-7", "Paint")
        # Carrying cost = 100 × 50 × 0.27 = 1,350
        expected = 100 * 50 * CARRYING_COST_RATE
        assert abs(health.carrying_cost_annual - expected) < 0.01

    def test_days_of_supply(self):
        sku = _make_sku(
            qty_on_hand=30,
            sales_last_30d=10,
        )
        health = self.scorer.score_sku(sku, "store-7", "Paint")
        # Daily sales = 120 / 365 ≈ 0.329
        # Days of supply = 30 / 0.329 ≈ 91
        assert 80 <= health.days_of_supply <= 100

    def test_zero_sales_days_of_supply(self):
        sku = _make_sku(qty_on_hand=100, sales_last_30d=0)
        health = self.scorer.score_sku(sku, "store-7")
        assert health.days_of_supply == 999.0

    def test_category_guessing(self):
        sku = _make_sku(sku_id="PNT-1001")
        health = self.scorer.score_sku(sku, "store-7")
        assert health.category == "Paint"

        sku2 = _make_sku(sku_id="ELC-4401")
        health2 = self.scorer.score_sku(sku2, "store-7")
        assert health2.category == "Electrical"


class TestCategoryAnalysis:
    def setup_method(self):
        self.scorer = InventoryHealthScorer()

    def test_aggregates_skus(self):
        skus = [
            _make_sku("PNT-001", qty_on_hand=20, sales_last_30d=10),
            _make_sku("PNT-002", qty_on_hand=30, sales_last_30d=5),
        ]
        healths = [self.scorer.score_sku(s, "store-7", "Paint") for s in skus]
        analysis = self.scorer.analyze_category(healths, "store-7", "Paint")
        assert analysis.sku_count == 2
        assert analysis.total_inventory_cost > 0

    def test_empty_category(self):
        analysis = self.scorer.analyze_category([], "store-7", "Empty")
        assert analysis.sku_count == 0
        assert analysis.gmroi == 0.0

    def test_classification_counts(self):
        skus = [
            _make_sku("A", qty_on_hand=10, sales_last_30d=20),  # fast
            _make_sku("B", qty_on_hand=100, sales_last_30d=0),  # dead
            _make_sku("C", qty_on_hand=40, sales_last_30d=10),  # healthy
        ]
        healths = [self.scorer.score_sku(s, "store-7", "Paint") for s in skus]
        analysis = self.scorer.analyze_category(healths, "store-7", "Paint")
        assert analysis.fast_mover_count == 1
        assert analysis.dead_count == 1
        assert analysis.healthy_count == 1


class TestStoreReport:
    def setup_method(self):
        self.scorer = InventoryHealthScorer()
        self.digest = Digest.model_validate_json(SAMPLE_JSON)

    def test_generates_report(self):
        report = self.scorer.score_from_digest(self.digest, "store-7")
        assert report.store_id == "store-7"
        assert report.total_inventory_value > 0

    def test_detects_dead_stock(self):
        report = self.scorer.score_from_digest(self.digest, "store-7")
        assert report.total_dead_stock_value > 0
        # SEA-1201 has 0 sales => dead stock = 100 × 50 = 5000
        assert report.total_dead_stock_value >= 5000

    def test_sku_details_populated(self):
        report = self.scorer.score_from_digest(self.digest, "store-7")
        assert len(report.sku_details) == 2

    def test_category_analyses_populated(self):
        report = self.scorer.score_from_digest(self.digest, "store-7")
        assert len(report.category_analyses) >= 1


class TestHealthAlerts:
    def setup_method(self):
        self.scorer = InventoryHealthScorer()
        self.digest = Digest.model_validate_json(SAMPLE_JSON)

    def test_generates_dead_stock_alert(self):
        report = self.scorer.score_from_digest(self.digest, "store-7")
        alerts = self.scorer.generate_alerts(report)
        dead_alerts = [
            a for a in alerts if a.alert_type == CoopAlertType.DEAD_STOCK_ALERT
        ]
        assert len(dead_alerts) >= 1
        assert dead_alerts[0].dollar_impact > 0

    def test_alert_has_recommendation(self):
        report = self.scorer.score_from_digest(self.digest, "store-7")
        alerts = self.scorer.generate_alerts(report)
        for alert in alerts:
            assert alert.recommendation
            assert "$" in alert.recommendation

    def test_alerts_sorted_by_impact(self):
        report = self.scorer.score_from_digest(self.digest, "store-7")
        alerts = self.scorer.generate_alerts(report)
        for i in range(len(alerts) - 1):
            assert alerts[i].dollar_impact >= alerts[i + 1].dollar_impact
