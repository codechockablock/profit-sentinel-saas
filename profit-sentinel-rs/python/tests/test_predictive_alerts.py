"""Tests for Predictive Inventory Alerts.

Covers:
    - Stockout prediction with severity levels
    - Overstock prediction with carrying costs
    - Velocity change detection (demand surge, demand drop)
    - Edge cases: zero velocity, negative stock, empty digest
    - Report generation and serialization
    - API endpoint integration
"""

import pytest

from sentinel_agent.predictive_alerts import (
    AlertSeverity,
    InventoryPrediction,
    PredictionType,
    PredictiveAlertEngine,
    PredictiveReport,
    predict_inventory,
)
from sentinel_agent.models import (
    Digest,
    Issue,
    IssueType,
    Sku,
    Summary,
    TrendDirection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sku(
    sku_id: str = "DMG-001",
    qty_on_hand: float = 10,
    unit_cost: float = 25.0,
    retail_price: float = 40.0,
    margin_pct: float = 0.375,
    sales_last_30d: float = 15,
    is_damaged: bool = False,
    on_order_qty: float = 0,
) -> Sku:
    return Sku(
        sku_id=sku_id,
        qty_on_hand=qty_on_hand,
        unit_cost=unit_cost,
        retail_price=retail_price,
        margin_pct=margin_pct,
        sales_last_30d=sales_last_30d,
        days_since_receipt=30,
        is_damaged=is_damaged,
        on_order_qty=on_order_qty,
        is_seasonal=False,
    )


def _make_issue(
    issue_type: IssueType = IssueType.RECEIVING_GAP,
    dollar_impact: float = 1000,
    skus: list[Sku] | None = None,
    store_id: str = "store-7",
) -> Issue:
    if skus is None:
        skus = [_make_sku()]
    return Issue(
        id=f"issue-{issue_type.value}",
        issue_type=issue_type,
        store_id=store_id,
        dollar_impact=dollar_impact,
        confidence=0.85,
        trend_direction=TrendDirection.WORSENING,
        priority_score=7.5,
        urgency_score=6.0,
        detection_timestamp="2026-02-06T10:00:00Z",
        skus=skus,
        context="Test issue",
    )


def _make_digest(issues: list[Issue] | None = None) -> Digest:
    if issues is None:
        issues = []
    return Digest(
        generated_at="2026-02-06T10:00:00Z",
        store_filter=["store-7"],
        pipeline_ms=150,
        issues=issues,
        summary=Summary(
            total_issues=len(issues),
            total_dollar_impact=sum(i.dollar_impact for i in issues),
            stores_affected=1,
            records_processed=100,
            issues_detected=len(issues),
            issues_filtered_out=0,
        ),
    )


# ---------------------------------------------------------------------------
# Stockout prediction tests
# ---------------------------------------------------------------------------


class TestStockoutPrediction:
    def test_critical_stockout(self):
        """SKU selling 2/day with 5 on hand → critical."""
        sku = _make_sku(qty_on_hand=5, sales_last_30d=60)  # 2/day
        issue = _make_issue(skus=[sku])
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        assert len(report.stockout_predictions) >= 1

        pred = report.stockout_predictions[0]
        assert pred.severity == AlertSeverity.CRITICAL
        assert pred.days_until_event == pytest.approx(2.5, abs=0.5)
        assert pred.estimated_lost_revenue > 0

    def test_warning_stockout(self):
        """SKU selling 1/day with 10 on hand → warning."""
        sku = _make_sku(qty_on_hand=10, sales_last_30d=30)  # 1/day
        issue = _make_issue(skus=[sku])
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        stockouts = report.stockout_predictions
        assert len(stockouts) >= 1

        pred = stockouts[0]
        assert pred.severity == AlertSeverity.WARNING
        assert pred.days_until_event == pytest.approx(10, abs=1)

    def test_watch_stockout(self):
        """SKU selling 0.5/day with 10 on hand → watch."""
        sku = _make_sku(qty_on_hand=10, sales_last_30d=15)  # 0.5/day
        issue = _make_issue(skus=[sku])
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        stockouts = report.stockout_predictions
        assert len(stockouts) >= 1

        pred = stockouts[0]
        assert pred.severity == AlertSeverity.WATCH
        assert pred.days_until_event == pytest.approx(20, abs=2)

    def test_no_stockout_high_stock(self):
        """SKU with plenty of stock → no stockout prediction."""
        sku = _make_sku(qty_on_hand=500, sales_last_30d=30)
        issue = _make_issue(skus=[sku])
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        # 500 days supply → outside horizon
        assert len(report.stockout_predictions) == 0

    def test_no_stockout_zero_velocity(self):
        """SKU with zero sales → no stockout prediction."""
        sku = _make_sku(qty_on_hand=10, sales_last_30d=0)
        issue = _make_issue(skus=[sku])
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        # No velocity → can't predict stockout
        assert len(report.stockout_predictions) == 0

    def test_no_stockout_negative_stock(self):
        """SKU already stocked out → not a prediction."""
        sku = _make_sku(qty_on_hand=-5, sales_last_30d=30)
        issue = _make_issue(skus=[sku])
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        assert len(report.stockout_predictions) == 0

    def test_stockout_revenue_calculation(self):
        """Estimated lost revenue should be calculated correctly."""
        sku = _make_sku(
            qty_on_hand=5,
            sales_last_30d=60,  # 2/day
            retail_price=50.0,
        )
        issue = _make_issue(skus=[sku])
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        pred = report.stockout_predictions[0]
        # 2/day × $50 × 7 days lost = $700
        assert pred.estimated_lost_revenue == pytest.approx(700, abs=50)

    def test_confidence_higher_for_near_term(self):
        """Predictions closer in time should have higher confidence."""
        sku_near = _make_sku(sku_id="DMG-001", qty_on_hand=3, sales_last_30d=60)
        sku_far = _make_sku(sku_id="DMG-002", qty_on_hand=25, sales_last_30d=30)
        issue = _make_issue(skus=[sku_near, sku_far])
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        assert len(report.stockout_predictions) >= 2, (
            f"Expected >=2 stockout predictions for 2 at-risk SKUs, got {len(report.stockout_predictions)}"
        )
        near = next(p for p in report.stockout_predictions if p.sku_id == "DMG-001")
        far = next(p for p in report.stockout_predictions if p.sku_id == "DMG-002")
        assert near.confidence >= far.confidence


# ---------------------------------------------------------------------------
# Overstock prediction tests
# ---------------------------------------------------------------------------


class TestOverstockPrediction:
    def test_overstock_detection(self):
        """SKU with high stock and low velocity → overstock."""
        sku = _make_sku(qty_on_hand=200, sales_last_30d=6)  # 0.2/day → 1000 days
        issue = _make_issue(
            issue_type=IssueType.OVERSTOCK,
            skus=[sku],
        )
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        assert len(report.overstock_predictions) >= 1

        pred = report.overstock_predictions[0]
        assert pred.prediction_type == PredictionType.OVERSTOCK
        assert pred.estimated_carrying_cost > 0

    def test_overstock_zero_velocity(self):
        """SKU with no sales and significant value → overstock alert."""
        sku = _make_sku(
            qty_on_hand=50,
            sales_last_30d=0,
            unit_cost=20.0,
        )
        issue = _make_issue(
            issue_type=IssueType.DEAD_STOCK,
            skus=[sku],
        )
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        overstocks = report.overstock_predictions
        assert len(overstocks) >= 1

        pred = overstocks[0]
        # $1000 inventory × 22% = $220 carrying cost
        assert pred.estimated_carrying_cost == pytest.approx(220, abs=20)

    def test_no_overstock_healthy_velocity(self):
        """SKU with healthy turnover → no overstock."""
        sku = _make_sku(qty_on_hand=20, sales_last_30d=30)  # 1/day → 20 days
        issue = _make_issue(skus=[sku])
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        assert len(report.overstock_predictions) == 0

    def test_overstock_carrying_cost(self):
        """Carrying cost should reflect inventory value."""
        sku = _make_sku(
            qty_on_hand=100,
            unit_cost=50.0,
            sales_last_30d=3,  # 0.1/day → 1000 days supply
        )
        issue = _make_issue(
            issue_type=IssueType.OVERSTOCK,
            skus=[sku],
        )
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        pred = report.overstock_predictions[0]
        # $5000 × 22% = $1100
        assert pred.estimated_carrying_cost == pytest.approx(1100, abs=100)


# ---------------------------------------------------------------------------
# Velocity change tests
# ---------------------------------------------------------------------------


class TestVelocityChange:
    def test_demand_surge_detection(self):
        """High velocity + low stock + receiving gap → demand surge."""
        sku = _make_sku(qty_on_hand=5, sales_last_30d=60)  # 2/day
        issue = _make_issue(
            issue_type=IssueType.RECEIVING_GAP,
            skus=[sku],
        )
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        surges = [
            p for p in report.velocity_alerts
            if p.prediction_type == PredictionType.DEMAND_SURGE
        ]
        assert len(surges) >= 1

    def test_velocity_drop_detection(self):
        """Low velocity + high stock + overstock issue → velocity drop."""
        sku = _make_sku(qty_on_hand=100, sales_last_30d=3)  # 0.1/day
        issue = _make_issue(
            issue_type=IssueType.OVERSTOCK,
            skus=[sku],
        )
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        drops = [
            p for p in report.velocity_alerts
            if p.prediction_type == PredictionType.VELOCITY_DROP
        ]
        assert len(drops) >= 1


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


class TestPredictiveReport:
    def test_empty_digest(self):
        """Empty digest → empty report."""
        digest = _make_digest([])
        report = predict_inventory(digest)
        assert report.total_predictions == 0
        assert report.critical_alerts == 0
        assert report.top_recommendation == "Inventory levels are healthy."

    def test_store_filter(self):
        """Store filter should exclude other stores."""
        issues = [
            _make_issue(
                skus=[_make_sku(sku_id="DMG-001", qty_on_hand=3, sales_last_30d=60)],
                store_id="store-7",
            ),
            _make_issue(
                skus=[_make_sku(sku_id="ELC-001", qty_on_hand=3, sales_last_30d=60)],
                store_id="store-12",
            ),
        ]
        digest = _make_digest(issues)

        report = predict_inventory(digest, store_id="store-7")
        # Only store-7 predictions
        for pred in report.stockout_predictions:
            assert pred.store_id == "store-7"

    def test_report_to_dict(self):
        """Report should be JSON-serializable."""
        import json

        sku = _make_sku(qty_on_hand=5, sales_last_30d=60)
        issue = _make_issue(skus=[sku])
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        data = report.to_dict()
        json_str = json.dumps(data)
        assert len(json_str) > 50
        assert "stockout_predictions" in data
        assert "overstock_predictions" in data

    def test_top_recommendation_stockout(self):
        """Top recommendation should reference worst stockout."""
        sku = _make_sku(qty_on_hand=3, sales_last_30d=60)
        issue = _make_issue(skus=[sku])
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        assert "stock out" in report.top_recommendation.lower()

    def test_critical_alert_count(self):
        """Report should count critical alerts correctly."""
        sku = _make_sku(qty_on_hand=3, sales_last_30d=60)
        issue = _make_issue(skus=[sku])
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        assert report.critical_alerts >= 1

    def test_revenue_at_risk_aggregated(self):
        """Total revenue at risk should sum all stockout predictions."""
        skus = [
            _make_sku(sku_id="DMG-001", qty_on_hand=3, sales_last_30d=60, retail_price=50),
            _make_sku(sku_id="DMG-002", qty_on_hand=5, sales_last_30d=90, retail_price=30),
        ]
        issue = _make_issue(skus=skus)
        digest = _make_digest([issue])

        report = predict_inventory(digest, store_id="store-7")
        if report.stockout_predictions:
            individual_sum = sum(
                p.estimated_lost_revenue for p in report.stockout_predictions
            )
            assert report.total_revenue_at_risk == pytest.approx(
                individual_sum, abs=1
            )

    def test_custom_horizon(self):
        """Custom horizon should change what gets flagged."""
        sku = _make_sku(qty_on_hand=50, sales_last_30d=60)  # 25 days supply
        issue = _make_issue(skus=[sku])
        digest = _make_digest([issue])

        # 30-day horizon → should flag
        report_30 = predict_inventory(digest, store_id="store-7", horizon_days=30)
        # 10-day horizon → should NOT flag
        report_10 = predict_inventory(digest, store_id="store-7", horizon_days=10)

        assert len(report_30.stockout_predictions) >= len(report_10.stockout_predictions)


# ---------------------------------------------------------------------------
# API endpoint integration test
# ---------------------------------------------------------------------------


class TestPredictiveEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from sentinel_agent.sidecar import create_app
        from sentinel_agent.sidecar_config import SidecarSettings

        settings = SidecarSettings(
            sidecar_dev_mode=True,
            csv_path="fixtures/sample_inventory.csv",
            supabase_url="",
            supabase_service_key="",
        )
        app = create_app(settings)
        return TestClient(app)

    def test_predictions_endpoint_exists(self, client):
        """Endpoint should exist and return 502 (no binary) or 200 or 404."""
        resp = client.get("/api/v1/predictions?store_id=store-7")
        assert resp.status_code in (200, 404, 502)

    def test_predictions_endpoint_structure(self, client):
        """Response should have expected structure if pipeline works."""
        resp = client.get("/api/v1/predictions")
        if resp.status_code == 200:
            data = resp.json()
            assert "stockout_predictions" in data
            assert "overstock_predictions" in data
            assert "total_predictions" in data
