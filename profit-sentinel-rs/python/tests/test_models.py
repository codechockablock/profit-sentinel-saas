"""Tests for Pydantic models and JSON contract parsing."""

import json

import pytest
from sentinel_agent.models import (
    CallPrep,
    Digest,
    Issue,
    IssueType,
    Sku,
    Summary,
    Task,
    TaskPriority,
    TrendDirection,
)

# ---------------------------------------------------------------------------
# Fixture: sample JSON matching the Rust server --json output
# ---------------------------------------------------------------------------

SAMPLE_JSON = json.dumps(
    {
        "generated_at": "2026-02-05T06:00:00+00:00",
        "store_filter": ["store-7", "store-12"],
        "pipeline_ms": 16,
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
                "context": "1 SKU with zero sales for 180+ days.",
            },
            {
                "id": "store-12-MarginErosion-001",
                "issue_type": "MarginErosion",
                "store_id": "store-12",
                "dollar_impact": 2422.25,
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
                    },
                    {
                        "sku_id": "PNT-1002",
                        "qty_on_hand": 35.0,
                        "unit_cost": 85.0,
                        "retail_price": 88.4,
                        "margin_pct": 0.04,
                        "sales_last_30d": 7.0,
                        "days_since_receipt": 25.0,
                        "is_damaged": False,
                        "on_order_qty": 0.0,
                        "is_seasonal": False,
                    },
                ],
                "context": "2 SKUs averaging 4% margin vs 35% benchmark.",
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
            "total_dollar_impact": 8526.75,
            "stores_affected": 2,
            "records_processed": 20,
            "issues_detected": 4,
            "issues_filtered_out": 0,
        },
    }
)


class TestDigestParsing:
    """Test that Pydantic models correctly parse Rust JSON output."""

    def test_parse_full_digest(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        assert len(digest.issues) == 3
        assert digest.pipeline_ms == 16
        assert digest.store_filter == ["store-7", "store-12"]

    def test_summary_fields(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        assert digest.summary.total_issues == 3
        assert digest.summary.total_dollar_impact == 8526.75
        assert digest.summary.stores_affected == 2
        assert digest.summary.records_processed == 20

    def test_issue_types_parse(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        types = [i.issue_type for i in digest.issues]
        assert IssueType.DEAD_STOCK in types
        assert IssueType.MARGIN_EROSION in types
        assert IssueType.NEGATIVE_INVENTORY in types

    def test_trend_directions_parse(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        assert digest.issues[0].trend_direction == TrendDirection.WORSENING
        assert digest.issues[2].trend_direction == TrendDirection.STABLE

    def test_sku_details(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        neg_inv = digest.issues[2]
        assert len(neg_inv.skus) == 1
        assert neg_inv.skus[0].sku_id == "ELC-4401"
        assert neg_inv.skus[0].qty_on_hand == -47.0
        assert neg_inv.skus[0].unit_cost == 23.5

    def test_multi_sku_issue(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        margin = digest.issues[1]
        assert len(margin.skus) == 2
        assert margin.skus[0].sku_id == "PNT-1001"
        assert margin.skus[1].sku_id == "PNT-1002"

    def test_dollar_impact_exact(self):
        """The critical test: -47 qty * $23.50 = $1,104.50."""
        digest = Digest.model_validate_json(SAMPLE_JSON)
        neg_inv = next(
            i for i in digest.issues if i.issue_type == IssueType.NEGATIVE_INVENTORY
        )
        assert abs(neg_inv.dollar_impact - 1104.50) < 0.01

    def test_generated_datetime(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        dt = digest.generated_datetime
        assert dt.year == 2026
        assert dt.month == 2
        assert dt.day == 5


class TestIssueProperties:
    """Test computed properties on Issue model."""

    def test_is_urgent(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        assert digest.issues[0].is_urgent  # score 11.37
        assert not digest.issues[2].is_urgent  # score 4.45

    def test_has_urgent_issues(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        assert digest.has_urgent_issues

    def test_dollar_display(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        assert digest.issues[0].dollar_display == "$5,000"
        assert digest.issues[1].dollar_display == "$2,422"

    def test_sku_count(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        assert digest.issues[0].sku_count == 1
        assert digest.issues[1].sku_count == 2

    def test_issues_by_store(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        by_store = digest.issues_by_store()
        assert len(by_store["store-7"]) == 2
        assert len(by_store["store-12"]) == 1

    def test_issues_by_type(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        by_type = digest.issues_by_type()
        assert len(by_type[IssueType.DEAD_STOCK]) == 1
        assert len(by_type[IssueType.MARGIN_EROSION]) == 1


class TestIssueTypeProperties:
    """Test IssueType enum properties."""

    def test_display_names(self):
        assert IssueType.DEAD_STOCK.display_name == "Dead Stock"
        assert IssueType.NEGATIVE_INVENTORY.display_name == "Negative Inventory"
        assert IssueType.MARGIN_EROSION.display_name == "Margin Erosion"

    def test_action_verbs(self):
        assert "markdown" in IssueType.DEAD_STOCK.action_verb.lower()
        assert "investigate" in IssueType.NEGATIVE_INVENTORY.action_verb.lower()

    def test_all_types_have_display_names(self):
        for issue_type in IssueType:
            assert issue_type.display_name  # not empty

    def test_all_types_have_action_verbs(self):
        for issue_type in IssueType:
            assert issue_type.action_verb  # not empty


class TestTrendDirection:
    """Test TrendDirection enum properties."""

    def test_arrows(self):
        assert TrendDirection.WORSENING.arrow == "\u2191"
        assert TrendDirection.STABLE.arrow == "\u2192"
        assert TrendDirection.IMPROVING.arrow == "\u2193"

    def test_descriptions(self):
        assert TrendDirection.WORSENING.description == "worsening"
        assert TrendDirection.STABLE.description == "stable"
        assert TrendDirection.IMPROVING.description == "improving"


class TestSkuProperties:
    """Test Sku computed properties."""

    def test_total_value(self):
        sku = Sku(
            sku_id="TEST-001",
            qty_on_hand=-47.0,
            unit_cost=23.5,
            retail_price=31.73,
            margin_pct=0.35,
            sales_last_30d=10.0,
            days_since_receipt=30.0,
            is_damaged=False,
            on_order_qty=0.0,
            is_seasonal=False,
        )
        assert abs(sku.total_value - 1104.5) < 0.01

    def test_margin_display(self):
        sku = Sku(
            sku_id="TEST-001",
            qty_on_hand=50.0,
            unit_cost=100.0,
            retail_price=105.0,
            margin_pct=0.05,
            sales_last_30d=10.0,
            days_since_receipt=30.0,
            is_damaged=False,
            on_order_qty=0.0,
            is_seasonal=False,
        )
        assert sku.margin_display == "5%"


class TestSummaryProperties:
    def test_total_dollar_display(self):
        digest = Digest.model_validate_json(SAMPLE_JSON)
        assert digest.summary.total_dollar_display == "$8,527"
