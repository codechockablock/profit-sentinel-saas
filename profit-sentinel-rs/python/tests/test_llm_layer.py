"""Tests for the template-based natural language rendering."""

import json

import pytest
from sentinel_agent.llm_layer import (
    format_dollars,
    format_qty,
    render_digest,
    render_issue_detail,
    render_issue_headline,
)
from sentinel_agent.models import Digest

# Reuse the same sample JSON from test_models
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


class TestFormatHelpers:
    def test_format_dollars_small(self):
        assert format_dollars(500) == "$500"

    def test_format_dollars_thousands(self):
        assert format_dollars(1104.5) == "$1,104"

    def test_format_dollars_large(self):
        assert format_dollars(8526.75) == "$8,527"

    def test_format_dollars_negative(self):
        assert format_dollars(-500) == "-$500"

    def test_format_dollars_zero(self):
        assert format_dollars(0) == "$0"

    def test_format_qty_positive(self):
        assert format_qty(100) == "100 units"

    def test_format_qty_negative(self):
        assert format_qty(-47) == "47 units short"


class TestIssueRendering:
    def setup_method(self):
        self.digest = Digest.model_validate_json(SAMPLE_JSON)

    def test_headline_dead_stock(self):
        headline = render_issue_headline(self.digest.issues[0], 1)
        assert "Store 7" in headline
        assert "Dead Stock" in headline
        assert "$5,000" in headline

    def test_headline_margin_erosion(self):
        headline = render_issue_headline(self.digest.issues[1], 2)
        assert "Store 12" in headline
        assert "Margin Erosion" in headline
        assert "$2,422" in headline

    def test_headline_negative_inventory(self):
        headline = render_issue_headline(self.digest.issues[2], 3)
        assert "Store 7" in headline
        assert "Negative Inventory" in headline
        assert "$1,104" in headline

    def test_detail_single_sku_dead_stock(self):
        detail = render_issue_detail(self.digest.issues[0])
        assert "SEA-1201" in detail
        assert "180" in detail  # days since receipt
        assert "zero sales" in detail

    def test_detail_multi_sku_margin_erosion(self):
        detail = render_issue_detail(self.digest.issues[1])
        assert "2 SKUs" in detail
        assert "35%" in detail  # benchmark

    def test_detail_negative_inventory_shows_qty(self):
        detail = render_issue_detail(self.digest.issues[2])
        assert "47" in detail  # units short
        assert "$23" in detail or "$24" in detail  # unit cost

    def test_detail_has_action(self):
        for issue in self.digest.issues:
            detail = render_issue_detail(issue)
            assert "[" in detail  # has action brackets


class TestDigestRendering:
    def setup_method(self):
        self.digest = Digest.model_validate_json(SAMPLE_JSON)

    def test_greeting(self):
        text = render_digest(self.digest)
        assert text.startswith("Good morning.")

    def test_item_count(self):
        text = render_digest(self.digest)
        assert "3 items need your attention" in text

    def test_total_exposure(self):
        text = render_digest(self.digest)
        assert "$8,527" in text

    def test_stores_affected(self):
        text = render_digest(self.digest)
        assert "2 stores" in text

    def test_all_issues_present(self):
        text = render_digest(self.digest)
        assert "Dead Stock" in text
        assert "Margin Erosion" in text
        assert "Negative Inventory" in text

    def test_pipeline_timing(self):
        text = render_digest(self.digest)
        assert "16ms" in text

    def test_empty_digest(self):
        """No issues should produce 'all clear' message."""
        empty_json = json.dumps(
            {
                "generated_at": "2026-02-05T06:00:00+00:00",
                "store_filter": [],
                "pipeline_ms": 5,
                "issues": [],
                "summary": {
                    "total_issues": 0,
                    "total_dollar_impact": 0.0,
                    "stores_affected": 0,
                    "records_processed": 20,
                    "issues_detected": 0,
                    "issues_filtered_out": 0,
                },
            }
        )
        digest = Digest.model_validate_json(empty_json)
        text = render_digest(digest)
        assert "All clear" in text or "No issues" in text
