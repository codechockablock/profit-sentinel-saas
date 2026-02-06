"""Tests for the VendorCallAssistant."""

import json

import pytest
from sentinel_agent.models import Digest, IssueType
from sentinel_agent.vendor_assist import VendorCallAssistant

SAMPLE_JSON = json.dumps(
    {
        "generated_at": "2026-02-05T06:00:00+00:00",
        "store_filter": ["store-3"],
        "pipeline_ms": 10,
        "issues": [
            {
                "id": "store-3-VendorShortShip-001",
                "issue_type": "VendorShortShip",
                "store_id": "store-3",
                "dollar_impact": 2250.0,
                "confidence": 0.89,
                "trend_direction": "Worsening",
                "priority_score": 9.83,
                "urgency_score": 0.9,
                "detection_timestamp": "2025-01-02T00:00:00Z",
                "skus": [
                    {
                        "sku_id": "DMG-0101",
                        "qty_on_hand": 10.0,
                        "unit_cost": 150.0,
                        "retail_price": 187.5,
                        "margin_pct": 0.25,
                        "sales_last_30d": 3.0,
                        "days_since_receipt": 60.0,
                        "is_damaged": True,
                        "on_order_qty": 20.0,
                        "is_seasonal": False,
                    }
                ],
                "context": "Damaged goods with active purchase orders.",
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
                ],
                "context": "Low margin items.",
            },
        ],
        "summary": {
            "total_issues": 2,
            "total_dollar_impact": 4672.25,
            "stores_affected": 2,
            "records_processed": 10,
            "issues_detected": 2,
            "issues_filtered_out": 0,
        },
    }
)


class TestVendorCallAssistant:
    def setup_method(self):
        self.assistant = VendorCallAssistant()
        self.digest = Digest.model_validate_json(SAMPLE_JSON)

    def test_prepare_call_vendor_short_ship(self):
        issue = self.digest.issues[0]
        prep = self.assistant.prepare_call(issue)

        assert prep.issue_id == "store-3-VendorShortShip-001"
        assert prep.store_id == "store-3"
        assert prep.total_dollar_impact == 2250.0
        assert len(prep.affected_skus) == 1
        assert prep.affected_skus[0].sku_id == "DMG-0101"

    def test_vendor_name_lookup(self):
        """DMG prefix should map to Martin's Supply Co."""
        issue = self.digest.issues[0]
        prep = self.assistant.prepare_call(issue)
        assert "Martin" in prep.vendor_name

    def test_vendor_name_lookup_paint(self):
        """PNT prefix should map to ColorMax."""
        issue = self.digest.issues[1]
        prep = self.assistant.prepare_call(issue)
        assert "Color" in prep.vendor_name or "Paint" in prep.vendor_name

    def test_talking_points_vendor_short_ship(self):
        issue = self.digest.issues[0]
        prep = self.assistant.prepare_call(issue)
        assert len(prep.talking_points) >= 2
        # Should mention damaged goods
        assert any("damaged" in p.lower() for p in prep.talking_points)

    def test_talking_points_margin_erosion(self):
        issue = self.digest.issues[1]
        prep = self.assistant.prepare_call(issue)
        assert len(prep.talking_points) >= 2
        # Should mention margin
        assert any("margin" in p.lower() for p in prep.talking_points)

    def test_questions_vendor_short_ship(self):
        issue = self.digest.issues[0]
        prep = self.assistant.prepare_call(issue)
        assert len(prep.questions_to_ask) >= 2
        # Should ask about quality or claims
        assert any(
            "quality" in q.lower() or "claim" in q.lower()
            for q in prep.questions_to_ask
        )

    def test_questions_mention_pending_order(self):
        """With on_order_qty > 0, should mention pending order."""
        issue = self.digest.issues[0]
        prep = self.assistant.prepare_call(issue)
        all_text = " ".join(prep.talking_points + prep.questions_to_ask)
        assert "order" in all_text.lower()

    def test_historical_context(self):
        issue = self.digest.issues[0]
        prep = self.assistant.prepare_call(issue)
        assert prep.historical_context  # not empty
        assert (
            "issue" in prep.historical_context.lower()
            or "claim" in prep.historical_context.lower()
        )

    def test_render_call_prep(self):
        issue = self.digest.issues[0]
        prep = self.assistant.prepare_call(issue)
        text = self.assistant.render(prep)

        assert "VENDOR CALL BRIEF" in text
        assert "Martin" in text
        assert "DMG-0101" in text
        assert "Talking Points" in text
        assert "Questions" in text
