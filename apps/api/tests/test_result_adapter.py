"""Tests for the Result Adapter (M2).

Tests the transformation from Rust sentinel-server DigestJson into the
legacy Python API response format expected by the React frontend.

Covers:
- Issue type mapping (Rust enum names → Python primitive keys)
- SKU detail transformation (SkuJson → ItemDetail)
- Context string generation (matches analysis.py exactly)
- Summary statistics aggregation
- Impact estimation from Rust dollar_impact values
- Cause diagnosis extraction
- All 11 primitives present in output (even if empty)
- Frontend contract validation (exact field shapes)
- Edge cases: empty digest, unknown issue types, missing fields
"""

import pytest

from src.services.result_adapter import (
    ALL_PRIMITIVES,
    LEAK_DISPLAY,
    RUST_TYPE_TO_PRIMITIVE,
    RustResultAdapter,
    _get_issue_context,
    _safe_float,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> RustResultAdapter:
    return RustResultAdapter()


@pytest.fixture
def sample_digest() -> dict:
    """Realistic Rust sentinel-server DigestJson output."""
    return {
        "generated_at": "2026-02-06T15:30:45.123456Z",
        "store_filter": ["default"],
        "pipeline_ms": 53022,
        "issues": [
            {
                "id": "issue-001",
                "issue_type": "MarginErosion",
                "store_id": "default",
                "dollar_impact": 47731.08,
                "confidence": 0.89,
                "trend_direction": "Stable",
                "priority_score": 8.5,
                "urgency_score": 7.2,
                "detection_timestamp": "2026-02-06T15:30:00Z",
                "skus": [
                    {
                        "sku_id": "G7958390",
                        "qty_on_hand": 10.0,
                        "unit_cost": 50.00,
                        "retail_price": 55.00,
                        "margin_pct": 0.0909,
                        "sales_last_30d": 5.0,
                        "days_since_receipt": 30.0,
                        "is_damaged": False,
                        "on_order_qty": 0.0,
                        "is_seasonal": False,
                    },
                    {
                        "sku_id": "G7958457",
                        "qty_on_hand": 25.0,
                        "unit_cost": 30.00,
                        "retail_price": 33.00,
                        "margin_pct": 0.0909,
                        "sales_last_30d": 12.0,
                        "days_since_receipt": 45.0,
                        "is_damaged": False,
                        "on_order_qty": 0.0,
                        "is_seasonal": False,
                    },
                ],
                "context": "7189 SKUs with margin below 20%",
                "root_cause": "MarginLeak",
                "root_cause_confidence": 0.87,
                "cause_scores": [
                    {"cause": "MarginLeak", "score": 2.15, "evidence_count": 3},
                    {"cause": "VendorIncrease", "score": 1.80, "evidence_count": 2},
                ],
                "root_cause_ambiguity": 0.15,
                "active_signals": ["low_margin", "high_cost"],
            },
            {
                "id": "issue-002",
                "issue_type": "NegativeInventory",
                "store_id": "default",
                "dollar_impact": 1250.50,
                "confidence": 0.95,
                "trend_direction": "Stable",
                "priority_score": 9.0,
                "urgency_score": 8.0,
                "detection_timestamp": "2026-02-06T15:30:00Z",
                "skus": [
                    {
                        "sku_id": "SKU-NEG1",
                        "qty_on_hand": -47.0,
                        "unit_cost": 23.50,
                        "retail_price": 31.73,
                        "margin_pct": 0.2593,
                        "sales_last_30d": 10.0,
                        "days_since_receipt": 30.0,
                        "is_damaged": False,
                        "on_order_qty": 0.0,
                        "is_seasonal": False,
                    },
                ],
                "context": "1 SKU with negative inventory",
                "root_cause": "Theft",
                "root_cause_confidence": 0.92,
                "cause_scores": [
                    {"cause": "Theft", "score": 2.50, "evidence_count": 4},
                    {"cause": "PricingError", "score": 0.80, "evidence_count": 1},
                ],
                "root_cause_ambiguity": 0.12,
                "active_signals": ["negative_qty"],
            },
            {
                "id": "issue-003",
                "issue_type": "ZeroCostAnomaly",
                "store_id": "default",
                "dollar_impact": 3125.20,
                "confidence": 0.76,
                "trend_direction": "Stable",
                "priority_score": 6.0,
                "urgency_score": 5.0,
                "detection_timestamp": "2026-02-06T15:30:00Z",
                "skus": [
                    {
                        "sku_id": "SKU-ZERO1",
                        "qty_on_hand": 100.0,
                        "unit_cost": 0.0,
                        "retail_price": 19.99,
                        "margin_pct": 1.0,
                        "sales_last_30d": 8.0,
                        "days_since_receipt": 60.0,
                        "is_damaged": False,
                        "on_order_qty": 0.0,
                        "is_seasonal": False,
                    },
                ],
                "context": "123 SKUs with zero cost",
                "root_cause": "PricingError",
                "root_cause_confidence": 0.70,
                "cause_scores": [
                    {"cause": "PricingError", "score": 1.80, "evidence_count": 2},
                ],
                "root_cause_ambiguity": 0.30,
                "active_signals": ["zero_cost"],
            },
        ],
        "summary": {
            "total_issues": 3,
            "total_dollar_impact": 52106.78,
            "stores_affected": 1,
            "records_processed": 36450,
            "issues_detected": 6,
            "issues_filtered_out": 3,
        },
    }


@pytest.fixture
def sample_rows() -> list[dict]:
    """Original DataFrame rows for enrichment."""
    return [
        {
            "sku": "G7958390",
            "description": "Premium Widget",
            "cost": 50.00,
            "revenue": 55.00,
            "quantity": 10.0,
            "sold": 5.0,
            "sub_total": 500.00,
        },
        {
            "sku": "SKU-NEG1",
            "description": "Negative Item",
            "cost": 23.50,
            "revenue": 31.73,
            "quantity": -47.0,
            "sold": 10.0,
            "sub_total": -1104.50,
        },
        {
            "sku": "SKU-ZERO1",
            "description": "Zero Cost Item",
            "cost": 0.0,
            "revenue": 19.99,
            "quantity": 100.0,
            "sold": 8.0,
            "sub_total": 0.0,
        },
    ]


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------


class TestTypeMapping:
    def test_all_rust_types_mapped(self):
        """Every Rust issue type should map to a Python primitive."""
        rust_types = [
            "NegativeInventory",
            "DeadStock",
            "MarginErosion",
            "ReceivingGap",
            "VendorShortShip",
            "PurchasingLeakage",
            "PatronageMiss",
            "ShrinkagePattern",
            "ZeroCostAnomaly",
            "PriceDiscrepancy",
            "Overstock",
        ]
        for rt in rust_types:
            assert rt in RUST_TYPE_TO_PRIMITIVE, f"Missing mapping for {rt}"

    def test_mapped_primitives_are_valid(self):
        """All mapped primitives should be in ALL_PRIMITIVES."""
        for primitive in RUST_TYPE_TO_PRIMITIVE.values():
            assert primitive in ALL_PRIMITIVES, f"{primitive} not in ALL_PRIMITIVES"

    def test_all_11_primitives_defined(self):
        assert len(ALL_PRIMITIVES) == 11

    def test_all_primitives_have_display(self):
        for prim in ALL_PRIMITIVES:
            assert prim in LEAK_DISPLAY, f"Missing display for {prim}"


# ---------------------------------------------------------------------------
# Core transformation
# ---------------------------------------------------------------------------


class TestTransform:
    def test_returns_all_top_level_keys(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        assert "leaks" in result
        assert "summary" in result
        assert "primitives_used" in result

    def test_all_11_primitives_in_leaks(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        for prim in ALL_PRIMITIVES:
            assert prim in result["leaks"], f"Missing primitive: {prim}"

    def test_primitives_used_matches(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        assert result["primitives_used"] == ALL_PRIMITIVES

    def test_margin_erosion_mapped(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        me = result["leaks"]["margin_erosion"]
        assert me["count"] == 2  # 2 SKUs in MarginErosion issue
        assert "G7958390" in me["top_items"]
        assert "G7958457" in me["top_items"]

    def test_negative_inventory_mapped(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        ni = result["leaks"]["negative_inventory"]
        assert ni["count"] == 1
        assert "SKU-NEG1" in ni["top_items"]

    def test_zero_cost_mapped(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        zc = result["leaks"]["zero_cost_anomaly"]
        assert zc["count"] == 1
        assert "SKU-ZERO1" in zc["top_items"]

    def test_empty_primitives_have_zero_count(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        # These primitives have no matching Rust issues
        for prim in [
            "low_stock",
            "dead_item",
            "negative_profit",
            "severe_inventory_deficit",
        ]:
            assert result["leaks"][prim]["count"] == 0
            assert result["leaks"][prim]["item_details"] == []


# ---------------------------------------------------------------------------
# LeakData shape validation
# ---------------------------------------------------------------------------


class TestLeakDataShape:
    """Verify each LeakData entry matches the frontend's TypeScript interface."""

    def test_all_fields_present(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        required_keys = {
            "top_items",
            "scores",
            "item_details",
            "count",
            "severity",
            "category",
            "recommendations",
            "title",
            "icon",
            "color",
            "priority",
        }
        for prim, leak in result["leaks"].items():
            for key in required_keys:
                assert key in leak, f"Missing key '{key}' in leaks[{prim}]"

    def test_severity_values(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        valid = {"critical", "high", "medium", "low", "info"}
        for prim, leak in result["leaks"].items():
            assert (
                leak["severity"] in valid
            ), f"Bad severity in {prim}: {leak['severity']}"

    def test_color_is_hex(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        for prim, leak in result["leaks"].items():
            assert leak["color"].startswith(
                "#"
            ), f"Bad color in {prim}: {leak['color']}"

    def test_recommendations_are_list(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        for prim, leak in result["leaks"].items():
            assert isinstance(leak["recommendations"], list)

    def test_priority_is_int(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        for prim, leak in result["leaks"].items():
            assert isinstance(leak["priority"], int)


# ---------------------------------------------------------------------------
# ItemDetail shape validation
# ---------------------------------------------------------------------------


class TestItemDetailShape:
    """Verify ItemDetail matches frontend TypeScript interface."""

    def test_all_fields_present(self, adapter, sample_digest, sample_rows):
        result = adapter.transform(
            sample_digest,
            total_rows=36452,
            analysis_time=2.5,
            original_rows=sample_rows,
        )
        me = result["leaks"]["margin_erosion"]
        required_keys = {
            "sku",
            "score",
            "description",
            "quantity",
            "cost",
            "revenue",
            "sold",
            "margin",
            "sub_total",
            "context",
        }
        for item in me["item_details"]:
            for key in required_keys:
                assert key in item, f"Missing key '{key}' in item {item.get('sku')}"

    def test_score_in_range(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        for prim, leak in result["leaks"].items():
            for item in leak["item_details"]:
                assert (
                    0 <= item["score"] <= 1.0
                ), f"Score {item['score']} out of range in {prim}/{item['sku']}"

    def test_enrichment_adds_description(self, adapter, sample_digest, sample_rows):
        result = adapter.transform(
            sample_digest,
            total_rows=36452,
            analysis_time=2.5,
            original_rows=sample_rows,
        )
        me = result["leaks"]["margin_erosion"]
        g79 = next(i for i in me["item_details"] if i["sku"] == "G7958390")
        assert g79["description"] == "Premium Widget"

    def test_enrichment_adds_sub_total(self, adapter, sample_digest, sample_rows):
        result = adapter.transform(
            sample_digest,
            total_rows=36452,
            analysis_time=2.5,
            original_rows=sample_rows,
        )
        me = result["leaks"]["margin_erosion"]
        g79 = next(i for i in me["item_details"] if i["sku"] == "G7958390")
        assert g79["sub_total"] == 500.00

    def test_margin_converted_to_percentage(self, adapter, sample_digest):
        """Rust margin is 0.09, display should be ~9.09%."""
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        me = result["leaks"]["margin_erosion"]
        g79 = next(i for i in me["item_details"] if i["sku"] == "G7958390")
        assert 9.0 < g79["margin"] < 10.0  # 0.0909 → 9.09%

    def test_context_is_string(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        for prim, leak in result["leaks"].items():
            for item in leak["item_details"]:
                assert isinstance(item["context"], str)
                assert len(item["context"]) > 0


# ---------------------------------------------------------------------------
# Summary shape validation
# ---------------------------------------------------------------------------


class TestSummaryShape:
    def test_all_summary_fields(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        summary = result["summary"]
        assert summary["total_rows_analyzed"] == 36452
        assert summary["analysis_time_seconds"] == 2.5
        assert isinstance(summary["total_items_flagged"], int)
        assert isinstance(summary["critical_issues"], int)
        assert isinstance(summary["high_issues"], int)
        assert "estimated_impact" in summary

    def test_estimated_impact_shape(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        impact = result["summary"]["estimated_impact"]
        assert impact["currency"] == "USD"
        assert isinstance(impact["low_estimate"], float)
        assert isinstance(impact["high_estimate"], float)
        assert isinstance(impact["breakdown"], dict)
        assert impact["low_estimate"] <= impact["high_estimate"]

    def test_all_primitives_in_breakdown(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        breakdown = result["summary"]["estimated_impact"]["breakdown"]
        for prim in ALL_PRIMITIVES:
            assert prim in breakdown, f"Missing {prim} in breakdown"

    def test_negative_inventory_excluded_from_estimate(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        breakdown = result["summary"]["estimated_impact"]["breakdown"]
        assert breakdown["negative_inventory"] == 0.0

    def test_dollar_impacts_from_rust(self, adapter, sample_digest):
        """Rust's dollar_impact values should flow through to breakdown."""
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        breakdown = result["summary"]["estimated_impact"]["breakdown"]
        assert breakdown["margin_erosion"] == 47731.08
        assert breakdown["zero_cost_anomaly"] == 3125.20

    def test_negative_inventory_alert(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        alert = result["summary"]["estimated_impact"]["negative_inventory_alert"]
        assert alert is not None
        assert alert["items_found"] == 1
        assert alert["requires_audit"] is True
        assert alert["excluded_from_annual_estimate"] is True

    def test_total_items_flagged(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        # 2 margin_erosion + 1 negative_inventory + 1 zero_cost = 4
        assert result["summary"]["total_items_flagged"] == 4


# ---------------------------------------------------------------------------
# Cause diagnosis
# ---------------------------------------------------------------------------


class TestCauseDiagnosis:
    def test_extracts_top_cause(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        assert "cause_diagnosis" in result
        cd = result["cause_diagnosis"]
        assert cd["top_cause"] is not None
        assert cd["confidence"] > 0

    def test_hypotheses_present(self, adapter, sample_digest):
        result = adapter.transform(sample_digest, total_rows=36452, analysis_time=2.5)
        cd = result["cause_diagnosis"]
        assert "hypotheses" in cd
        assert len(cd["hypotheses"]) > 0
        for h in cd["hypotheses"]:
            assert "cause" in h
            assert "probability" in h
            assert "evidence" in h

    def test_no_diagnosis_for_empty_issues(self, adapter):
        digest = {"issues": [], "summary": {}}
        result = adapter.transform(digest, total_rows=0, analysis_time=0.0)
        assert "cause_diagnosis" not in result


# ---------------------------------------------------------------------------
# Context generation
# ---------------------------------------------------------------------------


class TestContextGeneration:
    def test_negative_inventory_context(self):
        item = {"quantity": -5, "cost": 50.0, "revenue": 89.99}
        ctx = _get_issue_context("negative_inventory", item)
        assert "NEGATIVE" in ctx
        assert "$250.00" in ctx

    def test_margin_erosion_context(self):
        item = {"quantity": 10, "cost": 50.0, "revenue": 55.0, "sold": 5}
        ctx = _get_issue_context("margin_erosion", item)
        assert "9.1%" in ctx  # (55-50)/55 = 9.09%

    def test_zero_cost_context(self):
        item = {"quantity": 100, "cost": 0.0, "revenue": 19.99, "sold": 8}
        ctx = _get_issue_context("zero_cost_anomaly", item)
        assert "$0.00" in ctx
        assert "$19.99" in ctx

    def test_dead_item_zero_sales(self):
        item = {"quantity": 50, "cost": 10.0, "revenue": 20.0, "sold": 0}
        ctx = _get_issue_context("dead_item", item)
        assert "No sales" in ctx
        assert "$500.00" in ctx

    def test_negative_profit_context(self):
        item = {"quantity": 10, "cost": 25.0, "revenue": 20.0, "sold": 5}
        ctx = _get_issue_context("negative_profit", item)
        assert "LOSS" in ctx
        assert "$5.00" in ctx


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_digest(self, adapter):
        result = adapter.transform(
            {"issues": [], "summary": {}},
            total_rows=0,
            analysis_time=0.0,
        )
        assert all(result["leaks"][p]["count"] == 0 for p in ALL_PRIMITIVES)
        assert result["summary"]["total_items_flagged"] == 0

    def test_unknown_issue_type_logged(self, adapter, caplog):
        digest = {
            "issues": [
                {
                    "issue_type": "UnknownBugType",
                    "skus": [],
                    "confidence": 0.5,
                    "dollar_impact": 0,
                }
            ],
            "summary": {},
        }
        result = adapter.transform(digest, total_rows=10, analysis_time=0.1)
        # Should not crash, unknown type should be logged
        assert result["summary"]["total_items_flagged"] == 0

    def test_missing_sku_fields_use_defaults(self, adapter):
        digest = {
            "issues": [
                {
                    "issue_type": "MarginErosion",
                    "store_id": "default",
                    "dollar_impact": 100.0,
                    "confidence": 0.5,
                    "skus": [{"sku_id": "MINIMAL"}],  # Minimal SKU data
                    "cause_scores": [],
                }
            ],
            "summary": {},
        }
        result = adapter.transform(digest, total_rows=10, analysis_time=0.1)
        me = result["leaks"]["margin_erosion"]
        assert me["count"] == 1
        item = me["item_details"][0]
        assert item["sku"] == "MINIMAL"
        assert item["cost"] == 0.0
        assert item["revenue"] == 0.0

    def test_duplicate_skus_deduplicated(self, adapter):
        """Same SKU in multiple issues of same type should not duplicate."""
        digest = {
            "issues": [
                {
                    "issue_type": "MarginErosion",
                    "store_id": "default",
                    "dollar_impact": 50.0,
                    "confidence": 0.5,
                    "skus": [{"sku_id": "DUP-SKU", "qty_on_hand": 10}],
                    "cause_scores": [],
                },
                {
                    "issue_type": "MarginErosion",
                    "store_id": "store-2",
                    "dollar_impact": 50.0,
                    "confidence": 0.5,
                    "skus": [{"sku_id": "DUP-SKU", "qty_on_hand": 10}],
                    "cause_scores": [],
                },
            ],
            "summary": {},
        }
        result = adapter.transform(digest, total_rows=10, analysis_time=0.1)
        me = result["leaks"]["margin_erosion"]
        # Should be deduplicated
        assert me["count"] == 1

    def test_no_original_rows_works(self, adapter, sample_digest):
        """Should work without enrichment data."""
        result = adapter.transform(
            sample_digest,
            total_rows=36452,
            analysis_time=2.5,
            original_rows=None,
        )
        me = result["leaks"]["margin_erosion"]
        assert me["count"] == 2
        # Description should be empty without enrichment
        item = me["item_details"][0]
        assert item["description"] == ""

    def test_sanity_cap_applied(self, adapter):
        """Impact > $10M should be capped."""
        digest = {
            "issues": [
                {
                    "issue_type": "MarginErosion",
                    "store_id": "default",
                    "dollar_impact": 50_000_000,  # $50M — unrealistic
                    "confidence": 0.9,
                    "skus": [{"sku_id": "BIG"}],
                    "cause_scores": [],
                }
            ],
            "summary": {},
        }
        result = adapter.transform(digest, total_rows=100, analysis_time=0.1)
        impact = result["summary"]["estimated_impact"]
        assert impact["high_estimate"] <= 10_000_000


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_none(self):
        assert _safe_float(None) == 0.0

    def test_string(self):
        assert _safe_float("3.14") == 3.14

    def test_invalid(self):
        assert _safe_float("bad") == 0.0
