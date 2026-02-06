"""Tests for the NormalizedInventory → InventoryRecord pipeline bridge.

Tests cover:
- Field mapping (rename, type coercion)
- Derived field computation (margin_pct, days_since_receipt, sales_last_30d)
- Seasonal detection heuristic
- CSV output format (matches Rust InventoryRecord exactly)
- Bidirectional enrichment index
- Edge cases (zero retail, no dates, negative qty, empty records)
"""

import csv
import tempfile
from datetime import date
from pathlib import Path

import pytest
from sentinel_agent.adapters.base import NormalizedInventory
from sentinel_agent.adapters.bridge import PipelineBridge, to_pipeline_csv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bridge():
    """Bridge with a fixed reference date for deterministic tests."""
    return PipelineBridge(
        reference_date=date(2025, 7, 1),
        months_elapsed_ytd=6.0,
    )


@pytest.fixture
def sample_records():
    """Diverse set of NormalizedInventory records for testing."""
    return [
        NormalizedInventory(
            sku_id="ELC-4401",
            description="Wire Stripper Pro",
            vendor="ORGILL",
            vendor_sku="4401",
            qty_on_hand=-47,
            unit_cost=23.50,
            retail_price=31.73,
            last_receipt_date=date(2025, 6, 1),
            last_sale_date=date(2025, 6, 15),
            bin_location="A1-03",
            store_id="default-store",
            category="Electrical",
            department="15",
            barcode="029069752040",
            on_order_qty=0,
            sales_ytd=1200.00,
        ),
        NormalizedInventory(
            sku_id="SEA-1201",
            description="Dead Stock Widget",
            vendor="SEACOAST",
            qty_on_hand=100,
            unit_cost=50.00,
            retail_price=67.50,
            last_receipt_date=date(2025, 1, 2),
            store_id="default-store",
            sales_ytd=0.0,
        ),
        NormalizedInventory(
            sku_id="PNT-1001",
            description="Paint Brush Set",
            vendor="COLORMAX",
            qty_on_hand=50,
            unit_cost=100.00,
            retail_price=105.00,
            last_receipt_date=date(2025, 6, 1),
            store_id="default-store",
            department="Seasonal",
            sales_ytd=600.00,
        ),
        NormalizedInventory(
            sku_id="ZERO-RETAIL",
            description="No Retail Price Item",
            qty_on_hand=10,
            unit_cost=5.00,
            retail_price=0.0,
            store_id="default-store",
        ),
        NormalizedInventory(
            sku_id="NO-DATES",
            description="No Dates Item",
            qty_on_hand=25,
            unit_cost=15.00,
            retail_price=25.00,
            store_id="default-store",
            sales_ytd=300.00,
        ),
    ]


# ---------------------------------------------------------------------------
# Unit Tests — Field Mapping
# ---------------------------------------------------------------------------


class TestFieldMapping:
    def test_sku_renamed(self, bridge, sample_records):
        row = bridge.convert_record(sample_records[0])
        assert row["sku"] == "ELC-4401"
        assert "sku_id" not in row

    def test_store_id_preserved(self, bridge, sample_records):
        row = bridge.convert_record(sample_records[0])
        assert row["store_id"] == "default-store"

    def test_qty_as_float(self, bridge, sample_records):
        row = bridge.convert_record(sample_records[0])
        assert row["qty_on_hand"] == "-47.0"

    def test_negative_qty_preserved(self, bridge, sample_records):
        row = bridge.convert_record(sample_records[0])
        assert float(row["qty_on_hand"]) == -47.0

    def test_unit_cost_formatted(self, bridge, sample_records):
        row = bridge.convert_record(sample_records[0])
        assert row["unit_cost"] == "23.50"

    def test_retail_price_formatted(self, bridge, sample_records):
        row = bridge.convert_record(sample_records[0])
        assert row["retail_price"] == "31.73"

    def test_on_order_qty_as_float(self, bridge, sample_records):
        row = bridge.convert_record(sample_records[0])
        assert row["on_order_qty"] == "0.0"

    def test_is_damaged_default_false(self, bridge, sample_records):
        row = bridge.convert_record(sample_records[0])
        assert row["is_damaged"] == "false"

    def test_all_pipeline_columns_present(self, bridge, sample_records):
        row = bridge.convert_record(sample_records[0])
        expected_columns = {
            "store_id",
            "sku",
            "qty_on_hand",
            "unit_cost",
            "margin_pct",
            "sales_last_30d",
            "days_since_receipt",
            "retail_price",
            "is_damaged",
            "on_order_qty",
            "is_seasonal",
        }
        assert set(row.keys()) == expected_columns

    def test_no_extra_columns(self, bridge, sample_records):
        """Bridge must NOT include adapter-only fields (description, vendor, etc.)."""
        row = bridge.convert_record(sample_records[0])
        for forbidden in (
            "description",
            "vendor",
            "vendor_sku",
            "barcode",
            "bin_location",
            "category",
            "department",
            "last_receipt_date",
            "last_sale_date",
        ):
            assert forbidden not in row


# ---------------------------------------------------------------------------
# Unit Tests — Derived Fields
# ---------------------------------------------------------------------------


class TestDerivedFields:
    def test_margin_pct_normal(self, bridge, sample_records):
        """margin_pct = (retail - cost) / retail = (31.73 - 23.50) / 31.73 ≈ 0.2594"""
        row = bridge.convert_record(sample_records[0])
        margin = float(row["margin_pct"])
        expected = (31.73 - 23.50) / 31.73
        assert abs(margin - expected) < 0.001

    def test_margin_pct_zero_retail(self, bridge, sample_records):
        """When retail = 0, margin should be 0.0."""
        row = bridge.convert_record(sample_records[3])  # ZERO-RETAIL
        assert float(row["margin_pct"]) == 0.0

    def test_margin_pct_low_margin(self, bridge, sample_records):
        """PNT-1001: cost=100, retail=105, margin=(105-100)/105 ≈ 0.0476"""
        row = bridge.convert_record(sample_records[2])
        margin = float(row["margin_pct"])
        expected = (105.0 - 100.0) / 105.0
        assert abs(margin - expected) < 0.001

    def test_days_since_receipt_computed(self, bridge, sample_records):
        """ELC-4401: receipt date 2025-06-01, reference 2025-07-01 = 30 days."""
        row = bridge.convert_record(sample_records[0])
        assert float(row["days_since_receipt"]) == 30.0

    def test_days_since_receipt_old_stock(self, bridge, sample_records):
        """SEA-1201: receipt date 2025-01-02, reference 2025-07-01 = 180 days."""
        row = bridge.convert_record(sample_records[1])
        assert float(row["days_since_receipt"]) == 180.0

    def test_days_since_receipt_no_date(self, bridge, sample_records):
        """NO-DATES: no receipt date → default 365."""
        row = bridge.convert_record(sample_records[4])
        assert float(row["days_since_receipt"]) == 365.0

    def test_sales_last_30d_from_ytd(self, bridge, sample_records):
        """ELC-4401: sales_ytd=1200, 6 months → 200/month."""
        row = bridge.convert_record(sample_records[0])
        assert abs(float(row["sales_last_30d"]) - 200.0) < 0.01

    def test_sales_last_30d_zero_when_no_sales(self, bridge, sample_records):
        """SEA-1201: sales_ytd=0 → sales_last_30d=0."""
        row = bridge.convert_record(sample_records[1])
        assert float(row["sales_last_30d"]) == 0.0

    def test_sales_last_30d_nonzero(self, bridge, sample_records):
        """PNT-1001: sales_ytd=600, 6 months → 100/month."""
        row = bridge.convert_record(sample_records[2])
        assert abs(float(row["sales_last_30d"]) - 100.0) < 0.01


# ---------------------------------------------------------------------------
# Unit Tests — Seasonal Detection
# ---------------------------------------------------------------------------


class TestSeasonalDetection:
    def test_seasonal_department_keyword(self, bridge):
        rec = NormalizedInventory(
            sku_id="X",
            department="Seasonal",
            retail_price=10.0,
        )
        row = bridge.convert_record(rec)
        assert row["is_seasonal"] == "true"

    def test_seasonal_christmas(self, bridge):
        rec = NormalizedInventory(
            sku_id="X",
            department="Christmas Decorations",
            retail_price=10.0,
        )
        row = bridge.convert_record(rec)
        assert row["is_seasonal"] == "true"

    def test_seasonal_lawn_garden(self, bridge):
        rec = NormalizedInventory(
            sku_id="X",
            category="Lawn & Garden",
            retail_price=10.0,
        )
        row = bridge.convert_record(rec)
        assert row["is_seasonal"] == "true"

    def test_non_seasonal(self, bridge):
        rec = NormalizedInventory(
            sku_id="X",
            department="Plumbing",
            retail_price=10.0,
        )
        row = bridge.convert_record(rec)
        assert row["is_seasonal"] == "false"

    def test_no_department_not_seasonal(self, bridge):
        rec = NormalizedInventory(sku_id="X", retail_price=10.0)
        row = bridge.convert_record(rec)
        assert row["is_seasonal"] == "false"


# ---------------------------------------------------------------------------
# Unit Tests — CSV Output
# ---------------------------------------------------------------------------


class TestCSVOutput:
    def test_csv_has_correct_header(self, bridge, sample_records):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)

        try:
            bridge.to_pipeline_csv(sample_records, path)
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader)
            assert header == [
                "store_id",
                "sku",
                "qty_on_hand",
                "unit_cost",
                "margin_pct",
                "sales_last_30d",
                "days_since_receipt",
                "retail_price",
                "is_damaged",
                "on_order_qty",
                "is_seasonal",
            ]
        finally:
            path.unlink(missing_ok=True)

    def test_csv_row_count(self, bridge, sample_records):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)

        try:
            bridge.to_pipeline_csv(sample_records, path)
            with open(path) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                rows = list(reader)
            assert len(rows) == len(sample_records)
        finally:
            path.unlink(missing_ok=True)

    def test_csv_readable_by_rust_format(self, bridge, sample_records):
        """Verify CSV is parseable as the Rust InventoryRecord expects."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)

        try:
            bridge.to_pipeline_csv(sample_records, path)
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # All Rust f64 fields should be valid floats
                    for field in (
                        "qty_on_hand",
                        "unit_cost",
                        "margin_pct",
                        "sales_last_30d",
                        "days_since_receipt",
                        "retail_price",
                        "on_order_qty",
                    ):
                        float(row[field])  # Should not raise

                    # Bool fields should be "true" or "false"
                    assert row["is_damaged"] in ("true", "false")
                    assert row["is_seasonal"] in ("true", "false")
        finally:
            path.unlink(missing_ok=True)

    def test_temp_file_when_no_path(self, bridge, sample_records):
        path = bridge.to_pipeline_csv(sample_records)
        try:
            assert path.exists()
            assert path.suffix == ".csv"
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == len(sample_records) + 1  # header + records
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Unit Tests — Enrichment Index
# ---------------------------------------------------------------------------


class TestEnrichmentIndex:
    def test_index_by_store_sku(self, bridge, sample_records):
        index = bridge.build_enrichment_index(sample_records)
        rec = index["default-store::ELC-4401"]
        assert rec.description == "Wire Stripper Pro"
        assert rec.vendor == "ORGILL"

    def test_index_by_bare_sku(self, bridge, sample_records):
        index = bridge.build_enrichment_index(sample_records)
        rec = index["SEA-1201"]
        assert rec.description == "Dead Stock Widget"

    def test_all_records_indexed(self, bridge, sample_records):
        index = bridge.build_enrichment_index(sample_records)
        for rec in sample_records:
            key = f"{rec.store_id}::{rec.sku_id}"
            assert key in index

    def test_enrichment_returns_full_record(self, bridge, sample_records):
        index = bridge.build_enrichment_index(sample_records)
        rec = index["default-store::ELC-4401"]
        assert rec.bin_location == "A1-03"
        assert rec.barcode == "029069752040"
        assert rec.category == "Electrical"


# ---------------------------------------------------------------------------
# Unit Tests — Convenience Function
# ---------------------------------------------------------------------------


class TestConvenienceFunction:
    def test_to_pipeline_csv(self, sample_records):
        path = to_pipeline_csv(sample_records)
        try:
            assert path.exists()
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == len(sample_records)
            assert rows[0]["sku"] == "ELC-4401"
        finally:
            path.unlink(missing_ok=True)

    def test_with_custom_date(self, sample_records):
        path = to_pipeline_csv(
            sample_records,
            reference_date=date(2025, 7, 1),
        )
        try:
            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
            # ELC-4401: receipt 2025-06-01, ref 2025-07-01 = 30 days
            assert float(row["days_since_receipt"]) == 30.0
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_records_list(self, bridge):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            bridge.to_pipeline_csv([], path)
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)
            assert len(header) == 11
            assert len(rows) == 0
        finally:
            path.unlink(missing_ok=True)

    def test_future_receipt_date(self, bridge):
        """Receipt date in the future → days_since_receipt = 0."""
        rec = NormalizedInventory(
            sku_id="FUTURE",
            last_receipt_date=date(2025, 12, 31),
            retail_price=10.0,
        )
        row = bridge.convert_record(rec)
        assert float(row["days_since_receipt"]) == 0.0

    def test_very_old_receipt_date(self, bridge):
        """Receipt date years ago → large days_since_receipt."""
        rec = NormalizedInventory(
            sku_id="OLD",
            last_receipt_date=date(2020, 1, 1),
            retail_price=10.0,
        )
        row = bridge.convert_record(rec)
        days = float(row["days_since_receipt"])
        assert days > 1900  # > 5 years

    def test_negative_cost(self, bridge):
        """Negative cost (data error) should not crash."""
        rec = NormalizedInventory(
            sku_id="NEG-COST",
            unit_cost=-5.0,
            retail_price=10.0,
        )
        row = bridge.convert_record(rec)
        assert float(row["unit_cost"]) == -5.0
        # margin = (10 - (-5)) / 10 = 1.5 — data error but handled
        assert float(row["margin_pct"]) == 1.5

    def test_months_elapsed_auto_calculated(self):
        """Auto-calculate months_elapsed from reference_date."""
        bridge = PipelineBridge(reference_date=date(2025, 3, 15))
        assert bridge._months_elapsed == 3

    def test_months_elapsed_january(self):
        """January = 1 month elapsed (avoid divide-by-zero)."""
        bridge = PipelineBridge(reference_date=date(2025, 1, 15))
        assert bridge._months_elapsed == 1
