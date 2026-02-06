"""Tests for the Column Mapping Adapter (M1).

Tests the transformation from post-mapping pandas DataFrames to the strict
11-column CSV format expected by the Rust sentinel-server binary.

Covers:
- Paladin POS mapping (production CSV schema)
- Field resolution with aliases
- Margin computation (direct, calculated, percentage normalization)
- Date parsing and days_since_receipt computation
- Default values for missing fields
- Seasonal detection heuristic
- Edge cases: empty DataFrames, missing SKU, NaN handling
- CSV output validation (column order, types, formatting)
"""

import csv
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.services.column_adapter import (
    RUST_COLUMNS,
    ColumnAdapter,
    _detect_seasonal_from_category,
    _find_column,
    _parse_date,
    _safe_float,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paladin_df() -> pd.DataFrame:
    """Real Paladin POS DataFrame (post-mapping via df.rename())."""
    return pd.DataFrame(
        {
            "sku": ["ELC-4401", "SEA-1201", "PNT-1001", "HRD-9901", "GRD-5501"],
            "description": ["Electronics", "Sealant", "Paint", "Hardware", "Garden"],
            "quantity": [100, 0, 50, -5, 300],
            "cost": [23.50, 50.00, 100.00, 200.00, 25.00],
            "revenue": [31.73, 67.50, 105.00, 260.00, 33.75],
            "sold": [10, 0, 10, 5, 5],
            "margin": [25.9, 25.9, 4.8, 23.1, 25.9],  # Paladin uses percentage
            "sub_total": [2350.00, 0.00, 5000.00, -1000.00, 7500.00],
            "vendor": ["Vendor A", "Vendor B", "Vendor A", "Vendor C", "Vendor D"],
            "category": ["Electronics", "Plumbing", "Paint", "Tools", "Lawn & Garden"],
        }
    )


@pytest.fixture
def paladin_unmapped_df() -> pd.DataFrame:
    """Paladin POS DataFrame BEFORE rename (original column names)."""
    return pd.DataFrame(
        {
            "SKU": ["ELC-4401", "SEA-1201", "PNT-1001"],
            "Description ": ["Electronics", "Sealant", "Paint"],
            "In Stock Qty.": [100, 0, 50],
            "Cost": [23.50, 50.00, 100.00],
            "Retail": [31.73, 67.50, 105.00],
            "Sold": [10, 0, 10],
            "Profit Margin %": [25.9, 25.9, 4.8],
            "Category": ["Electronics", "Plumbing", "Paint"],
            "Last Pur.": ["2026-01-15", "2025-06-01", ""],
        }
    )


@pytest.fixture
def adapter() -> ColumnAdapter:
    """Standard adapter with fixed reference date for reproducibility."""
    return ColumnAdapter(reference_date=date(2026, 2, 6))


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Minimal DataFrame with just SKU and cost."""
    return pd.DataFrame(
        {
            "sku": ["A", "B", "C"],
            "cost": [10.0, 20.0, 30.0],
        }
    )


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_number(self):
        assert _safe_float("42.5") == 42.5

    def test_dollar_string(self):
        assert _safe_float("$1,234.56") == 1234.56

    def test_percentage_string(self):
        assert _safe_float("35%") == 35.0

    def test_none(self):
        assert _safe_float(None) == 0.0

    def test_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_empty_string(self):
        assert _safe_float("") == 0.0

    def test_dash(self):
        assert _safe_float("-") == 0.0

    def test_invalid(self):
        assert _safe_float("not a number") == 0.0

    def test_custom_default(self):
        assert _safe_float(None, default=99.0) == 99.0


# ---------------------------------------------------------------------------
# _find_column
# ---------------------------------------------------------------------------


class TestFindColumn:
    def test_exact_match(self):
        assert _find_column(["sku", "cost", "qty"], ["sku"]) == "sku"

    def test_alias_match(self):
        assert _find_column(["SKU", "Cost", "Qty."], ["sku", "SKU"]) == "SKU"

    def test_case_insensitive_fallback(self):
        assert _find_column(["SKU", "Cost"], ["sku"]) == "SKU"

    def test_no_match(self):
        assert _find_column(["foo", "bar"], ["sku", "barcode"]) is None

    def test_first_match_wins(self):
        result = _find_column(["revenue", "Retail"], ["revenue", "Retail"])
        assert result == "revenue"

    def test_strips_whitespace_in_matching(self):
        assert _find_column(["SKU ", "Cost"], ["sku"]) == "SKU "


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------


class TestParseDate:
    def test_iso_format(self):
        assert _parse_date("2026-01-15") == date(2026, 1, 15)

    def test_us_format(self):
        assert _parse_date("01/15/2026") == date(2026, 1, 15)

    def test_datetime_object(self):
        from datetime import datetime

        dt = datetime(2026, 1, 15, 10, 30)
        assert _parse_date(dt) == date(2026, 1, 15)

    def test_date_object(self):
        d = date(2026, 1, 15)
        assert _parse_date(d) == d

    def test_none(self):
        assert _parse_date(None) is None

    def test_nan(self):
        assert _parse_date(float("nan")) is None

    def test_empty_string(self):
        assert _parse_date("") is None

    def test_nat_string(self):
        assert _parse_date("NaT") is None

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2026-01-15")
        assert _parse_date(ts) == date(2026, 1, 15)


# ---------------------------------------------------------------------------
# _detect_seasonal_from_category
# ---------------------------------------------------------------------------


class TestDetectSeasonal:
    def test_lawn_and_garden(self):
        assert _detect_seasonal_from_category("Lawn & Garden") is True

    def test_christmas(self):
        assert _detect_seasonal_from_category("Christmas Decor") is True

    def test_electronics(self):
        assert _detect_seasonal_from_category("Electronics") is False

    def test_none(self):
        assert _detect_seasonal_from_category(None) is False

    def test_nan(self):
        assert _detect_seasonal_from_category(float("nan")) is False

    def test_pool(self):
        assert _detect_seasonal_from_category("Pool Supplies") is True

    def test_case_insensitive(self):
        assert _detect_seasonal_from_category("SEASONAL items") is True


# ---------------------------------------------------------------------------
# ColumnAdapter — Paladin POS (post-mapping)
# ---------------------------------------------------------------------------


class TestPaladinMapping:
    """Tests with Paladin POS data that has been through df.rename()."""

    def test_produces_11_column_csv(self, adapter, paladin_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == RUST_COLUMNS
            rows = list(reader)
        assert len(rows) == 5

    def test_sku_preserved(self, adapter, paladin_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        skus = [r["sku"] for r in rows]
        assert "ELC-4401" in skus
        assert "SEA-1201" in skus
        assert "GRD-5501" in skus

    def test_quantity_maps_to_qty_on_hand(self, adapter, paladin_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        elc = next(r for r in rows if r["sku"] == "ELC-4401")
        assert float(elc["qty_on_hand"]) == 100.0

    def test_negative_quantity_preserved(self, adapter, paladin_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        hrd = next(r for r in rows if r["sku"] == "HRD-9901")
        assert float(hrd["qty_on_hand"]) == -5.0

    def test_cost_maps_to_unit_cost(self, adapter, paladin_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        elc = next(r for r in rows if r["sku"] == "ELC-4401")
        assert float(elc["unit_cost"]) == 23.50

    def test_revenue_maps_to_retail_price(self, adapter, paladin_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        elc = next(r for r in rows if r["sku"] == "ELC-4401")
        assert float(elc["retail_price"]) == 31.73

    def test_sold_maps_to_sales_last_30d(self, adapter, paladin_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        elc = next(r for r in rows if r["sku"] == "ELC-4401")
        assert float(elc["sales_last_30d"]) == 10.0

    def test_margin_normalized_from_percentage(self, adapter, paladin_df, tmp_path):
        """Paladin margins are percentages (25.9) — adapter normalizes to 0.259."""
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        elc = next(r for r in rows if r["sku"] == "ELC-4401")
        margin = float(elc["margin_pct"])
        assert 0.25 < margin < 0.27  # 25.9% → ~0.259

    def test_store_id_defaults(self, adapter, paladin_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        for row in rows:
            assert row["store_id"] == "default"

    def test_is_damaged_defaults_false(self, adapter, paladin_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        for row in rows:
            assert row["is_damaged"] == "false"

    def test_seasonal_detected_from_category(self, adapter, paladin_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        grd = next(r for r in rows if r["sku"] == "GRD-5501")
        assert grd["is_seasonal"] == "true"  # "Lawn & Garden" → seasonal

        elc = next(r for r in rows if r["sku"] == "ELC-4401")
        assert elc["is_seasonal"] == "false"  # "Electronics" → not seasonal

    def test_on_order_defaults_zero(self, adapter, paladin_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        for row in rows:
            assert float(row["on_order_qty"]) == 0.0

    @staticmethod
    def _read_csv(path: Path) -> list[dict]:
        with open(path) as f:
            return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# ColumnAdapter — Unmapped Paladin (original column names)
# ---------------------------------------------------------------------------


class TestUnmappedPaladin:
    """Tests with original Paladin column names (before rename)."""

    def test_finds_original_columns(self, adapter, paladin_unmapped_df, tmp_path):
        """Adapter should find SKU, Cost, Retail, etc. via aliases."""
        csv_path = adapter.to_rust_csv(paladin_unmapped_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        assert len(rows) == 3
        assert rows[0]["sku"] == "ELC-4401"

    def test_in_stock_qty_resolved(self, adapter, paladin_unmapped_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_unmapped_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        assert float(rows[0]["qty_on_hand"]) == 100.0

    def test_last_pur_computes_days_since_receipt(
        self, adapter, paladin_unmapped_df, tmp_path
    ):
        """'Last Pur.' = '2026-01-15', reference = 2026-02-06 → 22 days."""
        csv_path = adapter.to_rust_csv(paladin_unmapped_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        elc = rows[0]
        assert float(elc["days_since_receipt"]) == 22.0

    def test_empty_date_uses_default(self, adapter, paladin_unmapped_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_unmapped_df, tmp_path / "out.csv")
        rows = self._read_csv(csv_path)
        pnt = rows[2]  # PNT-1001 has empty date
        assert float(pnt["days_since_receipt"]) == 30.0  # default

    @staticmethod
    def _read_csv(path: Path) -> list[dict]:
        with open(path) as f:
            return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Margin computation
# ---------------------------------------------------------------------------


class TestMarginComputation:
    def test_margin_calculated_when_no_margin_column(self, tmp_path):
        """When no margin column exists, compute from cost and revenue."""
        df = pd.DataFrame(
            {
                "sku": ["A"],
                "cost": [10.0],
                "revenue": [20.0],
            }
        )
        adapter = ColumnAdapter()
        csv_path = adapter.to_rust_csv(df, tmp_path / "out.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        # (20 - 10) / 20 = 0.50
        assert abs(float(rows[0]["margin_pct"]) - 0.50) < 0.01

    def test_margin_zero_when_no_data(self, tmp_path):
        """When both cost and revenue are zero, margin should be 0."""
        df = pd.DataFrame(
            {
                "sku": ["A"],
                "cost": [0.0],
                "revenue": [0.0],
            }
        )
        adapter = ColumnAdapter()
        csv_path = adapter.to_rust_csv(df, tmp_path / "out.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["margin_pct"]) == 0.0

    def test_margin_percentage_normalized(self, tmp_path):
        """Margins > 1.0 should be divided by 100 (percentage → decimal)."""
        df = pd.DataFrame(
            {
                "sku": ["A"],
                "margin": [35.0],  # 35% as percentage
                "cost": [10.0],
                "revenue": [20.0],
            }
        )
        adapter = ColumnAdapter()
        csv_path = adapter.to_rust_csv(df, tmp_path / "out.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert abs(float(rows[0]["margin_pct"]) - 0.35) < 0.01

    def test_margin_decimal_preserved(self, tmp_path):
        """Margins <= 1.0 should be used as-is (already decimal)."""
        df = pd.DataFrame(
            {
                "sku": ["A"],
                "margin": [0.35],
                "cost": [10.0],
                "revenue": [20.0],
            }
        )
        adapter = ColumnAdapter()
        csv_path = adapter.to_rust_csv(df, tmp_path / "out.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert abs(float(rows[0]["margin_pct"]) - 0.35) < 0.01


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_dataframe(self, adapter, tmp_path):
        df = pd.DataFrame({"sku": [], "cost": []})
        csv_path = adapter.to_rust_csv(df, tmp_path / "out.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0

    def test_no_sku_column_raises(self, adapter, tmp_path):
        df = pd.DataFrame({"foo": [1], "bar": [2]})
        with pytest.raises(ValueError, match="No SKU column"):
            adapter.to_rust_csv(df, tmp_path / "out.csv")

    def test_nan_sku_rows_skipped(self, adapter, tmp_path):
        df = pd.DataFrame(
            {
                "sku": ["A", None, "C", float("nan"), ""],
                "cost": [10, 20, 30, 40, 50],
            }
        )
        csv_path = adapter.to_rust_csv(df, tmp_path / "out.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # Only A and C
        assert rows[0]["sku"] == "A"
        assert rows[1]["sku"] == "C"

    def test_nan_numeric_defaults_to_zero(self, adapter, tmp_path):
        df = pd.DataFrame(
            {
                "sku": ["A"],
                "cost": [float("nan")],
                "revenue": [None],
            }
        )
        csv_path = adapter.to_rust_csv(df, tmp_path / "out.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["unit_cost"]) == 0.0
        assert float(rows[0]["retail_price"]) == 0.0

    def test_dollar_signs_stripped(self, adapter, tmp_path):
        df = pd.DataFrame(
            {
                "sku": ["A"],
                "cost": ["$23.50"],
                "revenue": ["$31.73"],
            }
        )
        csv_path = adapter.to_rust_csv(df, tmp_path / "out.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["unit_cost"]) == 23.50
        assert float(rows[0]["retail_price"]) == 31.73

    def test_commas_in_numbers_stripped(self, adapter, tmp_path):
        df = pd.DataFrame(
            {
                "sku": ["A"],
                "cost": ["1,234.56"],
            }
        )
        csv_path = adapter.to_rust_csv(df, tmp_path / "out.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["unit_cost"]) == 1234.56

    def test_large_dataframe(self, adapter, tmp_path):
        """Verify adapter handles large DataFrames efficiently."""
        n = 10_000
        df = pd.DataFrame(
            {
                "sku": [f"SKU-{i:05d}" for i in range(n)],
                "quantity": range(n),
                "cost": [10.0] * n,
                "revenue": [20.0] * n,
                "sold": [5] * n,
            }
        )
        csv_path = adapter.to_rust_csv(df, tmp_path / "out.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n


# ---------------------------------------------------------------------------
# Custom store_id and defaults
# ---------------------------------------------------------------------------


class TestCustomDefaults:
    def test_custom_store_id(self, tmp_path):
        adapter = ColumnAdapter(default_store_id="store-7")
        df = pd.DataFrame({"sku": ["A"], "cost": [10.0]})
        csv_path = adapter.to_rust_csv(df, tmp_path / "out.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["store_id"] == "store-7"

    def test_store_from_column(self, tmp_path):
        adapter = ColumnAdapter()
        df = pd.DataFrame(
            {
                "sku": ["A", "B"],
                "cost": [10.0, 20.0],
                "store_id": ["alpha", "beta"],
            }
        )
        csv_path = adapter.to_rust_csv(df, tmp_path / "out.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["store_id"] == "alpha"
        assert rows[1]["store_id"] == "beta"

    def test_custom_days_since_receipt_default(self, tmp_path):
        adapter = ColumnAdapter(default_days_since_receipt=90.0)
        df = pd.DataFrame({"sku": ["A"], "cost": [10.0]})
        csv_path = adapter.to_rust_csv(df, tmp_path / "out.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["days_since_receipt"]) == 90.0


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_temp_file_cleanup(self, adapter, paladin_df):
        csv_path = adapter.to_rust_csv(paladin_df)
        assert csv_path.exists()
        adapter.cleanup()
        assert not csv_path.exists()

    def test_cleanup_idempotent(self, adapter, paladin_df):
        adapter.to_rust_csv(paladin_df)
        adapter.cleanup()
        adapter.cleanup()  # Should not raise


# ---------------------------------------------------------------------------
# Column mapping report
# ---------------------------------------------------------------------------


class TestMappingReport:
    def test_report_shows_resolved(self, adapter, paladin_df):
        report = adapter.get_column_mapping_report(paladin_df)
        assert "sku" in report["resolved"]
        assert "qty_on_hand" in report["resolved"]
        assert "unit_cost" in report["resolved"]
        assert "retail_price" in report["resolved"]

    def test_report_shows_defaults(self, adapter, paladin_df):
        report = adapter.get_column_mapping_report(paladin_df)
        default_fields = [d["rust_field"] for d in report["defaults"]]
        assert "store_id" in default_fields  # No store column in paladin_df
        assert "days_since_receipt" in default_fields  # No date column

    def test_report_warns_missing_sku(self, adapter):
        df = pd.DataFrame({"foo": [1], "bar": [2]})
        report = adapter.get_column_mapping_report(df)
        assert any("SKU" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# CSV format validation (Rust compatibility)
# ---------------------------------------------------------------------------


class TestRustCsvFormat:
    """Verify output CSV is valid for Rust's csv crate + serde deserializer."""

    def test_column_order_exact(self, adapter, paladin_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        with open(csv_path) as f:
            header = f.readline().strip()
        assert header == ",".join(RUST_COLUMNS)

    def test_booleans_are_lowercase(self, adapter, paladin_df, tmp_path):
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["is_damaged"] in ("true", "false")
                assert row["is_seasonal"] in ("true", "false")

    def test_no_empty_fields(self, adapter, paladin_df, tmp_path):
        """Rust's serde can't parse empty strings as f64."""
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for field in RUST_COLUMNS:
                    assert row[field] != "", f"Empty field: {field} in row {row['sku']}"

    def test_floats_parseable(self, adapter, paladin_df, tmp_path):
        """All numeric fields should be parseable as float."""
        float_fields = [
            "qty_on_hand",
            "unit_cost",
            "margin_pct",
            "sales_last_30d",
            "days_since_receipt",
            "retail_price",
            "on_order_qty",
        ]
        csv_path = adapter.to_rust_csv(paladin_df, tmp_path / "out.csv")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for field in float_fields:
                    try:
                        float(row[field])
                    except ValueError:
                        pytest.fail(
                            f"Field {field}='{row[field]}' not parseable as float "
                            f"in row {row['sku']}"
                        )
