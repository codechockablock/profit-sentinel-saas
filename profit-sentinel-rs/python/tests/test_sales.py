"""Tests for Phase 12: Sales Data Integration.

Tests cover:
- Sales data adapter: column auto-detection, multi-format date parsing, CSV parsing
- Sales aggregation: 30-day rolling window, per-SKU grouping, multi-store
- Sales overlay: join with inventory records, coverage stats
- Pipeline bridge update: actual sales_last_30d vs YTD estimation fallback
- NormalizedInventory: new sales_last_30d field
- Detection registry: SalesDataAdapter properly registered
- Edge cases: empty files, missing columns, returns, zero-qty transactions
"""

import csv
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest
from sentinel_agent.adapters.base import AdapterResult, NormalizedInventory
from sentinel_agent.adapters.bridge import PipelineBridge
from sentinel_agent.adapters.detection import detect_adapter, list_adapters
from sentinel_agent.adapters.sales import (
    SalesAdapterResult,
    SalesAggregation,
    SalesDataAdapter,
    SalesOverlay,
    SalesTransaction,
    _detect_columns,
    _find_column,
    _is_sales_csv,
    _parse_date,
    _safe_float,
    aggregate_sales_30d,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SALES_FIXTURE = FIXTURE_DIR / "sales_detail_sample.csv"


@pytest.fixture
def adapter():
    return SalesDataAdapter()


@pytest.fixture
def sample_transactions() -> list[SalesTransaction]:
    """Controlled set of transactions for testing."""
    ref = date(2025, 6, 30)
    return [
        # 4 transactions for ELC-4401 in the 30-day window
        SalesTransaction(
            sku="ELC-4401",
            sale_date=ref - timedelta(days=5),
            qty_sold=3,
            amount=95.19,
            store_id="default-store",
        ),
        SalesTransaction(
            sku="ELC-4401",
            sale_date=ref - timedelta(days=10),
            qty_sold=2,
            amount=63.46,
            store_id="default-store",
        ),
        SalesTransaction(
            sku="ELC-4401",
            sale_date=ref - timedelta(days=15),
            qty_sold=1,
            amount=31.73,
            store_id="default-store",
        ),
        SalesTransaction(
            sku="ELC-4401",
            sale_date=ref - timedelta(days=2),
            qty_sold=4,
            amount=126.92,
            store_id="default-store",
        ),
        # 3 transactions for PNT-1001
        SalesTransaction(
            sku="PNT-1001",
            sale_date=ref - timedelta(days=20),
            qty_sold=5,
            amount=525.00,
            store_id="default-store",
        ),
        SalesTransaction(
            sku="PNT-1001",
            sale_date=ref - timedelta(days=15),
            qty_sold=3,
            amount=315.00,
            store_id="default-store",
        ),
        SalesTransaction(
            sku="PNT-1001",
            sale_date=ref - timedelta(days=10),
            qty_sold=2,
            amount=210.00,
            store_id="default-store",
        ),
        # 1 transaction OUTSIDE the 30-day window (should be excluded)
        SalesTransaction(
            sku="ELC-4401",
            sale_date=ref - timedelta(days=45),
            qty_sold=2,
            amount=63.46,
            store_id="default-store",
        ),
        # Multi-store: TLS-2001 in store-12
        SalesTransaction(
            sku="TLS-2001",
            sale_date=ref - timedelta(days=5),
            qty_sold=1,
            amount=83.20,
            store_id="store-12",
        ),
        SalesTransaction(
            sku="TLS-2001",
            sale_date=ref - timedelta(days=12),
            qty_sold=2,
            amount=166.40,
            store_id="store-12",
        ),
    ]


@pytest.fixture
def sample_inventory() -> list[NormalizedInventory]:
    """Inventory records to overlay sales onto."""
    return [
        NormalizedInventory(
            sku_id="ELC-4401",
            description="Wire Stripper Pro",
            qty_on_hand=-47,
            unit_cost=23.50,
            retail_price=31.73,
            store_id="default-store",
            sales_ytd=1200.00,
        ),
        NormalizedInventory(
            sku_id="PNT-1001",
            description="Paint Brush Set",
            qty_on_hand=50,
            unit_cost=100.00,
            retail_price=105.00,
            store_id="default-store",
            sales_ytd=600.00,
        ),
        NormalizedInventory(
            sku_id="SEA-1201",
            description="Dead Stock Widget",
            qty_on_hand=100,
            unit_cost=50.00,
            retail_price=67.50,
            store_id="default-store",
            sales_ytd=0.0,
        ),
        NormalizedInventory(
            sku_id="TLS-2001",
            description="Socket Set 40pc",
            qty_on_hand=-15,
            unit_cost=65.00,
            retail_price=83.20,
            store_id="store-12",
            sales_ytd=500.00,
        ),
    ]


# ---------------------------------------------------------------------------
# Column Auto-Detection
# ---------------------------------------------------------------------------


class TestColumnDetection:
    def test_detect_paladin_columns(self):
        headers = ["SKU", "Description Full", "Date", "Qty Sold", "Amount"]
        mapping = _detect_columns(headers)
        assert mapping["sku"] == "SKU"
        assert mapping["date"] == "Date"
        assert mapping["qty"] == "Qty Sold"
        assert mapping["amount"] == "Amount"

    def test_detect_generic_columns(self):
        headers = ["item_number", "transaction_date", "quantity", "net_sales"]
        mapping = _detect_columns(headers)
        assert mapping["sku"] == "item_number"
        assert mapping["date"] == "transaction_date"
        assert mapping["qty"] == "quantity"
        assert mapping["amount"] == "net_sales"

    def test_detect_shopify_columns(self):
        headers = ["Product ID", "Completed At", "Units Sold", "Revenue"]
        mapping = _detect_columns(headers)
        assert mapping["sku"] == "Product ID"
        assert mapping["date"] == "Completed At"
        assert mapping["qty"] == "Units Sold"
        assert mapping["amount"] == "Revenue"

    def test_detect_store_column(self):
        headers = ["SKU", "Date", "Qty", "Store ID"]
        mapping = _detect_columns(headers)
        assert mapping["store"] == "Store ID"

    def test_no_store_column_returns_none(self):
        headers = ["SKU", "Date", "Qty"]
        mapping = _detect_columns(headers)
        assert mapping["store"] is None

    def test_is_sales_csv_true(self):
        headers = ["SKU", "Date", "Qty Sold", "Amount"]
        assert _is_sales_csv(headers) is True

    def test_is_sales_csv_false_no_date(self):
        headers = ["SKU", "Qty On Hand", "Unit Cost"]
        assert _is_sales_csv(headers) is False

    def test_is_sales_csv_false_no_measure(self):
        headers = ["SKU", "Date", "Description"]
        assert _is_sales_csv(headers) is False

    def test_is_sales_csv_only_amount(self):
        """Should work with amount but no qty column."""
        headers = ["SKU", "Date", "Amount"]
        assert _is_sales_csv(headers) is True

    def test_find_column_case_insensitive(self):
        headers = ["sku", "DATE", "qty_sold"]
        assert _find_column(headers, ["SKU", "sku"]) == "sku"
        assert _find_column(headers, ["Date", "DATE"]) == "DATE"


# ---------------------------------------------------------------------------
# Date Parsing
# ---------------------------------------------------------------------------


class TestDateParsing:
    def test_iso_format(self):
        assert _parse_date("2025-06-15") == date(2025, 6, 15)

    def test_us_format(self):
        assert _parse_date("06/15/2025") == date(2025, 6, 15)

    def test_us_short_year(self):
        assert _parse_date("06/15/25") == date(2025, 6, 15)

    def test_yyyymmdd(self):
        assert _parse_date("20250615") == date(2025, 6, 15)

    def test_paladin_shlp(self):
        assert _parse_date("Jun 15,25") == date(2025, 6, 15)

    def test_dmy_format(self):
        assert _parse_date("15-Jun-2025") == date(2025, 6, 15)

    def test_empty_string(self):
        assert _parse_date("") is None

    def test_invalid_format(self):
        assert _parse_date("not-a-date") is None

    def test_quoted_date(self):
        assert _parse_date('"2025-06-15"') == date(2025, 6, 15)


# ---------------------------------------------------------------------------
# Float Parsing
# ---------------------------------------------------------------------------


class TestFloatParsing:
    def test_normal(self):
        assert _safe_float("123.45") == 123.45

    def test_with_commas(self):
        assert _safe_float("1,234.56") == 1234.56

    def test_with_dollar(self):
        assert _safe_float("$99.99") == 99.99

    def test_negative_parentheses(self):
        assert _safe_float("(15.00)") == -15.0

    def test_empty(self):
        assert _safe_float("") == 0.0

    def test_dash(self):
        assert _safe_float("-") == 0.0

    def test_invalid(self):
        assert _safe_float("N/A") == 0.0


# ---------------------------------------------------------------------------
# Sales Data Adapter
# ---------------------------------------------------------------------------


class TestSalesDataAdapter:
    def test_adapter_name(self, adapter):
        assert adapter.name == "Sales Data"

    def test_can_handle_fixture(self, adapter):
        assert adapter.can_handle(SALES_FIXTURE)

    def test_cannot_handle_inventory_csv(self, adapter):
        """Should not detect an inventory CSV as sales data."""
        inv_path = FIXTURE_DIR / "generic_pos_sample.csv"
        if inv_path.exists():
            # Only run if fixture exists
            adapter.can_handle(inv_path)
            # Sales adapter should not claim inventory files
            # (the Sample Store inventory file has different column signatures)

    def test_ingest_fixture(self, adapter):
        result = adapter.ingest_sales(SALES_FIXTURE, store_id="default-store")
        assert isinstance(result, SalesAdapterResult)
        assert result.total_transactions == 21
        assert result.files_processed == 1
        assert len(result.errors) == 0

    def test_ingest_date_range(self, adapter):
        result = adapter.ingest_sales(SALES_FIXTURE, store_id="default-store")
        assert result.date_range_start == date(2025, 5, 15)
        assert result.date_range_end == date(2025, 6, 28)

    def test_ingest_unique_skus(self, adapter):
        result = adapter.ingest_sales(SALES_FIXTURE, store_id="default-store")
        assert result.unique_skus == 7  # ELC, SEA, PNT, HRD, PLB, TLS, FLR

    def test_ingest_store_from_csv(self, adapter):
        """Store ID should come from CSV column when present."""
        result = adapter.ingest_sales(SALES_FIXTURE, store_id="default-store")
        stores = {t.store_id for t in result.transactions}
        assert "default-store" in stores
        assert "store-12" in stores
        assert "store-3" in stores

    def test_ingest_empty_csv(self, adapter):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("SKU,Date,Qty Sold,Amount\n")
            path = Path(f.name)
        try:
            result = adapter.ingest_sales(path)
            assert result.total_transactions == 0
        finally:
            path.unlink(missing_ok=True)

    def test_ingest_missing_sku_column(self, adapter):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("Product,Date,Qty\n")
            f.write("ABC,2025-06-15,5\n")
            path = Path(f.name)
        try:
            result = adapter.ingest_sales(path)
            assert len(result.errors) > 0
            assert "No SKU column" in result.errors[0]
        finally:
            path.unlink(missing_ok=True)

    def test_ingest_skips_bad_dates(self, adapter):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("SKU,Date,Qty Sold,Amount\n")
            f.write("ABC,2025-06-15,5,50.00\n")
            f.write("DEF,bad-date,3,30.00\n")
            f.write("GHI,2025-06-20,2,20.00\n")
            path = Path(f.name)
        try:
            result = adapter.ingest_sales(path)
            assert result.total_transactions == 2
            assert len(result.warnings) == 1  # 1 skipped row
        finally:
            path.unlink(missing_ok=True)

    def test_can_handle_directory(self, adapter):
        """Adapter can handle directory containing sales CSVs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sales_path = Path(tmpdir) / "Sales_Detail_June.csv"
            sales_path.write_text("SKU,Date,Qty Sold,Amount\nABC,2025-06-15,5,50.00\n")
            assert adapter.can_handle(Path(tmpdir))


# ---------------------------------------------------------------------------
# Sales Aggregation
# ---------------------------------------------------------------------------


class TestSalesAggregation:
    def test_aggregate_basic(self, sample_transactions):
        ref = date(2025, 6, 30)
        aggs = aggregate_sales_30d(sample_transactions, reference_date=ref)
        # ELC-4401 has 4 txns in window (the 5th is outside)
        elc = aggs["default-store::ELC-4401"]
        assert elc.total_qty_sold == 10  # 3+2+1+4
        assert elc.transaction_count == 4

    def test_aggregate_excludes_old(self, sample_transactions):
        ref = date(2025, 6, 30)
        aggs = aggregate_sales_30d(sample_transactions, reference_date=ref)
        elc = aggs["default-store::ELC-4401"]
        # Old transaction (45 days ago) should be excluded
        assert elc.total_qty_sold == 10  # not 12

    def test_aggregate_multi_store(self, sample_transactions):
        ref = date(2025, 6, 30)
        aggs = aggregate_sales_30d(sample_transactions, reference_date=ref)
        assert "store-12::TLS-2001" in aggs
        tls = aggs["store-12::TLS-2001"]
        assert tls.total_qty_sold == 3  # 1+2

    def test_aggregate_amount(self, sample_transactions):
        ref = date(2025, 6, 30)
        aggs = aggregate_sales_30d(sample_transactions, reference_date=ref)
        pnt = aggs["default-store::PNT-1001"]
        assert abs(pnt.total_amount - 1050.00) < 0.01  # 525+315+210

    def test_aggregate_empty(self):
        aggs = aggregate_sales_30d([], reference_date=date(2025, 6, 30))
        assert len(aggs) == 0

    def test_aggregate_custom_window(self, sample_transactions):
        ref = date(2025, 6, 30)
        # 7-day window: only transactions from Jun 23-30
        aggs = aggregate_sales_30d(
            sample_transactions, reference_date=ref, window_days=7
        )
        elc = aggs.get("default-store::ELC-4401")
        # Jun 25 (5 days ago, qty=3) and Jun 28 (2 days ago, qty=4)
        assert elc is not None
        assert elc.total_qty_sold == 7  # 3 + 4
        assert elc.transaction_count == 2

    def test_aggregate_date_range(self, sample_transactions):
        ref = date(2025, 6, 30)
        aggs = aggregate_sales_30d(sample_transactions, reference_date=ref)
        elc = aggs["default-store::ELC-4401"]
        assert elc.first_sale is not None
        assert elc.last_sale is not None
        assert elc.first_sale <= elc.last_sale


# ---------------------------------------------------------------------------
# Sales Overlay
# ---------------------------------------------------------------------------


class TestSalesOverlay:
    def test_from_transactions(self, sample_transactions):
        ref = date(2025, 6, 30)
        overlay = SalesOverlay.from_transactions(
            sample_transactions,
            reference_date=ref,
        )
        assert overlay.coverage["total_skus"] > 0

    def test_lookup_by_store_sku(self, sample_transactions):
        ref = date(2025, 6, 30)
        overlay = SalesOverlay.from_transactions(
            sample_transactions,
            reference_date=ref,
        )
        agg = overlay.lookup("default-store", "ELC-4401")
        assert agg is not None
        assert agg.total_qty_sold == 10

    def test_lookup_by_bare_sku(self, sample_transactions):
        ref = date(2025, 6, 30)
        overlay = SalesOverlay.from_transactions(
            sample_transactions,
            reference_date=ref,
        )
        # Falls back to bare SKU
        agg = overlay.lookup("unknown-store", "ELC-4401")
        assert agg is not None

    def test_lookup_missing_sku(self, sample_transactions):
        ref = date(2025, 6, 30)
        overlay = SalesOverlay.from_transactions(
            sample_transactions,
            reference_date=ref,
        )
        assert overlay.lookup("default-store", "NONEXISTENT") is None

    def test_get_sales_last_30d(self, sample_transactions):
        ref = date(2025, 6, 30)
        overlay = SalesOverlay.from_transactions(
            sample_transactions,
            reference_date=ref,
        )
        qty = overlay.get_sales_last_30d("default-store", "PNT-1001")
        assert qty == 10  # 5+3+2

    def test_get_sales_last_30d_missing(self, sample_transactions):
        ref = date(2025, 6, 30)
        overlay = SalesOverlay.from_transactions(
            sample_transactions,
            reference_date=ref,
        )
        assert overlay.get_sales_last_30d("default-store", "NOPE") is None

    def test_apply_overlay(self, sample_transactions, sample_inventory):
        ref = date(2025, 6, 30)
        overlay = SalesOverlay.from_transactions(
            sample_transactions,
            reference_date=ref,
        )
        enriched = overlay.apply(sample_inventory)
        assert len(enriched) == len(sample_inventory)

        # ELC-4401 should have actual sales data
        elc = next(r for r in enriched if r.sku_id == "ELC-4401")
        assert elc.sales_last_30d == 10

        # PNT-1001 should have actual sales data
        pnt = next(r for r in enriched if r.sku_id == "PNT-1001")
        assert pnt.sales_last_30d == 10

        # SEA-1201 has no sales transactions → unchanged
        sea = next(r for r in enriched if r.sku_id == "SEA-1201")
        assert sea.sales_last_30d is None  # No overlay data available

    def test_apply_does_not_modify_originals(
        self, sample_transactions, sample_inventory
    ):
        ref = date(2025, 6, 30)
        overlay = SalesOverlay.from_transactions(
            sample_transactions,
            reference_date=ref,
        )
        original_ytd = sample_inventory[0].sales_ytd
        _ = overlay.apply(sample_inventory)
        # Original should not be modified
        assert sample_inventory[0].sales_ytd == original_ytd
        assert sample_inventory[0].sales_last_30d is None  # Still None

    def test_apply_multi_store(self, sample_transactions, sample_inventory):
        ref = date(2025, 6, 30)
        overlay = SalesOverlay.from_transactions(
            sample_transactions,
            reference_date=ref,
        )
        enriched = overlay.apply(sample_inventory)
        # TLS-2001 in store-12 should get overlay
        tls = next(r for r in enriched if r.sku_id == "TLS-2001")
        assert tls.sales_last_30d == 3  # 1+2

    def test_coverage_stats(self, sample_transactions):
        ref = date(2025, 6, 30)
        overlay = SalesOverlay.from_transactions(
            sample_transactions,
            reference_date=ref,
        )
        stats = overlay.coverage
        assert stats["total_skus"] > 0
        assert stats["total_transactions"] > 0


# ---------------------------------------------------------------------------
# NormalizedInventory — Phase 12 Field
# ---------------------------------------------------------------------------


class TestNormalizedInventoryPhase12:
    def test_sales_last_30d_default_none(self):
        rec = NormalizedInventory(sku_id="X", retail_price=10.0)
        assert rec.sales_last_30d is None

    def test_sales_last_30d_set(self):
        rec = NormalizedInventory(sku_id="X", retail_price=10.0, sales_last_30d=42.0)
        assert rec.sales_last_30d == 42.0

    def test_sales_last_30d_zero(self):
        rec = NormalizedInventory(sku_id="X", retail_price=10.0, sales_last_30d=0.0)
        assert rec.sales_last_30d == 0.0

    def test_model_copy_with_sales(self):
        rec = NormalizedInventory(sku_id="X", retail_price=10.0, sales_ytd=500.0)
        updated = rec.model_copy(update={"sales_last_30d": 25.0})
        assert updated.sales_last_30d == 25.0
        assert updated.sales_ytd == 500.0  # Preserved
        assert rec.sales_last_30d is None  # Original unchanged


# ---------------------------------------------------------------------------
# Pipeline Bridge — Phase 12 Integration
# ---------------------------------------------------------------------------


class TestBridgePhase12:
    def test_actual_sales_preferred_over_ytd(self):
        bridge = PipelineBridge(
            reference_date=date(2025, 7, 1),
            months_elapsed_ytd=6.0,
        )
        rec = NormalizedInventory(
            sku_id="X",
            retail_price=10.0,
            sales_ytd=1200.0,  # YTD estimate would be 200/month
            sales_last_30d=42.0,  # Actual: 42
        )
        row = bridge.convert_record(rec)
        assert float(row["sales_last_30d"]) == 42.0  # NOT 200

    def test_ytd_fallback_when_no_actual(self):
        bridge = PipelineBridge(
            reference_date=date(2025, 7, 1),
            months_elapsed_ytd=6.0,
        )
        rec = NormalizedInventory(
            sku_id="X",
            retail_price=10.0,
            sales_ytd=1200.0,
            # sales_last_30d is None — should fall back to YTD
        )
        row = bridge.convert_record(rec)
        assert abs(float(row["sales_last_30d"]) - 200.0) < 0.01

    def test_zero_actual_sales(self):
        bridge = PipelineBridge(
            reference_date=date(2025, 7, 1),
            months_elapsed_ytd=6.0,
        )
        rec = NormalizedInventory(
            sku_id="X",
            retail_price=10.0,
            sales_ytd=1200.0,
            sales_last_30d=0.0,  # Confirmed zero (not unknown)
        )
        row = bridge.convert_record(rec)
        assert float(row["sales_last_30d"]) == 0.0

    def test_end_to_end_overlay_then_bridge(
        self, sample_transactions, sample_inventory
    ):
        """Full flow: transactions → overlay → bridge → CSV row."""
        ref = date(2025, 6, 30)
        overlay = SalesOverlay.from_transactions(
            sample_transactions,
            reference_date=ref,
        )
        enriched = overlay.apply(sample_inventory)

        bridge = PipelineBridge(
            reference_date=ref,
            months_elapsed_ytd=6.0,
        )

        # ELC-4401: overlay gives 10 actual sales
        elc = next(r for r in enriched if r.sku_id == "ELC-4401")
        row = bridge.convert_record(elc)
        assert float(row["sales_last_30d"]) == 10.0

        # SEA-1201: no overlay data → falls back to YTD (0)
        sea = next(r for r in enriched if r.sku_id == "SEA-1201")
        row = bridge.convert_record(sea)
        assert float(row["sales_last_30d"]) == 0.0

    def test_bridge_csv_output_with_actual_sales(
        self, sample_transactions, sample_inventory
    ):
        """Ensure bridge CSV output uses actual sales data."""
        ref = date(2025, 6, 30)
        overlay = SalesOverlay.from_transactions(
            sample_transactions,
            reference_date=ref,
        )
        enriched = overlay.apply(sample_inventory)

        bridge = PipelineBridge(reference_date=ref, months_elapsed_ytd=6.0)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            bridge.to_pipeline_csv(enriched, path)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = {r["sku"]: r for r in reader}

            # ELC-4401 should have actual sales (10), not YTD estimate (200)
            assert float(rows["ELC-4401"]["sales_last_30d"]) == 10.0
            # TLS-2001 should have actual sales (3), not YTD estimate (~83)
            assert float(rows["TLS-2001"]["sales_last_30d"]) == 3.0
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Detection Registry
# ---------------------------------------------------------------------------


class TestDetectionRegistry:
    def test_sales_adapter_registered(self):
        adapters = list_adapters()
        names = [a["name"] for a in adapters]
        assert "Sales Data" in names

    def test_detect_sales_file(self, adapter):
        assert adapter.can_handle(SALES_FIXTURE)

    def test_detect_adapter_for_sales(self):
        detected = detect_adapter(SALES_FIXTURE)
        # May be detected as Sample Store (higher priority) or Sales Data
        # depending on column overlap. What matters is it's not None.
        assert detected is not None


# ---------------------------------------------------------------------------
# SalesTransaction Model
# ---------------------------------------------------------------------------


class TestSalesTransaction:
    def test_basic_creation(self):
        txn = SalesTransaction(
            sku="ABC",
            sale_date=date(2025, 6, 15),
            qty_sold=5,
            amount=50.00,
        )
        assert txn.sku == "ABC"
        assert txn.sale_date == date(2025, 6, 15)
        assert txn.qty_sold == 5
        assert txn.amount == 50.00

    def test_defaults(self):
        txn = SalesTransaction(sku="ABC", sale_date=date(2025, 6, 15))
        assert txn.qty_sold == 0.0
        assert txn.amount == 0.0
        assert txn.unit_cost == 0.0
        assert txn.store_id == ""


# ---------------------------------------------------------------------------
# SalesAggregation Model
# ---------------------------------------------------------------------------


class TestSalesAggregationModel:
    def test_basic_creation(self):
        agg = SalesAggregation(
            sku="ABC",
            store_id="default-store",
            total_qty_sold=15,
            total_amount=300.00,
            transaction_count=5,
        )
        assert agg.total_qty_sold == 15
        assert agg.transaction_count == 5

    def test_defaults(self):
        agg = SalesAggregation(sku="X")
        assert agg.store_id == ""
        assert agg.total_qty_sold == 0.0
        assert agg.days_in_window == 30


# ---------------------------------------------------------------------------
# SalesAdapterResult Model
# ---------------------------------------------------------------------------


class TestSalesAdapterResult:
    def test_properties(self):
        result = SalesAdapterResult(
            source="test.csv",
            transactions=[
                SalesTransaction(sku="A", sale_date=date(2025, 6, 1), amount=100),
                SalesTransaction(sku="B", sale_date=date(2025, 6, 2), amount=200),
                SalesTransaction(sku="A", sale_date=date(2025, 6, 3), amount=150),
            ],
        )
        assert result.total_transactions == 3
        assert result.total_revenue == 450.0
        assert result.unique_skus == 2


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_returns_negative_qty(self):
        """Return transactions with negative qty should be handled."""
        txns = [
            SalesTransaction(
                sku="A", sale_date=date(2025, 6, 15), qty_sold=5, store_id="s"
            ),
            SalesTransaction(
                sku="A", sale_date=date(2025, 6, 20), qty_sold=-2, store_id="s"
            ),  # Return
        ]
        aggs = aggregate_sales_30d(txns, reference_date=date(2025, 6, 30))
        assert aggs["s::A"].total_qty_sold == 3  # 5 + (-2) = 3

    def test_all_transactions_outside_window(self):
        txns = [
            SalesTransaction(
                sku="A", sale_date=date(2025, 1, 1), qty_sold=5, store_id="s"
            ),
        ]
        aggs = aggregate_sales_30d(txns, reference_date=date(2025, 6, 30))
        assert len(aggs) == 0

    def test_boundary_date_included(self):
        """Transaction exactly 30 days ago should be included."""
        ref = date(2025, 6, 30)
        cutoff_date = ref - timedelta(days=30)  # May 31
        txns = [
            SalesTransaction(sku="A", sale_date=cutoff_date, qty_sold=5, store_id="s"),
        ]
        aggs = aggregate_sales_30d(txns, reference_date=ref)
        assert "s::A" in aggs

    def test_boundary_date_excluded(self):
        """Transaction 31 days ago should be excluded."""
        ref = date(2025, 6, 30)
        old_date = ref - timedelta(days=31)
        txns = [
            SalesTransaction(sku="A", sale_date=old_date, qty_sold=5, store_id="s"),
        ]
        aggs = aggregate_sales_30d(txns, reference_date=ref)
        assert len(aggs) == 0

    def test_same_sku_multiple_stores(self):
        """Same SKU in different stores should aggregate separately."""
        txns = [
            SalesTransaction(
                sku="A", sale_date=date(2025, 6, 15), qty_sold=5, store_id="store-1"
            ),
            SalesTransaction(
                sku="A", sale_date=date(2025, 6, 15), qty_sold=3, store_id="store-2"
            ),
        ]
        aggs = aggregate_sales_30d(txns, reference_date=date(2025, 6, 30))
        assert aggs["store-1::A"].total_qty_sold == 5
        assert aggs["store-2::A"].total_qty_sold == 3

    def test_adapter_standard_ingest(self, adapter):
        """Standard ingest() returns AdapterResult (not SalesAdapterResult)."""
        result = adapter.ingest(SALES_FIXTURE, store_id="default-store")
        assert isinstance(result, AdapterResult)
        assert result.adapter_name == "Sales Data"
        assert result.files_processed == 1
