"""Tests for the multi-vendor data adapter layer.

Tests cover:
- Base models (NormalizedInventory, PurchaseOrder, POLineItem)
- Orgill PO parser (header extraction, line items, short-ships)
- Default POS inventory mapper (column mapping, date parsing)
- Auto-detection (correct adapter selection)
- Adapter result aggregation (summaries, fill rates)
"""

import csv
import tempfile
from datetime import date
from pathlib import Path

import pytest
from sentinel_agent.adapters.base import (
    AdapterResult,
    NormalizedInventory,
    POLineItem,
    POStatus,
    PurchaseOrder,
)
from sentinel_agent.adapters.detection import (
    detect_adapter,
    detect_and_ingest,
    list_adapters,
)
from sentinel_agent.adapters.generic_pos import GenericPosAdapter
from sentinel_agent.adapters.orgill import OrgillPOAdapter

# ---------------------------------------------------------------------------
# Fixtures — sample data files
# ---------------------------------------------------------------------------


SAMPLE_ORGILL_PO = """\
Shipto: 462283                     ,Billto: 462283                     ,Order: 7959433
TEST STORE COMMUNITY HD            ,TEST STORE COMMUNITY HD            ,Cust PO:
4700 LIMESTONE ROAD                ,4700 LIMESTONE ROAD                ,Date: 12/02/2025
                                   ,UNIT 1B                            ,Status: INVOICED
UNIT 1B                            ,WILMINGTON        DE 19808-0000    ,Via: INWO IN  C/T1 WILMINGTON DE
WILMINGTON        DE 19808-0000    ,Phone: 302-598-0454                ,Terms: NET DEC. 25TH
Phone: 302-598-0454                ,Cubic Size (cu ft):   433.92       ,USD Amount:   21726.53
Comments
1:                                                             ,2:
3:                                                             ,4:
5:                                                             ,6:
7:                                                             ,8:
Notes: Asterisk after Qty Filled indicates item not filled or partially filled.
       Asterisk after Unit Cost indicates special price.
Line ,  Retail,Item   , ,Ord Qty,Unit ,Description                   , Unit Cost,Spc, Ext Cost,Prod Care,Qty Fill,Out,Shelf Pk,UPC Code    ,POS            ,Crd Inv,Retail Dept Description       ,Dept,Vendor Item Num,PickLne,Country of Origin             ,Resvd Qty,Xref Item,Itm Weight,French Description,Cust Item Desc,Velocity Code,Rtl Sens Code,
00895,        ,0000000, ,        ,     ,STOP CHARGE (E)               ,    55.000 ,   ,   55.00 ,         ,         ,   ,        ,            ,               ,       ,                              ,    ,               ,0000000,                              ,         ,         ,     .0000,                              ,                                                            , ,00,
00480,    8.99,0010413, ,      1 ,CD   ,DW1583 WOOD SPADE BIT 1-1/8X6 ,     3.470 ,   ,    3.47 ,         ,       1 ,   ,       1,028877475684,028877475684   ,       ,POWER TOOLS & ACC             ,  15,               ,0000483,UNITED STATES                 ,        1,         ,     .3800,FORET   3 POINTES PERCAGE BOIS,                                                            ,F,06,
00120,   12.99,0102129,Y,      4 ,EA   ,26-19063 CHICKADEE FEED 4LB   ,     9.170 ,   ,     .00 ,         ,       0 ,*  ,       8,088685190636,               ,       ,PET & WILD BIRD               ,  58,               ,0000120,UNITED STATES                 ,         ,         ,    3.5000,                              ,                                                            ,D,04,
00233,  219.99,0022913, ,      1 ,EA   ,788AA104.020 CADT3 TTG RT HT  ,   164.780 ,*  ,  164.78 ,         ,       1 ,   ,       1,193406305050,193406305050   ,       ,PLUMBING                      ,  40,               ,0000235,MEXICO                        ,        1,         ,   94.9000,                              ,                                                            ,N,02,
00062,   25.99,0083204, ,      5 ,BX6  ,50604 FIRELOG 3HR BRN TM 2.5LB,    19.730 ,   ,   98.65 ,         ,       5 ,   ,       1,041137506041,041137506041   ,       ,HEATING & COOLING             ,  45,               ,0000062,UNITED STATES                 ,        5,         ,   26.0000,                              ,                                                            ,A,02,
"""


SAMPLE_POS_CSV = """\
SKU,Vendor SKU,Vendor,Alt. Vendor/Mfgr.,Cat.,Dpt.,BIN,Description Full,Description Short,Qty.,Qty On Hand,Std. Cost,Inventory @Cost,Avg. Cost,Inventory @AvgCost,Retail,Inventory @Retail,Margin @Cost,Sug. Retail,Retail Dif.,On Hold,Sales,$ Sold,Last Sale,Returns,$ Returned,Last Return,Last Ordered,On Order,Last Purchase,Min.,Max.,Pkg.,Whse.,Mfgr. SKU,Barcode,Alt. Barcode ,Level Option,Level Type,Level 1 Retail,Level 2 Retail,Level 3 Retail,Level 4 Retail,Level 1 % Off/Over,Level 2 % Off/Over,Level 3 % Off/Over,Level 4 % Off/Over,Promtion From,Promotion To,Promo. Retail,Promo. Margin %,Tax Account,
TEST-SKU-001,VS001,TESTVENDOR,MFG1,101,15,A1-01,Test Widget Full Name,Test Widget,10,10,5.99,59.90,5.99,59.90,9.99,99.90,40.07,9.99,,0,25,249.75,20250601,2,19.98,20250301,20250515,5,20250515,2,10,1,,MFG-001,123456789012,,OFF,Retail,0.000,0.000,0.000,0.000,0,0,0,0,,,,,Y,
TEST-SKU-002,VS002,TESTVENDOR,MFG2,103,25,B2-05,Another Product Full,Another Product,-3,0,12.50,0.00,12.50,0.00,19.99,,36.47,,,0,5,99.95,20250815,0,0.00,,,0,,0,0,1,,MFG-002,987654321098,,OFF,Retail,0.000,0.000,0.000,0.000,0,0,0,0,,,,,Y,
TEST-SKU-003,,,,,,,,Empty Fields Item,0,0,0.00,0.00,0.00,0.00,0.00,0.00,,,,0,0,0.00,,0,0.00,,,0,,0,0,1,,,,,OFF,Retail,0.000,0.000,0.000,0.000,0,0,0,0,,,,,Y,
"""


SAMPLE_SHLP_CSV = """\
SKU,Description,Barcode,Vendor,Vendor SKU,Alt. Vendor,Alt. Vendor SKU,Mfgr. Reference,Mfgr. SKU,Last Sale,Real Date,Gross Sales,Gross Cost,Gross Profit,Profit Margin%,Avg. Cost,Stock,Min,Max,MQ,MQ Type,Year Total,Report Total,Jan,Last Dec,Last Nov,Last Oct,Last Sep,Last Aug,Last Jul,Last Jun,Last May,Last Apr,Last Mar,Last Feb,Last Jan
G1956663,COLORED KEY CAPS BULK/200,029069752040,NATIONAL HARDWARE,1956663,17376,KT134,17376,KT134,"Jun 02,25",20250602,22.12,0.00,22.12,,0.00,40124,,,,,28,28,,,,,,,,5,2,2,6,8,5
G6134043,SCANHOOK FASTWIST 10IN R3,037193107186,ORGILL,6134043,68304,R35-10,68304,R35-10,,,,,,,0.66,10400,,,,,,,,,,,,,,,,,,,
"""


@pytest.fixture
def orgill_po_file(tmp_path):
    """Write a sample Orgill PO to a temp file."""
    f = tmp_path / "C04MLE (test).csv"
    f.write_text(SAMPLE_ORGILL_PO)
    return f


@pytest.fixture
def orgill_po_dir(tmp_path):
    """Directory with multiple sample Orgill PO files."""
    for i in range(3):
        f = tmp_path / f"C04MLE ({i}).csv"
        f.write_text(SAMPLE_ORGILL_PO)
    return tmp_path


@pytest.fixture
def pos_inventory_file(tmp_path):
    """Write a sample POS inventory CSV to a temp file."""
    f = tmp_path / "custom_1.csv"
    f.write_text(SAMPLE_POS_CSV)
    return f


@pytest.fixture
def pos_inventory_dir(tmp_path):
    """Directory with POS inventory files."""
    f = tmp_path / "custom_1.csv"
    f.write_text(SAMPLE_POS_CSV)
    return tmp_path


@pytest.fixture
def shlp_file(tmp_path):
    """Write a sample SHLP CSV to a temp file."""
    f = tmp_path / "Inventory_Report_AllSKUs_SHLP_YTD.csv"
    f.write_text(SAMPLE_SHLP_CSV)
    return f


# ---------------------------------------------------------------------------
# Base Model Tests
# ---------------------------------------------------------------------------


class TestNormalizedInventory:
    def test_margin_calculation(self):
        rec = NormalizedInventory(
            sku_id="SKU-001",
            unit_cost=5.99,
            retail_price=9.99,
        )
        assert abs(rec.margin_pct - 0.4004) < 0.001

    def test_margin_zero_retail(self):
        rec = NormalizedInventory(sku_id="SKU-001", retail_price=0)
        assert rec.margin_pct == 0.0

    def test_inventory_value(self):
        rec = NormalizedInventory(
            sku_id="SKU-001",
            qty_on_hand=10,
            unit_cost=5.99,
        )
        assert abs(rec.inventory_value_at_cost - 59.90) < 0.01

    def test_defaults(self):
        rec = NormalizedInventory(sku_id="SKU-001")
        assert rec.qty_on_hand == 0
        assert rec.unit_cost == 0.0
        assert rec.store_id == "default-store"
        assert rec.vendor is None


class TestPOLineItem:
    def test_short_ship_qty(self):
        item = POLineItem(
            line_number=1,
            sku_id="123",
            qty_ordered=10,
            qty_filled=3,
            unit_cost=5.0,
            is_short_ship=True,
        )
        assert item.short_ship_qty == 7
        assert item.short_ship_value == 35.0

    def test_full_fill(self):
        item = POLineItem(
            line_number=1,
            sku_id="123",
            qty_ordered=5,
            qty_filled=5,
            unit_cost=10.0,
        )
        assert item.short_ship_qty == 0
        assert item.fill_rate == 1.0

    def test_zero_ordered(self):
        item = POLineItem(
            line_number=1,
            sku_id="123",
            qty_ordered=0,
            qty_filled=0,
        )
        assert item.fill_rate == 1.0


class TestPurchaseOrder:
    def test_fill_rate(self):
        po = PurchaseOrder(
            po_number="PO-001",
            line_items=[
                POLineItem(
                    line_number=1,
                    sku_id="A",
                    qty_ordered=10,
                    qty_filled=8,
                    unit_cost=5.0,
                    ext_cost=40.0,
                ),
                POLineItem(
                    line_number=2,
                    sku_id="B",
                    qty_ordered=5,
                    qty_filled=5,
                    unit_cost=10.0,
                    ext_cost=50.0,
                ),
            ],
        )
        assert abs(po.fill_rate - 13 / 15) < 0.01

    def test_short_ship_count(self):
        po = PurchaseOrder(
            po_number="PO-001",
            line_items=[
                POLineItem(
                    line_number=1,
                    sku_id="A",
                    qty_ordered=10,
                    qty_filled=0,
                    unit_cost=5.0,
                    ext_cost=0.0,
                    is_short_ship=True,
                ),
                POLineItem(
                    line_number=2,
                    sku_id="B",
                    qty_ordered=5,
                    qty_filled=5,
                    unit_cost=10.0,
                    ext_cost=50.0,
                ),
            ],
        )
        assert po.short_ship_count == 1
        assert po.total_short_ship_value == 50.0

    def test_product_line_items_excludes_service(self):
        po = PurchaseOrder(
            po_number="PO-001",
            line_items=[
                POLineItem(line_number=1, sku_id="0000000", description="STOP CHARGE"),
                POLineItem(line_number=2, sku_id="1234567", description="Widget"),
            ],
        )
        products = po.product_line_items
        assert len(products) == 1
        assert products[0].sku_id == "1234567"


class TestAdapterResult:
    def test_summary_with_inventory(self):
        result = AdapterResult(
            source="/test",
            adapter_name="Test",
            inventory_records=[
                NormalizedInventory(sku_id="A", qty_on_hand=10, unit_cost=5.0),
                NormalizedInventory(sku_id="B", qty_on_hand=20, unit_cost=3.0),
            ],
            files_processed=1,
        )
        assert result.total_inventory_records == 2
        assert "Inventory records: 2" in result.summary
        assert "$110.00" in result.summary

    def test_summary_with_pos(self):
        result = AdapterResult(
            source="/test",
            adapter_name="Test",
            purchase_orders=[
                PurchaseOrder(
                    po_number="PO-1",
                    line_items=[
                        POLineItem(
                            line_number=1,
                            sku_id="A",
                            qty_ordered=10,
                            qty_filled=8,
                            unit_cost=5.0,
                            ext_cost=40.0,
                            is_short_ship=True,
                        ),
                    ],
                ),
            ],
            files_processed=1,
        )
        assert result.total_purchase_orders == 1
        assert result.total_short_ships == 1
        assert "Short-ships: 1" in result.summary


# ---------------------------------------------------------------------------
# Orgill PO Parser Tests
# ---------------------------------------------------------------------------


class TestOrgillPOAdapter:
    def test_can_handle_file(self, orgill_po_file):
        adapter = OrgillPOAdapter()
        assert adapter.can_handle(orgill_po_file)

    def test_can_handle_directory(self, orgill_po_dir):
        adapter = OrgillPOAdapter()
        assert adapter.can_handle(orgill_po_dir)

    def test_cannot_handle_pos_inventory(self, pos_inventory_file):
        adapter = OrgillPOAdapter()
        assert not adapter.can_handle(pos_inventory_file)

    def test_parse_single_file(self, orgill_po_file):
        adapter = OrgillPOAdapter()
        result = adapter.ingest(orgill_po_file)
        assert result.total_purchase_orders == 1
        assert not result.has_errors

    def test_parse_directory(self, orgill_po_dir):
        adapter = OrgillPOAdapter()
        result = adapter.ingest(orgill_po_dir)
        assert result.total_purchase_orders == 3
        assert result.files_processed == 3

    def test_header_extraction(self, orgill_po_file):
        adapter = OrgillPOAdapter()
        result = adapter.ingest(orgill_po_file)
        po = result.purchase_orders[0]
        assert po.po_number == "7959433"
        assert po.order_date == date(2025, 12, 2)
        assert po.status == POStatus.INVOICED
        assert po.terms == "NET DEC. 25TH"
        assert abs(po.usd_amount - 21726.53) < 0.01

    def test_line_items_parsed(self, orgill_po_file):
        adapter = OrgillPOAdapter()
        result = adapter.ingest(orgill_po_file)
        po = result.purchase_orders[0]
        # Sample has: 1 service line + 4 product lines
        assert po.total_line_items == 5
        products = po.product_line_items
        assert len(products) == 4

    def test_short_ship_detected(self, orgill_po_file):
        adapter = OrgillPOAdapter()
        result = adapter.ingest(orgill_po_file)
        po = result.purchase_orders[0]

        short_items = [i for i in po.line_items if i.is_short_ship]
        assert len(short_items) == 1

        chickadee = short_items[0]
        assert chickadee.sku_id == "0102129"
        assert chickadee.qty_ordered == 4
        assert chickadee.qty_filled == 0
        assert abs(chickadee.unit_cost - 9.17) < 0.01
        assert abs(chickadee.short_ship_value - 36.68) < 0.01

    def test_special_price_detected(self, orgill_po_file):
        adapter = OrgillPOAdapter()
        result = adapter.ingest(orgill_po_file)
        po = result.purchase_orders[0]

        special = [i for i in po.line_items if i.is_special_price]
        assert len(special) == 1
        assert special[0].sku_id == "0022913"  # The plumbing item

    def test_department_parsed(self, orgill_po_file):
        adapter = OrgillPOAdapter()
        result = adapter.ingest(orgill_po_file)
        po = result.purchase_orders[0]

        products = po.product_line_items
        # First product: POWER TOOLS & ACC, dept 15
        assert "POWER TOOLS" in products[0].department

    def test_adapter_name(self):
        adapter = OrgillPOAdapter()
        assert adapter.name == "Orgill PO"


# ---------------------------------------------------------------------------
# Default POS Inventory Tests
# ---------------------------------------------------------------------------


class TestDefaultPosAdapter:
    def test_can_handle_file(self, pos_inventory_file):
        adapter = GenericPosAdapter()
        assert adapter.can_handle(pos_inventory_file)

    def test_can_handle_directory(self, pos_inventory_dir):
        adapter = GenericPosAdapter()
        assert adapter.can_handle(pos_inventory_dir)

    def test_cannot_handle_orgill(self, orgill_po_file):
        adapter = GenericPosAdapter()
        assert not adapter.can_handle(orgill_po_file)

    def test_parse_inventory(self, pos_inventory_file):
        adapter = GenericPosAdapter()
        result = adapter.ingest(pos_inventory_file)
        assert result.total_inventory_records == 3
        assert not result.has_errors

    def test_column_mapping(self, pos_inventory_file):
        adapter = GenericPosAdapter()
        result = adapter.ingest(pos_inventory_file)
        rec = result.inventory_records[0]

        assert rec.sku_id == "TEST-SKU-001"
        assert rec.vendor == "TESTVENDOR"
        assert rec.vendor_sku == "VS001"
        assert rec.qty_on_hand == 10
        assert abs(rec.unit_cost - 5.99) < 0.01
        assert abs(rec.retail_price - 9.99) < 0.01
        assert rec.bin_location == "A1-01"
        assert rec.category == "101"
        assert rec.department == "15"
        assert rec.barcode == "123456789012"
        assert rec.on_order_qty == 5
        assert rec.min_qty == 2
        assert rec.max_qty == 10
        assert rec.description == "Test Widget Full Name"

    def test_date_parsing_yyyymmdd(self, pos_inventory_file):
        adapter = GenericPosAdapter()
        result = adapter.ingest(pos_inventory_file)
        rec = result.inventory_records[0]
        assert rec.last_sale_date == date(2025, 6, 1)
        assert rec.last_receipt_date == date(2025, 5, 15)

    def test_sales_ytd(self, pos_inventory_file):
        adapter = GenericPosAdapter()
        result = adapter.ingest(pos_inventory_file)
        rec = result.inventory_records[0]
        assert abs(rec.sales_ytd - 249.75) < 0.01

    def test_negative_qty_on_hand(self, pos_inventory_file):
        adapter = GenericPosAdapter()
        result = adapter.ingest(pos_inventory_file)
        # Second item has Qty On Hand = 0, Qty. = -3
        rec = result.inventory_records[1]
        assert rec.sku_id == "TEST-SKU-002"
        # Adapter prefers "Qty." over "Qty On Hand" because "Qty."
        # captures negatives critical for NegativeInventory detection.
        # In real data: "Qty." has 3,958 negatives, "Qty On Hand" has zero.
        assert rec.qty_on_hand == -3

    def test_empty_fields_handled(self, pos_inventory_file):
        adapter = GenericPosAdapter()
        result = adapter.ingest(pos_inventory_file)
        rec = result.inventory_records[2]
        assert rec.sku_id == "TEST-SKU-003"
        assert rec.vendor is None
        assert rec.unit_cost == 0.0

    def test_store_id_propagation(self, pos_inventory_file):
        adapter = GenericPosAdapter()
        result = adapter.ingest(pos_inventory_file, store_id="my-store")
        for rec in result.inventory_records:
            assert rec.store_id == "my-store"

    def test_shlp_format(self, shlp_file):
        adapter = GenericPosAdapter()
        result = adapter.ingest(shlp_file)
        assert result.total_inventory_records == 2
        rec = result.inventory_records[0]
        assert rec.sku_id == "G1956663"
        assert rec.vendor == "NATIONAL HARDWARE"
        assert rec.qty_on_hand == 40124
        assert abs(rec.unit_cost - 0.0) < 0.01  # Avg. Cost = 0.00

    def test_shlp_date_parsing(self, shlp_file):
        adapter = GenericPosAdapter()
        result = adapter.ingest(shlp_file)
        rec = result.inventory_records[0]
        assert rec.last_sale_date == date(2025, 6, 2)
        assert rec.last_receipt_date == date(2025, 6, 2)  # Real Date column

    def test_adapter_name(self):
        adapter = GenericPosAdapter()
        assert adapter.name == "Generic POS Inventory"

    def test_directory_picks_custom1(self, pos_inventory_dir):
        """When directory has custom_1.csv, it should be picked first."""
        adapter = GenericPosAdapter()
        result = adapter.ingest(pos_inventory_dir)
        assert result.total_inventory_records == 3  # 3 rows in our fixture


# ---------------------------------------------------------------------------
# Auto-Detection Tests
# ---------------------------------------------------------------------------


class TestDetection:
    def test_detect_orgill(self, orgill_po_file):
        adapter = detect_adapter(orgill_po_file)
        assert adapter is not None
        assert adapter.name == "Orgill PO"

    def test_detect_orgill_dir(self, orgill_po_dir):
        adapter = detect_adapter(orgill_po_dir)
        assert adapter is not None
        assert adapter.name == "Orgill PO"

    def test_detect_pos_inventory(self, pos_inventory_file):
        adapter = detect_adapter(pos_inventory_file)
        assert adapter is not None
        assert adapter.name == "Generic POS Inventory"

    def test_detect_unknown(self, tmp_path):
        f = tmp_path / "random.csv"
        f.write_text("a,b,c\n1,2,3\n")
        adapter = detect_adapter(f)
        assert adapter is None

    def test_detect_and_ingest_orgill(self, orgill_po_file):
        result = detect_and_ingest(orgill_po_file)
        assert result.adapter_name == "Orgill PO"
        assert result.total_purchase_orders == 1

    def test_detect_and_ingest_pos_inventory(self, pos_inventory_file):
        result = detect_and_ingest(pos_inventory_file)
        assert result.adapter_name == "Generic POS Inventory"
        assert result.total_inventory_records == 3

    def test_detect_and_ingest_unknown(self, tmp_path):
        f = tmp_path / "random.csv"
        f.write_text("a,b,c\n1,2,3\n")
        result = detect_and_ingest(f)
        assert result.has_errors
        assert "No adapter found" in result.errors[0]

    def test_list_adapters(self):
        adapters = list_adapters()
        assert len(adapters) == 5
        names = [a["name"] for a in adapters]
        assert "Orgill PO" in names
        assert "Generic POS Inventory" in names
        assert "Sales Data" in names
        assert "Do It Best" in names
        assert "Ace Hardware" in names


# ---------------------------------------------------------------------------
# Stub Adapter Tests
# ---------------------------------------------------------------------------


class TestStubAdapters:
    def test_do_it_best_not_implemented(self, tmp_path):
        from sentinel_agent.adapters.do_it_best import DoItBestAdapter

        adapter = DoItBestAdapter()
        assert adapter.name == "Do It Best"
        assert not adapter.can_handle(tmp_path)
        result = adapter.ingest(tmp_path)
        assert result.has_errors

    def test_ace_not_implemented(self, tmp_path):
        from sentinel_agent.adapters.ace import AceAdapter

        adapter = AceAdapter()
        assert adapter.name == "Ace Hardware"
        assert not adapter.can_handle(tmp_path)
        result = adapter.ingest(tmp_path)
        assert result.has_errors


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_orgill_empty_directory(self, tmp_path):
        adapter = OrgillPOAdapter()
        result = adapter.ingest(tmp_path)
        assert result.total_purchase_orders == 0
        assert result.files_processed == 0

    def test_orgill_truncated_file(self, tmp_path):
        f = tmp_path / "truncated.csv"
        f.write_text("Shipto: 462283\nOnly one line\n")
        adapter = OrgillPOAdapter()
        result = adapter.ingest(f)
        assert result.total_purchase_orders == 0

    def test_pos_inventory_empty_csv(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("SKU,Vendor SKU,Vendor,Qty On Hand,Avg. Cost,Retail\n")
        adapter = GenericPosAdapter()
        result = adapter.ingest(f)
        assert result.total_inventory_records == 0

    def test_nonexistent_path(self):
        adapter = OrgillPOAdapter()
        result = adapter.ingest(Path("/nonexistent/path"))
        assert result.has_errors

    def test_margin_pct_normal(self):
        rec = NormalizedInventory(sku_id="A", unit_cost=7.0, retail_price=10.0)
        assert abs(rec.margin_pct - 0.30) < 0.01

    def test_margin_pct_negative_cost(self):
        """Negative cost scenario (data error, but should not crash)."""
        rec = NormalizedInventory(sku_id="A", unit_cost=-5.0, retail_price=10.0)
        # (-5 - 10) / 10 = -1.5 — data error but handled gracefully
        assert rec.margin_pct == 1.5

    def test_po_fill_rate_no_items(self):
        po = PurchaseOrder(po_number="EMPTY")
        assert po.fill_rate == 1.0
        assert po.total_ordered_value == 0.0


# ---------------------------------------------------------------------------
# Real-Data Negative Inventory Tests
# ---------------------------------------------------------------------------

_REAL_CUSTOM1 = Path("/Users/joseph/Downloads/custom_1.csv")
_REAL_REPORTS_DIR = Path("/Users/joseph/Downloads/Reports")


@pytest.mark.skipif(
    not _REAL_CUSTOM1.exists(),
    reason="Real custom_1.csv not available",
)
class TestRealNegativeInventory:
    """Tests against real data to verify negative inventory handling.

    The real custom_1.csv has 156,157 data rows. The 'Qty.' column contains
    3,958 negative entries which are critical for NegativeInventory detection
    in the Rust pipeline. The 'Qty On Hand' column has ZERO negatives.
    """

    def test_negative_inventory_count(self):
        """Verify the adapter captures negative inventory from Qty. column."""
        adapter = GenericPosAdapter()
        result = adapter.ingest(_REAL_CUSTOM1)
        assert not result.has_errors, f"Errors: {result.errors[:5]}"

        negative_records = [r for r in result.inventory_records if r.qty_on_hand < 0]
        # Real data has 3,958 negative entries in Qty. column
        assert len(negative_records) >= 3900, (
            f"Expected ~3,958 negative inventory records, got {len(negative_records)}. "
            "The adapter may not be reading the Qty. column correctly."
        )
        assert (
            len(negative_records) <= 4100
        ), f"Got {len(negative_records)} negatives — suspiciously high"

    def test_total_record_count(self):
        """Verify all 156K+ records are parsed."""
        adapter = GenericPosAdapter()
        result = adapter.ingest(_REAL_CUSTOM1)
        assert (
            result.total_inventory_records >= 156000
        ), f"Expected ~156,157 records, got {result.total_inventory_records}"

    def test_no_false_negatives_from_qty_on_hand(self):
        """Verify we aren't accidentally reading Qty On Hand as negative.

        In real data, Qty On Hand never has negatives — only Qty. does.
        This tests that the adapter is reading the right column.
        """
        adapter = GenericPosAdapter()
        result = adapter.ingest(_REAL_CUSTOM1)

        # Sample a few negative records and verify they have valid SKUs
        negative_records = [r for r in result.inventory_records if r.qty_on_hand < 0]
        for rec in negative_records[:10]:
            assert rec.sku_id, "Negative record should have a SKU"
            assert rec.qty_on_hand < 0
            # Most negatives are small (sold-not-received), typically -1 to -50
            # but some could be larger data anomalies
            assert rec.qty_on_hand >= -10000, (
                f"SKU {rec.sku_id} has qty_on_hand={rec.qty_on_hand} — "
                "suspiciously large negative"
            )

    def test_positive_inventory_values(self):
        """Verify records with positive inventory have sane values."""
        adapter = GenericPosAdapter()
        result = adapter.ingest(_REAL_CUSTOM1)

        positive_records = [r for r in result.inventory_records if r.qty_on_hand > 0]
        assert len(positive_records) > 10000, "Should have many positive records"

        # Spot-check: positive records should often have costs and retail prices
        has_cost = sum(1 for r in positive_records if r.unit_cost > 0)
        assert (
            has_cost > len(positive_records) * 0.5
        ), "Most positive-inventory items should have a unit cost"
