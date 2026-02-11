"""Generic POS inventory adapter.

Maps a common POS export format to the canonical NormalizedInventory
schema. Supports two main CSV layouts:

1. custom_1.csv / Full inventory report — Full inventory
   with 50+ columns (Qty On Hand, Std. Cost, Retail, etc.)

2. SHLP_YTD report — Orgill-sourced inventory report
   with monthly sales columns (Jan, Feb, ..., Year Total)

Column mapping (custom_1 / full inventory):
    SKU                → sku_id
    Description Full   → description
    Vendor             → vendor
    Vendor SKU         → vendor_sku
    Qty On Hand        → qty_on_hand  (note: column 10, "Qty." is different)
    Avg. Cost          → unit_cost
    Retail             → retail_price
    Last Sale          → last_sale_date (format: YYYYMMDD)
    Last Purchase      → last_receipt_date
    BIN                → bin_location
    Cat.               → category
    Dpt.               → department
    Barcode            → barcode
    On Order           → on_order_qty
    Min.               → min_qty
    Max.               → max_qty
    Sales              → sales_ytd
    Cost (P&L)         → cost_ytd

Date formats:
    custom_1: YYYYMMDD (e.g. "20250311")
    SHLP_YTD: "Mon DD,YY" (e.g. "Jun 02,25")
"""

from __future__ import annotations

import csv
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path

from ..base import AdapterResult, BaseAdapter, NormalizedInventory

logger = logging.getLogger("sentinel.adapters.generic_pos")

# Detection signatures
_CUSTOM1_COLUMNS = {"SKU", "Vendor SKU", "Vendor", "Qty On Hand", "Avg. Cost", "Retail"}
_SHLP_COLUMNS = {"SKU", "Vendor", "Vendor SKU", "Stock", "Avg. Cost", "Gross Sales"}


class GenericPosAdapter(BaseAdapter):
    """Adapter for generic POS inventory exports."""

    @property
    def name(self) -> str:
        return "Generic POS Inventory"

    def can_handle(self, path: Path) -> bool:
        """Detect compatible inventory by column signature."""
        path = Path(path)

        if path.is_dir():
            # Look for known filenames
            known_files = [
                "Inventory_Report_AllSKUs_SHLP_YTD.csv",
                "custom_1.csv",
                "SKU_Profit_LossYTD.csv",
            ]
            for name in known_files:
                if (path / name).exists():
                    return True
            # Check any CSV for column signatures
            csvs = list(path.glob("*.csv"))
            return any(self._check_file(c) for c in csvs[:3])

        if path.is_file() and path.suffix.lower() == ".csv":
            return self._check_file(path)

        return False

    def _check_file(self, path: Path) -> bool:
        """Check if a CSV has compatible column signatures."""
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                header = f.readline()
            columns = {c.strip().strip('"') for c in header.split(",")}
            return bool(columns & _CUSTOM1_COLUMNS) or bool(columns & _SHLP_COLUMNS)
        except (OSError, UnicodeDecodeError):
            return False

    def ingest(self, path: Path, store_id: str = "default-store") -> AdapterResult:
        """Parse POS inventory files.

        Supports:
        - Single CSV file
        - Directory containing inventory CSVs (auto-detects best file)
        """
        start = time.monotonic()
        path = Path(path)

        errors: list[str] = []
        warnings: list[str] = []
        records: list[NormalizedInventory] = []
        files_processed = 0

        if path.is_dir():
            # Priority order for inventory data
            candidates = [
                path / "custom_1.csv",
                path / "Inventory_Report_AllSKUs_SHLP_YTD.csv",
                path / "Inventory_Report_GreaterThanZero_AllSKUs.csv",
            ]
            target = None
            for c in candidates:
                if c.exists():
                    target = c
                    break

            if not target:
                # Try any CSV with matching columns
                for csv_file in sorted(path.glob("*.csv")):
                    if self._check_file(csv_file):
                        target = csv_file
                        break

            if not target:
                return AdapterResult(
                    source=str(path),
                    adapter_name=self.name,
                    errors=["No compatible inventory CSV found in directory"],
                )

            result = self._parse_inventory_csv(target, store_id)
            records = result["records"]
            errors.extend(result["errors"])
            warnings.extend(result["warnings"])
            files_processed = 1

        elif path.is_file():
            # Detect format from header
            header_cols = self._read_header(path)
            if "Qty On Hand" in header_cols or "Qty." in header_cols:
                result = self._parse_inventory_csv(path, store_id)
            elif "Stock" in header_cols:
                result = self._parse_shlp_csv(path, store_id)
            else:
                result = self._parse_inventory_csv(path, store_id)

            records = result["records"]
            errors.extend(result["errors"])
            warnings.extend(result["warnings"])
            files_processed = 1

        elapsed_ms = int((time.monotonic() - start) * 1000)

        return AdapterResult(
            source=str(path),
            adapter_name=self.name,
            inventory_records=records,
            errors=errors,
            warnings=warnings,
            files_processed=files_processed,
            processing_time_ms=elapsed_ms,
        )

    def _read_header(self, path: Path) -> set[str]:
        """Read CSV header and return column names."""
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, [])
            return {c.strip() for c in header}
        except (OSError, UnicodeDecodeError, StopIteration):
            return set()

    def _parse_inventory_csv(
        self,
        path: Path,
        store_id: str,
    ) -> dict:
        """Parse custom_1.csv / full inventory format.

        These files have columns:
        SKU, Vendor SKU, Vendor, Alt. Vendor/Mfgr., Cat., Dpt., BIN,
        Description Full, Description Short, Qty., Qty On Hand, Std. Cost,
        Inventory @Cost, Avg. Cost, Inventory @AvgCost, Retail,
        Inventory @Retail, Margin @Cost, Sug. Retail, Retail Dif., On Hold,
        Sales, $ Sold, Last Sale, Returns, $ Returned, Last Return,
        Last Ordered, On Order, Last Purchase, Min., Max., Pkg., Whse.,
        Mfgr. SKU, Barcode, ...
        """
        records: list[NormalizedInventory] = []
        errors: list[str] = []
        warnings: list[str] = []
        skipped = 0

        with open(path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)

            for row_num, row in enumerate(reader, start=2):
                try:
                    record = self._map_inventory_row(row, store_id)
                    if record:
                        records.append(record)
                    else:
                        skipped += 1
                except Exception as e:
                    if len(errors) < 50:  # Cap error messages
                        errors.append(f"Row {row_num}: {e}")

        if skipped > 0:
            warnings.append(f"Skipped {skipped} rows with empty SKU")

        return {"records": records, "errors": errors, "warnings": warnings}

    def _map_inventory_row(
        self,
        row: dict,
        store_id: str,
    ) -> NormalizedInventory | None:
        """Map a single CSV row to NormalizedInventory."""
        sku = row.get("SKU", "").strip()
        if not sku:
            return None

        # Parse qty_on_hand — prefer "Qty." (includes negatives from sold-not-received),
        # fall back to "Qty On Hand" (physical count, typically non-negative).
        # The negatives are critical for the pipeline's NegativeInventory detection.
        qty_raw = _safe_int(row.get("Qty.", ""))
        if qty_raw is None:
            qty_raw = _safe_int(row.get("Qty On Hand", ""))
        qty_on_hand = qty_raw if qty_raw is not None else 0

        # Parse cost — prefer Avg. Cost, fall back to Std. Cost
        avg_cost = _safe_float(row.get("Avg. Cost", ""))
        unit_cost = (
            avg_cost if avg_cost != 0.0 else _safe_float(row.get("Std. Cost", ""))
        )

        # Parse retail price
        retail = _safe_float(row.get("Retail", ""))

        # Parse dates
        last_sale = _parse_date_yyyymmdd(row.get("Last Sale", ""))
        last_receipt = _parse_date_yyyymmdd(row.get("Last Purchase", ""))

        # Parse sales data
        sales_ytd = _safe_float(row.get("$ Sold", "").replace(",", ""))

        return NormalizedInventory(
            sku_id=sku,
            description=row.get("Description Full", "").strip()
            or row.get("Description Short", "").strip()
            or None,
            vendor=row.get("Vendor", "").strip() or None,
            vendor_sku=row.get("Vendor SKU", "").strip() or None,
            qty_on_hand=qty_on_hand,
            unit_cost=unit_cost,
            retail_price=retail,
            last_sale_date=last_sale,
            last_receipt_date=last_receipt,
            bin_location=row.get("BIN", "").strip() or None,
            store_id=store_id,
            category=row.get("Cat.", "").strip() or None,
            department=row.get("Dpt.", "").strip() or None,
            barcode=row.get("Barcode", "").strip() or None,
            on_order_qty=_safe_int(row.get("On Order", "")) or 0,
            min_qty=_safe_int(row.get("Min.", "")) or 0,
            max_qty=_safe_int(row.get("Max.", "")) or 0,
            sales_ytd=sales_ytd,
        )

    def _parse_shlp_csv(
        self,
        path: Path,
        store_id: str,
    ) -> dict:
        """Parse Inventory_Report_AllSKUs_SHLP_YTD.csv format.

        These files have monthly sales columns (Jan, Feb, ...) and
        use "Stock" instead of "Qty On Hand".
        """
        records: list[NormalizedInventory] = []
        errors: list[str] = []
        warnings: list[str] = []
        skipped = 0

        with open(path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)

            for row_num, row in enumerate(reader, start=2):
                try:
                    record = self._map_shlp_row(row, store_id)
                    if record:
                        records.append(record)
                    else:
                        skipped += 1
                except Exception as e:
                    if len(errors) < 50:
                        errors.append(f"Row {row_num}: {e}")

        if skipped > 0:
            warnings.append(f"Skipped {skipped} rows with empty SKU")

        return {"records": records, "errors": errors, "warnings": warnings}

    def _map_shlp_row(
        self,
        row: dict,
        store_id: str,
    ) -> NormalizedInventory | None:
        """Map a SHLP-format row to NormalizedInventory."""
        sku = row.get("SKU", "").strip()
        if not sku:
            return None

        qty_on_hand = _safe_int(row.get("Stock", "")) or 0
        unit_cost = _safe_float(row.get("Avg. Cost", ""))
        sales_ytd = _safe_float(row.get("Gross Sales", "").replace(",", ""))

        # Parse "Mon DD,YY" date format
        last_sale = _parse_date_mon_dd_yy(row.get("Last Sale", ""))
        last_receipt_raw = row.get("Real Date", "")
        last_receipt = _parse_date_yyyymmdd(last_receipt_raw)

        # Calculate year total from monthly columns
        year_total = _safe_float(row.get("Year Total", "").replace(",", ""))

        return NormalizedInventory(
            sku_id=sku,
            description=row.get("Description", "").strip() or None,
            vendor=row.get("Vendor", "").strip() or None,
            vendor_sku=row.get("Vendor SKU", "").strip() or None,
            qty_on_hand=qty_on_hand,
            unit_cost=unit_cost,
            retail_price=0.0,  # Not available in SHLP format
            last_sale_date=last_sale,
            last_receipt_date=last_receipt,
            store_id=store_id,
            sales_ytd=sales_ytd or year_total,
        )


# ---------------------------------------------------------------------------
# Date parsing helpers
# ---------------------------------------------------------------------------


def _parse_date_yyyymmdd(s: str) -> date | None:
    """Parse YYYYMMDD format (e.g. '20250311')."""
    s = s.strip()
    if not s or len(s) < 8:
        return None
    try:
        return datetime.strptime(s[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _parse_date_mon_dd_yy(s: str) -> date | None:
    """Parse 'Mon DD,YY' format (e.g. 'Jun 02,25')."""
    s = s.strip().strip('"')
    if not s:
        return None
    try:
        return datetime.strptime(s, "%b %d,%y").date()
    except ValueError:
        return None


def _safe_int(s: str) -> int | None:
    """Parse int, returning None on failure."""
    s = s.strip()
    if not s:
        return None
    try:
        return int(float(s))  # Handle "3.0" -> 3
    except (ValueError, TypeError):
        return None


def _safe_float(s: str) -> float:
    """Parse float, returning 0.0 on failure."""
    s = s.strip()
    if not s:
        return 0.0
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return 0.0
