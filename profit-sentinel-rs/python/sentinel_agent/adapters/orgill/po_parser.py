"""Orgill PO parser.

Parses the non-standard Orgill CSV format:
  - Rows 1-14: Header block (ship-to, order #, date, terms, comments)
  - Row 15: Column headers for line items
  - Rows 16+: Line items with fixed-width-ish CSV columns

Short-ship detection:
  - '*' in the "Out" column (index 12) indicates item not filled or partially filled
  - '*' after "Spc" column (index 8) indicates special pricing

Header extraction:
  - Row 1: Shipto, Billto, Order number
  - Row 3: Date
  - Row 4: Status
  - Row 6: Terms
  - Row 7: USD Amount
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime
from pathlib import Path

from ..base import (
    AdapterResult,
    BaseAdapter,
    POLineItem,
    POStatus,
    PurchaseOrder,
)

logger = logging.getLogger("sentinel.adapters.orgill")

# Orgill PO files start with "Shipto:" in line 1
_ORGILL_SIGNATURE = "Shipto:"

# Service line items that are not actual product
_SERVICE_SKUS = {"0000000"}


class OrgillPOAdapter(BaseAdapter):
    """Adapter for Orgill purchase order CSVs."""

    @property
    def name(self) -> str:
        return "Orgill PO"

    def can_handle(self, path: Path) -> bool:
        """Detect Orgill PO files by the 'Shipto:' signature in line 1."""
        path = Path(path)

        if path.is_dir():
            # Check any CSV in the directory
            csvs = list(path.glob("*.csv"))
            if not csvs:
                return False
            return self._check_file(csvs[0])

        if path.is_file() and path.suffix.lower() == ".csv":
            return self._check_file(path)

        return False

    def _check_file(self, path: Path) -> bool:
        """Check if a file starts with the Orgill signature."""
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                first_line = f.readline()
            return first_line.strip().startswith(_ORGILL_SIGNATURE)
        except (OSError, UnicodeDecodeError):
            return False

    def ingest(self, path: Path, store_id: str = "default-store") -> AdapterResult:
        """Parse Orgill PO file(s) and return normalized PurchaseOrder records.

        Args:
            path: Path to a single PO CSV or a directory of PO CSVs.
            store_id: Store identifier.

        Returns:
            AdapterResult with purchase_orders populated.
        """
        start = time.monotonic()
        path = Path(path)

        errors: list[str] = []
        warnings: list[str] = []
        purchase_orders: list[PurchaseOrder] = []

        if path.is_dir():
            csv_files = sorted(path.glob("*.csv"))
        elif path.is_file():
            csv_files = [path]
        else:
            return AdapterResult(
                source=str(path),
                adapter_name=self.name,
                errors=[f"Path not found: {path}"],
            )

        for csv_file in csv_files:
            try:
                po = self._parse_single_po(csv_file)
                if po:
                    purchase_orders.append(po)
            except Exception as e:
                errors.append(f"{csv_file.name}: {e}")

        elapsed_ms = int((time.monotonic() - start) * 1000)

        return AdapterResult(
            source=str(path),
            adapter_name=self.name,
            purchase_orders=purchase_orders,
            errors=errors,
            warnings=warnings,
            files_processed=len(csv_files),
            processing_time_ms=elapsed_ms,
        )

    def _parse_single_po(self, path: Path) -> PurchaseOrder | None:
        """Parse a single Orgill PO CSV file."""
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if len(lines) < 16:
            return None

        # --- Parse header block (rows 1-14) ---
        header = self._parse_header(lines[:14])

        # --- Parse line items (row 16+, row 15 is column headers) ---
        line_items = self._parse_line_items(lines[15:])

        return PurchaseOrder(
            po_number=header.get("order", path.stem),
            vendor="Orgill",
            order_date=header.get("date"),
            status=header.get("status", POStatus.UNKNOWN),
            terms=header.get("terms"),
            ship_to=header.get("ship_to"),
            bill_to=header.get("bill_to"),
            usd_amount=header.get("usd_amount", 0.0),
            line_items=line_items,
        )

    def _parse_header(self, header_lines: list[str]) -> dict:
        """Extract structured data from the 14-line header block."""
        result: dict = {}

        if not header_lines:
            return result

        # Row 1: "Shipto: 462283 ,Billto: 462283 ,Order: 7959433"
        row1 = header_lines[0]
        parts = row1.split(",")
        for part in parts:
            part = part.strip()
            if part.startswith("Shipto:"):
                result["ship_to"] = part[7:].strip()
            elif part.startswith("Billto:"):
                result["bill_to"] = part[7:].strip()
            elif part.startswith("Order:"):
                result["order"] = part[6:].strip()

        # Row 3 (index 2): "...Date: 12/02/2025"
        if len(header_lines) > 2:
            date_match = re.search(r"Date:\s*(\d{2}/\d{2}/\d{4})", header_lines[2])
            if date_match:
                try:
                    result["date"] = datetime.strptime(
                        date_match.group(1), "%m/%d/%Y"
                    ).date()
                except ValueError:
                    pass

        # Row 4 (index 3): "...Status: INVOICED"
        if len(header_lines) > 3:
            status_match = re.search(r"Status:\s*(\S+)", header_lines[3])
            if status_match:
                status_str = status_match.group(1).upper()
                try:
                    result["status"] = POStatus(status_str)
                except ValueError:
                    result["status"] = POStatus.UNKNOWN

        # Row 6 (index 5): "...Terms: NET DEC. 25TH"
        if len(header_lines) > 5:
            terms_match = re.search(r"Terms:\s*(.+)", header_lines[5])
            if terms_match:
                result["terms"] = terms_match.group(1).strip()

        # Row 7 (index 6): "...USD Amount:   21726.53"
        if len(header_lines) > 6:
            amount_match = re.search(r"USD Amount:\s*([\d,]+\.?\d*)", header_lines[6])
            if amount_match:
                try:
                    result["usd_amount"] = float(amount_match.group(1).replace(",", ""))
                except ValueError:
                    pass

        return result

    def _parse_line_items(self, data_lines: list[str]) -> list[POLineItem]:
        """Parse line item rows from position-aware CSV fields.

        Orgill CSV is comma-separated but values are padded with spaces.
        Column order (0-indexed):
            0: Line
            1: Retail
            2: Item (SKU)
            3: (flag column, Y or blank)
            4: Ord Qty
            5: Unit
            6: Description
            7: Unit Cost
            8: Spc (special price marker, * if special)
            9: Ext Cost
            10: Prod Care
            11: Qty Fill
            12: Out (* if short-shipped)
            13: Shelf Pk
            14: UPC Code
            15: POS
            16: Crd Inv
            17: Retail Dept Description
            18: Dept
            19: Vendor Item Num
            20: PickLne
            21: Country of Origin
            ...
        """
        items: list[POLineItem] = []

        for line in data_lines:
            line = line.rstrip("\n\r")
            if not line.strip():
                continue

            parts = line.split(",")
            if len(parts) < 13:
                continue

            try:
                item = self._parse_line_item(parts)
                if item:
                    items.append(item)
            except Exception as e:
                logger.debug("Failed to parse line: %s — %s", line[:80], e)
                continue

        return items

    def _parse_line_item(self, parts: list[str]) -> POLineItem | None:
        """Parse a single line item from split CSV parts."""
        # Strip all parts
        parts = [p.strip() for p in parts]

        line_num = _safe_int(parts[0])
        if line_num is None:
            return None

        sku_id = parts[2] if len(parts) > 2 else ""
        retail = _safe_float(parts[1]) if len(parts) > 1 else 0.0
        ord_qty = _safe_int(parts[4]) if len(parts) > 4 else 0
        unit_measure = parts[5] if len(parts) > 5 else None
        description = parts[6] if len(parts) > 6 else ""
        unit_cost = _safe_float(parts[7]) if len(parts) > 7 else 0.0
        spc = parts[8] if len(parts) > 8 else ""
        ext_cost = _safe_float(parts[9]) if len(parts) > 9 else 0.0
        qty_fill = _safe_int(parts[11]) if len(parts) > 11 else 0
        out_marker = parts[12] if len(parts) > 12 else ""
        parts[13] if len(parts) > 13 else ""
        upc = parts[14] if len(parts) > 14 else None
        dept_desc = parts[17] if len(parts) > 17 else None
        dept_code = parts[18] if len(parts) > 18 else None
        vendor_item = parts[19] if len(parts) > 19 else None
        country = parts[21] if len(parts) > 21 else None

        is_short = "*" in out_marker
        is_special = "*" in spc

        return POLineItem(
            line_number=line_num,
            sku_id=sku_id,
            description=description,
            qty_ordered=ord_qty or 0,
            qty_filled=qty_fill or 0,
            unit_cost=unit_cost or 0.0,
            ext_cost=ext_cost or 0.0,
            retail_price=retail or 0.0,
            upc=upc if upc else None,
            department=dept_desc if dept_desc else None,
            department_code=dept_code if dept_code else None,
            unit_of_measure=unit_measure if unit_measure else None,
            is_short_ship=is_short,
            is_special_price=is_special,
            country_of_origin=country if country else None,
            vendor_item_number=vendor_item if vendor_item else None,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_int(s: str) -> int | None:
    """Parse an integer from a possibly padded string."""
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        # Try removing non-digit characters
        digits = re.sub(r"[^\d-]", "", s)
        if digits:
            try:
                return int(digits)
            except ValueError:
                return None
        return None


def _safe_float(s: str) -> float:
    """Parse a float from a possibly padded string."""
    s = s.strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        # Try removing non-numeric characters (except . and -)
        cleaned = re.sub(r"[^\d.\-]", "", s)
        if cleaned:
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
        return 0.0
