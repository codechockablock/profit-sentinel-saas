"""Sales data adapter — Phase 12: Sales Data Integration.

Parses transaction-level sales data from POS exports and aggregates
per-SKU sales metrics for the Rust pipeline. Replaces the rough
``sales_ytd / months_elapsed`` estimation in the bridge with actual
30-day rolling sales figures.

Supported formats:
    1. **Paladin Transaction Export** — daily sales CSV with columns:
       Date, SKU, Description, Qty Sold, Amount, Cost, Customer
    2. **Generic Sales Detail** — minimal: sku, date, qty_sold, amount
    3. **Monthly Columnar** — already handled by Sample Store SHLP parser;
       this module focuses on transaction-level data.

Usage:
    from sentinel_agent.adapters.sales import (
        SalesDataAdapter,
        SalesOverlay,
        aggregate_sales_30d,
    )

    adapter = SalesDataAdapter()
    result = adapter.ingest(Path("sales_detail.csv"), store_id="default-store")

    # Overlay onto existing inventory records
    overlay = SalesOverlay.from_transactions(result.transactions)
    enriched = overlay.apply(inventory_records)
    # Now each record's sales_last_30d is populated from actual data

Column detection:
    The adapter auto-detects columns via alias matching (same pattern as
    POS_COLUMN_MAPPING_REFERENCE.md). Aliases cover Paladin, Eagle, Spruce,
    Lightspeed, Shopify, Square, and generic exports.
"""

from __future__ import annotations

import csv
import logging
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from .base import AdapterResult, BaseAdapter, NormalizedInventory

logger = logging.getLogger("sentinel.adapters.sales")


# ---------------------------------------------------------------------------
# Column alias tables (matches POS_COLUMN_MAPPING_REFERENCE.md)
# ---------------------------------------------------------------------------

_SKU_ALIASES = [
    "sku",
    "SKU",
    "item_number",
    "Item Number",
    "Item",
    "partnumber",
    "Part Number",
    "product_id",
    "Product ID",
    "item_code",
    "ItemCode",
    "barcode",
    "upc",
    "UPC",
    "item_no",
    "Item No.",
    "Stock Number",
    "PLU",
    "Item Sku",
]

_DATE_ALIASES = [
    "date",
    "Date",
    "transaction_date",
    "Transaction Date",
    "sale_date",
    "Sale Date",
    "sold_date",
    "Sold Date",
    "invoice_date",
    "Invoice Date",
    "completed_at",
    "Completed At",
    "created_at",
    "Sold On",
    "Txn Date",
]

_QTY_ALIASES = [
    "qty_sold",
    "Qty Sold",
    "quantity",
    "Quantity",
    "qty",
    "Qty",
    "units_sold",
    "Units Sold",
    "quantity_sold",
    "Net Qty",
    "Gross Qty",
    "net_qty",
    "gross_qty",
    "Units",
]

_AMOUNT_ALIASES = [
    "amount",
    "Amount",
    "total",
    "Total",
    "net_sales",
    "Net Sales",
    "gross_sales",
    "Gross Sales",
    "ext_price",
    "Ext. Price",
    "extended_price",
    "sales_amount",
    "Sales Amount",
    "line_total",
    "Line Total",
    "revenue",
    "Revenue",
    "$ Sold",
    "Sold Amt",
    "Sale Amount",
]

_COST_ALIASES = [
    "cost",
    "Cost",
    "unit_cost",
    "Unit Cost",
    "Avg. Cost",
    "avg_cost",
    "Ext. Cost",
    "ext_cost",
    "cost_amount",
    "Cost Amount",
    "COGS",
    "cogs",
]

_STORE_ALIASES = [
    "store_id",
    "Store ID",
    "store",
    "Store",
    "location",
    "Location",
    "outlet",
    "Outlet",
    "site",
    "Site",
    "branch",
    "Branch",
]

# Detection: at minimum we need SKU + Date + (Qty or Amount)
_REQUIRED_GROUPS = [_SKU_ALIASES, _DATE_ALIASES]
_OPTIONAL_GROUPS = [_QTY_ALIASES, _AMOUNT_ALIASES]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SalesTransaction(BaseModel):
    """A single sales transaction line item."""

    sku: str
    sale_date: date
    qty_sold: float = 0.0
    amount: float = 0.0
    unit_cost: float = 0.0
    store_id: str = ""


class SalesAggregation(BaseModel):
    """Per-SKU sales aggregation for a rolling window."""

    sku: str
    store_id: str = ""
    total_qty_sold: float = 0.0
    total_amount: float = 0.0
    total_cost: float = 0.0
    transaction_count: int = 0
    first_sale: date | None = None
    last_sale: date | None = None
    days_in_window: int = 30


class SalesAdapterResult(BaseModel):
    """Extended adapter result with transaction-level data."""

    source: str
    adapter_name: str = "Sales Data"
    transactions: list[SalesTransaction] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    files_processed: int = 0
    processing_time_ms: int = 0
    date_range_start: date | None = None
    date_range_end: date | None = None

    @property
    def total_transactions(self) -> int:
        return len(self.transactions)

    @property
    def total_revenue(self) -> float:
        return sum(t.amount for t in self.transactions)

    @property
    def unique_skus(self) -> int:
        return len({t.sku for t in self.transactions})


# ---------------------------------------------------------------------------
# Column auto-detection
# ---------------------------------------------------------------------------


def _find_column(headers: list[str], aliases: list[str]) -> str | None:
    """Find the first header that matches any alias (case-insensitive)."""
    header_lower = {h.strip().lower(): h.strip() for h in headers}
    for alias in aliases:
        key = alias.strip().lower()
        if key in header_lower:
            return header_lower[key]
    return None


def _detect_columns(headers: list[str]) -> dict[str, str | None]:
    """Auto-detect column mapping from CSV headers."""
    return {
        "sku": _find_column(headers, _SKU_ALIASES),
        "date": _find_column(headers, _DATE_ALIASES),
        "qty": _find_column(headers, _QTY_ALIASES),
        "amount": _find_column(headers, _AMOUNT_ALIASES),
        "cost": _find_column(headers, _COST_ALIASES),
        "store": _find_column(headers, _STORE_ALIASES),
    }


def _is_sales_csv(headers: list[str]) -> bool:
    """Check if headers look like a sales transaction CSV."""
    mapping = _detect_columns(headers)
    has_required = all(mapping[k] is not None for k in ("sku", "date"))
    has_measure = mapping["qty"] is not None or mapping["amount"] is not None
    return has_required and has_measure


# ---------------------------------------------------------------------------
# Date parsing (multi-format)
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d",  # 2025-03-15
    "%m/%d/%Y",  # 03/15/2025
    "%m/%d/%y",  # 03/15/25
    "%Y%m%d",  # 20250315
    "%d-%b-%Y",  # 15-Mar-2025
    "%d-%b-%y",  # 15-Mar-25
    "%b %d,%y",  # Mar 15,25 (Paladin SHLP)
    "%b %d, %Y",  # Mar 15, 2025
    "%m-%d-%Y",  # 03-15-2025
    "%Y/%m/%d",  # 2025/03/15
]


def _parse_date(s: str) -> date | None:
    """Parse a date string trying multiple formats."""
    s = s.strip().strip('"').strip("'")
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _safe_float(s: str) -> float:
    """Parse a float, handling commas and dollar signs."""
    s = s.strip().strip('"').strip("'")
    if not s or s == "-":
        return 0.0
    # Remove dollar signs, commas, parentheses (negative)
    s = s.replace("$", "").replace(",", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Sales Data Adapter
# ---------------------------------------------------------------------------


class SalesDataAdapter(BaseAdapter):
    """Adapter for transaction-level sales data exports.

    Parses sales CSVs from various POS systems, auto-detecting column
    names via alias matching. Produces SalesTransaction records that
    can be aggregated and overlaid onto inventory records.
    """

    @property
    def name(self) -> str:
        return "Sales Data"

    def can_handle(self, path: Path) -> bool:
        """Detect sales transaction CSV by column signature."""
        path = Path(path)

        if path.is_dir():
            # Look for known sales filenames
            known_patterns = [
                "Sales_Detail*.csv",
                "sales_detail*.csv",
                "transactions*.csv",
                "Transactions*.csv",
                "sales_export*.csv",
                "Sales_Export*.csv",
            ]
            for pat in known_patterns:
                matches = list(path.glob(pat))
                if matches:
                    return True
            # Check first few CSVs for sales column signatures
            csvs = sorted(path.glob("*.csv"))
            for csv_file in csvs[:5]:
                if self._check_file(csv_file):
                    return True
            return False

        if path.is_file() and path.suffix.lower() == ".csv":
            return self._check_file(path)

        return False

    def _check_file(self, path: Path) -> bool:
        """Check if a CSV has sales transaction column signatures."""
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                header_line = f.readline()
            headers = [h.strip() for h in header_line.split(",")]
            return _is_sales_csv(headers)
        except Exception:
            return False

    def ingest(
        self,
        path: Path,
        store_id: str = "default-store",
    ) -> AdapterResult:
        """Parse sales data — returns standard AdapterResult.

        For transaction-level access, use ``ingest_sales()`` instead.
        """
        sales_result = self.ingest_sales(path, store_id=store_id)
        # Convert to standard AdapterResult (no inventory records, just metadata)
        return AdapterResult(
            source=sales_result.source,
            adapter_name=sales_result.adapter_name,
            errors=sales_result.errors,
            warnings=sales_result.warnings,
            files_processed=sales_result.files_processed,
            processing_time_ms=sales_result.processing_time_ms,
        )

    def ingest_sales(
        self,
        path: Path,
        store_id: str = "default-store",
    ) -> SalesAdapterResult:
        """Parse sales transactions from CSV.

        Returns:
            SalesAdapterResult with transaction-level data.
        """
        start = time.monotonic()
        path = Path(path)

        transactions: list[SalesTransaction] = []
        errors: list[str] = []
        warnings: list[str] = []
        files_processed = 0

        if path.is_dir():
            # Collect all sales CSVs
            targets: list[Path] = []
            for pat in [
                "Sales_Detail*.csv",
                "sales_detail*.csv",
                "transactions*.csv",
                "sales_export*.csv",
            ]:
                targets.extend(path.glob(pat))
            if not targets:
                # Fall back to any CSV with sales columns
                for csv_file in sorted(path.glob("*.csv")):
                    if self._check_file(csv_file):
                        targets.append(csv_file)
                        break

            for target in targets:
                result = self._parse_sales_csv(target, store_id)
                transactions.extend(result["transactions"])
                errors.extend(result["errors"])
                warnings.extend(result["warnings"])
                files_processed += 1

        elif path.is_file():
            result = self._parse_sales_csv(path, store_id)
            transactions = result["transactions"]
            errors = result["errors"]
            warnings = result["warnings"]
            files_processed = 1

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Compute date range
        dates = [t.sale_date for t in transactions]
        date_start = min(dates) if dates else None
        date_end = max(dates) if dates else None

        return SalesAdapterResult(
            source=str(path),
            transactions=transactions,
            errors=errors,
            warnings=warnings,
            files_processed=files_processed,
            processing_time_ms=elapsed_ms,
            date_range_start=date_start,
            date_range_end=date_end,
        )

    def _parse_sales_csv(
        self,
        path: Path,
        store_id: str,
    ) -> dict:
        """Parse a single sales CSV file."""
        transactions: list[SalesTransaction] = []
        errors: list[str] = []
        warnings: list[str] = []
        skipped = 0

        with open(path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return {
                    "transactions": [],
                    "errors": [f"No header row found in {path}"],
                    "warnings": [],
                }

            headers = list(reader.fieldnames)
            mapping = _detect_columns(headers)

            if mapping["sku"] is None:
                return {
                    "transactions": [],
                    "errors": [f"No SKU column found in {path}"],
                    "warnings": [],
                }
            if mapping["date"] is None:
                return {
                    "transactions": [],
                    "errors": [f"No date column found in {path}"],
                    "warnings": [],
                }

            sku_col = mapping["sku"]
            date_col = mapping["date"]
            qty_col = mapping["qty"]
            amount_col = mapping["amount"]
            cost_col = mapping["cost"]
            store_col = mapping["store"]

            for row_num, row in enumerate(reader, start=2):
                try:
                    sku = row.get(sku_col, "").strip()
                    if not sku:
                        skipped += 1
                        continue

                    sale_date = _parse_date(row.get(date_col, ""))
                    if sale_date is None:
                        skipped += 1
                        continue

                    qty_sold = _safe_float(row.get(qty_col, "")) if qty_col else 0.0
                    amount = _safe_float(row.get(amount_col, "")) if amount_col else 0.0
                    cost = _safe_float(row.get(cost_col, "")) if cost_col else 0.0
                    txn_store = (
                        row.get(store_col, "").strip() if store_col else store_id
                    )

                    transactions.append(
                        SalesTransaction(
                            sku=sku,
                            sale_date=sale_date,
                            qty_sold=qty_sold,
                            amount=amount,
                            unit_cost=cost,
                            store_id=txn_store or store_id,
                        )
                    )

                except Exception as e:
                    if len(errors) < 50:
                        errors.append(f"Row {row_num}: {e}")

        if skipped > 0:
            warnings.append(
                f"Skipped {skipped} rows with empty SKU or unparseable date"
            )

        logger.info(
            "Sales: parsed %d transactions from %s (skipped %d)",
            len(transactions),
            path.name,
            skipped,
        )

        return {"transactions": transactions, "errors": errors, "warnings": warnings}


# ---------------------------------------------------------------------------
# Sales Aggregation
# ---------------------------------------------------------------------------


def aggregate_sales_30d(
    transactions: list[SalesTransaction],
    reference_date: date | None = None,
    window_days: int = 30,
) -> dict[str, SalesAggregation]:
    """Aggregate transactions into a per-SKU 30-day rolling window.

    Args:
        transactions: List of sales transactions.
        reference_date: End date of the window (defaults to today).
        window_days: Rolling window size in days.

    Returns:
        Dict keyed by "{store_id}::{sku}" → SalesAggregation.
    """
    ref = reference_date or date.today()
    cutoff = ref - timedelta(days=window_days)

    # Group by (store_id, sku)
    buckets: dict[str, list[SalesTransaction]] = defaultdict(list)
    for txn in transactions:
        if cutoff <= txn.sale_date <= ref:
            key = f"{txn.store_id}::{txn.sku}"
            buckets[key].append(txn)

    result: dict[str, SalesAggregation] = {}
    for key, txns in buckets.items():
        store_id, sku = key.split("::", 1)
        dates = [t.sale_date for t in txns]
        result[key] = SalesAggregation(
            sku=sku,
            store_id=store_id,
            total_qty_sold=sum(t.qty_sold for t in txns),
            total_amount=sum(t.amount for t in txns),
            total_cost=sum(t.unit_cost * t.qty_sold for t in txns if t.unit_cost > 0),
            transaction_count=len(txns),
            first_sale=min(dates),
            last_sale=max(dates),
            days_in_window=window_days,
        )

    return result


# ---------------------------------------------------------------------------
# Sales Overlay — joins sales data onto inventory records
# ---------------------------------------------------------------------------


class SalesOverlay:
    """Overlays actual sales data onto NormalizedInventory records.

    Replaces the estimated ``sales_ytd / months_elapsed`` calculation
    in the bridge with actual 30-day sales figures from transaction data.

    Usage:
        overlay = SalesOverlay.from_transactions(transactions, ref_date)
        enriched = overlay.apply(inventory_records)
    """

    def __init__(
        self,
        aggregations: dict[str, SalesAggregation],
    ):
        """Initialize with pre-computed aggregations.

        Args:
            aggregations: Dict keyed by "{store_id}::{sku}" → SalesAggregation.
        """
        self._aggregations = aggregations
        # Also build bare-sku index for single-store convenience
        self._bare_sku_index: dict[str, SalesAggregation] = {}
        for key, agg in aggregations.items():
            if agg.sku not in self._bare_sku_index:
                self._bare_sku_index[agg.sku] = agg

    @classmethod
    def from_transactions(
        cls,
        transactions: list[SalesTransaction],
        reference_date: date | None = None,
        window_days: int = 30,
    ) -> SalesOverlay:
        """Create overlay from raw transactions.

        Args:
            transactions: Sales transactions from the adapter.
            reference_date: End of the 30-day window (defaults to today).
            window_days: Rolling window size.
        """
        aggregations = aggregate_sales_30d(
            transactions,
            reference_date=reference_date,
            window_days=window_days,
        )
        return cls(aggregations)

    def lookup(self, store_id: str, sku: str) -> SalesAggregation | None:
        """Look up sales aggregation for a specific store + SKU.

        Falls back to bare SKU if store-qualified key not found.
        """
        key = f"{store_id}::{sku}"
        if key in self._aggregations:
            return self._aggregations[key]
        return self._bare_sku_index.get(sku)

    def get_sales_last_30d(self, store_id: str, sku: str) -> float | None:
        """Get 30-day sales quantity for a specific SKU.

        Returns:
            Sales quantity in the 30-day window, or None if no data.
        """
        agg = self.lookup(store_id, sku)
        if agg is not None:
            return agg.total_qty_sold
        return None

    def apply(
        self,
        records: list[NormalizedInventory],
    ) -> list[NormalizedInventory]:
        """Apply sales overlay to inventory records.

        For each record, if actual 30-day sales data is available,
        updates the ``sales_last_30d`` field. Records without sales
        data in the window get ``sales_last_30d = 0.0`` (confirmed
        zero sales, not unknown).

        Returns:
            New list of NormalizedInventory with updated sales data.
            Original records are not modified.
        """
        enriched: list[NormalizedInventory] = []
        matched = 0
        unmatched = 0

        for rec in records:
            agg = self.lookup(rec.store_id, rec.sku_id)
            if agg is not None:
                # Create updated record with actual sales data
                updated = rec.model_copy(
                    update={
                        "sales_last_30d": agg.total_qty_sold,
                    }
                )
                enriched.append(updated)
                matched += 1
            else:
                # No sales in window — could mean zero sales or no data
                # Default: if we have sales data at all, SKUs not in it have 0 sales
                enriched.append(rec)
                unmatched += 1

        logger.info(
            "Sales overlay: %d matched, %d unmatched (of %d records)",
            matched,
            unmatched,
            len(records),
        )

        return enriched

    @property
    def coverage(self) -> dict[str, int]:
        """Statistics about the overlay data."""
        return {
            "total_skus": len(self._aggregations),
            "total_transactions": sum(
                a.transaction_count for a in self._aggregations.values()
            ),
        }
