"""Column Mapping Adapter — Transform any POS DataFrame → Rust pipeline CSV.

This is the critical bridge (M1) that enables the Rust analysis pipeline
to consume data from ANY POS system. It sits between the existing column
mapping step (which renames user columns to normalized names like "sku",
"quantity", "cost", "revenue") and the Rust binary (which expects a strict
11-column CSV matching InventoryRecord).

Data flow:
    User CSV  →  AI mapping  →  df.rename()  →  THIS ADAPTER  →  Rust CSV
    (29 cols)    (Grok)         (normalized)     (11 cols)        (sentinel-server)

The adapter handles:
- Field renaming: Python normalized names → Rust field names
- Derived fields: margin_pct from (revenue - cost) / revenue
- Date computation: days_since_receipt from date fields
- Default values: store_id="default", is_damaged=false, etc.
- Type coercion: everything to str for CSV output
- Seasonal detection: heuristic from category/department names

Usage:
    from services.column_adapter import ColumnAdapter

    adapter = ColumnAdapter()
    csv_path = adapter.to_rust_csv(df)
    # → temp file ready for sentinel-server subprocess
    # adapter.cleanup() when done
"""

from __future__ import annotations

import csv
import logging
import tempfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Rust InventoryRecord CSV column order (must match exactly)
RUST_COLUMNS = [
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

# ---------------------------------------------------------------------------
# Python normalized field → Rust field mapping
# Left: what df.rename() produces    Right: what Rust expects
# ---------------------------------------------------------------------------
FIELD_MAP: dict[str, str] = {
    "sku": "sku",
    "quantity": "qty_on_hand",
    "cost": "unit_cost",
    "revenue": "retail_price",
    "sold": "sales_last_30d",
    "margin": "margin_pct",
}

# Additional aliases that might appear after rename or from original columns
SKU_ALIASES = [
    "sku",
    "SKU",
    "barcode",
    "Barcode",
    "product_id",
    "item_id",
    "upc",
    "UPC",
    "item_no",
    "partnumber",
]

QUANTITY_ALIASES = [
    "quantity",
    "qty",
    "Qty.",
    "In Stock Qty.",
    "qoh",
    "on_hand",
    "stock",
    "qty_on_hand",
]

COST_ALIASES = [
    "cost",
    "Cost",
    "unit_cost",
    "avg_cost",
    "cogs",
]

REVENUE_ALIASES = [
    "revenue",
    "Retail",
    "retail",
    "retail_price",
    "price",
    "sell_price",
    "msrp",
]

SOLD_ALIASES = [
    "sold",
    "Sold",
    "units_sold",
    "qty_sold",
    "sales_qty",
    "movement",
    "sales_last_30d",
]

MARGIN_ALIASES = [
    "margin",
    "Profit Margin %",
    "margin_pct",
    "gp_pct",
    "profit_margin",
]

STORE_ALIASES = [
    "store_id",
    "Store",
    "store",
    "Location",
    "location",
    "Branch",
    "branch",
    "Site",
    "site",
]

ON_ORDER_ALIASES = [
    "on_order_qty",
    "on_order",
    "On Order",
    "qty_on_order",
    "purchase_order_qty",
    "po_qty",
]

DAMAGED_ALIASES = [
    "is_damaged",
    "damaged",
    "Damaged",
    "damage_flag",
]

SEASONAL_ALIASES = [
    "is_seasonal",
    "seasonal",
    "Seasonal",
]

CATEGORY_ALIASES = [
    "category",
    "Category",
    "product_category",
    "dept",
    "Dpt.",
    "department",
    "Department",
]

# Date fields for computing days_since_receipt
DATE_ALIASES = [
    "last_purchase_date",
    "Last Pur.",
    "last_pur",
    "Last Purchase",
    "last_receipt",
    "Last Received",
    "last_received_date",
    "Real Perpetual Inventory Date",
    "Last Perpetual Inventory Date",
]

# Keywords that indicate a seasonal product category
SEASONAL_KEYWORDS = frozenset(
    {
        "seasonal",
        "christmas",
        "holiday",
        "halloween",
        "easter",
        "spring",
        "summer",
        "winter",
        "fall",
        "outdoor living",
        "lawn & garden",
        "lawn and garden",
        "snow",
        "ice melt",
        "pool",
        "patio",
        "garden",
        "fireworks",
        "valentine",
    }
)


def _find_column(df_columns: list[str], aliases: list[str]) -> str | None:
    """Find the first matching column name from a list of aliases."""
    col_set = set(df_columns)
    for alias in aliases:
        if alias in col_set:
            return alias
    # Case-insensitive fallback
    col_lower = {c.lower().strip(): c for c in df_columns}
    for alias in aliases:
        if alias.lower().strip() in col_lower:
            return col_lower[alias.lower().strip()]
    return None


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    if isinstance(val, int | float):
        return float(val)
    try:
        cleaned = str(val).replace("$", "").replace(",", "").replace("%", "").strip()
        if cleaned == "" or cleaned == "-":
            return default
        return float(cleaned)
    except (ValueError, TypeError):
        return default


def _parse_date(val) -> date | None:
    """Try to parse a date from various formats."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    # Check Pandas Timestamp first (before datetime/date, since Timestamp is a subclass)
    if isinstance(val, pd.Timestamp):
        return val.date() if not pd.isna(val) else None
    # datetime before date (datetime is a subclass of date)
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    s = str(val).strip()
    if not s or s.lower() in ("", "nan", "nat", "none", "-"):
        return None
    # Try common date formats
    for fmt in (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y/%m/%d",
        "%d-%m-%Y",
    ):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # Try pandas timestamp parsing as last resort
    try:
        ts = pd.Timestamp(s)
        if not pd.isna(ts):
            return ts.date()
    except Exception:
        pass
    return None


def _detect_seasonal_from_category(category_val) -> bool:
    """Detect seasonal items from category/department names."""
    if category_val is None or (
        isinstance(category_val, float) and pd.isna(category_val)
    ):
        return False
    text = str(category_val).lower().strip()
    if not text:
        return False
    for kw in SEASONAL_KEYWORDS:
        if kw in text:
            return True
    return False


class ColumnAdapter:
    """Transforms a post-mapping pandas DataFrame into Rust pipeline CSV.

    The adapter is stateless per invocation — call to_rust_csv() to get
    a temp file path, then pass that to sentinel-server. Call cleanup()
    when done to delete the temp file.

    Example:
        adapter = ColumnAdapter()
        csv_path = adapter.to_rust_csv(df)
        result = subprocess.run(["sentinel-server", csv_path, "--json"])
        adapter.cleanup()
    """

    def __init__(
        self,
        reference_date: date | None = None,
        default_store_id: str = "default",
        default_days_since_receipt: float = 30.0,
    ):
        self.reference_date = reference_date or date.today()
        self.default_store_id = default_store_id
        self.default_days_since_receipt = default_days_since_receipt
        self._temp_path: Path | None = None

    def to_rust_csv(
        self,
        df: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """Convert a post-mapping DataFrame to Rust pipeline CSV.

        Args:
            df: DataFrame with columns renamed by the mapping step.
                Expected to have at least 'sku' (or a SKU alias).
            output_path: Where to write CSV. If None, uses a temp file.

        Returns:
            Path to the written CSV file.

        Raises:
            ValueError: If no SKU column can be found.
        """
        columns = df.columns.tolist()

        # Resolve column names for each Rust field
        sku_col = _find_column(columns, SKU_ALIASES)
        qty_col = _find_column(columns, QUANTITY_ALIASES)
        cost_col = _find_column(columns, COST_ALIASES)
        revenue_col = _find_column(columns, REVENUE_ALIASES)
        sold_col = _find_column(columns, SOLD_ALIASES)
        margin_col = _find_column(columns, MARGIN_ALIASES)
        store_col = _find_column(columns, STORE_ALIASES)
        on_order_col = _find_column(columns, ON_ORDER_ALIASES)
        damaged_col = _find_column(columns, DAMAGED_ALIASES)
        seasonal_col = _find_column(columns, SEASONAL_ALIASES)
        category_col = _find_column(columns, CATEGORY_ALIASES)
        date_col = _find_column(columns, DATE_ALIASES)

        if sku_col is None:
            raise ValueError(
                "No SKU column found in DataFrame. "
                f"Available columns: {columns}. "
                f"Expected one of: {SKU_ALIASES}"
            )

        logger.info(
            "Column adapter resolved: sku=%s, qty=%s, cost=%s, revenue=%s, "
            "sold=%s, margin=%s, store=%s, date=%s, category=%s",
            sku_col,
            qty_col,
            cost_col,
            revenue_col,
            sold_col,
            margin_col,
            store_col,
            date_col,
            category_col,
        )

        # Prepare output path
        if output_path is None:
            fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="sentinel_adapter_")
            import os

            os.close(fd)
            output_path = Path(tmp)
            self._temp_path = output_path
        else:
            output_path = Path(output_path)

        converted = 0
        skipped = 0

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RUST_COLUMNS)
            writer.writeheader()

            for _, row in df.iterrows():
                try:
                    rust_row = self._convert_row(
                        row,
                        sku_col=sku_col,
                        qty_col=qty_col,
                        cost_col=cost_col,
                        revenue_col=revenue_col,
                        sold_col=sold_col,
                        margin_col=margin_col,
                        store_col=store_col,
                        on_order_col=on_order_col,
                        damaged_col=damaged_col,
                        seasonal_col=seasonal_col,
                        category_col=category_col,
                        date_col=date_col,
                    )
                    if rust_row is not None:
                        writer.writerow(rust_row)
                        converted += 1
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1
                    if skipped <= 10:
                        logger.warning("Skipped row during adapter conversion: %s", e)

        logger.info(
            "Column adapter: %d rows converted, %d skipped → %s",
            converted,
            skipped,
            output_path,
        )

        return output_path

    def _convert_row(
        self,
        row: pd.Series,
        *,
        sku_col: str,
        qty_col: str | None,
        cost_col: str | None,
        revenue_col: str | None,
        sold_col: str | None,
        margin_col: str | None,
        store_col: str | None,
        on_order_col: str | None,
        damaged_col: str | None,
        seasonal_col: str | None,
        category_col: str | None,
        date_col: str | None,
    ) -> dict[str, str] | None:
        """Convert a single DataFrame row to a Rust CSV row dict.

        Returns None if the row should be skipped (e.g., no SKU).
        """
        # SKU (required)
        sku = str(row[sku_col]).strip() if pd.notna(row[sku_col]) else ""
        if not sku or sku.lower() in ("nan", "none", ""):
            return None

        # Numeric fields (with defaults)
        qty_on_hand = _safe_float(row.get(qty_col) if qty_col else None, 0.0)
        unit_cost = _safe_float(row.get(cost_col) if cost_col else None, 0.0)
        retail_price = _safe_float(row.get(revenue_col) if revenue_col else None, 0.0)
        sales_last_30d = _safe_float(row.get(sold_col) if sold_col else None, 0.0)
        on_order_qty = _safe_float(row.get(on_order_col) if on_order_col else None, 0.0)

        # Margin: prefer direct column, compute from cost/revenue if missing
        if margin_col is not None:
            raw_margin = _safe_float(row.get(margin_col), None)
            if raw_margin is not None:
                # Normalize: if > 1, it's a percentage (e.g., 35 → 0.35)
                margin_pct = raw_margin / 100.0 if raw_margin > 1.0 else raw_margin
            elif retail_price > 0 and unit_cost >= 0:
                margin_pct = (retail_price - unit_cost) / retail_price
            else:
                margin_pct = 0.0
        elif retail_price > 0 and unit_cost >= 0:
            margin_pct = (retail_price - unit_cost) / retail_price
        else:
            margin_pct = 0.0

        # Store ID
        if store_col is not None and pd.notna(row.get(store_col)):
            store_id = str(row[store_col]).strip()
            if not store_id or store_id.lower() in ("nan", "none", ""):
                store_id = self.default_store_id
        else:
            store_id = self.default_store_id

        # Days since receipt: compute from date column
        if date_col is not None:
            parsed_date = _parse_date(row.get(date_col))
            if parsed_date is not None:
                delta = (self.reference_date - parsed_date).days
                days_since_receipt = max(0.0, float(delta))
            else:
                days_since_receipt = self.default_days_since_receipt
        else:
            days_since_receipt = self.default_days_since_receipt

        # is_damaged: check column or default to false
        if damaged_col is not None and pd.notna(row.get(damaged_col)):
            val = str(row[damaged_col]).lower().strip()
            is_damaged = val in ("true", "1", "yes", "y")
        else:
            is_damaged = False

        # is_seasonal: check column, then heuristic from category
        if seasonal_col is not None and pd.notna(row.get(seasonal_col)):
            val = str(row[seasonal_col]).lower().strip()
            is_seasonal = val in ("true", "1", "yes", "y")
        elif category_col is not None:
            is_seasonal = _detect_seasonal_from_category(row.get(category_col))
        else:
            is_seasonal = False

        return {
            "store_id": store_id,
            "sku": sku,
            "qty_on_hand": str(float(qty_on_hand)),
            "unit_cost": f"{unit_cost:.2f}",
            "margin_pct": f"{margin_pct:.4f}",
            "sales_last_30d": f"{sales_last_30d:.2f}",
            "days_since_receipt": f"{days_since_receipt:.0f}",
            "retail_price": f"{retail_price:.2f}",
            "is_damaged": str(is_damaged).lower(),
            "on_order_qty": str(float(on_order_qty)),
            "is_seasonal": str(is_seasonal).lower(),
        }

    def cleanup(self) -> None:
        """Delete the temp CSV file if one was created."""
        if self._temp_path and self._temp_path.exists():
            try:
                self._temp_path.unlink()
                logger.debug("Cleaned up temp file: %s", self._temp_path)
            except OSError as e:
                logger.warning(
                    "Failed to clean up temp file %s: %s", self._temp_path, e
                )
            self._temp_path = None

    def get_column_mapping_report(self, df: pd.DataFrame) -> dict:
        """Generate a report of how DataFrame columns map to Rust fields.

        Useful for debugging and UI display.

        Returns:
            Dict with resolved mappings, defaults used, and warnings.
        """
        columns = df.columns.tolist()
        report = {
            "resolved": {},
            "defaults": [],
            "warnings": [],
        }

        mappings = [
            ("sku", SKU_ALIASES),
            ("qty_on_hand", QUANTITY_ALIASES),
            ("unit_cost", COST_ALIASES),
            ("retail_price", REVENUE_ALIASES),
            ("sales_last_30d", SOLD_ALIASES),
            ("margin_pct", MARGIN_ALIASES),
            ("store_id", STORE_ALIASES),
            ("on_order_qty", ON_ORDER_ALIASES),
            ("is_damaged", DAMAGED_ALIASES),
            ("is_seasonal", SEASONAL_ALIASES),
            ("days_since_receipt", DATE_ALIASES),
        ]

        for rust_field, aliases in mappings:
            found = _find_column(columns, aliases)
            if found:
                report["resolved"][rust_field] = found
            else:
                default_val = {
                    "sku": "REQUIRED",
                    "qty_on_hand": "0.0",
                    "unit_cost": "0.0",
                    "retail_price": "0.0",
                    "sales_last_30d": "0.0",
                    "margin_pct": "computed or 0.0",
                    "store_id": self.default_store_id,
                    "on_order_qty": "0.0",
                    "is_damaged": "false",
                    "is_seasonal": "heuristic from category",
                    "days_since_receipt": str(self.default_days_since_receipt),
                }.get(rust_field, "unknown")
                report["defaults"].append(
                    {
                        "rust_field": rust_field,
                        "default_value": default_val,
                    }
                )
                if rust_field == "sku":
                    report["warnings"].append("No SKU column found — adapter will fail")

        return report
