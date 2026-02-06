"""
Compatibility layer for core module imports.

Provides fallback implementations when core.py is not available (e.g., in CI/CD).
This allows VSA evidence module to work independently for testing.
"""

from __future__ import annotations

from typing import Any

# Try to import from core module, otherwise use fallbacks
try:
    from ..core import (
        CATEGORY_ALIASES,
        COST_ALIASES,
        MARGIN_ALIASES,
        QTY_DIFF_ALIASES,
        QUANTITY_ALIASES,
        REVENUE_ALIASES,
        SKU_ALIASES,
        SOLD_ALIASES,
        _get_field,
        _safe_float,
    )

    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

    # Fallback aliases - common column names across POS systems
    SKU_ALIASES = frozenset(
        {
            "sku",
            "SKU",
            "item_number",
            "item_id",
            "upc",
            "UPC",
            "product_id",
            "ItemNumber",
            "Item Number",
        }
    )

    CATEGORY_ALIASES = frozenset(
        {
            "category",
            "Category",
            "dept",
            "department",
            "Department",
            "class",
            "Class",
            "product_category",
        }
    )

    QUANTITY_ALIASES = frozenset(
        {
            "quantity",
            "Quantity",
            "qty",
            "Qty",
            "QTY",
            "on_hand",
            "OnHand",
            "stock",
            "Stock",
            "inventory",
        }
    )

    COST_ALIASES = frozenset(
        {
            "cost",
            "Cost",
            "unit_cost",
            "UnitCost",
            "avg_cost",
            "AvgCost",
            "item_cost",
            "wholesale",
        }
    )

    REVENUE_ALIASES = frozenset(
        {
            "revenue",
            "Revenue",
            "retail",
            "Retail",
            "price",
            "Price",
            "sell_price",
            "SellPrice",
            "unit_price",
        }
    )

    SOLD_ALIASES = frozenset(
        {
            "sold",
            "Sold",
            "units_sold",
            "UnitsSold",
            "qty_sold",
            "QtySold",
            "sales_qty",
        }
    )

    QTY_DIFF_ALIASES = frozenset(
        {
            "qty_difference",
            "qty_diff",
            "QtyDiff",
            "variance",
            "Variance",
            "shrinkage",
            "Shrinkage",
            "adjustment",
        }
    )

    MARGIN_ALIASES = frozenset(
        {
            "margin",
            "Margin",
            "gross_margin",
            "GrossMargin",
            "margin_pct",
            "MarginPct",
            "profit_margin",
        }
    )

    def _get_field(
        row: dict[str, Any], aliases: frozenset[str], default: Any = None
    ) -> Any:
        """
        Get field value from row using multiple alias names.

        Args:
            row: Data row dictionary
            aliases: Set of possible column names
            default: Default value if not found

        Returns:
            Field value or default
        """
        for alias in aliases:
            if alias in row:
                return row[alias]
        return default

    def _safe_float(value: Any, default: float = 0.0) -> float:
        """
        Safely convert value to float.

        Args:
            value: Value to convert
            default: Default if conversion fails

        Returns:
            Float value or default
        """
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default


__all__ = [
    "_CORE_AVAILABLE",
    "CATEGORY_ALIASES",
    "COST_ALIASES",
    "MARGIN_ALIASES",
    "QTY_DIFF_ALIASES",
    "QUANTITY_ALIASES",
    "REVENUE_ALIASES",
    "SKU_ALIASES",
    "SOLD_ALIASES",
    "_get_field",
    "_safe_float",
]
