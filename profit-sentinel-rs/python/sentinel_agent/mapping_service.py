"""Column Mapping Service for the sidecar API.

Provides intelligent column mapping using Anthropic Claude with
aggressive heuristic fallback. Ported from _legacy/apps/api/src/services/mapping.py
with Claude replacing Grok for AI-powered mapping.
"""

from __future__ import annotations

import json
import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Standard fields and their aliases for heuristic matching
STANDARD_FIELDS: dict[str, list[str]] = {
    "sku": [
        "sku",
        "SKU",
        "item_id",
        "product_id",
        "upc",
        "UPC",
        "barcode",
        "Barcode",
        "partnumber",
        "item_no",
        "itemlookupcode",
        "handle",
        "variant_sku",
        "token",
        "clover_id",
        "ean",
        "stock_code",
        "item_code",
        "material_number",
    ],
    "quantity": [
        "quantity",
        "qty",
        "Qty.",
        "qoh",
        "on_hand",
        "onhand",
        "in_stock",
        "In Stock Qty.",
        "stock",
        "inventory",
        "qty_on_hand",
        "qty_avail",
        "net_qty",
        "available",
        "current_stock",
        "new_quantity",
        "qty_oh",
        "physical_qty",
        "book_qty",
        "bal",
        "balance",
    ],
    "cost": [
        "cost",
        "Cost",
        "cogs",
        "cost_price",
        "unit_cost",
        "avg_cost",
        "average_cost",
        "standard_cost",
        "lst_cost",
        "last_cost",
        "default_cost",
        "vendor_cost",
        "supply_price",
        "purchase_cost",
        "buy_price",
        "wholesale_cost",
        "landed_cost",
    ],
    "revenue": [
        "revenue",
        "retail",
        "Retail",
        "retail_price",
        "price",
        "sell_price",
        "selling_price",
        "sale_price",
        "unit_price",
        "msrp",
        "list_price",
        "sug. retail",
        "Sug. Retail",
        "suggested_retail",
        "regular_price",
        "default_price",
        "variant_price",
        "pos_price",
    ],
    "sold": [
        "sold",
        "Sold",
        "units_sold",
        "qty_sold",
        "quantity_sold",
        "sales_qty",
        "total_sold",
        "sold_qty",
        "sold_last_month",
        "sold_30_days",
        "sales_units",
        "unit_sales",
        "movement",
    ],
    "margin": [
        "margin",
        "Profit Margin %",
        "margin_pct",
        "profit_margin",
        "gp",
        "gross_profit",
        "gp_pct",
        "gp_percent",
        "profit_pct",
        "markup",
        "markup_pct",
    ],
    "description": [
        "description",
        "Description",
        "desc",
        "item_name",
        "product_name",
        "name",
        "title",
        "item_description",
    ],
    "category": [
        "category",
        "Category",
        "product_category",
        "dept",
        "Dpt.",
        "department",
        "Department",
        "class",
        "sub_category",
    ],
    "vendor": [
        "vendor",
        "Vendor",
        "supplier",
        "manufacturer",
        "brand",
        "vendor_name",
        "supplier_name",
    ],
    "sub_total": [
        "sub_total",
        "subtotal",
        "total",
        "ext_total",
        "inventory_value",
        "stock_value",
        "on_hand_value",
    ],
    "last_sale_date": [
        "last_sale_date",
        "Last Sale",
        "last_sold",
        "last_sale",
        "last_activity",
        "last_trans_date",
    ],
    "last_purchase_date": [
        "last_purchase_date",
        "Last Pur.",
        "last_pur",
        "Last Purchase",
        "last_receipt",
        "Last Received",
        "last_received_date",
    ],
    "qty_difference": [
        "qty_difference",
        "Qty. Difference",
        "difference",
        "variance",
        "qty_variance",
        "shrinkage",
        "adjustment",
        "discrepancy",
    ],
    "on_order_qty": [
        "on_order_qty",
        "on_order",
        "On Order",
        "qty_on_order",
        "purchase_order_qty",
        "po_qty",
    ],
    "reorder_point": [
        "reorder_point",
        "reorder_level",
        "min_qty",
        "minimum_qty",
        "safety_stock",
        "par_level",
    ],
    "store_id": [
        "store_id",
        "Store",
        "store",
        "Location",
        "location",
        "Branch",
        "branch",
        "Site",
        "site",
    ],
    "date": [
        "date",
        "Date",
        "transaction_date",
        "trans_date",
        "sale_date",
    ],
}

FIELD_IMPORTANCE: dict[str, int] = {
    "quantity": 5,
    "cost": 5,
    "revenue": 5,
    "sold": 5,
    "sku": 5,
    "margin": 4,
    "qty_difference": 4,
    "sub_total": 4,
    "last_sale_date": 4,
    "last_purchase_date": 4,
    "description": 3,
    "category": 3,
    "vendor": 3,
    "store_id": 3,
    "on_order_qty": 3,
    "reorder_point": 2,
    "date": 2,
}


class MappingService:
    """Service for intelligent column mapping across POS systems."""

    def __init__(self):
        self._alias_lookup: dict[str, str] = {}
        self._build_alias_lookup()

    def _build_alias_lookup(self):
        for std_field, aliases in STANDARD_FIELDS.items():
            for alias in aliases:
                normalized = self._normalize_column(alias)
                if normalized not in self._alias_lookup:
                    self._alias_lookup[normalized] = std_field

    def _normalize_column(self, col: str) -> str:
        if not col:
            return ""
        norm = col.strip().lower()
        norm = re.sub(r"[\s._\-#$%()]+", "", norm)
        return norm

    def suggest_mapping(
        self, df: pd.DataFrame, filename: str, anthropic_api_key: str = ""
    ) -> dict:
        """Suggest column mappings for uploaded data."""
        columns = list(df.columns)
        sample = df.head(10).to_dict(orient="records")

        if anthropic_api_key:
            suggestions = self._ai_mapping(anthropic_api_key, columns, sample, filename)
        else:
            suggestions = self._aggressive_heuristic_mapping(columns, sample)
            suggestions["notes"] = (
                "AI unavailable - using aggressive heuristic matching"
            )

        if "confidence" not in suggestions:
            suggestions["confidence"] = {
                col: 1.0 if suggestions["mapping"].get(col) else 0.0 for col in columns
            }

        importance = {}
        for col, target in suggestions["mapping"].items():
            if target:
                importance[col] = FIELD_IMPORTANCE.get(target, 1)
            else:
                importance[col] = 0

        mapped_targets = set(v for v in suggestions["mapping"].values() if v)
        critical_missing = [
            f
            for f in ["quantity", "cost", "revenue", "sold", "sku"]
            if f not in mapped_targets
        ]

        return {
            "original_columns": columns,
            "sample_data": sample,
            "suggestions": suggestions["mapping"],
            "confidences": suggestions["confidence"],
            "importance": importance,
            "critical_missing": critical_missing,
            "notes": suggestions.get("notes", ""),
        }

    def _ai_mapping(
        self,
        api_key: str,
        columns: list[str],
        sample: list[dict],
        filename: str,
    ) -> dict:
        """Use Anthropic Claude for intelligent column mapping."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)

            prompt = f"""You are Profit Sentinel's expert column mapper for retail POS/ERP data.
Map messy CSV columns to standard fields using BOTH column names AND sample values.

CRITICAL: This data must be mapped for profit leak detection. Prioritize these fields:
- quantity (QOH, stock on hand, inventory levels)
- cost (unit cost, COGS, purchase price)
- revenue (retail price, sell price, MSRP)
- sold (units sold, sales qty)
- sku (item ID, UPC, barcode)
- margin (profit %, GP%)

Uploaded file: {filename}
Columns: {columns}
Sample data (first 5 rows):
{json.dumps(sample[:5], indent=2)}

Standard fields to map to:
{json.dumps({k: v[:5] for k, v in STANDARD_FIELDS.items()}, indent=2)}

RULES:
1. Use sample VALUES to disambiguate ambiguous names
2. Date patterns (MM/DD/YYYY) -> date fields
3. "Qty." with dots -> quantity
4. "Sug. Retail" -> revenue
5. Only map if confidence > 0.5
6. Use EXACT standard field names or null

Return ONLY valid JSON:
{{"mapping": {{"Column Name": "standard_field" or null}}, "confidence": {{"Column Name": 0.0-1.0}}, "notes": "Brief explanation"}}"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()

            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()

            result = json.loads(content)
            result["notes"] = f"AI-powered mapping (Claude): {result.get('notes', '')}"
            return result

        except Exception as e:
            logger.warning(f"Claude mapping failed: {e}")
            result = self._aggressive_heuristic_mapping(columns, sample)
            result["notes"] = f"AI failed ({str(e)[:50]}) - heuristic fallback"
            return result

    def _aggressive_heuristic_mapping(
        self, columns: list[str], sample: list[dict] | None = None
    ) -> dict:
        """Aggressive heuristic mapping with multi-tier matching."""
        mapping = {}
        confidence = {}
        used_targets: set[str] = set()

        for col in columns:
            target, conf = self._match_column(col, sample, used_targets)
            mapping[col] = target
            confidence[col] = conf
            if target:
                used_targets.add(target)

        return {
            "mapping": mapping,
            "confidence": confidence,
            "notes": "Aggressive multi-tier heuristic matching",
        }

    def _match_column(
        self, col: str, sample: list[dict] | None, used_targets: set[str]
    ) -> tuple[str | None, float]:
        """Match a single column using multi-tier strategy."""
        normalized = self._normalize_column(col)

        # Tier 1: Exact match in alias lookup
        if normalized in self._alias_lookup:
            target = self._alias_lookup[normalized]
            if target not in used_targets:
                return target, 0.95

        # Tier 2: Partial/substring match
        best_match = None
        best_score = 0.0

        for alias, target in self._alias_lookup.items():
            if target in used_targets:
                continue
            if len(alias) >= 3 and len(normalized) >= 3:
                if alias in normalized or normalized in alias:
                    score = min(len(alias), len(normalized)) / max(
                        len(alias), len(normalized)
                    )
                    if score > best_score:
                        best_score = score
                        best_match = target

        if best_match and best_score > 0.5:
            return best_match, min(best_score, 0.85)

        # Tier 3: Sample value analysis
        if sample:
            inferred = self._infer_from_sample(col, sample, used_targets)
            if inferred:
                return inferred, 0.6

        return None, 0.0

    def _infer_from_sample(
        self, col: str, sample: list[dict], used_targets: set[str]
    ) -> str | None:
        """Infer field type from sample values."""
        values = [str(row.get(col, "")) for row in sample if row.get(col)]
        if not values:
            return None

        col_lower = col.lower()

        # Currency patterns
        currency_pattern = r"^\$?[\d,]+\.?\d*$"
        if all(
            re.match(currency_pattern, v.replace(",", "").replace("$", ""))
            for v in values[:5]
            if v
        ):
            if any(
                hint in col_lower
                for hint in ["cost", "cogs", "buy", "purchase", "wholesale"]
            ):
                if "cost" not in used_targets:
                    return "cost"
            elif any(
                hint in col_lower for hint in ["retail", "price", "sell", "msrp", "sug"]
            ):
                if "revenue" not in used_targets:
                    return "revenue"

        # Date patterns
        date_patterns = [
            r"\d{1,2}/\d{1,2}/\d{2,4}",
            r"\d{4}-\d{2}-\d{2}",
        ]
        for pattern in date_patterns:
            if any(re.match(pattern, v) for v in values[:5] if v):
                if any(hint in col_lower for hint in ["sale", "sold", "last"]):
                    if "last_sale_date" not in used_targets:
                        return "last_sale_date"
                elif any(hint in col_lower for hint in ["pur", "recv", "receiv"]):
                    if "last_purchase_date" not in used_targets:
                        return "last_purchase_date"

        return None
