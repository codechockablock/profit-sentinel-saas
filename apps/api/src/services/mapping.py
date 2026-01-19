"""
Column Mapping Service - Universal POS System Support.

Provides intelligent column mapping using AI with aggressive heuristic fallback.
Supports 15+ major POS systems with fuzzy matching and synonym detection.
"""

import json
import logging
import re

import pandas as pd

from ..config import FIELD_IMPORTANCE, STANDARD_FIELDS
from ..dependencies import get_grok_client

logger = logging.getLogger(__name__)


class MappingService:
    """Service for intelligent column mapping across all major POS systems."""

    def __init__(self):
        """Initialize with precomputed lookup tables for fast matching."""
        # Build reverse lookup: normalized_alias -> standard_field
        self._alias_lookup: dict[str, str] = {}
        self._build_alias_lookup()

    def _build_alias_lookup(self):
        """Build normalized alias lookup table for O(1) matching."""
        for std_field, aliases in STANDARD_FIELDS.items():
            for alias in aliases:
                # Normalize: lowercase, strip, remove spaces/dots/underscores
                normalized = self._normalize_column(alias)
                if normalized not in self._alias_lookup:
                    self._alias_lookup[normalized] = std_field

    def _normalize_column(self, col: str) -> str:
        """
        Normalize column name for matching.

        Handles: case, spaces, dots, underscores, special chars.
        """
        if not col:
            return ""
        # Lowercase, strip whitespace
        norm = col.strip().lower()
        # Remove common separators and special chars
        norm = re.sub(r'[\s._\-#$%()]+', '', norm)
        return norm

    def suggest_mapping(self, df: pd.DataFrame, filename: str) -> dict:
        """
        Suggest column mappings for uploaded data.

        Uses Grok AI when available, falls back to aggressive heuristics.
        Prioritizes critical fields (rating 4-5) for leak detection.

        Args:
            df: DataFrame with sample data
            filename: Original filename

        Returns:
            Mapping suggestions with confidence scores and field importance
        """
        columns = list(df.columns)
        sample = df.head(10).to_dict(orient='records')

        grok_client = get_grok_client()
        if grok_client:
            suggestions = self._ai_mapping(grok_client, columns, sample, filename)
        else:
            suggestions = self._aggressive_heuristic_mapping(columns, sample)
            suggestions["notes"] = "AI unavailable - using aggressive heuristic matching"

        # Ensure confidence scores exist
        if "confidence" not in suggestions:
            suggestions["confidence"] = {
                col: 1.0 if suggestions["mapping"].get(col) else 0.0
                for col in columns
            }

        # Add field importance ratings to help frontend prioritize
        importance = {}
        for col, target in suggestions["mapping"].items():
            if target:
                importance[col] = FIELD_IMPORTANCE.get(target, 1)
            else:
                importance[col] = 0

        # Identify critical unmapped fields
        mapped_targets = set(v for v in suggestions["mapping"].values() if v)
        critical_missing = [
            f for f in ["quantity", "cost", "revenue", "sold", "sku"]
            if f not in mapped_targets
        ]

        return {
            "original_columns": columns,
            "sample_data": sample,
            "suggestions": suggestions["mapping"],
            "confidences": suggestions["confidence"],
            "importance": importance,
            "critical_missing": critical_missing,
            "notes": suggestions.get("notes", "")
        }

    def _ai_mapping(
        self,
        client,
        columns: list[str],
        sample: list[dict],
        filename: str
    ) -> dict:
        """Use Grok AI for intelligent column mapping with POS expertise."""
        prompt = f"""
You are Profit Sentinel's expert column mapper for retail POS/ERP data.
Your job: Map messy CSV columns to standard fields using BOTH column names AND sample values.

CRITICAL: This data must be mapped for profit leak detection. Prioritize these fields:
- quantity (QOH, stock on hand, inventory levels)
- cost (unit cost, COGS, purchase price)
- revenue (retail price, sell price, MSRP)
- sold (units sold, sales qty)
- sku (item ID, UPC, barcode)
- margin (profit %, GP%)
- qty_difference (variance, shrinkage)
- last_sale_date (last sold, last activity)

Uploaded file: {filename}
Columns: {columns}
Sample data (first 10 rows):
{json.dumps(sample[:5], indent=2)}

Standard fields to map to:
{json.dumps({k: v[:5] for k, v in STANDARD_FIELDS.items()}, indent=2)}

RULES:
1. Use sample VALUES to disambiguate ambiguous names (e.g., numbers -> quantity, $XX.XX -> cost/revenue)
2. Date patterns (MM/DD/YYYY) -> date fields
3. Negative numbers in qty-like columns -> qty_difference or return_flag
4. "Qty." with dots, "In Stock Qty." -> quantity
5. "Sug. Retail", "Sug Retail" -> revenue
6. "Last Pur.", "Last Sale" -> last_purchase_date or last_sale_date
7. Only map if confidence > 0.5
8. Use EXACT standard field names or null

Return ONLY valid JSON:
{{
  "mapping": {{"Column Name": "standard_field" or null}},
  "confidence": {{"Column Name": 0.0-1.0}},
  "notes": "Brief explanation"
}}
"""

        try:
            response = client.chat.completions.create(
                model="grok-3",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Lower temp for more consistent mapping
                max_tokens=2048
            )
            content = response.choices[0].message.content.strip()

            # Clean JSON from markdown code blocks
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()

            result = json.loads(content)
            result["notes"] = f"AI-powered mapping: {result.get('notes', '')}"
            return result

        except Exception as e:
            logger.warning(f"Grok mapping failed: {e}")
            result = self._aggressive_heuristic_mapping(columns, sample)
            result["notes"] = f"AI failed ({str(e)[:50]}) - heuristic fallback"
            return result

    def _aggressive_heuristic_mapping(
        self,
        columns: list[str],
        sample: list[dict] | None = None
    ) -> dict:
        """
        Aggressive heuristic mapping with multi-tier matching.

        Tiers:
        1. Exact normalized match against alias lookup
        2. Partial/substring match (for compound names like "In Stock Qty.")
        3. Sample value analysis (if provided)
        """
        mapping = {}
        confidence = {}
        used_targets = set()

        for col in columns:
            target, conf = self._match_column(col, sample, used_targets)
            mapping[col] = target
            confidence[col] = conf
            if target:
                used_targets.add(target)

        return {
            "mapping": mapping,
            "confidence": confidence,
            "notes": "Aggressive multi-tier heuristic matching"
        }

    def _match_column(
        self,
        col: str,
        sample: list[dict] | None,
        used_targets: set
    ) -> tuple[str | None, float]:
        """
        Match a single column using multi-tier strategy.

        Returns (target_field, confidence).
        """
        normalized = self._normalize_column(col)

        # Tier 1: Exact match in alias lookup
        if normalized in self._alias_lookup:
            target = self._alias_lookup[normalized]
            if target not in used_targets:
                return target, 0.95

        # Tier 2: Partial match - check if any alias is substring of column (or vice versa)
        best_match = None
        best_score = 0.0

        for alias, target in self._alias_lookup.items():
            if target in used_targets:
                continue

            # Check substring match both ways
            if len(alias) >= 3 and len(normalized) >= 3:
                if alias in normalized or normalized in alias:
                    score = min(len(alias), len(normalized)) / max(len(alias), len(normalized))
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
        self,
        col: str,
        sample: list[dict],
        used_targets: set
    ) -> str | None:
        """
        Infer field type from sample values.

        Heuristics:
        - Currency values ($XX.XX) -> cost or revenue (based on col name hints)
        - Date patterns -> date fields
        - Negative numbers -> qty_difference
        - Integer-ish values -> quantity or sold
        - Alphanumeric codes -> sku
        """
        if not sample:
            return None

        # Get sample values for this column
        values = [str(row.get(col, '')) for row in sample if row.get(col)]
        if not values:
            return None

        col_lower = col.lower()

        # Check for currency patterns
        currency_pattern = r'^\$?[\d,]+\.?\d*$'
        if all(re.match(currency_pattern, v.replace(',', '').replace('$', '')) for v in values[:5] if v):
            # Disambiguate cost vs revenue based on column name
            if any(hint in col_lower for hint in ['cost', 'cogs', 'buy', 'purchase', 'wholesale']):
                if 'cost' not in used_targets:
                    return 'cost'
            elif any(hint in col_lower for hint in ['retail', 'price', 'sell', 'msrp', 'sug']):
                if 'revenue' not in used_targets:
                    return 'revenue'

        # Check for date patterns
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2}',          # YYYY-MM-DD
            r'\d{1,2}-\d{1,2}-\d{2,4}',    # MM-DD-YYYY
        ]
        for pattern in date_patterns:
            if any(re.match(pattern, v) for v in values[:5] if v):
                if any(hint in col_lower for hint in ['sale', 'sold', 'last']):
                    if 'last_sale_date' not in used_targets:
                        return 'last_sale_date'
                elif any(hint in col_lower for hint in ['pur', 'recv', 'receiv']):
                    if 'last_purchase_date' not in used_targets:
                        return 'last_purchase_date'
                elif 'date' not in used_targets:
                    return 'date'

        # Check for negative numbers (shrinkage/variance)
        if any(v.startswith('-') for v in values[:5] if v):
            if 'qty_difference' not in used_targets:
                return 'qty_difference'

        # Check for pure integers (quantity candidates)
        try:
            int_values = [int(float(v.replace(',', ''))) for v in values[:5] if v]
            if int_values and all(v == int(v) for v in [float(x.replace(',', '')) for x in values[:5] if x]):
                if any(hint in col_lower for hint in ['qty', 'stock', 'hand', 'inv', 'avail']):
                    if 'quantity' not in used_targets:
                        return 'quantity'
                elif any(hint in col_lower for hint in ['sold', 'sale']):
                    if 'sold' not in used_targets:
                        return 'sold'
        except (ValueError, TypeError):
            pass

        # Check for alphanumeric codes (SKU candidates)
        if all(re.match(r'^[A-Za-z0-9\-_]+$', v) for v in values[:5] if v):
            if len(values[0]) > 3 and len(values[0]) < 30:
                if any(hint in col_lower for hint in ['sku', 'code', 'item', 'product', 'upc', 'barcode']):
                    if 'sku' not in used_targets:
                        return 'sku'

        return None

    def validate_mapping_completeness(self, mapping: dict[str, str]) -> dict:
        """
        Validate mapping has critical fields for leak detection.

        Returns warnings and suggestions for unmapped critical fields.
        """
        mapped = set(v for v in mapping.values() if v)

        critical_fields = {
            'quantity': "Required for stock level analysis",
            'cost': "Required for margin calculation",
            'revenue': "Required for margin calculation",
            'sku': "Required for item identification",
        }

        important_fields = {
            'sold': "Enables velocity/dead stock detection",
            'last_sale_date': "Enables dead item detection",
            'qty_difference': "Enables shrinkage detection",
            'margin': "Direct margin analysis (optional if cost+revenue mapped)",
        }

        errors = []
        warnings = []

        for field, reason in critical_fields.items():
            if field not in mapped:
                errors.append(f"Missing critical field '{field}': {reason}")

        for field, reason in important_fields.items():
            if field not in mapped:
                warnings.append(f"Consider mapping '{field}': {reason}")

        # Special case: margin can be calculated if cost and revenue are present
        if 'margin' not in mapped and 'cost' in mapped and 'revenue' in mapped:
            # Remove margin warning if we can calculate it
            warnings = [w for w in warnings if 'margin' not in w]

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "mapped_count": len(mapped),
            "critical_mapped": sum(1 for f in critical_fields if f in mapped),
            "critical_total": len(critical_fields)
        }
