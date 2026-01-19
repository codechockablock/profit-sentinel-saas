"""
Analysis Service - Aggressive Profit Leak Detection.

Runs VSA-based profit leak detection with 8 primitives, $ impact estimation,
and actionable recommendations. Supports data from any POS system.

CRITICAL: Uses request-scoped AnalysisContext to ensure isolation.
Each analyze() call creates a fresh context - no cross-request contamination.
"""

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _record_metrics(
    success: bool,
    rows: int,
    leaks: int,
    duration_ms: float,
    primitive_counts: dict[str, int],
):
    """Record analysis metrics (best-effort, won't fail analysis)."""
    try:
        from ..routes.metrics import record_analysis_metrics
        record_analysis_metrics(
            success=success,
            rows_processed=rows,
            leaks_detected=leaks,
            duration_ms=duration_ms,
            primitive_counts=primitive_counts,
        )
    except Exception as e:
        logger.debug(f"Failed to record metrics: {e}")


# Leak type display metadata
LEAK_DISPLAY = {
    "low_stock": {
        "title": "Low Stock Risk",
        "icon": "alert-triangle",
        "color": "#f59e0b",  # amber
        "priority": 1,
    },
    "high_margin_leak": {
        "title": "Margin Leak",
        "icon": "trending-down",
        "color": "#ef4444",  # red
        "priority": 2,
    },
    "dead_item": {
        "title": "Dead Inventory",
        "icon": "package-x",
        "color": "#6b7280",  # gray
        "priority": 4,
    },
    "negative_inventory": {
        "title": "Negative Inventory",
        "icon": "alert-circle",
        "color": "#dc2626",  # red-600
        "priority": 3,
    },
    "overstock": {
        "title": "Overstock",
        "icon": "boxes",
        "color": "#3b82f6",  # blue
        "priority": 5,
    },
    "price_discrepancy": {
        "title": "Price Discrepancy",
        "icon": "tag",
        "color": "#8b5cf6",  # violet
        "priority": 6,
    },
    "shrinkage_pattern": {
        "title": "Shrinkage Pattern",
        "icon": "shield-alert",
        "color": "#f97316",  # orange
        "priority": 7,
    },
    "margin_erosion": {
        "title": "Margin Erosion",
        "icon": "trending-down",
        "color": "#ec4899",  # pink
        "priority": 8,
    },
}


class AnalysisService:
    """Service for VSA-based profit leak analysis with 8 detection primitives."""

    # All 8 analysis primitives - ordered by typical impact severity
    PRIMITIVES = [
        "high_margin_leak",      # Critical - direct profit loss
        "negative_inventory",    # Critical - data integrity / theft
        "low_stock",             # High - lost sales
        "shrinkage_pattern",     # High - inventory loss
        "margin_erosion",        # High - profitability trend
        "dead_item",             # Medium - capital tied up
        "overstock",             # Medium - cash flow
        "price_discrepancy",     # Warning - pricing integrity
    ]

    def __init__(self):
        """Initialize analysis service with VSA engine."""
        try:
            from sentinel_engine import (
                LEAK_METADATA,
                bundle_pos_facts,
                get_all_primitives,
                get_primitive_metadata,
                query_bundle,
            )
            from sentinel_engine.context import create_analysis_context

            self._bundle_pos_facts = bundle_pos_facts
            self._query_bundle = query_bundle
            self._get_primitive_metadata = get_primitive_metadata
            self._get_all_primitives = get_all_primitives
            self._create_context = create_analysis_context
            self._leak_metadata = LEAK_METADATA
            self._engine_available = True
            logger.info("Sentinel engine loaded successfully (8 primitives, context-isolated)")
        except ImportError as e:
            logger.warning(f"Sentinel engine not available: {e}")
            self._engine_available = False
            self._leak_metadata = {}
            self._create_context = None

    def analyze(self, rows: list[dict]) -> dict:
        """
        Analyze POS data for profit leaks using all 8 primitives.

        CRITICAL: Creates a fresh AnalysisContext for each call.
        This ensures complete isolation between concurrent requests.

        Args:
            rows: List of row dictionaries from POS data

        Returns:
            Comprehensive leak analysis with:
            - top_items and scores per primitive
            - metadata (severity, category, recommendations)
            - summary statistics
        """
        if not self._engine_available:
            return self._mock_analysis()

        start_time = time.time()

        # Create fresh context for this request - CRITICAL for isolation
        ctx = self._create_context()
        logger.debug(f"Created analysis context: {ctx.get_summary()}")

        try:
            # Bundle facts with aggressive detection
            bundle_start = time.time()
            bundle = self._bundle_pos_facts(ctx, rows)
            logger.info(f"Bundled {len(rows)} rows in {time.time() - bundle_start:.2f}s")

            # Query each primitive
            leaks = {}
            total_items_flagged = 0
            critical_count = 0
            high_count = 0

            for primitive in self.PRIMITIVES:
                query_start = time.time()
                items, scores = self._query_bundle(ctx, bundle, primitive)

                # Filter to meaningful scores (> 0.1 similarity threshold)
                filtered_items = []
                filtered_scores = []
                for item, score in zip(items, scores):
                    if score > 0.1:  # Threshold for relevance
                        filtered_items.append(item)
                        filtered_scores.append(float(score))

                # Get metadata for this primitive
                metadata = self._leak_metadata.get(primitive, {})
                display = LEAK_DISPLAY.get(primitive, {})

                leaks[primitive] = {
                    "top_items": filtered_items[:20],
                    "scores": filtered_scores[:20],
                    "count": len(filtered_items),
                    "severity": metadata.get("severity", "info"),
                    "category": metadata.get("category", "Unknown"),
                    "recommendations": metadata.get("recommendations", []),
                    "title": display.get("title", primitive.replace("_", " ").title()),
                    "icon": display.get("icon", "alert"),
                    "color": display.get("color", "#6b7280"),
                    "priority": display.get("priority", 99),
                }

                # Track summary stats
                total_items_flagged += len(filtered_items)
                if metadata.get("severity") == "critical":
                    critical_count += len(filtered_items)
                elif metadata.get("severity") == "high":
                    high_count += len(filtered_items)

                elapsed = time.time() - query_start
                logger.info(f"Query {primitive}: {len(filtered_items)} items in {elapsed:.2f}s")

            # Calculate estimated $ impact (simplified - based on available data)
            estimated_impact = self._estimate_total_impact(rows, leaks)

            total_time = time.time() - start_time
            logger.info(f"Full analysis complete in {total_time:.2f}s")

            # Record metrics (best-effort)
            primitive_counts = {p: leaks[p]["count"] for p in leaks}
            _record_metrics(
                success=True,
                rows=len(rows),
                leaks=total_items_flagged,
                duration_ms=total_time * 1000,
                primitive_counts=primitive_counts,
            )

            return {
                "leaks": leaks,
                "summary": {
                    "total_rows_analyzed": len(rows),
                    "total_items_flagged": total_items_flagged,
                    "critical_issues": critical_count,
                    "high_issues": high_count,
                    "estimated_impact": estimated_impact,
                    "analysis_time_seconds": round(total_time, 2),
                },
                "primitives_used": self.PRIMITIVES,
            }
        finally:
            # Cleanup context (optional - GC handles it, but explicit is better)
            if ctx is not None:
                ctx.reset()
                logger.debug("Analysis context cleaned up")

    def _estimate_total_impact(self, rows: list[dict], leaks: dict) -> dict:
        """
        Estimate $ impact of detected leaks.

        This is a simplified estimation - actual impact requires
        deeper analysis with historical data.
        """
        impact = {
            "currency": "USD",
            "low_estimate": 0.0,
            "high_estimate": 0.0,
            "breakdown": {},
        }

        # Build lookup of row data by SKU for impact calculation
        row_lookup = {}
        for row in rows:
            sku = self._get_sku(row)
            if sku:
                row_lookup[sku.lower()] = row

        for primitive, data in leaks.items():
            primitive_impact = 0.0
            for item in data.get("top_items", [])[:10]:  # Top 10 for estimation
                row = row_lookup.get(item.lower())
                if row:
                    item_impact = self._calculate_item_impact(primitive, row)
                    primitive_impact += item_impact

            impact["breakdown"][primitive] = round(primitive_impact, 2)
            impact["low_estimate"] += primitive_impact * 0.7
            impact["high_estimate"] += primitive_impact * 1.3

        impact["low_estimate"] = round(impact["low_estimate"], 2)
        impact["high_estimate"] = round(impact["high_estimate"], 2)

        return impact

    def _calculate_item_impact(self, primitive: str, row: dict) -> float:
        """Calculate estimated $ impact for a single item."""
        cost = self._safe_float(row.get("cost", row.get("Cost", 0)))
        revenue = self._safe_float(row.get("revenue", row.get("Retail", row.get("retail", 0))))
        quantity = self._safe_float(row.get("quantity", row.get("Qty.", row.get("In Stock Qty.", 0))))
        sold = self._safe_float(row.get("sold", row.get("Sold", 0)))

        if primitive == "high_margin_leak":
            # Impact = margin shortfall * revenue
            if revenue > 0 and cost > 0:
                actual_margin = (revenue - cost) / revenue
                expected_margin = 0.30  # 30% expected
                if actual_margin < expected_margin:
                    return (expected_margin - actual_margin) * revenue * max(sold, 1)

        elif primitive == "negative_inventory":
            # Impact = lost inventory value
            return abs(quantity) * cost if quantity < 0 else 0

        elif primitive == "low_stock":
            # Impact = potential lost sales (assume 50% lost)
            if 0 < quantity < 10:
                margin_per_unit = (revenue - cost) if revenue > cost else revenue * 0.3
                return margin_per_unit * 5  # Assume 5 lost sales

        elif primitive == "dead_item":
            # Impact = capital tied up
            return quantity * cost if quantity > 0 else 0

        elif primitive == "overstock":
            # Impact = carrying cost (assume 20% annual, monthly = 1.67%)
            excess = max(0, quantity - 30)  # Assume 30 is optimal
            return excess * cost * 0.0167

        elif primitive == "shrinkage_pattern":
            # Impact = shrinkage value
            diff = self._safe_float(row.get("qty_difference", row.get("Qty. Difference", 0)))
            return abs(diff) * cost if diff < 0 else 0

        elif primitive == "margin_erosion":
            # Impact = margin shortfall relative to average
            if revenue > 0 and cost > 0:
                actual_margin = (revenue - cost) / revenue
                if actual_margin < 0.25:  # Below 25%
                    return (0.25 - actual_margin) * revenue

        elif primitive == "price_discrepancy":
            # Impact = revenue leakage
            sug_retail = self._safe_float(row.get("sug. retail", row.get("Sug. Retail", row.get("msrp", 0))))
            if sug_retail > 0 and revenue > 0 and revenue < sug_retail:
                return (sug_retail - revenue) * max(sold, 1)

        return 0.0

    def _get_sku(self, row: dict) -> str | None:
        """Extract SKU from row using common aliases."""
        sku_keys = ["sku", "SKU", "product_id", "item_id", "upc", "barcode", "item_no", "partnumber"]
        for key in sku_keys:
            if key in row and row[key]:
                return str(row[key])
        return None

    def _safe_float(self, val: Any) -> float:
        """Safely convert to float."""
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace("$", "").replace(",", "").strip())
        except (ValueError, TypeError):
            return 0.0

    def _mock_analysis(self) -> dict:
        """
        Return realistic mock analysis results when engine is unavailable.

        Useful for frontend development and testing.
        """
        logger.warning("Using mock analysis - sentinel engine not available")

        mock_items = [
            "SKU-001", "SKU-002", "SKU-003", "SKU-004", "SKU-005",
            "ITEM-A100", "ITEM-B200", "PROD-X1", "PROD-Y2", "PROD-Z3",
        ]
        mock_scores = [0.95, 0.87, 0.82, 0.78, 0.71, 0.65, 0.58, 0.52, 0.45, 0.38]

        leaks = {}
        for primitive in self.PRIMITIVES:
            display = LEAK_DISPLAY.get(primitive, {})
            metadata = {
                "severity": "high" if "margin" in primitive or "negative" in primitive else "medium",
                "category": primitive.replace("_", " ").title(),
                "recommendations": [
                    "Review flagged items",
                    "Verify data accuracy",
                    "Take corrective action",
                ]
            }

            leaks[primitive] = {
                "top_items": mock_items[:5],
                "scores": mock_scores[:5],
                "count": 5,
                "severity": metadata["severity"],
                "category": metadata["category"],
                "recommendations": metadata["recommendations"],
                "title": display.get("title", primitive.replace("_", " ").title()),
                "icon": display.get("icon", "alert"),
                "color": display.get("color", "#6b7280"),
                "priority": display.get("priority", 99),
            }

        return {
            "leaks": leaks,
            "summary": {
                "total_rows_analyzed": 0,
                "total_items_flagged": 40,
                "critical_issues": 10,
                "high_issues": 15,
                "estimated_impact": {
                    "currency": "USD",
                    "low_estimate": 5000.00,
                    "high_estimate": 15000.00,
                    "breakdown": {p: 1000.0 for p in self.PRIMITIVES[:5]},
                },
                "analysis_time_seconds": 0.01,
            },
            "primitives_used": self.PRIMITIVES,
            "mock": True,
        }

    def get_available_primitives(self) -> list[str]:
        """Return list of available analysis primitives."""
        return self.PRIMITIVES.copy()

    def get_primitive_info(self, primitive: str) -> dict | None:
        """Get detailed info about a specific primitive."""
        if primitive not in self.PRIMITIVES:
            return None

        display = LEAK_DISPLAY.get(primitive, {})
        metadata = self._leak_metadata.get(primitive, {}) if self._engine_available else {}

        return {
            "key": primitive,
            "title": display.get("title", primitive.replace("_", " ").title()),
            "severity": metadata.get("severity", "info"),
            "category": metadata.get("category", "Unknown"),
            "recommendations": metadata.get("recommendations", []),
            "icon": display.get("icon", "alert"),
            "color": display.get("color", "#6b7280"),
        }
