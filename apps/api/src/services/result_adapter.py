"""Result Adapter — Transform Rust sentinel-server JSON → Legacy API response.

This is M2 of the production migration. It takes the JSON output from the
Rust sentinel-server binary (DigestJson) and transforms it into the exact
same shape that AnalysisService.analyze() returns, so the React frontend
sees zero difference.

The mapping is non-trivial because the two systems have fundamentally
different output structures:

    Rust: Issues are grouped per store, each containing multiple SKUs.
          issue_type uses Rust enum names: "NegativeInventory", "MarginErosion"

    Python: Leaks are grouped by primitive, each with top_items/scores/item_details.
            primitive names use snake_case: "negative_inventory", "margin_erosion"

This adapter bridges the gap with:
- Issue type name mapping (Rust enum → Python primitive key)
- SKU detail flattening (Rust SkuJson → Python ItemDetail)
- Context generation (same logic as _get_issue_context)
- Summary statistics aggregation
- Impact estimation from Rust dollar_impact values
- Severity/display metadata injection from LEAK_DISPLAY

Usage:
    from services.result_adapter import RustResultAdapter

    adapter = RustResultAdapter()
    legacy_response = adapter.transform(
        digest=rust_json,
        total_rows=36452,
        analysis_time=2.5,
        original_rows=rows,  # For enrichment (description, sub_total)
    )
    # legacy_response has exact same shape as AnalysisService.analyze()
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rust issue_type → Python primitive key
# ---------------------------------------------------------------------------
RUST_TYPE_TO_PRIMITIVE: dict[str, str] = {
    "NegativeInventory": "negative_inventory",
    "DeadStock": "dead_item",
    "MarginErosion": "margin_erosion",
    "ReceivingGap": "high_margin_leak",  # Closest mapping: receiving gap → margin leak
    "VendorShortShip": "shrinkage_pattern",  # Vendor short-ship → shrinkage
    "PurchasingLeakage": "high_margin_leak",
    "PatronageMiss": "overstock",  # Patronage miss → overstock
    "ShrinkagePattern": "shrinkage_pattern",
    "ZeroCostAnomaly": "zero_cost_anomaly",
    "PriceDiscrepancy": "price_discrepancy",
    "Overstock": "overstock",
}

# Reverse: Python primitive → which Rust types map to it
PRIMITIVE_RUST_TYPES: dict[str, list[str]] = {}
for _rt, _prim in RUST_TYPE_TO_PRIMITIVE.items():
    PRIMITIVE_RUST_TYPES.setdefault(_prim, []).append(_rt)

# Display metadata — matches LEAK_DISPLAY from analysis.py exactly
LEAK_DISPLAY: dict[str, dict[str, Any]] = {
    "low_stock": {
        "title": "Low Stock Risk",
        "icon": "alert-triangle",
        "color": "#f59e0b",
        "priority": 1,
        "severity": "high",
        "category": "Inventory Risk",
    },
    "high_margin_leak": {
        "title": "Margin Leak",
        "icon": "trending-down",
        "color": "#ef4444",
        "priority": 2,
        "severity": "critical",
        "category": "Profitability",
    },
    "negative_inventory": {
        "title": "Negative Inventory",
        "icon": "alert-circle",
        "color": "#dc2626",
        "priority": 3,
        "severity": "critical",
        "category": "Data Integrity",
    },
    "dead_item": {
        "title": "Dead Inventory",
        "icon": "package-x",
        "color": "#6b7280",
        "priority": 4,
        "severity": "medium",
        "category": "Cash Flow",
    },
    "overstock": {
        "title": "Overstock",
        "icon": "boxes",
        "color": "#3b82f6",
        "priority": 5,
        "severity": "medium",
        "category": "Cash Flow",
    },
    "price_discrepancy": {
        "title": "Price Discrepancy",
        "icon": "tag",
        "color": "#8b5cf6",
        "priority": 6,
        "severity": "low",
        "category": "Pricing Integrity",
    },
    "shrinkage_pattern": {
        "title": "Shrinkage Pattern",
        "icon": "shield-alert",
        "color": "#f97316",
        "priority": 7,
        "severity": "high",
        "category": "Loss Prevention",
    },
    "margin_erosion": {
        "title": "Margin Erosion",
        "icon": "trending-down",
        "color": "#ec4899",
        "priority": 8,
        "severity": "high",
        "category": "Profitability",
    },
    "zero_cost_anomaly": {
        "title": "Zero Cost Anomaly",
        "icon": "alert-triangle",
        "color": "#eab308",
        "priority": 9,
        "severity": "high",
        "category": "Data Quality",
    },
    "negative_profit": {
        "title": "Negative Profit",
        "icon": "alert-circle",
        "color": "#dc2626",
        "priority": 10,
        "severity": "critical",
        "category": "Profitability",
    },
    "severe_inventory_deficit": {
        "title": "Severe Inventory Deficit",
        "icon": "alert-circle",
        "color": "#d946ef",
        "priority": 11,
        "severity": "critical",
        "category": "Inventory Risk",
    },
}

# Recommendations per primitive
RECOMMENDATIONS: dict[str, list[str]] = {
    "high_margin_leak": [
        "Review and update retail prices for flagged items",
        "Check if vendor costs have increased recently",
        "Verify no unauthorized discounts are being applied",
        "Consider bundling low-margin items with high-margin ones",
    ],
    "negative_inventory": [
        "Immediately investigate each negative SKU",
        "Check if receiving was skipped for recent shipments",
        "Review POS transaction logs for voids/returns",
        "Consider physical count to verify actual stock",
        "Train staff on proper receiving procedures",
    ],
    "low_stock": [
        "Place emergency orders for critical items",
        "Set up automatic reorder points in your POS",
        "Review lead times with vendors",
        "Consider safety stock for top sellers",
    ],
    "dead_item": [
        "Run clearance sale on dead items",
        "Return to vendor if possible",
        "Bundle with popular items",
        "Donate for tax write-off",
        "Don't reorder - let it sell through",
    ],
    "overstock": [
        "Slow down or pause reorders",
        "Run promotion to move excess",
        "Negotiate return to vendor",
        "Transfer to other locations if applicable",
    ],
    "price_discrepancy": [
        "Audit POS price file against vendor suggested retail",
        "Check for expired promotional pricing",
        "Verify shelf tags match system prices",
        "Review price override patterns",
    ],
    "shrinkage_pattern": [
        "Conduct cycle counts on flagged items",
        "Review security footage for high-value items",
        "Check vendor deliveries match invoices",
        "Implement better receiving verification",
        "Consider locked display for theft-prone items",
    ],
    "margin_erosion": [
        "Compare current vs historical margins",
        "Negotiate better vendor pricing",
        "Review discount patterns and promotions",
        "Consider discontinuing chronically low-margin items",
    ],
    "zero_cost_anomaly": [
        "Update cost data from recent vendor invoices",
        "Check if items were received without cost entry",
        "Review vendor price lists for current costs",
        "Set up automatic cost updates from purchasing",
    ],
    "negative_profit": [
        "Immediately raise prices or stop selling",
        "Verify cost and price data are correct",
        "Check for pricing errors or unauthorized discounts",
        "Consider removing from active inventory",
    ],
    "severe_inventory_deficit": [
        "Emergency reorder immediately",
        "Check if stock is misplaced in store/warehouse",
        "Contact vendor for expedited shipping",
        "Consider sourcing from alternative suppliers",
    ],
}

# All 11 primitives in standard order
ALL_PRIMITIVES = [
    "high_margin_leak",
    "negative_inventory",
    "negative_profit",
    "severe_inventory_deficit",
    "low_stock",
    "shrinkage_pattern",
    "margin_erosion",
    "zero_cost_anomaly",
    "dead_item",
    "overstock",
    "price_discrepancy",
]

# Impact caps (match analysis.py)
MAX_IMPACTABLE_UNITS = 100
MAX_PER_ITEM_IMPACT = 1000
MAX_DEAD_ITEM_IMPACT = 5000
MAX_OVERSTOCK_IMPACT = 2000
MAX_SHRINKAGE_IMPACT = 2000
MAX_MARGIN_IMPACT = 5000


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _get_issue_context(primitive: str, item: dict) -> str:
    """Generate human-readable context. Mirrors analysis.py._get_issue_context."""
    qty = _safe_float(item.get("quantity", 0))
    cost = _safe_float(item.get("cost", 0))
    revenue = _safe_float(item.get("revenue", 0))
    sold = _safe_float(item.get("sold", 0))
    margin = _safe_float(item.get("margin", 0))
    sub_total = _safe_float(item.get("sub_total", 0))

    if primitive == "high_margin_leak":
        if revenue > 0 and cost > 0:
            actual_margin = (revenue - cost) / revenue * 100
            return (
                f"Margin is only {actual_margin:.1f}% (Cost: ${cost:.2f}, "
                f"Retail: ${revenue:.2f}). QOH: {qty:.0f}, Sold: {sold:.0f}. "
                f"Consider repricing or reviewing vendor costs."
            )
        return f"Low margin detected. QOH: {qty:.0f}, Sold: {sold:.0f}."

    elif primitive == "negative_inventory":
        return (
            f"Showing {qty:.0f} units (NEGATIVE). Value: ${abs(qty * cost):.2f}. "
            f"Possible causes: overselling, theft, data entry error. "
            f"Audit immediately."
        )

    elif primitive == "low_stock":
        return (
            f"Only {qty:.0f} units left but sold {sold:.0f} recently. "
            f"Risk of stockout. Reorder cost: ${cost:.2f}/unit. "
            f"Reorder urgently to avoid lost sales."
        )

    elif primitive == "dead_item":
        months_supply = qty / sold if sold > 0 else float("inf")
        capital_tied = qty * cost
        return (
            f"QOH: {qty:.0f}, Sold: {sold:.0f}. "
            f"{'No sales' if sold == 0 else f'{months_supply:.0f}+ months supply'}. "
            f"${capital_tied:.2f} capital tied up. Consider clearance."
        )

    elif primitive == "overstock":
        months_supply = qty / sold if sold > 0 else float("inf")
        return (
            f"QOH: {qty:.0f} vs Sold: {sold:.0f} = "
            f"{months_supply:.0f}+ months of inventory. "
            f"Inventory value: ${sub_total:.2f}. Reduce reorder qty."
        )

    elif primitive == "shrinkage_pattern":
        return (
            f"High value (${sub_total:.2f}) with low margin ({margin:.1f}%) "
            f"and minimal sales ({sold:.0f}). QOH: {qty:.0f}. "
            f"Investigate for potential shrinkage or theft."
        )

    elif primitive == "margin_erosion":
        if revenue > 0 and cost > 0:
            actual_margin = (revenue - cost) / revenue * 100
            return (
                f"Margin eroded to {actual_margin:.1f}%. "
                f"Cost ${cost:.2f} is {cost / revenue * 100:.0f}% of retail ${revenue:.2f}. "
                f"QOH: {qty:.0f}, Sold: {sold:.0f}. Review pricing."
            )
        return f"Margin erosion detected. QOH: {qty:.0f}."

    elif primitive == "price_discrepancy":
        if cost == 0:
            return (
                f"Zero cost recorded but retail is ${revenue:.2f}. "
                f"QOH: {qty:.0f}. Verify cost data is correct."
            )
        if revenue > 0 and cost > revenue:
            return (
                f"Selling BELOW cost! Cost: ${cost:.2f}, Retail: ${revenue:.2f}. "
                f"QOH: {qty:.0f}, Sold: {sold:.0f}. Fix pricing immediately."
            )
        return f"Price data anomaly detected. QOH: {qty:.0f}."

    elif primitive == "zero_cost_anomaly":
        return (
            f"Cost is $0.00 but selling at ${revenue:.2f}. "
            f"QOH: {qty:.0f}, Sold: {sold:.0f}. "
            f"Cannot calculate true margin without cost data."
        )

    elif primitive == "negative_profit":
        loss_per_unit = cost - revenue
        return (
            f"Selling at a LOSS: Cost ${cost:.2f} > Retail ${revenue:.2f} "
            f"(losing ${loss_per_unit:.2f}/unit). "
            f"QOH: {qty:.0f}, Sold: {sold:.0f}."
        )

    elif primitive == "severe_inventory_deficit":
        return (
            f"Only {qty:.0f} units left with high demand ({sold:.0f} sold). "
            f"Critical stock shortage. Emergency reorder needed."
        )

    return f"Issue detected. QOH: {qty:.0f}, Cost: ${cost:.2f}, Retail: ${revenue:.2f}."


class RustResultAdapter:
    """Transforms Rust sentinel-server DigestJson → Python AnalysisResult.

    The output is byte-for-byte compatible with AnalysisService.analyze()
    so the React frontend renders identically.
    """

    def transform(
        self,
        digest: dict,
        total_rows: int,
        analysis_time: float,
        original_rows: list[dict] | None = None,
    ) -> dict:
        """Transform Rust DigestJson into legacy API response.

        Args:
            digest: Parsed JSON from sentinel-server --json output.
            total_rows: Total rows in the original DataFrame.
            analysis_time: Total elapsed time for the full pipeline (seconds).
            original_rows: Optional original DataFrame rows (list of dicts)
                for enrichment with description, sub_total, etc.

        Returns:
            Dict matching AnalysisService.analyze() output shape exactly.
        """
        # Build SKU enrichment lookup from original rows
        enrichment = self._build_enrichment(original_rows) if original_rows else {}

        # Group Rust issues by Python primitive
        primitive_issues = self._group_by_primitive(digest.get("issues", []))

        # Build leaks dict for all 11 primitives
        leaks = {}
        total_items_flagged = 0
        critical_count = 0
        high_count = 0

        for primitive in ALL_PRIMITIVES:
            issues = primitive_issues.get(primitive, [])
            display = LEAK_DISPLAY.get(primitive, {})

            # Collect all SKUs across issues for this primitive
            all_skus = []
            for issue in issues:
                for sku_data in issue.get("skus", []):
                    sku_id = sku_data.get("sku_id", "")
                    if sku_id and sku_id not in {s["sku"] for s in all_skus}:
                        all_skus.append(
                            self._sku_to_item_detail(
                                sku_data, primitive, enrichment, issue
                            )
                        )

            # Sort by score descending, take top 20
            all_skus.sort(key=lambda x: x["score"], reverse=True)
            top_items = all_skus[:20]
            total_count = len(all_skus)

            leaks[primitive] = {
                "top_items": [item["sku"] for item in top_items],
                "scores": [item["score"] for item in top_items],
                "item_details": top_items,
                "count": total_count,
                "severity": display.get("severity", "info"),
                "category": display.get("category", "Unknown"),
                "recommendations": RECOMMENDATIONS.get(primitive, []),
                "title": display.get("title", primitive.replace("_", " ").title()),
                "icon": display.get("icon", "alert"),
                "color": display.get("color", "#6b7280"),
                "priority": display.get("priority", 99),
            }

            total_items_flagged += total_count
            if display.get("severity") == "critical":
                critical_count += total_count
            elif display.get("severity") == "high":
                high_count += total_count

        # Build impact estimation from Rust dollar_impact values
        estimated_impact = self._estimate_impact(digest, leaks, enrichment)

        # Build cause diagnosis from Rust root cause data
        cause_diagnosis = self._extract_cause_diagnosis(digest.get("issues", []))

        result = {
            "leaks": leaks,
            "summary": {
                "total_rows_analyzed": total_rows,
                "total_items_flagged": total_items_flagged,
                "critical_issues": critical_count,
                "high_issues": high_count,
                "estimated_impact": estimated_impact,
                "analysis_time_seconds": round(analysis_time, 2),
            },
            "primitives_used": ALL_PRIMITIVES,
        }

        if cause_diagnosis:
            result["cause_diagnosis"] = cause_diagnosis

        return result

    def _group_by_primitive(self, issues: list[dict]) -> dict[str, list[dict]]:
        """Group Rust issues by Python primitive key."""
        grouped: dict[str, list[dict]] = {}
        for issue in issues:
            rust_type = issue.get("issue_type", "")
            primitive = RUST_TYPE_TO_PRIMITIVE.get(rust_type)
            if primitive:
                grouped.setdefault(primitive, []).append(issue)
            else:
                logger.warning("Unknown Rust issue_type: %s", rust_type)
        return grouped

    def _sku_to_item_detail(
        self,
        sku_data: dict,
        primitive: str,
        enrichment: dict,
        issue: dict,
    ) -> dict:
        """Convert Rust SkuJson → Python ItemDetail."""
        sku_id = sku_data.get("sku_id", "")
        enriched = enrichment.get(sku_id, {})

        qty = _safe_float(sku_data.get("qty_on_hand", 0))
        cost = _safe_float(sku_data.get("unit_cost", 0))
        retail = _safe_float(sku_data.get("retail_price", 0))
        sold = _safe_float(sku_data.get("sales_last_30d", 0))
        margin_pct = _safe_float(sku_data.get("margin_pct", 0))

        # Convert margin from decimal to percentage for display
        margin_display = margin_pct * 100 if margin_pct <= 1.0 else margin_pct

        # Description from enrichment, or empty
        description = str(enriched.get("description", ""))[:50]

        # Sub-total from enrichment, or compute
        sub_total = _safe_float(enriched.get("sub_total", qty * cost))

        # Build the item dict for context generation
        item_for_context = {
            "quantity": qty,
            "cost": cost,
            "revenue": retail,
            "sold": sold,
            "margin": margin_display,
            "sub_total": sub_total,
        }

        # Score: use Rust issue confidence, distributed across SKUs
        confidence = _safe_float(issue.get("confidence", 0.5))
        # Give each SKU a slightly different score based on its data severity
        sku_score = self._compute_sku_score(primitive, sku_data, confidence)

        return {
            "sku": sku_id,
            "score": round(sku_score, 4),
            "description": description,
            "quantity": qty,
            "cost": cost,
            "revenue": retail,
            "sold": sold,
            "margin": margin_display,
            "sub_total": sub_total,
            "context": _get_issue_context(primitive, item_for_context),
        }

    def _compute_sku_score(
        self, primitive: str, sku_data: dict, base_confidence: float
    ) -> float:
        """Compute per-SKU severity score (0.0-1.0).

        Uses the Rust issue confidence as a base and adjusts based on
        how extreme the SKU's data is.
        """
        qty = _safe_float(sku_data.get("qty_on_hand", 0))
        cost = _safe_float(sku_data.get("unit_cost", 0))
        retail = _safe_float(sku_data.get("retail_price", 0))
        margin = _safe_float(sku_data.get("margin_pct", 0))
        sold = _safe_float(sku_data.get("sales_last_30d", 0))

        # Start with issue confidence
        score = base_confidence

        if primitive == "negative_inventory":
            # More negative → higher score
            if qty < 0:
                severity = min(abs(qty) / 100, 1.0)
                score = max(score, 0.7 + 0.3 * severity)

        elif primitive == "margin_erosion" or primitive == "high_margin_leak":
            # Lower margin → higher score
            if margin < 0.2:
                score = max(score, 0.6 + 0.4 * (1 - margin / 0.2))

        elif primitive == "zero_cost_anomaly":
            # Zero cost with higher retail → higher score
            if cost == 0 and retail > 0:
                score = max(score, 0.7 + min(retail / 100, 0.3))

        elif primitive == "price_discrepancy":
            # Bigger cost > retail gap → higher score
            if cost > retail and retail > 0:
                gap_pct = (cost - retail) / retail
                score = max(score, 0.6 + min(gap_pct, 0.4))

        elif primitive == "dead_item":
            # More stock with zero sales → higher score
            if sold == 0 and qty > 0:
                value = qty * cost
                score = max(score, 0.5 + min(value / 5000, 0.5))

        elif primitive == "overstock":
            if sold > 0:
                months_supply = qty / sold
                score = max(score, 0.4 + min(months_supply / 24, 0.6))
            elif qty > 0:
                score = max(score, 0.6)

        elif primitive == "shrinkage_pattern":
            value = qty * cost
            if value > 0 and margin < 0.15:
                score = max(score, 0.6 + min(value / 10000, 0.4))

        elif primitive == "negative_profit":
            if cost > retail:
                loss = cost - retail
                score = max(score, 0.8 + min(loss / 50, 0.2))

        return min(score, 1.0)

    def _build_enrichment(self, rows: list[dict]) -> dict[str, dict]:
        """Build SKU→row lookup for enrichment data (description, sub_total)."""
        lookup: dict[str, dict] = {}
        if not rows:
            return lookup
        for row in rows:
            # Try multiple SKU key names
            sku = None
            for key in ("sku", "SKU", "barcode", "Barcode", "product_id"):
                val = row.get(key)
                if val is not None and str(val).strip():
                    sku = str(val).strip()
                    break
            if sku:
                lookup[sku] = row
        return lookup

    def _estimate_impact(
        self,
        digest: dict,
        leaks: dict,
        enrichment: dict,
    ) -> dict:
        """Estimate dollar impact from Rust issue data.

        Uses Rust's dollar_impact values directly (which are computed from
        actual inventory data), rather than the old heuristic estimation.
        """
        impact: dict[str, Any] = {
            "currency": "USD",
            "low_estimate": 0.0,
            "high_estimate": 0.0,
            "breakdown": {},
            "negative_inventory_alert": None,
        }

        # Group Rust issues by primitive for dollar impact aggregation
        primitive_impacts: dict[str, float] = {}
        for issue in digest.get("issues", []):
            rust_type = issue.get("issue_type", "")
            primitive = RUST_TYPE_TO_PRIMITIVE.get(rust_type)
            if primitive:
                dollar_impact = _safe_float(issue.get("dollar_impact", 0))
                primitive_impacts[primitive] = (
                    primitive_impacts.get(primitive, 0) + dollar_impact
                )

        # Build negative inventory alert
        neg_inv_leak = leaks.get("negative_inventory", {})
        neg_inv_count = neg_inv_leak.get("count", 0)
        if neg_inv_count > 0:
            # Compute untracked COGS from item details
            untracked_cogs = 0.0
            raw_untracked = 0.0
            for item in neg_inv_leak.get("item_details", []):
                qty = _safe_float(item.get("quantity", 0))
                cost = _safe_float(item.get("cost", 0))
                if qty < 0:
                    raw_untracked += abs(qty) * cost
                    capped_qty = min(abs(qty), MAX_IMPACTABLE_UNITS)
                    untracked_cogs += min(capped_qty * cost, MAX_PER_ITEM_IMPACT)

            is_anomalous = raw_untracked > 100_000
            impact["negative_inventory_alert"] = {
                "items_found": neg_inv_count,
                "potential_untracked_cogs": round(untracked_cogs, 2),
                "raw_data_anomaly_value": (
                    round(raw_untracked, 2) if is_anomalous else None
                ),
                "is_anomalous": is_anomalous,
                "threshold_exceeded": is_anomalous,
                "requires_audit": True,
                "excluded_from_annual_estimate": True,
                "note": (
                    "Capped estimate - actual data suggests system sync issues"
                    if is_anomalous
                    else None
                ),
            }

        # Fill breakdown for all primitives
        for primitive in ALL_PRIMITIVES:
            if primitive == "negative_inventory":
                impact["breakdown"][primitive] = 0.0  # Excluded from annual estimate
                continue

            rust_impact = primitive_impacts.get(primitive, 0.0)
            impact["breakdown"][primitive] = round(rust_impact, 2)
            impact["low_estimate"] += rust_impact * 0.7
            impact["high_estimate"] += rust_impact * 1.3

        # Sanity cap
        max_reasonable = 10_000_000
        if impact["high_estimate"] > max_reasonable:
            scale = max_reasonable / impact["high_estimate"]
            impact["low_estimate"] *= scale
            impact["high_estimate"] = max_reasonable
            for k in impact["breakdown"]:
                impact["breakdown"][k] = round(impact["breakdown"][k] * scale, 2)

        impact["low_estimate"] = round(impact["low_estimate"], 2)
        impact["high_estimate"] = round(impact["high_estimate"], 2)

        return impact

    def _extract_cause_diagnosis(self, issues: list[dict]) -> dict | None:
        """Extract cause diagnosis from Rust root_cause data.

        Aggregates root cause information across all issues into the
        format expected by the frontend's CauseDiagnosis interface.
        """
        if not issues:
            return None

        # Find the highest-confidence root cause across all issues
        best_cause = None
        best_confidence = 0.0
        all_cause_scores: dict[str, list[float]] = {}

        for issue in issues:
            root_cause = issue.get("root_cause")
            confidence = _safe_float(issue.get("root_cause_confidence", 0))

            if root_cause and confidence > best_confidence:
                best_cause = root_cause
                best_confidence = confidence

            # Collect all cause scores for hypotheses
            for cs in issue.get("cause_scores", []):
                cause = cs.get("cause", "")
                score = _safe_float(cs.get("score", 0))
                if cause:
                    all_cause_scores.setdefault(cause, []).append(score)

        if not best_cause:
            return None

        # Build hypotheses from averaged cause scores
        hypotheses = []
        for cause, scores in sorted(
            all_cause_scores.items(),
            key=lambda x: -sum(x[1]) / len(x[1]),
        )[:3]:
            avg_score = sum(scores) / len(scores)
            # Normalize to probability-like range
            probability = min(avg_score / max(best_confidence * 2, 1.0), 1.0)
            hypotheses.append(
                {
                    "cause": cause,
                    "probability": round(probability, 3),
                    "evidence": [
                        f"Detected across {len(scores)} issue(s)",
                        f"Average evidence score: {avg_score:.2f}",
                    ],
                }
            )

        return {
            "top_cause": best_cause,
            "confidence": round(best_confidence, 3),
            "hypotheses": hypotheses,
        }
