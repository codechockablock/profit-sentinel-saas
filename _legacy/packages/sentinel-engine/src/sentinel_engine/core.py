"""
Sentinel Engine Core - Request-scoped VSA-based Profit Leak Detection.

This module implements the hyperdimensional computing (VSA) resonator for
detecting profit leaks across retail inventory data from ANY POS system.

CRITICAL: All functions receive an AnalysisContext parameter.
No module-level mutable state is used. Each request is isolated.

v2.1 Changes (Request Isolation):
- All functions now require AnalysisContext parameter
- No global codebook_dict - state is per-context
- Primitives are cached per-context, not globally
- Thread-safe for concurrent requests

v2.0 Features:
- 8 detection primitives
- Aggressive thresholds tuned for real-world retail data
- Universal column synonym handling
- $ impact estimation
- Recommended actions per leak type
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from .context import AnalysisContext, DiracVector

logger = logging.getLogger(__name__)


# =============================================================================
# COLUMN SYNONYM MAPS - Universal POS Support (Immutable - safe to share)
# Handles: Paladin, Spruce, Epicor, Square, Lightspeed, Clover, Shopify,
#          NCR Counterpoint, Microsoft Dynamics RMS, and generic exports
# =============================================================================

QUANTITY_ALIASES = [
    "quantity",
    "qty",
    "qoh",
    "on_hand",
    "onhand",
    "in_stock",
    "instock",
    "stock",
    "inventory",
    "inv_qty",
    "stock_qty",
    "stock_level",
    "in stock qty.",
    "qty.",
    "stockonhand",
    "qty_on_hnd",
    "qty_avail",
    "net_qty",
    "available",
    "current_stock",
    "units_on_hand",
    "quantity_on_hand",
    "bal",
    "balance",
    "variant_inventory_qty",
    "new_quantity",
    "qty_oh",
    "physical_qty",
    "book_qty",
]

COST_ALIASES = [
    "cost",
    "cogs",
    "cost_price",
    "unit_cost",
    "unitcost",
    "avg_cost",
    "avgcost",
    "average_cost",
    "standard_cost",
    "item_cost",
    "product_cost",
    "mktcost",
    "lst_cost",
    "last_cost",
    "default_cost",
    "vendor_cost",
    "supply_price",
    "cost_per_item",
    "posting_cost",
    "landed_cost",
    "purchase_cost",
    "buy_price",
    "wholesale_cost",
    "invoice_cost",
]

REVENUE_ALIASES = [
    "revenue",
    "retail",
    "retail_price",
    "price",
    "sell_price",
    "selling_price",
    "sale_price",
    "unit_price",
    "msrp",
    "list_price",
    "sug. retail",
    "sug retail",
    "suggested_retail",
    "total_sale",
    "ext_price",
    "line_total",
    "amount",
    "gross_sales",
    "netprice",
    "net_price",
    "prc_1",
    "reg_prc",
    "regular_price",
    "pricea",
    "default_price",
    "variant_price",
    "compare_at_price",
    "pos_price",
    "current_price",
    "active_price",
]

SOLD_ALIASES = [
    "sold",
    "units_sold",
    "qty_sold",
    "quantity_sold",
    "sales_qty",
    "total_sold",
    "sold_qty",
    "sold_last_week",
    "sold_last_month",
    "sold_30_days",
    "sold_7_days",
    "sold_ytd",
    "ytd_sold",
    "mtd_sold",
    "sales_units",
    "unit_sales",
    "movement",
    "turns",
    "turnover",
]

QTY_DIFF_ALIASES = [
    "qty_difference",
    "qty. difference",
    "difference",
    "variance",
    "qty_variance",
    "inventory_variance",
    "stock_variance",
    "count_variance",
    "adjustment",
    "qty_adjustment",
    "shrinkage",
    "shrink",
    "loss",
    "qty_loss",
    "discrepancy",
    "audit_variance",
    "over_short",
]

MARGIN_ALIASES = [
    "margin",
    "profit_margin",
    "margin_pct",
    "margin_percent",
    "margin_%",
    "margin%",
    "gp",
    "gross_profit",
    "gp_pct",
    "gp_percent",
    "gp_%",
    "gp%",
    "profit_pct",
    "profit_percent",
    "markup",
    "markup_pct",
    "markup_percent",
]

LAST_SALE_ALIASES = [
    "last_sale",
    "last_sold",
    "last_sale_date",
    "last_sold_date",
    "date_last_sold",
    "last_transaction",
    "last sale",
    "last sold",
    "last_activity",
    "last_movement",
    "recent_sale",
    "most_recent_sale",
    "lst_maint_dt",
]

SKU_ALIASES = [
    "sku",
    "product_id",
    "item_id",
    "upc",
    "barcode",
    "plu",
    "item_number",
    "item_no",
    "partnumber",
    "itemlookupcode",
    "system_id",
    "custom_sku",
    "handle",
    "variant_sku",
    "clover_id",
    "article_code",
    "stock_code",
    "item_code",
    "material_number",
]

VENDOR_ALIASES = [
    "vendor",
    "supplier",
    "vendor_name",
    "supplier_name",
    "manufacturer",
    "mfg",
    "mfr",
    "brand",
    "supplier_number1",
    "item_vend_no",
    "vendor_id",
    "default_vendor_name",
    "primary_vendor",
    "distributor",
    "wholesaler",
]

CATEGORY_ALIASES = [
    "category",
    "department",
    "dept",
    "dpt",
    "dpt.",
    "product_type",
    "class",
    "group",
    "deptid",
    "classid",
    "categ_cod",
    "subcategory",
    "type",
    "tags",
    "labels",
    "division",
    "family",
]

REORDER_ALIASES = [
    "reorder_point",
    "reorder_level",
    "min_qty",
    "minimum_qty",
    "safety_stock",
    "par_level",
    "low_stock_alert",
    "stock_alert",
    "minorderqty",
    "stock_min",
]


# =============================================================================
# DEFAULT THRESHOLDS (Immutable - can be overridden per-analysis)
# =============================================================================

THRESHOLDS = {
    # Low Stock
    "low_stock_qty": 10,
    "low_stock_critical": 3,
    # Dead Item
    "dead_item_days": 60,
    "dead_item_sold_threshold": 2,
    # Margin Leak
    "margin_leak_threshold": 0.25,
    "margin_critical_threshold": 0.10,
    "negative_margin_threshold": 0.0,
    # Overstock (calibrated v2.1.3: use qty/sold ratio instead of days_supply)
    # Strategy: qty > sold * 200 means 200+ months of inventory
    "overstock_qty_threshold": 100,  # Minimum qty to consider
    "overstock_qty_to_sold_ratio": 200,  # qty/sold ratio threshold
    "overstock_velocity_threshold": 0.5,  # Legacy, kept for backward compat
    "overstock_days_supply": 270,  # Legacy, kept for backward compat
    # Shrinkage
    "shrinkage_threshold": -1,
    "shrinkage_critical": -10,
    # Price Discrepancy (calibrated: 15% -> 30% variance to reduce false positives)
    "price_discrepancy_threshold": 0.30,
}


# =============================================================================
# LEAK METADATA (Immutable - safe to share)
# =============================================================================

LEAK_METADATA = {
    "low_stock": {
        "severity": "high",
        "category": "Lost Sales",
        "impact_formula": "lost_units * margin_per_unit",
        "recommendations": [
            "Review reorder point settings",
            "Check supplier lead times",
            "Analyze demand forecast accuracy",
            "Consider safety stock increase",
        ],
    },
    "high_margin_leak": {
        "severity": "critical",
        "category": "Margin Erosion",
        "impact_formula": "(expected_margin - actual_margin) * revenue",
        "recommendations": [
            "Audit recent cost increases from vendors",
            "Check for stuck promotional pricing",
            "Review competitor price matching policies",
            "Investigate unauthorized discounting",
        ],
    },
    "dead_item": {
        "severity": "medium",
        "category": "Dead Capital",
        "impact_formula": "quantity * cost",
        "recommendations": [
            "Consider markdown or clearance pricing",
            "Evaluate vendor return options",
            "Check if item is still vendor-active",
            "Review seasonal patterns",
        ],
    },
    "negative_inventory": {
        "severity": "critical",
        "category": "Data Integrity / Shrinkage",
        "impact_formula": "abs(quantity) * cost",
        "recommendations": [
            "Perform immediate physical count",
            "Review recent receiving records",
            "Check for unprocessed purchase orders",
            "Audit void/return transactions",
        ],
    },
    "overstock": {
        "severity": "medium",
        "category": "Cash Flow / Carrying Cost",
        "impact_formula": "excess_qty * cost * carrying_rate",
        "recommendations": [
            "Reduce future order quantities",
            "Create promotional bundles",
            "Evaluate cross-location transfers",
            "Review forecast vs actual demand",
        ],
    },
    "price_discrepancy": {
        "severity": "warning",
        "category": "Pricing Integrity",
        "impact_formula": "(suggested_retail - actual_retail) * sold_qty",
        "recommendations": [
            "Verify retail price is intentional",
            "Check for promotional pricing errors",
            "Update suggested retail if outdated",
            "Review price change authorization logs",
        ],
    },
    "shrinkage_pattern": {
        "severity": "high",
        "category": "Inventory Loss",
        "impact_formula": "shrinkage_qty * cost",
        "recommendations": [
            "Review receiving accuracy",
            "Check for theft patterns",
            "Audit cycle count procedures",
            "Investigate damage/spoilage tracking",
        ],
    },
    "margin_erosion": {
        "severity": "high",
        "category": "Profitability Trend",
        "impact_formula": "margin_drop * revenue",
        "recommendations": [
            "Analyze cost trend over time",
            "Review vendor contract terms",
            "Evaluate competitive pricing pressure",
            "Check for promotional frequency issues",
        ],
    },
    # New primitives (v2.2 - discovered via structure learning on real retail data)
    "zero_cost_anomaly": {
        "severity": "critical",
        "category": "Data Integrity / Costing",
        "impact_formula": "revenue * estimated_margin",
        "recommendations": [
            "Check receiving records for missing cost entry",
            "Review PO cost vs item master cost",
            "Verify vendor invoice was processed",
            "Update item cost from recent purchase",
        ],
    },
    "negative_profit": {
        "severity": "critical",
        "category": "Margin Loss",
        "impact_formula": "abs(gross_profit)",
        "recommendations": [
            "Verify cost data is current",
            "Check for clearance/promotional pricing",
            "Review recent vendor cost increases",
            "Evaluate if item should be discontinued",
        ],
    },
    "severe_inventory_deficit": {
        "severity": "critical",
        "category": "Inventory Crisis",
        "impact_formula": "abs(quantity) * cost",
        "recommendations": [
            "Immediate physical count required",
            "Audit all transactions for this SKU",
            "Check for system sync issues (POS/inventory)",
            "Review theft/shrinkage patterns",
            "Escalate to management - high priority",
        ],
    },
}


# =============================================================================
# HELPER FUNCTIONS (Stateless - safe to share)
# =============================================================================


def _normalize_key(key: str) -> str:
    """Normalize a key for comparison. Cached via module-level dict."""
    return key.lower().replace(" ", "").replace(".", "").replace("_", "")


# Pre-normalized alias lookup tables (computed once at module load)
# Maps normalized alias -> original alias list reference
_NORMALIZED_QUANTITY_ALIASES: frozenset[str] = frozenset(
    _normalize_key(a) for a in QUANTITY_ALIASES
)
_NORMALIZED_COST_ALIASES: frozenset[str] = frozenset(
    _normalize_key(a) for a in COST_ALIASES
)
_NORMALIZED_REVENUE_ALIASES: frozenset[str] = frozenset(
    _normalize_key(a) for a in REVENUE_ALIASES
)
_NORMALIZED_SOLD_ALIASES: frozenset[str] = frozenset(
    _normalize_key(a) for a in SOLD_ALIASES
)
_NORMALIZED_QTY_DIFF_ALIASES: frozenset[str] = frozenset(
    _normalize_key(a) for a in QTY_DIFF_ALIASES
)
_NORMALIZED_MARGIN_ALIASES: frozenset[str] = frozenset(
    _normalize_key(a) for a in MARGIN_ALIASES
)
_NORMALIZED_LAST_SALE_ALIASES: frozenset[str] = frozenset(
    _normalize_key(a) for a in LAST_SALE_ALIASES
)
_NORMALIZED_SKU_ALIASES: frozenset[str] = frozenset(
    _normalize_key(a) for a in SKU_ALIASES
)
_NORMALIZED_VENDOR_ALIASES: frozenset[str] = frozenset(
    _normalize_key(a) for a in VENDOR_ALIASES
)
_NORMALIZED_CATEGORY_ALIASES: frozenset[str] = frozenset(
    _normalize_key(a) for a in CATEGORY_ALIASES
)
_NORMALIZED_REORDER_ALIASES: frozenset[str] = frozenset(
    _normalize_key(a) for a in REORDER_ALIASES
)
# Additional alias lists used in bundling (defined as module-level for id() stability)
DESC_ALIASES: list[str] = ["description", "desc", "name", "item", "title"]
SUGRETAIL_ALIASES: list[str] = [
    "sugretail",
    "sugretail",
    "suggestedretail",
    "msrp",
    "listprice",
]

_NORMALIZED_DESC_ALIASES: frozenset[str] = frozenset(
    _normalize_key(a) for a in DESC_ALIASES
)
_NORMALIZED_SUGRETAIL_ALIASES: frozenset[str] = frozenset(
    _normalize_key(a) for a in SUGRETAIL_ALIASES
)

# Map from alias list identity to pre-normalized frozenset
_ALIAS_TO_NORMALIZED: dict[int, frozenset[str]] = {
    id(QUANTITY_ALIASES): _NORMALIZED_QUANTITY_ALIASES,
    id(COST_ALIASES): _NORMALIZED_COST_ALIASES,
    id(REVENUE_ALIASES): _NORMALIZED_REVENUE_ALIASES,
    id(SOLD_ALIASES): _NORMALIZED_SOLD_ALIASES,
    id(QTY_DIFF_ALIASES): _NORMALIZED_QTY_DIFF_ALIASES,
    id(MARGIN_ALIASES): _NORMALIZED_MARGIN_ALIASES,
    id(LAST_SALE_ALIASES): _NORMALIZED_LAST_SALE_ALIASES,
    id(SKU_ALIASES): _NORMALIZED_SKU_ALIASES,
    id(VENDOR_ALIASES): _NORMALIZED_VENDOR_ALIASES,
    id(CATEGORY_ALIASES): _NORMALIZED_CATEGORY_ALIASES,
    id(REORDER_ALIASES): _NORMALIZED_REORDER_ALIASES,
    id(DESC_ALIASES): _NORMALIZED_DESC_ALIASES,
    id(SUGRETAIL_ALIASES): _NORMALIZED_SUGRETAIL_ALIASES,
}


def _preprocess_row(row: dict) -> dict:
    """
    Normalize row keys once, reuse for all field lookups.

    Returns dict mapping normalized_key -> value for O(1) lookup.
    """
    return {_normalize_key(k): v for k, v in row.items()}


def _get_field_fast(row_normalized: dict, aliases: list[str], default: Any = 0) -> Any:
    """
    Fast O(1) lookup using pre-normalized row and pre-normalized aliases.

    Performance: Uses frozenset intersection instead of iterating aliases.
    For 156K rows × 15 fields, this eliminates ~46M string normalizations.
    """
    # Try to get pre-normalized alias set (O(1) dict lookup)
    normalized_set = _ALIAS_TO_NORMALIZED.get(id(aliases))

    if normalized_set is not None:
        # Fast path: use set intersection
        matching_keys = normalized_set & row_normalized.keys()
        for key in matching_keys:
            val = row_normalized[key]
            if val is not None and str(val).strip() != "":
                return val
    else:
        # Slow path for ad-hoc alias lists: normalize on demand
        for alias in aliases:
            norm_alias = _normalize_key(alias)
            if norm_alias in row_normalized:
                val = row_normalized[norm_alias]
                if val is not None and str(val).strip() != "":
                    return val

    return default


def _get_field(row: dict, aliases: list[str], default: Any = 0) -> Any:
    """
    Extract field value from row using multiple alias names.
    Case-insensitive matching with normalization.

    NOTE: For bulk operations, use _preprocess_row() once per row,
    then _get_field_fast() for all lookups. This avoids repeated dict creation.
    """
    row_normalized = _preprocess_row(row)
    return _get_field_fast(row_normalized, aliases, default)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float, handling currency strings."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        # Remove currency symbols and commas
        cleaned = str(val).replace("$", "").replace(",", "").replace("%", "").strip()
        if cleaned == "" or cleaned.lower() in ("nan", "null", "none", "-"):
            return default
        return float(cleaned)
    except (ValueError, TypeError):
        return default


def _parse_date(val: Any) -> datetime | None:
    """Parse date from various formats."""
    if val is None or str(val).strip() == "":
        return None

    date_str = str(val).strip()

    formats = [
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%m-%d-%Y",
        "%d/%m/%Y",
        "%m/%d/%y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m.%d.%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


# =============================================================================
# RESONATOR (Context-aware)
# =============================================================================


@torch.no_grad()
def convergence_lock_resonator(
    ctx: AnalysisContext,
    unbound_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Context-aware convergence lock resonator with early-stopping.

    Uses multi-step iterative resonance with attention-weighted sparse projections.
    All state comes from the context - no global variables.

    Production v3.0: Added early-stopping when delta < convergence_threshold (default 1e-4).
    This typically converges in 20-40 iterations instead of full 100, providing 3-5x speedup.

    Args:
        ctx: Analysis context with codebook and parameters
        unbound_batch: Query vectors to clean up, shape (batch, dimensions)

    Returns:
        Cleaned query vectors, same shape as input
    """
    codebook_tensor = ctx.get_codebook_tensor()
    if codebook_tensor is None:
        return unbound_batch

    x = ctx.normalize(unbound_batch)
    x_prev = x.clone()  # Track previous state for convergence check
    F_mat = codebook_tensor.T.to(ctx.dtype)

    total_iters = 0
    for step in range(ctx.multi_steps):
        for inner in range(ctx.iters // ctx.multi_steps):
            sims = torch.real(torch.conj(F_mat).T @ x.T).T
            sims = torch.maximum(sims, torch.zeros_like(sims))

            attn_weights = torch.softmax(sims * 5.0, dim=-1)
            sims = sims**ctx.power * torch.exp(sims / 5.0)

            topk_vals, topk_idx = torch.topk(
                sims, min(ctx.top_k, sims.shape[-1]), dim=-1
            )
            sims_sparse = torch.zeros_like(sims)
            sims_sparse.scatter_(-1, topk_idx, topk_vals)
            sims = sims_sparse.to(ctx.dtype)

            projection = F_mat @ (attn_weights * sims).T
            projection = projection.T

            x = ctx.alpha * x + (1 - ctx.alpha) * projection
            x = ctx.normalize(x)

            total_iters += 1

            # Early-stopping: check convergence every iteration
            delta = torch.norm(x - x_prev).item()
            if delta < ctx.convergence_threshold:
                logger.debug(
                    f"Resonator converged at iteration {total_iters} "
                    f"(delta={delta:.2e} < threshold={ctx.convergence_threshold:.2e})"
                )
                return x

            x_prev = x.clone()

    logger.debug(f"Resonator completed {total_iters} iterations (no early convergence)")
    return x


# =============================================================================
# BUNDLE POS FACTS (Context-aware)
# =============================================================================


def bundle_pos_facts(
    ctx: AnalysisContext,
    rows: list[dict],
    thresholds: dict | None = None,
) -> torch.Tensor:
    """
    Bundle POS data facts into hypervector with aggressive leak detection.

    Handles data from ANY POS system using universal column aliases.
    Binds primitives for 11 different leak types with tuned thresholds.

    IMPORTANT: All state is stored in ctx, not in global variables.

    Performance optimizations (v3.4):
    - BATCH PRE-COMPUTE: All SKU vectors computed upfront in batch
    - PRE-EXTRACT: All fields extracted once before main loop
    - SKIP NON-SKU CODEBOOK: Don't track description/vendor/category vectors
    - SINGLE PASS bundling after stats collection

    Target: 36K rows in <10s on 4 vCPU Fargate

    Args:
        ctx: Analysis context (mutated: codebook populated, leak_counts updated)
        rows: List of row dictionaries from POS data
        thresholds: Optional custom thresholds (uses THRESHOLDS if None)

    Returns:
        Normalized bundle hypervector
    """
    start_time = time.time()
    num_rows = len(rows)
    logger.info(f"Bundling facts for {num_rows} rows (v3.4 optimized)")

    if thresholds is None:
        thresholds = THRESHOLDS

    primitives = ctx.get_primitives()
    bundle = ctx.zeros()

    # =================================================================
    # PHASE 0: BATCH PRE-EXTRACT all fields (v3.4 optimization)
    # =================================================================
    preprocess_start = time.time()

    # Pre-extract all row data in a single pass
    extracted_data = []
    unique_skus = set()

    for row in rows:
        row_norm = _preprocess_row(row)

        sku = str(_get_field_fast(row_norm, SKU_ALIASES, "unknown_sku"))
        if sku != "unknown_sku":
            unique_skus.add(sku)

        extracted_data.append(
            {
                "sku": sku,
                "category": str(
                    _get_field_fast(row_norm, CATEGORY_ALIASES, "unknown_category")
                ),
                "quantity": _safe_float(_get_field_fast(row_norm, QUANTITY_ALIASES, 0)),
                "qty_diff": _safe_float(_get_field_fast(row_norm, QTY_DIFF_ALIASES, 0)),
                "cost": _safe_float(_get_field_fast(row_norm, COST_ALIASES, 0)),
                "revenue": _safe_float(_get_field_fast(row_norm, REVENUE_ALIASES, 0)),
                "sold": _safe_float(_get_field_fast(row_norm, SOLD_ALIASES, 0)),
                "margin_direct": _safe_float(
                    _get_field_fast(row_norm, MARGIN_ALIASES, None)
                ),
                "last_sale_date": _parse_date(
                    _get_field_fast(row_norm, LAST_SALE_ALIASES, None)
                ),
                "sug_retail": _safe_float(
                    _get_field_fast(row_norm, SUGRETAIL_ALIASES, 0)
                ),
                "reorder_point": _safe_float(
                    _get_field_fast(row_norm, REORDER_ALIASES, 0)
                ),
            }
        )

    preprocess_time = time.time() - preprocess_start
    logger.info(
        f"Pre-extracted {len(extracted_data)} rows, {len(unique_skus)} unique SKUs in {preprocess_time:.2f}s"
    )

    # =================================================================
    # PHASE 0.5: BATCH PRE-COMPUTE SKU vectors (v3.6 optimization)
    # =================================================================
    codebook_start = time.time()

    # v3.6: Use batched vector creation (88s -> ~2s for 36K SKUs)
    sku_list_for_batch = list(unique_skus)
    if hasattr(ctx, "batch_populate_codebook"):
        ctx.batch_populate_codebook(sku_list_for_batch, is_sku=True)
    else:
        # Fallback for older context versions
        for sku in unique_skus:
            ctx.add_to_codebook(sku, is_sku=True)
            ctx.get_or_create(sku)

    codebook_time = time.time() - codebook_start
    logger.info(
        f"v3.6: Pre-computed {len(unique_skus)} SKU vectors in {codebook_time:.2f}s"
    )

    # =================================================================
    # PHASE 1: Collect stats from first N rows
    # =================================================================
    sample_size = max(1000, num_rows // 10)  # At least 1000 or 10%
    sample_quantities: list[float] = []
    sample_margins: list[float] = []
    sample_sold: list[float] = []
    category_margins: dict[str, list[float]] = {}

    # Stats (computed after sample_size rows)
    avg_qty: float = 20.0
    avg_margin: float = 0.3
    avg_sold: float = 10.0
    category_avg_margin: dict[str, float] = {}
    adaptive_dead_threshold: float = thresholds["dead_item_sold_threshold"]

    # =================================================================
    # PHASE 1: Collect stats from first N rows using pre-extracted data
    # =================================================================
    stats_start = time.time()
    now = datetime.now()

    for i, data in enumerate(extracted_data[:sample_size]):
        quantity = data["quantity"]
        cost = data["cost"]
        revenue = data["revenue"]
        sold = data["sold"]
        category = data["category"]

        # Calculate margin
        margin_direct = data["margin_direct"]
        if margin_direct is not None and margin_direct != 0:
            margin = margin_direct / 100 if margin_direct > 1 else margin_direct
        elif revenue > 0:
            margin = (revenue - cost) / revenue
        else:
            margin = 0

        if quantity > 0:
            sample_quantities.append(quantity)
        if revenue > 0 and cost > 0:
            sample_margins.append(margin)
            cat_key = category.strip().lower()
            if cat_key not in category_margins:
                category_margins[cat_key] = []
            category_margins[cat_key].append(margin)
        if sold >= 0:
            sample_sold.append(sold)

    # Compute stats
    avg_qty = (
        sum(sample_quantities) / len(sample_quantities) if sample_quantities else 20.0
    )
    avg_margin = sum(sample_margins) / len(sample_margins) if sample_margins else 0.3
    avg_sold = sum(sample_sold) / len(sample_sold) if sample_sold else 10.0

    ctx.update_stats(avg_qty, avg_margin, avg_sold)

    for cat, cat_margins in category_margins.items():
        if len(cat_margins) >= 3:
            category_avg_margin[cat] = sum(cat_margins) / len(cat_margins)
        else:
            category_avg_margin[cat] = avg_margin

    adaptive_dead_threshold = max(
        avg_sold * 0.1, thresholds["dead_item_sold_threshold"]
    )

    stats_time = time.time() - stats_start
    logger.info(
        f"Stats computed in {stats_time:.2f}s - Avg QOH: {avg_qty:.1f}, "
        f"Avg Margin: {avg_margin:.2%}, Avg Sold: {avg_sold:.1f}"
    )

    # =================================================================
    # PHASE 2: BATCHED BUNDLING (v3.5 FAISS optimization)
    # =================================================================
    # Instead of processing one row at a time with tensor ops,
    # we collect (sku_idx, primitive_idx, strength) tuples first,
    # then do a single batched tensor operation.
    # =================================================================
    bundle_start = time.time()

    # Build SKU index mapping for batch operations
    sku_list = list(unique_skus)
    sku_to_idx = {sku: idx for idx, sku in enumerate(sku_list)}

    # Primitive name to index mapping
    primitive_names = [
        "low_stock",
        "high_margin_leak",
        "dead_item",
        "negative_inventory",
        "overstock",
        "price_discrepancy",
        "shrinkage_pattern",
        "margin_erosion",
        "zero_cost_anomaly",
        "negative_profit",
        "severe_inventory_deficit",
        "high_velocity",
    ]
    primitive_to_idx = {name: idx for idx, name in enumerate(primitive_names)}

    # Collect all (sku_idx, primitive_idx, strength) in one pass
    # This avoids 36K tensor operations by batching
    bindings: list[tuple[int, int, float]] = []  # (sku_idx, primitive_idx, strength)
    leak_counts_local: dict[str, int] = {}

    # =====================================================================
    # PHASE 2a: VECTORIZED BINDING COLLECTION (v3.8 numpy optimization)
    # =====================================================================
    # Build numpy arrays from extracted_data (one-time cost, ~1s for 36K rows)
    n = len(extracted_data)

    # Extract all fields into numpy arrays
    skus = [d["sku"] for d in extracted_data]
    sku_indices = np.array([sku_to_idx.get(s, -1) for s in skus], dtype=np.int32)
    valid = sku_indices >= 0  # mask for rows with valid SKUs

    quantities = np.array([d["quantity"] for d in extracted_data], dtype=np.float64)
    qty_diffs = np.array([d["qty_diff"] for d in extracted_data], dtype=np.float64)
    costs = np.array([d["cost"] for d in extracted_data], dtype=np.float64)
    revenues = np.array([d["revenue"] for d in extracted_data], dtype=np.float64)
    sold_arr = np.array([d["sold"] for d in extracted_data], dtype=np.float64)
    sug_retails = np.array([d["sug_retail"] for d in extracted_data], dtype=np.float64)
    reorder_points = np.array(
        [d["reorder_point"] for d in extracted_data], dtype=np.float64
    )

    # Compute margin array (vectorized)
    margin_directs = np.array(
        [
            d["margin_direct"] if d["margin_direct"] is not None else 0.0
            for d in extracted_data
        ],
        dtype=np.float64,
    )
    margins = np.where(
        margin_directs != 0,
        np.where(margin_directs > 1, margin_directs / 100, margin_directs),
        np.where(revenues > 0, (revenues - costs) / np.maximum(revenues, 1e-10), 0.0),
    )

    # Compute days_since_sale array (-1 means no date)
    days_arr = np.full(n, -1.0, dtype=np.float64)
    for i, d in enumerate(extracted_data):
        if d["last_sale_date"] is not None:
            days_arr[i] = (now - d["last_sale_date"]).days

    has_date = days_arr >= 0

    # Category average margin lookup (vectorized)
    categories = [d["category"].strip().lower() for d in extracted_data]
    cat_avg_margins = np.array(
        [category_avg_margin.get(c, avg_margin) for c in categories], dtype=np.float64
    )

    # =====================================================================
    # VECTORIZED PRIMITIVE DETECTION
    # =====================================================================
    def _collect(prim_name: str, mask: np.ndarray, strengths: np.ndarray) -> None:
        """Collect bindings for one primitive from boolean mask + strength array."""
        combined = valid & mask
        idx = np.where(combined)[0]
        if len(idx) == 0:
            return
        prim_idx = primitive_to_idx[prim_name]
        for i in idx:
            bindings.append((int(sku_indices[i]), prim_idx, float(strengths[i])))
        leak_counts_local[prim_name] = leak_counts_local.get(prim_name, 0) + len(idx)

    # PRIMITIVE 1: LOW STOCK
    low_mask = (quantities > 0) & (quantities <= thresholds["low_stock_qty"])
    low_str = np.where(
        quantities <= thresholds["low_stock_critical"],
        2.0,
        (thresholds["low_stock_qty"] - quantities) / thresholds["low_stock_qty"],
    )
    # Reorder point boost
    low_str = np.where(
        (reorder_points > 0) & (quantities < reorder_points),
        low_str * 1.5,
        low_str,
    )
    _collect("low_stock", low_mask, low_str)

    # PRIMITIVE 2: HIGH MARGIN LEAK (3 tiers)
    has_revenue = revenues > 0
    cat_leak_thresh = cat_avg_margins * 0.5

    # Tier 1: negative margin
    neg_margin_mask = has_revenue & (margins < thresholds["negative_margin_threshold"])
    neg_margin_str = 3.0 + np.abs(margins) * 5
    _collect("high_margin_leak", neg_margin_mask, neg_margin_str)

    # Tier 2: critical margin (exclude already-caught tier 1)
    crit_margin_mask = (
        has_revenue
        & ~neg_margin_mask
        & (margins < thresholds["margin_critical_threshold"])
    )
    crit_margin_str = 2.0 + (thresholds["margin_critical_threshold"] - margins) * 10
    _collect("high_margin_leak", crit_margin_mask, crit_margin_str)

    # Tier 3: below category average (exclude tier 1 and 2)
    cat_margin_mask = (
        has_revenue & ~neg_margin_mask & ~crit_margin_mask & (margins < cat_leak_thresh)
    )
    cat_margin_str = np.where(
        cat_leak_thresh > 0,
        (cat_leak_thresh - margins) / np.maximum(cat_leak_thresh, 1e-10) * 2,
        0.0,
    )
    _collect("high_margin_leak", cat_margin_mask, cat_margin_str)

    # PRIMITIVE 3: DEAD ITEM (two paths)
    # Path A: has date, days > threshold, low sold, qty > 0
    dead_a_mask = (
        has_date
        & (days_arr > thresholds["dead_item_days"])
        & (quantities > 0)
        & (sold_arr < adaptive_dead_threshold)
    )
    dead_a_str = np.minimum(days_arr / thresholds["dead_item_days"], 3.0)
    dead_a_str = np.where(quantities > 10, dead_a_str * 1.5, dead_a_str)
    _collect("dead_item", dead_a_mask, dead_a_str)

    # Path B: no date or date > 30 days, low sold, qty > 0, NOT already caught by path A
    dead_b_mask = (
        ~dead_a_mask
        & (sold_arr < adaptive_dead_threshold)
        & (quantities > 0)
        & (~has_date | (days_arr > 30))
    )
    dead_b_str = np.maximum(
        0.5,
        (adaptive_dead_threshold - sold_arr) / np.maximum(adaptive_dead_threshold, 1.0),
    )
    dead_b_str = np.where(quantities > 10, dead_b_str * 1.5, dead_b_str)
    _collect("dead_item", dead_b_mask, dead_b_str)

    # PRIMITIVE 4: NEGATIVE INVENTORY
    neg_inv_mask = (qty_diffs < thresholds["shrinkage_threshold"]) | (quantities < 0)
    magnitudes = np.where(qty_diffs < 0, np.abs(qty_diffs), np.abs(quantities))
    neg_inv_str = np.where(
        magnitudes > abs(thresholds["shrinkage_critical"]),
        3.0,
        np.minimum(magnitudes / 10.0, 2.0) + 0.5,
    )
    _collect("negative_inventory", neg_inv_mask, neg_inv_str)

    # PRIMITIVE 5: OVERSTOCK
    qty_to_sold = np.where(sold_arr > 0, quantities / np.maximum(sold_arr, 1e-10), 0.0)
    overstock_mask = (
        (sold_arr > 0)
        & (quantities > thresholds["overstock_qty_threshold"])
        & (qty_to_sold > thresholds["overstock_qty_to_sold_ratio"])
    )
    daily_velocity = sold_arr / 30.0
    overstock_str = np.minimum(qty_to_sold / 100, 3.0)
    overstock_str = np.where(
        daily_velocity < thresholds["overstock_velocity_threshold"],
        overstock_str * 1.5,
        overstock_str,
    )
    _collect("overstock", overstock_mask, overstock_str)

    # PRIMITIVE 6: PRICE DISCREPANCY
    price_var = np.where(
        (sug_retails > 0) & (revenues > 0),
        np.abs(revenues - sug_retails) / np.maximum(sug_retails, 1e-10),
        0.0,
    )
    price_mask = (
        (sug_retails > 0)
        & (revenues > 0)
        & (price_var > thresholds["price_discrepancy_threshold"])
    )
    price_str = np.minimum(price_var * 3, 2.0)
    price_str = np.where(revenues < sug_retails, price_str * 1.5, price_str)
    _collect("price_discrepancy", price_mask, price_str)

    # PRIMITIVE 7: SHRINKAGE PATTERN
    shrink_pct = np.where(
        quantities > 0,
        np.abs(qty_diffs) / np.maximum(quantities, 1e-10),
        np.abs(qty_diffs),
    )
    shrink_mask = (qty_diffs < thresholds["shrinkage_threshold"]) & (shrink_pct > 0.05)
    shrink_str = np.minimum(shrink_pct * 10, 3.0)
    _collect("shrinkage_pattern", shrink_mask, shrink_str)

    # PRIMITIVE 8: MARGIN EROSION
    erosion_mask = has_revenue & (margins < avg_margin * 0.7)
    erosion_rate = np.where(avg_margin > 0, (avg_margin - margins) / avg_margin, 0.0)
    erosion_str = erosion_rate * 2.5
    _collect("margin_erosion", erosion_mask, erosion_str)

    # PRIMITIVE 9: ZERO COST ANOMALY
    zero_cost_mask = has_revenue & (costs == 0)
    zero_cost_str = np.minimum(revenues / 100, 3.0) + 1.0
    _collect("zero_cost_anomaly", zero_cost_mask, zero_cost_str)

    # PRIMITIVE 10: NEGATIVE PROFIT
    gross_profit = np.where(
        sold_arr > 0, revenues - (costs * sold_arr), revenues - costs
    )
    neg_profit_mask = gross_profit < 0
    neg_profit_str = np.minimum(np.abs(gross_profit) / 100, 3.0) + 1.0
    _collect("negative_profit", neg_profit_mask, neg_profit_str)

    # PRIMITIVE 11: SEVERE INVENTORY DEFICIT
    severe_mask = quantities < -100
    severe_str = np.minimum(np.abs(quantities) / 100, 5.0) + 2.0
    _collect("severe_inventory_deficit", severe_mask, severe_str)

    # HIGH VELOCITY (composite signal)
    hv_mask = sold_arr > avg_sold * 2
    hv_str = np.full(n, 0.5)
    _collect("high_velocity", hv_mask, hv_str)

    collect_time = time.time() - bundle_start
    logger.info(f"v3.5: Collected {len(bindings)} bindings in {collect_time:.2f}s")

    # =================================================================
    # PHASE 2b: BATCHED TENSOR OPERATIONS (v3.7 memory-optimized)
    # =================================================================
    # Process bindings per-primitive without building giant SKU tensor
    tensor_start = time.time()

    # Group bindings by primitive for batch processing
    from collections import defaultdict

    primitive_bindings: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for sku_idx, prim_idx, strength in bindings:
        primitive_bindings[prim_idx].append((sku_idx, strength))

    # Process each primitive - lookup SKU vectors directly from codebook
    # This avoids creating a 36K x 4K tensor (1.2GB) in memory
    for prim_idx, prim_binds in primitive_bindings.items():
        if not prim_binds:
            continue

        # Get primitive vector
        prim_name = primitive_names[prim_idx]
        prim_vec = primitives[prim_name]

        # Process bindings in chunks to avoid memory issues
        CHUNK_SIZE = 5000
        for chunk_start in range(0, len(prim_binds), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(prim_binds))
            chunk_binds = prim_binds[chunk_start:chunk_end]

            # Get SKU vectors for this chunk directly from codebook
            sku_vecs_list = [ctx.codebook[sku_list[b[0]]] for b in chunk_binds]
            sku_vecs_batch = torch.stack(sku_vecs_list)

            # Get strengths for this chunk
            strengths = torch.tensor(
                [b[1] for b in chunk_binds], dtype=torch.float32, device=ctx.device
            )

            # Batched binding: prim_vec * sku_vecs * strengths
            bound_vecs = (
                prim_vec * sku_vecs_batch * strengths.unsqueeze(1).to(ctx.dtype)
            )

            # Sum and add to bundle
            bundle += bound_vecs.sum(dim=0)

            # Free memory
            del sku_vecs_list, sku_vecs_batch, strengths, bound_vecs

    # Update leak counts
    for name, count in leak_counts_local.items():
        ctx.leak_counts[name] = ctx.leak_counts.get(name, 0) + count

    tensor_time = time.time() - tensor_start
    bundle_phase_time = time.time() - bundle_start
    logger.info(
        f"v3.7: Tensor ops in {tensor_time:.2f}s, total bundling in {bundle_phase_time:.2f}s"
    )

    ctx.rows_processed += num_rows

    elapsed = time.time() - start_time
    logger.info(
        f"Bundling complete in {elapsed:.2f}s "
        f"(codebook: {len(ctx.codebook)}, leaks: {sum(ctx.leak_counts.values())})"
    )

    # Single normalize at the end
    return ctx.normalize(bundle)


# =============================================================================
# QUERY BUNDLE (Context-aware)
# =============================================================================


def query_bundle(
    ctx: AnalysisContext,
    bundle: torch.Tensor,
    primitive_key: str,
    top_k: int = 20,
) -> tuple[list[str], list[float]]:
    """
    Query bundle for items matching a primitive.

    Uses resonator to lock onto items associated with the given leak type.
    All state comes from the context.

    v3.5 OPTIMIZATION: Uses FAISS for O(log n) similarity search.

    Args:
        ctx: Analysis context with populated codebook
        bundle: The bundled hypervector from bundle_pos_facts
        primitive_key: Key for the primitive to query (e.g., "low_stock")
        top_k: Number of top results to return (default 20)

    Returns:
        Tuple of (list of item names, list of similarity scores)
    """
    start_time = time.time()

    primitive = ctx.get_primitive(primitive_key)
    if primitive is None:
        logger.warning(f"Unknown primitive: {primitive_key}")
        return [], []

    # Unbind primitive from bundle
    unbound = primitive.conj() * bundle
    unbound = ctx.normalize(unbound)
    unbound_batch = unbound.unsqueeze(0)

    # Run resonator
    cleaned = convergence_lock_resonator(ctx, unbound_batch)[0]

    # v3.5: Use FAISS search if available
    if hasattr(ctx, "faiss_search") and getattr(ctx, "use_faiss", False):
        results, scores = ctx.faiss_search(cleaned, top_k)
    else:
        # Fallback: Manual matrix multiplication
        codebook_tensor = ctx.get_codebook_tensor()
        if codebook_tensor is None:
            return [], []

        F_mat = codebook_tensor.T.to(ctx.dtype)
        raw_sims = torch.real(torch.conj(F_mat).T @ cleaned)

        # Get top matches
        k = min(top_k, len(ctx.codebook))
        topk_vals, topk_idx = torch.topk(raw_sims, k)

        codebook_keys = ctx.get_codebook_keys()
        results = [codebook_keys[i] for i in topk_idx.tolist()]
        scores = topk_vals.tolist()

    elapsed = time.time() - start_time
    logger.debug(
        f"Query {primitive_key} complete in {elapsed:.2f}s ({len(results)} results)"
    )

    return results, scores


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_primitive_metadata(primitive_key: str) -> dict:
    """Get metadata for a primitive including recommendations."""
    return LEAK_METADATA.get(
        primitive_key,
        {
            "severity": "info",
            "category": "Unknown",
            "recommendations": ["Review manually"],
        },
    )


def get_all_primitives() -> list[str]:
    """Get list of all available primitive keys."""
    return [
        # Original 8 primitives
        "low_stock",
        "high_margin_leak",
        "dead_item",
        "negative_inventory",
        "overstock",
        "price_discrepancy",
        "shrinkage_pattern",
        "margin_erosion",
        # New primitives (v2.2)
        "zero_cost_anomaly",
        "negative_profit",
        "severe_inventory_deficit",
    ]


def get_thresholds() -> dict[str, Any]:
    """Get default thresholds (for reference/documentation)."""
    return THRESHOLDS.copy()


# =============================================================================
# GLM (GEOMETRIC LANGUAGE MODEL) COMPONENTS
# =============================================================================
# These functions implement the complete GLM architecture:
# - Causal/temporal binding with asymmetry
# - Multi-hop reasoning with per-hop cleanup
# - Probabilistic superposition
# - Forward/backward chaining
# - Analogy solving
# - LLM grounding interface


# =============================================================================
# CAUSAL/TEMPORAL PRIMITIVES (Asymmetric Binding)
# =============================================================================


def causal_bind(
    ctx: AnalysisContext,
    cause: torch.Tensor,
    effect: torch.Tensor,
) -> torch.Tensor:
    """
    Asymmetric binding for causal relationships.

    Unlike symmetric bind (a ⊗ b = b ⊗ a), causal relationships
    have direction: cause → effect ≠ effect → cause.

    Implementation: permute(cause) ⊗ causes_primitive ⊗ effect

    This ensures unbinding with cause retrieves effect,
    but unbinding with effect does NOT cleanly retrieve cause.

    Mathematical basis:
        causal_bind(A, B) = permute(A) ⊗ causes ⊗ B
        unbind_cause(causal_bind(A, B), A) ≈ B  (clean)
        unbind_effect(causal_bind(A, B), B) ≈ noise (not clean)

    Args:
        ctx: Analysis context
        cause: Hypervector representing the cause
        effect: Hypervector representing the effect

    Returns:
        Causally-bound hypervector encoding cause → effect
    """
    primitives = ctx.get_primitives()
    causes_primitive = primitives["causes"]

    # Permute cause to break symmetry
    permuted_cause = ctx.permute(cause)

    # Bind: permute(cause) ⊗ causes ⊗ effect
    return ctx.normalize(permuted_cause * causes_primitive * effect)


def temporal_bind(
    ctx: AnalysisContext,
    event_before: torch.Tensor,
    event_after: torch.Tensor,
    relation: str = "before",  # "before", "after", "during"
) -> torch.Tensor:
    """
    Encode temporal relationships with direction.

    Uses permutation to encode temporal order, preserving the
    asymmetry of time (before ≠ after).

    Args:
        ctx: Analysis context
        event_before: The earlier event
        event_after: The later event
        relation: Type of temporal relation ("before", "after", "during")

    Returns:
        Bound temporal relationship vector
    """
    primitives = ctx.get_primitives()
    temporal_primitive = primitives.get(relation, primitives["before"])

    # Use permutation to encode order
    if relation == "before":
        return ctx.normalize(
            ctx.permute(event_before) * temporal_primitive * event_after
        )
    elif relation == "after":
        return ctx.normalize(
            event_before * temporal_primitive * ctx.permute(event_after)
        )
    else:  # during - symmetric
        return ctx.normalize(event_before * temporal_primitive * event_after)


def unbind_cause(
    ctx: AnalysisContext,
    causal_vector: torch.Tensor,
    known_cause: torch.Tensor,
) -> torch.Tensor:
    """
    Retrieve effect from causal binding given the cause.

    This is the "forward" direction of causal inference:
    Given that A caused something, what was the effect?

    Args:
        ctx: Analysis context
        causal_vector: The causally-bound vector
        known_cause: The known cause vector

    Returns:
        Approximate effect vector (clean if cause is correct)
    """
    primitives = ctx.get_primitives()
    causes_primitive = primitives["causes"]

    permuted_cause = ctx.permute(known_cause)
    return ctx.normalize(
        causal_vector * permuted_cause.conj() * causes_primitive.conj()
    )


def unbind_effect(
    ctx: AnalysisContext,
    causal_vector: torch.Tensor,
    known_effect: torch.Tensor,
) -> torch.Tensor:
    """
    Attempt to retrieve cause from effect - NOTE: This is intentionally noisy.

    The asymmetric binding means this returns a degraded signal,
    which is correct behavior for causal reasoning: you can't
    reliably infer cause from effect without additional information.

    This reflects the epistemological reality that correlation
    (observing an effect) doesn't imply causation (knowing the cause).

    Args:
        ctx: Analysis context
        causal_vector: The causally-bound vector
        known_effect: The known effect vector

    Returns:
        Noisy/degraded cause vector (intentionally unreliable)
    """
    primitives = ctx.get_primitives()
    causes_primitive = primitives["causes"]

    # This will be noisy because we can't cleanly un-permute
    return ctx.normalize(causal_vector * known_effect.conj() * causes_primitive.conj())


# =============================================================================
# MULTI-HOP REASONING WITH CLEANUP
# =============================================================================


def multi_hop_query(
    ctx: AnalysisContext,
    bundle: torch.Tensor,
    start_entity: str,
    relation_chain: list[str],
    min_confidence: float = 0.40,
) -> list[dict] | None:
    """
    Multi-hop reasoning with per-hop cleanup to prevent noise accumulation.

    This is the key protocol that makes multi-hop reasoning work:
    After each hop, we use the resonator to "clean up" the noisy
    result vector back to the nearest codebook entry, then continue
    from that clean symbol.

    WITHOUT cleanup: SNR degrades exponentially with each hop
    WITH cleanup: SNR remains constant per hop

    This is critical for reasoning chains longer than 2-3 hops.

    Args:
        ctx: Analysis context
        bundle: The bundled knowledge
        start_entity: Starting entity name
        relation_chain: List of relation primitives to traverse
        min_confidence: Minimum similarity to continue chain

    Returns:
        List of traversal steps with entities and confidence, or None if chain breaks

    Example:
        # Find: rain → causes → ? → causes → ?
        result = multi_hop_query(ctx, bundle, "rain", ["causes", "causes"])
        # Returns: [
        #   {"entity": "rain", "similarity": 1.0, "confidence_level": "start"},
        #   {"entity": "wet_ground", "similarity": 0.72, "confidence_level": "high"},
        #   {"entity": "slipping", "similarity": 0.65, "confidence_level": "high"},
        # ]
    """
    primitives = ctx.get_primitives()
    traversal = []

    # Start with clean entity vector
    current_vec = ctx.get_or_create(start_entity)
    traversal.append(
        {
            "entity": start_entity,
            "similarity": 1.0,
            "confidence_level": "start",
            "hop": 0,
        }
    )

    for hop_idx, relation in enumerate(relation_chain):
        relation_vec = primitives.get(relation)
        if relation_vec is None:
            logger.warning(f"Unknown relation: {relation}")
            return None

        # Step 1: Unbind current entity with relation to get query
        if relation == "causes":
            # Use asymmetric unbinding for causal relations
            query = unbind_cause(ctx, bundle, current_vec)
        else:
            # Standard symmetric unbind
            query = ctx.normalize(bundle * current_vec.conj() * relation_vec.conj())

        # Step 2: Run resonator to clean up noisy query
        query_batch = query.unsqueeze(0)
        cleaned = convergence_lock_resonator(ctx, query_batch)[0]

        # Step 3: Find nearest codebook entry (the "cleanup" step)
        codebook_tensor = ctx.get_codebook_tensor()
        if codebook_tensor is None:
            return traversal  # Can't continue without codebook

        similarities = torch.real(torch.conj(codebook_tensor) @ cleaned)
        best_idx = torch.argmax(similarities).item()
        best_similarity = similarities[best_idx].item()

        # Step 4: Validate confidence
        codebook_keys = ctx.get_codebook_keys()
        next_entity = codebook_keys[best_idx]
        next_vec = ctx.get_or_create(next_entity)

        is_valid, sim, confidence = ctx.validate_claim(bundle, next_vec, min_confidence)

        if not is_valid:
            logger.info(
                f"Chain broke at hop {hop_idx + 1}: "
                f"similarity {best_similarity:.3f} < threshold {min_confidence}"
            )
            traversal.append(
                {
                    "entity": None,
                    "similarity": best_similarity,
                    "confidence_level": "chain_broken",
                    "hop": hop_idx + 1,
                }
            )
            return traversal

        # Step 5: Record and continue from clean entity
        traversal.append(
            {
                "entity": next_entity,
                "similarity": best_similarity,
                "confidence_level": confidence,
                "hop": hop_idx + 1,
            }
        )

        # CRITICAL: Use clean codebook vector, not noisy query result
        current_vec = next_vec

    return traversal


# =============================================================================
# PROBABILISTIC SUPERPOSITION
# =============================================================================


@dataclass
class HypothesisState:
    """
    Probabilistic superposition of multiple hypotheses.

    In quantum mechanics, a particle can be in superposition of states.
    In GLM, a query can have multiple candidate answers with different
    confidence weights until "collapsed" by validation.

    This enables reasoning under uncertainty - we can maintain multiple
    possible interpretations until geometric evidence favors one.

    Attributes:
        hypotheses: List of (name, vector, confidence) tuples
        collapsed: Whether superposition has been collapsed
        collapsed_to: The winning hypothesis after collapse
    """

    hypotheses: list[tuple[str, torch.Tensor, float]] = field(default_factory=list)
    collapsed: bool = False
    collapsed_to: str | None = None

    def add_hypothesis(
        self, name: str, vector: torch.Tensor, confidence: float
    ) -> None:
        """Add a hypothesis to the superposition."""
        if self.collapsed:
            raise ValueError("Cannot add to collapsed state")
        self.hypotheses.append((name, vector, confidence))

    def get_superposition_vector(self, ctx: AnalysisContext) -> torch.Tensor:
        """
        Get weighted bundle of all hypotheses.

        Higher confidence hypotheses contribute more to the superposition.

        Args:
            ctx: Analysis context for normalization

        Returns:
            Weighted bundle of all hypothesis vectors
        """
        if not self.hypotheses:
            raise ValueError("No hypotheses in superposition")

        weighted_sum = sum(vec * conf for _, vec, conf in self.hypotheses)
        return ctx.normalize(weighted_sum)

    def collapse(
        self,
        ctx: AnalysisContext,
        bundle: torch.Tensor,
        threshold: float = 0.40,
    ) -> str | None:
        """
        Collapse superposition to single hypothesis via geometric validation.

        The hypothesis with highest geometric similarity to the bundle wins,
        but only if it exceeds the confidence threshold.

        Args:
            ctx: Analysis context
            bundle: Knowledge bundle to validate against
            threshold: Minimum confidence for acceptance

        Returns:
            Name of winning hypothesis, or None if all fail validation
        """
        if self.collapsed:
            return self.collapsed_to

        best_name = None
        best_score = -1.0

        for name, vec, prior_conf in self.hypotheses:
            is_valid, similarity, _ = ctx.validate_claim(bundle, vec, threshold)
            # Combine prior confidence with geometric validation
            combined_score = prior_conf * similarity

            if is_valid and combined_score > best_score:
                best_score = combined_score
                best_name = name

        self.collapsed = True
        self.collapsed_to = best_name
        return best_name


def p_sup_collapse(
    ctx: AnalysisContext,
    bundle: torch.Tensor,
    candidates: list[tuple[str, torch.Tensor]],
    threshold: float = 0.40,
) -> tuple[str | None, float, list[dict]]:
    """
    Probabilistic superposition collapse - choose best grounded hypothesis.

    Functional interface for hypothesis collapse without maintaining state.

    Args:
        ctx: Analysis context
        bundle: Knowledge bundle to validate against
        candidates: List of (name, vector) candidate hypotheses
        threshold: Minimum confidence for any hypothesis to be accepted

    Returns:
        Tuple of (winner_name, winner_confidence, all_scores)
        Returns (None, 0.0, scores) if no hypothesis exceeds threshold
    """
    scores = []

    for name, vec in candidates:
        is_valid, similarity, confidence_level = ctx.validate_claim(
            bundle, vec, threshold
        )
        scores.append(
            {
                "name": name,
                "similarity": similarity,
                "confidence_level": confidence_level,
                "is_valid": is_valid,
            }
        )

    # Sort by similarity descending
    scores.sort(key=lambda x: x["similarity"], reverse=True)

    # Return best valid hypothesis
    for score in scores:
        if score["is_valid"]:
            return (score["name"], score["similarity"], scores)

    return (None, 0.0, scores)


# =============================================================================
# FORWARD AND BACKWARD CHAINING
# =============================================================================


def forward_chain(
    ctx: AnalysisContext,
    bundle: torch.Tensor,
    facts: list[str],
    rules: list[tuple[list[str], str]],  # (antecedents, consequent)
    max_iterations: int = 10,
    min_confidence: float = 0.40,
) -> list[dict]:
    """
    Forward chaining inference - derive new facts from existing facts and rules.

    This implements classic forward chaining from symbolic AI, but with
    geometric validation at each step. New facts are only derived if
    the inference is geometrically grounded in the bundle.

    Args:
        ctx: Analysis context
        bundle: Knowledge bundle
        facts: List of known fact entity names
        rules: List of (antecedent_list, consequent) rules
        max_iterations: Maximum inference iterations (prevents infinite loops)
        min_confidence: Minimum confidence to accept derived fact

    Returns:
        List of derived facts with confidence levels

    Example:
        facts = ["rain"]
        rules = [
            (["rain"], "wet_ground"),
            (["wet_ground"], "slipping_hazard"),
        ]
        derived = forward_chain(ctx, bundle, facts, rules)
        # Returns: [
        #   {"fact": "wet_ground", "confidence": 0.72, "derivation": ["rain"]},
        #   {"fact": "slipping_hazard", "confidence": 0.68, "derivation": ["wet_ground"]},
        # ]
    """
    primitives = ctx.get_primitives()
    implies_vec = primitives["implies"]

    known_facts = set(facts)
    derived = []

    for iteration in range(max_iterations):
        new_facts_this_round = []

        for antecedents, consequent in rules:
            # Skip if consequent already known
            if consequent in known_facts:
                continue

            # Check if all antecedents are known
            if not all(ant in known_facts for ant in antecedents):
                continue

            # Build implication encoding
            antecedent_bundle = ctx.get_or_create(antecedents[0])
            for ant in antecedents[1:]:
                antecedent_bundle = ctx.normalize(
                    antecedent_bundle * ctx.get_or_create(ant)
                )

            # Query: check if implication exists in bundle
            consequent_vec = ctx.get_or_create(consequent)
            rule_vec = ctx.normalize(implies_vec * antecedent_bundle * consequent_vec)

            # Validate this derivation against the bundle
            is_valid, similarity, confidence = ctx.validate_claim(
                bundle, rule_vec, min_confidence
            )

            if is_valid:
                new_facts_this_round.append(
                    {
                        "fact": consequent,
                        "similarity": similarity,
                        "confidence_level": confidence,
                        "derivation": antecedents,
                        "iteration": iteration,
                    }
                )
                known_facts.add(consequent)

        if not new_facts_this_round:
            break  # Fixed point reached

        derived.extend(new_facts_this_round)

    return derived


def backward_chain(
    ctx: AnalysisContext,
    bundle: torch.Tensor,
    goal: str,
    rules: list[tuple[list[str], str]],
    known_facts: set[str],
    min_confidence: float = 0.40,
    max_depth: int = 5,
    _depth: int = 0,
) -> tuple[bool, list[str], float]:
    """
    Backward chaining inference - find facts needed to prove a goal.

    Works backwards from the goal to find what premises are needed
    and whether they can be established from known facts.

    Args:
        ctx: Analysis context
        bundle: Knowledge bundle
        goal: Goal to prove
        rules: List of (antecedent_list, consequent) rules
        known_facts: Set of already known facts
        min_confidence: Minimum confidence for proofs
        max_depth: Maximum recursion depth
        _depth: Current recursion depth (internal)

    Returns:
        Tuple of (proven, proof_path, confidence)
    """
    # Base case: goal is already known
    if goal in known_facts:
        return (True, [goal], 1.0)

    # Depth limit
    if _depth >= max_depth:
        return (False, [], 0.0)

    # Find rules that conclude the goal
    for antecedents, consequent in rules:
        if consequent != goal:
            continue

        # Try to prove all antecedents
        all_proven = True
        proof_path = []
        min_conf = 1.0

        for ant in antecedents:
            proven, sub_path, conf = backward_chain(
                ctx,
                bundle,
                ant,
                rules,
                known_facts,
                min_confidence,
                max_depth,
                _depth + 1,
            )

            if not proven:
                all_proven = False
                break

            proof_path.extend(sub_path)
            min_conf = min(min_conf, conf)

        if all_proven:
            # Validate the final derivation
            goal_vec = ctx.get_or_create(goal)
            is_valid, similarity, _ = ctx.validate_claim(
                bundle, goal_vec, min_confidence
            )

            if is_valid:
                proof_path.append(goal)
                return (True, proof_path, min(min_conf, similarity))

    return (False, [], 0.0)


# =============================================================================
# ANALOGY SOLVING
# =============================================================================


def solve_analogy(
    ctx: AnalysisContext,
    bundle: torch.Tensor,
    a: str,
    b: str,
    c: str,
    top_k: int = 5,
    min_confidence: float = 0.40,
) -> list[dict] | None:
    """
    Solve analogy: A is to B as C is to ?

    Uses the classic VSA analogy formula:
        ? = C ⊗ B ⊗ A*

    Where A* is the conjugate (inverse for binding).

    This is one of the remarkable properties of VSA: analogies emerge
    naturally from the algebra of binding and unbinding.

    Args:
        ctx: Analysis context
        bundle: Knowledge bundle (optional context for validation)
        a, b, c: Analogy terms (a:b :: c:?)
        top_k: Number of candidates to return
        min_confidence: Minimum confidence threshold

    Returns:
        List of candidate answers with confidence, or None if no valid answers

    Example:
        # king:queen :: man:?
        result = solve_analogy(ctx, bundle, "king", "queen", "man")
        # Returns: [{"entity": "woman", "similarity": 0.73, ...}]
    """
    a_vec = ctx.get_or_create(a)
    b_vec = ctx.get_or_create(b)
    c_vec = ctx.get_or_create(c)

    # Analogy formula: ? = c ⊗ b ⊗ a*
    query = ctx.normalize(c_vec * b_vec * a_vec.conj())

    # Clean up via resonator
    query_batch = query.unsqueeze(0)
    cleaned = convergence_lock_resonator(ctx, query_batch)[0]

    # Find nearest codebook entries
    codebook_tensor = ctx.get_codebook_tensor()
    if codebook_tensor is None:
        return None

    similarities = torch.real(torch.conj(codebook_tensor) @ cleaned)
    topk_vals, topk_idx = torch.topk(similarities, min(top_k * 2, len(similarities)))

    codebook_keys = ctx.get_codebook_keys()

    results = []
    for idx, sim in zip(topk_idx.tolist(), topk_vals.tolist()):
        entity = codebook_keys[idx]

        # Skip input terms
        if entity in [a, b, c]:
            continue

        entity_vec = ctx.get_or_create(entity)
        is_valid, similarity, confidence = ctx.validate_claim(
            bundle, entity_vec, min_confidence
        )

        if is_valid:
            results.append(
                {
                    "entity": entity,
                    "similarity": similarity,
                    "confidence_level": confidence,
                }
            )

        if len(results) >= top_k:
            break

    return results if results else None


# =============================================================================
# GLM INTERFACE (LLM ↔ VSA Communication Protocol)
# =============================================================================


@dataclass
class GLMRequest:
    """
    Structured request from LLM to GLM.

    The LLM proposes operations; the GLM executes and validates.
    This provides a clean interface for LLM grounding.

    Attributes:
        operation: Type of operation ("query", "validate", "chain", "analogy", "derive")
        parameters: Operation-specific parameters
        require_grounding: Whether to enforce geometric validation
        min_confidence: Minimum similarity for acceptance
    """

    operation: str  # "query", "validate", "chain", "analogy", "derive"
    parameters: dict
    require_grounding: bool = True
    min_confidence: float = 0.40


@dataclass
class GLMResponse:
    """
    Structured response from GLM to LLM.

    Includes geometric validation results so LLM knows
    what claims are grounded vs. speculation.

    Attributes:
        success: Whether operation succeeded
        results: Operation results (if any)
        confidence_summary: Summary of confidence levels
        grounding_failures: Claims that failed validation
        metadata: Additional operation metadata
    """

    success: bool
    results: list[dict] | None
    confidence_summary: dict
    grounding_failures: list[dict]
    metadata: dict


class GLMInterface:
    """
    LLM ↔ GLM Grounding Interface.

    This is the core of the Geometric Language Model architecture:

        User Query → LLM (parse intent) → GLMInterface → VSA (geometric reasoning)
                                               ↓
        Response ← LLM (interpret) ← GLMInterface ← Validation Gate

    The LLM cannot bypass geometric validation. All claims must
    pass through the grounding interface.

    This ensures that:
    1. Invalid claims are rejected at the geometry level
    2. All grounding decisions are logged for audit
    3. The LLM receives clear signals about what is supported vs. speculative

    Attributes:
        ctx: Analysis context
        bundle: Knowledge bundle
        grounding_log: Audit log of all grounding decisions
    """

    def __init__(self, ctx: AnalysisContext, bundle: torch.Tensor):
        """
        Initialize GLM interface.

        Args:
            ctx: Analysis context with populated codebook
            bundle: Bundled knowledge hypervector
        """
        self.ctx = ctx
        self.bundle = bundle
        self.grounding_log: list[dict] = []

    def process_request(self, request: GLMRequest) -> GLMResponse:
        """
        Process LLM request through geometric validation.

        This is the enforcement point - the LLM proposes,
        the geometry validates.

        Args:
            request: Structured GLM request

        Returns:
            GLM response with validation results
        """
        try:
            if request.operation == "query":
                return self._handle_query(request)
            elif request.operation == "validate":
                return self._handle_validate(request)
            elif request.operation == "chain":
                return self._handle_chain(request)
            elif request.operation == "analogy":
                return self._handle_analogy(request)
            elif request.operation == "derive":
                return self._handle_derive(request)
            else:
                return GLMResponse(
                    success=False,
                    results=None,
                    confidence_summary={
                        "error": f"Unknown operation: {request.operation}"
                    },
                    grounding_failures=[],
                    metadata={},
                )
        except Exception as e:
            logger.error(f"GLM request failed: {e}")
            return GLMResponse(
                success=False,
                results=None,
                confidence_summary={"error": str(e)},
                grounding_failures=[],
                metadata={},
            )

    def _handle_query(self, request: GLMRequest) -> GLMResponse:
        """Handle primitive query with grounding."""
        primitive = request.parameters.get("primitive")
        top_k = request.parameters.get("top_k", 20)

        if request.require_grounding:
            results = self.ctx.grounded_query(
                self.bundle, primitive, request.min_confidence, top_k
            )
        else:
            items, scores = query_bundle(self.ctx, self.bundle, primitive, top_k)
            results = [{"entity": e, "similarity": s} for e, s in zip(items, scores)]

        # Log for audit trail
        self.grounding_log.append(
            {
                "operation": "query",
                "primitive": primitive,
                "results_count": len(results) if results else 0,
                "grounded": request.require_grounding,
            }
        )

        return GLMResponse(
            success=results is not None,
            results=results,
            confidence_summary=self._summarize_confidence(results),
            grounding_failures=[],
            metadata={"primitive": primitive},
        )

    def _handle_validate(self, request: GLMRequest) -> GLMResponse:
        """Handle claim validation."""
        claim_entity = request.parameters.get("claim")
        claim_vec = self.ctx.get_or_create(claim_entity)

        is_valid, similarity, confidence = self.ctx.validate_claim(
            self.bundle, claim_vec, request.min_confidence
        )

        result = {
            "claim": claim_entity,
            "is_valid": is_valid,
            "similarity": similarity,
            "confidence_level": confidence,
        }

        grounding_failures = [] if is_valid else [result]

        self.grounding_log.append(
            {
                "operation": "validate",
                "claim": claim_entity,
                "is_valid": is_valid,
                "similarity": similarity,
            }
        )

        return GLMResponse(
            success=True,
            results=[result],
            confidence_summary={"validation": confidence},
            grounding_failures=grounding_failures,
            metadata={},
        )

    def _handle_chain(self, request: GLMRequest) -> GLMResponse:
        """Handle multi-hop chain query."""
        start = request.parameters.get("start")
        relations = request.parameters.get("relations", [])

        results = multi_hop_query(
            self.ctx, self.bundle, start, relations, request.min_confidence
        )

        # Check for chain breaks
        grounding_failures = [
            r for r in (results or []) if r.get("confidence_level") == "chain_broken"
        ]

        self.grounding_log.append(
            {
                "operation": "chain",
                "start": start,
                "relations": relations,
                "chain_length": len(results) - 1 if results else 0,
                "broken": bool(grounding_failures),
            }
        )

        return GLMResponse(
            success=results is not None and not grounding_failures,
            results=results,
            confidence_summary=self._summarize_confidence(results),
            grounding_failures=grounding_failures,
            metadata={"chain_length": len(relations)},
        )

    def _handle_analogy(self, request: GLMRequest) -> GLMResponse:
        """Handle analogy solving."""
        a = request.parameters.get("a")
        b = request.parameters.get("b")
        c = request.parameters.get("c")
        top_k = request.parameters.get("top_k", 5)

        results = solve_analogy(
            self.ctx, self.bundle, a, b, c, top_k, request.min_confidence
        )

        self.grounding_log.append(
            {
                "operation": "analogy",
                "analogy": f"{a}:{b}::{c}:?",
                "results_count": len(results) if results else 0,
            }
        )

        return GLMResponse(
            success=results is not None,
            results=results,
            confidence_summary=self._summarize_confidence(results),
            grounding_failures=[],
            metadata={"analogy": f"{a}:{b}::{c}:?"},
        )

    def _handle_derive(self, request: GLMRequest) -> GLMResponse:
        """Handle forward chaining derivation."""
        facts = request.parameters.get("facts", [])
        rules = request.parameters.get("rules", [])

        results = forward_chain(
            self.ctx, self.bundle, facts, rules, min_confidence=request.min_confidence
        )

        self.grounding_log.append(
            {
                "operation": "derive",
                "initial_facts": len(facts),
                "rules": len(rules),
                "derived_facts": len(results),
            }
        )

        return GLMResponse(
            success=True,
            results=results,
            confidence_summary=self._summarize_confidence(results),
            grounding_failures=[],
            metadata={"initial_facts": len(facts), "rules": len(rules)},
        )

    def _summarize_confidence(self, results: list[dict] | None) -> dict:
        """Summarize confidence levels across results."""
        if not results:
            return {"total": 0, "distribution": {}}

        distribution = {}
        total_similarity = 0.0

        for r in results:
            level = r.get("confidence_level", "unknown")
            distribution[level] = distribution.get(level, 0) + 1
            total_similarity += r.get("similarity", 0)

        return {
            "total": len(results),
            "distribution": distribution,
            "avg_similarity": total_similarity / len(results) if results else 0,
        }

    def get_grounding_audit(self) -> list[dict]:
        """Return audit log of all grounding decisions."""
        return self.grounding_log.copy()


# =============================================================================
# DIRAC VSA CAUSAL OPERATIONS
# =============================================================================


def dirac_causal_bind(
    ctx: AnalysisContext,
    cause_name: str,
    effect_name: str,
) -> DiracVector:
    """
    Create a causal binding: cause → effect using Dirac VSA.

    This creates an asymmetric binding where:
    - unbind_cause(result, cause) cleanly recovers effect
    - unbind_effect(result, effect) is intentionally noisy

    This captures the fundamental asymmetry of causation:
    knowing the cause lets you predict the effect,
    but knowing the effect doesn't cleanly determine the cause.

    Args:
        ctx: AnalysisContext with configured dimensions/device
        cause_name: Name of the cause entity (will be hashed)
        effect_name: Name of the effect entity (will be hashed)

    Returns:
        DiracVector representing cause → effect binding

    Example:
        >>> ctx = create_analysis_context()
        >>> binding = dirac_causal_bind(ctx, "price_drop", "margin_erosion")
        >>> # Can cleanly recover effect from cause
        >>> dvsa = ctx.get_dirac_vsa()
        >>> cause = dvsa.seed_hash("price_drop")
        >>> recovered_effect = dvsa.unbind_cause(binding, cause)
    """
    from .context import DiracVector

    dvsa = ctx.get_dirac_vsa()
    cause = dvsa.seed_hash(cause_name)
    effect = dvsa.seed_hash(effect_name)

    return dvsa.bind(cause, effect, symmetric=False)


def dirac_multi_hop_query(
    ctx: AnalysisContext,
    start_entity: str,
    causal_chain: list[tuple[str, str]],
) -> list[DiracVector]:
    """
    Execute multi-hop causal reasoning via Dirac VSA.

    Given a starting entity and a chain of cause→effect relationships,
    traces through the chain to find all intermediate states.

    This is the Dirac equivalent of multi-hop reasoning:
    A → B → C can be queried by:
    1. bind(A, B) gives A→B
    2. unbind_cause(A→B, A) gives B
    3. bind(B, C) gives B→C
    4. unbind_cause(B→C, B) gives C

    Args:
        ctx: AnalysisContext with configured dimensions/device
        start_entity: Name of the starting entity
        causal_chain: List of (cause, effect) tuples defining the chain

    Returns:
        List of DiracVectors for each hop in the chain

    Example:
        >>> ctx = create_analysis_context()
        >>> # Trace: supplier_delay → stockout → lost_sales → margin_loss
        >>> chain = [
        ...     ("supplier_delay", "stockout"),
        ...     ("stockout", "lost_sales"),
        ...     ("lost_sales", "margin_loss"),
        ... ]
        >>> hops = dirac_multi_hop_query(ctx, "supplier_delay", chain)
        >>> # hops[0] = supplier_delay → stockout binding
        >>> # hops[1] = stockout → lost_sales binding
        >>> # hops[2] = lost_sales → margin_loss binding
    """
    from .context import DiracVector

    dvsa = ctx.get_dirac_vsa()
    results = []

    current_entity = dvsa.seed_hash(start_entity)

    for cause_name, effect_name in causal_chain:
        cause = dvsa.seed_hash(cause_name)
        effect = dvsa.seed_hash(effect_name)

        # Create causal binding
        binding = dvsa.bind(cause, effect, symmetric=False)
        results.append(binding)

        # Recover effect to continue chain
        recovered = dvsa.unbind_cause(binding, cause)
        current_entity = recovered

    return results


def dirac_validate_causal_claim(
    ctx: AnalysisContext,
    proposed_cause: str,
    proposed_effect: str,
    known_bindings: list[DiracVector],
    threshold: float = 0.5,
) -> tuple[bool, float, str]:
    """
    Validate whether a proposed causal relationship is supported.

    This is the Dirac VSA equivalent of GLM validation:
    checks whether the geometry supports the claimed causation.

    Args:
        ctx: AnalysisContext with configured dimensions/device
        proposed_cause: Name of the proposed cause
        proposed_effect: Name of the proposed effect
        known_bindings: List of known causal DiracVectors to check against
        threshold: Minimum similarity for validation

    Returns:
        Tuple of (is_valid, similarity, confidence_level)

    Example:
        >>> ctx = create_analysis_context()
        >>> known = [dirac_causal_bind(ctx, "overstock", "dead_inventory")]
        >>> is_valid, sim, conf = dirac_validate_causal_claim(
        ...     ctx, "overstock", "dead_inventory", known
        ... )
        >>> # Should be valid since this exact relationship is known
    """
    from .context import VALIDATION_THRESHOLDS

    dvsa = ctx.get_dirac_vsa()

    # Create the proposed binding
    proposed = dvsa.bind(
        dvsa.seed_hash(proposed_cause),
        dvsa.seed_hash(proposed_effect),
        symmetric=False,
    )

    # Check similarity against all known bindings
    max_similarity = 0.0
    for known in known_bindings:
        sim = dvsa.similarity(proposed, known)
        max_similarity = max(max_similarity, sim)

    # Determine confidence level
    if max_similarity < VALIDATION_THRESHOLDS["rejection_threshold"]:
        return (False, max_similarity, "rejected")
    elif max_similarity < VALIDATION_THRESHOLDS["low_confidence"]:
        return (False, max_similarity, "noise_floor")
    elif max_similarity < VALIDATION_THRESHOLDS["moderate_confidence"]:
        return (max_similarity >= threshold, max_similarity, "low_confidence")
    elif max_similarity < VALIDATION_THRESHOLDS["high_confidence"]:
        return (True, max_similarity, "moderate_confidence")
    elif max_similarity < VALIDATION_THRESHOLDS["very_high_confidence"]:
        return (True, max_similarity, "high_confidence")
    else:
        return (True, max_similarity, "very_high_confidence")


def dirac_entropy_audit(
    ctx: AnalysisContext,
    operations: list[DiracVector],
) -> dict:
    """
    Audit entropy accumulation across a series of Dirac operations.

    Useful for verifying that entropy is monotonically increasing
    and tracking information loss through a reasoning chain.

    Args:
        ctx: AnalysisContext (for configuration)
        operations: List of DiracVectors from sequential operations

    Returns:
        Dict with entropy statistics:
        - entropies: List of entropy values
        - is_monotonic: Whether entropy never decreases
        - total_increase: Total entropy accumulated
        - avg_per_step: Average entropy increase per operation

    Example:
        >>> ctx = create_analysis_context()
        >>> dvsa = ctx.get_dirac_vsa()
        >>> a = dvsa.seed_hash("A")  # entropy = 0
        >>> b = dvsa.seed_hash("B")  # entropy = 0
        >>> ab = dvsa.bind(a, b)     # entropy > 0
        >>> c = dvsa.seed_hash("C")
        >>> abc = dvsa.bind(ab, c)   # entropy > ab.entropy
        >>> audit = dirac_entropy_audit(ctx, [a, b, ab, c, abc])
    """
    if not operations:
        return {
            "entropies": [],
            "is_monotonic": True,
            "total_increase": 0.0,
            "avg_per_step": 0.0,
        }

    entropies = [op.entropy for op in operations]

    # Check monotonicity
    is_monotonic = all(
        entropies[i] <= entropies[i + 1] for i in range(len(entropies) - 1)
    )

    total_increase = entropies[-1] - entropies[0] if entropies else 0.0
    avg_per_step = total_increase / (len(entropies) - 1) if len(entropies) > 1 else 0.0

    return {
        "entropies": entropies,
        "is_monotonic": is_monotonic,
        "total_increase": total_increase,
        "avg_per_step": avg_per_step,
    }


# =============================================================================
# BACKWARD COMPATIBILITY LAYER
# =============================================================================
# These functions maintain the old API signature for gradual migration.
# They create a temporary context internally. NOT RECOMMENDED for production.
# Will be removed in v3.0.

_COMPAT_WARNING_SHOWN = False


def _show_compat_warning():
    """Show deprecation warning once."""
    global _COMPAT_WARNING_SHOWN
    if not _COMPAT_WARNING_SHOWN:
        logger.warning(
            "Using deprecated global-state API. Please migrate to context-based API: "
            "ctx = create_analysis_context(); bundle_pos_facts(ctx, rows)"
        )
        _COMPAT_WARNING_SHOWN = True


# Legacy global for backwards compatibility - DO NOT USE IN NEW CODE
# This is only here to prevent immediate breakage of existing code
codebook_dict = None  # Placeholder - actual codebook is in context


def reset_codebook():
    """
    Legacy function - now a no-op.

    In the new API, contexts are garbage collected automatically.
    This function exists only for backward compatibility.
    """
    _show_compat_warning()
    logger.info("reset_codebook() called - this is now a no-op with context-based API")


# Legacy seed_hash for compatibility
def seed_hash(string: str, d: int = 16384) -> torch.Tensor:
    """
    Legacy seed_hash - creates ephemeral context.

    DEPRECATED: Use ctx.seed_hash() instead.
    """
    _show_compat_warning()
    from .context import create_analysis_context

    ctx = create_analysis_context(dimensions=d, use_gpu=True)
    return ctx.seed_hash(string)


def normalize_torch(v: torch.Tensor) -> torch.Tensor:
    """
    Legacy normalize - stateless so still safe.

    Provided for backward compatibility.
    """
    norm = torch.norm(v, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-8)
    return v / norm


def add_to_codebook(entity: str, d: int = 16384):
    """
    Legacy add_to_codebook - now a no-op with warning.

    DEPRECATED: Use ctx.add_to_codebook() instead.
    """
    _show_compat_warning()
    logger.warning(f"add_to_codebook('{entity}') called - this is now a no-op")


def convergence_lock_resonator_gpu(
    unbound_batch: torch.Tensor,
    iters: int = 450,
    alpha: float = 0.85,
    power: float = 0.64,
) -> torch.Tensor:
    """
    Legacy resonator - creates ephemeral context.

    DEPRECATED: Use convergence_lock_resonator(ctx, unbound_batch) instead.
    """
    _show_compat_warning()
    # This won't work properly without a real codebook,
    # but we provide it for API compatibility
    return unbound_batch


# Note: PRIMITIVES is now generated per-context, but we provide a module-level
# reference for backward compatibility with code that imports PRIMITIVES directly.
# This will be a dict of None values - actual vectors come from context.
PRIMITIVES = {
    # Original 8 primitives
    "low_stock": None,
    "high_margin_leak": None,
    "dead_item": None,
    "negative_inventory": None,
    "overstock": None,
    "price_discrepancy": None,
    "shrinkage_pattern": None,
    "margin_erosion": None,
    # New primitives (v2.2)
    "zero_cost_anomaly": None,
    "negative_profit": None,
    "severe_inventory_deficit": None,
    # Utility primitives
    "high_velocity": None,
    "seasonal": None,
}
