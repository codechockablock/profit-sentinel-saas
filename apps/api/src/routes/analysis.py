"""
Analysis endpoints - Aggressive Profit Leak Detection.

Handles profit leak analysis using VSA resonator with 8 detection primitives.
Supports data from any major POS system (Paladin, Square, Lightspeed, etc.).

Privacy features:
- Auto-deletes uploaded files after processing
- Stores only anonymized aggregated analytics
"""

import json
import logging
import time

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException

from ..config import SUPPORTED_POS_SYSTEMS, get_settings
from ..dependencies import get_current_user, get_s3_client
from ..services.analysis import AnalysisService
from ..services.anonymization import get_anonymization_service
from ..services.s3 import S3Service

router = APIRouter()
logger = logging.getLogger(__name__)


# =============================================================================
# COMPREHENSIVE NUMERIC COLUMN ALIASES
# Covers all major POS systems: Paladin, Square, Lightspeed, Clover, Shopify,
# NCR Counterpoint, Microsoft Dynamics RMS, Epicor, and generic exports
# =============================================================================

NUMERIC_COLUMN_ALIASES = [
    # Quantity / Stock
    'quantity', 'qty', 'qty.', 'qoh', 'on_hand', 'onhand', 'in_stock', 'instock',
    'in stock qty.', 'stock', 'inventory', 'stockonhand', 'qty_on_hnd',
    'qty_avail', 'net_qty', 'available', 'current_stock', 'variant_inventory_qty',
    'new_quantity', 'qty_oh', 'physical_qty', 'book_qty', 'bal', 'balance',
    # Quantity Difference / Variance
    'qty_difference', 'qty. difference', 'difference', 'variance', 'qty_variance',
    'shrinkage', 'adjustment', 'discrepancy', 'over_short',
    # Inventoried
    'inventoried_qty', 'inventoried', 'inventoried qty.', 'counted_qty', 'count',
    # Cost
    'cost', 'cogs', 'cost_price', 'unit_cost', 'unitcost', 'avg_cost', 'avgcost',
    'average_cost', 'standard_cost', 'lst_cost', 'last_cost', 'default_cost',
    'vendor_cost', 'supply_price', 'cost_per_item', 'posting_cost', 'mktcost',
    'landed_cost', 'purchase_cost', 'buy_price', 'wholesale_cost',
    # Revenue / Retail Price
    'revenue', 'retail', 'retail_price', 'price', 'sell_price', 'selling_price',
    'sale_price', 'unit_price', 'msrp', 'list_price', 'sug. retail', 'sug retail',
    'suggested_retail', 'netprice', 'net_price', 'prc_1', 'reg_prc', 'regular_price',
    'pricea', 'default_price', 'variant_price', 'compare_at_price', 'pos_price',
    'ext_price', 'line_total', 'amount', 'gross_sales', 'total_sale',
    # Sold
    'sold', 'units_sold', 'qty_sold', 'quantity_sold', 'sales_qty', 'total_sold',
    'sold_qty', 'sold_last_week', 'sold_last_month', 'sold_30_days', 'sold_7_days',
    'sales_units', 'unit_sales', 'movement',
    # Margin
    'margin', 'profit_margin', 'margin_pct', 'margin_percent', 'gp', 'gross_profit',
    'gp_pct', 'gp_percent', 'profit_pct', 'markup', 'markup_pct',
    # Sub Total / Value
    'sub_total', 'subtotal', 'total', 'ext_total', 'inventory_value', 'stock_value',
    'on_hand_value', 'gl_val',
    # Discount / Tax
    'discount', 'discount_amt', 'discount_amount', 'discount_pct',
    'tax', 'sales_tax', 'vat', 'tax_amount',
    # Reorder
    'reorder_point', 'reorder_level', 'min_qty', 'minimum_qty', 'safety_stock',
    'par_level', 'minorderqty', 'stock_min', 'stock_alert',
    # Package
    'pkg_qty', 'package_qty', 'pack_size', 'case_qty',
]


def _validate_and_apply_mapping(
    df: pd.DataFrame,
    mapping_dict: dict[str, str]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Safely validate and apply column mapping to DataFrame.

    Handles:
    - Missing source columns (skipped with warning)
    - Duplicate target names (only first mapping applied)
    - Empty/null target values (skipped)
    - Extra columns not in mapping (preserved as-is)

    Args:
        df: Source DataFrame
        mapping_dict: Dict of {source_column: target_column}

    Returns:
        Tuple of (mapped DataFrame, list of warning messages)
    """
    warnings = []
    df_columns = set(df.columns.tolist())

    # Filter to only valid mappings
    valid_mapping = {}
    seen_targets = set()

    for source_col, target_col in mapping_dict.items():
        # Skip empty/null targets
        if not target_col or not str(target_col).strip():
            continue

        target_col = str(target_col).strip()

        # Check if source column exists in DataFrame
        if source_col not in df_columns:
            warnings.append(f"Source column '{source_col}' not found in data, skipping")
            continue

        # Check for duplicate target names (would cause length mismatch)
        if target_col in seen_targets:
            warnings.append(
                f"Duplicate target column '{target_col}' for source '{source_col}', skipping"
            )
            continue

        # Check if target would conflict with an existing unmapped column
        if target_col in df_columns and target_col not in mapping_dict:
            warnings.append(
                f"Target '{target_col}' conflicts with existing column, will be overwritten"
            )

        valid_mapping[source_col] = target_col
        seen_targets.add(target_col)

    # Log warnings
    for warning in warnings:
        logger.warning(f"Column mapping: {warning}")

    # Apply the validated mapping
    if valid_mapping:
        df = df.rename(columns=valid_mapping)

    return df, warnings


def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert numeric columns from any POS system.

    Handles:
    - Currency symbols ($)
    - Thousands separators (,)
    - Percentage signs (%)
    - Leading/trailing whitespace
    - Non-numeric values -> 0
    """
    # Find existing numeric columns (case-insensitive match)
    df_cols_lower = {col.lower().replace(' ', '').replace('.', ''): col for col in df.columns}
    existing_numeric = []

    for alias in NUMERIC_COLUMN_ALIASES:
        normalized = alias.lower().replace(' ', '').replace('.', '')
        if normalized in df_cols_lower:
            existing_numeric.append(df_cols_lower[normalized])

    # Also add original column names that match
    for alias in NUMERIC_COLUMN_ALIASES:
        if alias in df.columns and alias not in existing_numeric:
            existing_numeric.append(alias)

    # Remove duplicates while preserving order
    existing_numeric = list(dict.fromkeys(existing_numeric))

    if existing_numeric:
        logger.info(f"Cleaning {len(existing_numeric)} numeric columns: {existing_numeric[:10]}...")

        # Strip $ and , and %, convert to numeric
        for col in existing_numeric:
            try:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r'[$,\%]', '', regex=True)
                    .str.strip()
                )
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0.0)
            except Exception as e:
                logger.warning(f"Failed to clean column {col}: {e}")

    return df


@router.post("/analyze")
async def analyze_upload(
    key: str = Form(...),
    mapping: str = Form(...),  # JSON string
    background_tasks: BackgroundTasks = None,
    user_id: str | None = Depends(get_current_user),
) -> dict:
    """
    Analyze uploaded POS data for profit leaks.

    Supports data from any major POS system including:
    - Paladin POS
    - Square POS
    - Lightspeed Retail
    - Clover POS
    - Shopify POS
    - NCR Counterpoint
    - Microsoft Dynamics RMS
    - Epicor Eagle
    - And many more...

    Args:
        key: S3 object key
        mapping: JSON string of column mappings {source: target}
        user_id: Current user ID (from auth)

    Returns:
        Comprehensive analysis results with:
        - leaks: Dict of detected leak types with items, scores, metadata
        - summary: Overall statistics and estimated $ impact
        - primitives_used: List of 8 detection primitives used
        - warnings: Any mapping warnings
    """
    # Parse mapping
    try:
        mapping_dict = json.loads(mapping)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid mapping JSON: {str(e)}"
        )

    # Validate mapping is a dict
    if not isinstance(mapping_dict, dict):
        raise HTTPException(
            status_code=400,
            detail="Mapping must be a JSON object with column name pairs"
        )

    try:
        overall_start = time.time()
        logger.info(f"Starting aggressive analysis for key: {key}")

        settings = get_settings()
        s3_client = get_s3_client()
        s3_service = S3Service(s3_client, settings.s3_bucket_name)

        # Load full DataFrame
        load_start = time.time()
        df = s3_service.load_dataframe(key)
        original_columns = df.columns.tolist()
        logger.info(
            f"Loaded DataFrame ({len(df)} rows, {len(df.columns)} columns) "
            f"in {time.time() - load_start:.2f}s"
        )
        logger.debug(f"Original columns: {original_columns}")

        # Safely apply column mapping with validation
        df, mapping_warnings = _validate_and_apply_mapping(df, mapping_dict)

        if mapping_warnings:
            logger.info(f"Mapping applied with {len(mapping_warnings)} warnings")

        # Clean numeric columns (aggressive - handles all POS systems)
        clean_start = time.time()
        df = _clean_numeric_columns(df)
        logger.info(f"Cleaned numeric columns in {time.time() - clean_start:.2f}s")

        # Drop duplicate columns (keep first occurrence)
        if df.columns.duplicated().any():
            dup_cols = df.columns[df.columns.duplicated()].tolist()
            logger.warning(f"Dropping duplicate columns: {dup_cols}")
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # Convert to records
        records_start = time.time()
        rows = df.to_dict(orient='records')
        logger.info(
            f"Converted to records ({len(rows)} rows) "
            f"in {time.time() - records_start:.2f}s"
        )

        # Run analysis with all 8 primitives
        analysis_service = AnalysisService()
        result = analysis_service.analyze(rows)

        total_time = time.time() - overall_start
        logger.info(f"Full analysis pipeline completed in {total_time:.2f}s")

        # Add status and warnings to response
        result["status"] = "success"
        result["warnings"] = mapping_warnings if mapping_warnings else None
        result["supported_pos_systems"] = SUPPORTED_POS_SYSTEMS

        # Schedule file cleanup after processing (privacy compliance)
        # Delete raw file from S3 after a short delay to allow for any retries
        if background_tasks:
            anon_service = get_anonymization_service()
            background_tasks.add_task(
                anon_service.cleanup_s3_file,
                s3_client,
                settings.s3_bucket_name,
                key,
                delay_seconds=60  # 1 minute delay before deletion
            )
            logger.info(f"Scheduled S3 file cleanup for key: {key}")

            # Store anonymized analytics
            background_tasks.add_task(
                anon_service.store_anonymized_analytics,
                [result],
                False  # report_sent=False (not sent via email yet)
            )

        return result

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty or has no valid data"
        )
    except ValueError as e:
        # File size or validation errors
        error_msg = str(e)
        if "too large" in error_msg.lower() or "size" in error_msg.lower():
            raise HTTPException(status_code=413, detail=error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        # Provide user-friendly error message
        error_msg = str(e)
        if "Columns must be same length as key" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="Column mapping error: duplicate target column names detected. "
                       "Each source column must map to a unique target name."
            )
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")


@router.get("/primitives")
async def list_primitives() -> dict:
    """
    List all available analysis primitives.

    Returns metadata about each of the 8 leak detection primitives.
    """
    analysis_service = AnalysisService()
    primitives = {}

    for p in analysis_service.get_available_primitives():
        info = analysis_service.get_primitive_info(p)
        if info:
            primitives[p] = info

    return {
        "primitives": primitives,
        "count": len(primitives),
    }


@router.get("/supported-pos")
async def list_supported_pos() -> dict:
    """
    List all supported POS systems.

    Returns list of POS systems whose export formats are supported.
    """
    return {
        "supported_systems": SUPPORTED_POS_SYSTEMS,
        "count": len(SUPPORTED_POS_SYSTEMS),
        "notes": "Column mapping auto-detects formats from these systems",
    }
