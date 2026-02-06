"""
Analysis endpoints - Aggressive Profit Leak Detection.

Handles profit leak analysis using VSA resonator with 11 detection primitives.
Supports data from any major POS system (Paladin, Square, Lightspeed, etc.).

Privacy features:
- Auto-deletes uploaded files after processing
- Stores only anonymized aggregated analytics
"""

import json
import logging
import time

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..config import SUPPORTED_POS_SYSTEMS, get_settings
from ..dependencies import get_current_user, get_s3_client, require_pro_tier
from ..services.analysis import AnalysisService
from ..services.anonymization import get_anonymization_service
from ..services.column_adapter import ColumnAdapter

# v3.7: Replaced GuardDuty with lightweight file validator
from ..services.file_validator import get_file_validator as get_virus_scanner
from ..services.result_adapter import RustResultAdapter
from ..services.s3 import S3Service

router = APIRouter()

# Rate limiter for this router
limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger(__name__)


# =============================================================================
# COMPREHENSIVE NUMERIC COLUMN ALIASES
# Covers all major POS systems: Paladin, Square, Lightspeed, Clover, Shopify,
# NCR Counterpoint, Microsoft Dynamics RMS, Epicor, and generic exports
# =============================================================================

NUMERIC_COLUMN_ALIASES = [
    # Quantity / Stock
    "quantity",
    "qty",
    "qty.",
    "qoh",
    "on_hand",
    "onhand",
    "in_stock",
    "instock",
    "in stock qty.",
    "stock",
    "inventory",
    "stockonhand",
    "qty_on_hnd",
    "qty_avail",
    "net_qty",
    "available",
    "current_stock",
    "variant_inventory_qty",
    "new_quantity",
    "qty_oh",
    "physical_qty",
    "book_qty",
    "bal",
    "balance",
    # Quantity Difference / Variance
    "qty_difference",
    "qty. difference",
    "difference",
    "variance",
    "qty_variance",
    "shrinkage",
    "adjustment",
    "discrepancy",
    "over_short",
    # Inventoried
    "inventoried_qty",
    "inventoried",
    "inventoried qty.",
    "counted_qty",
    "count",
    # Cost
    "cost",
    "cogs",
    "cost_price",
    "unit_cost",
    "unitcost",
    "avg_cost",
    "avgcost",
    "average_cost",
    "standard_cost",
    "lst_cost",
    "last_cost",
    "default_cost",
    "vendor_cost",
    "supply_price",
    "cost_per_item",
    "posting_cost",
    "mktcost",
    "landed_cost",
    "purchase_cost",
    "buy_price",
    "wholesale_cost",
    # Revenue / Retail Price
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
    "ext_price",
    "line_total",
    "amount",
    "gross_sales",
    "total_sale",
    # Sold
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
    "sales_units",
    "unit_sales",
    "movement",
    # Margin
    "margin",
    "profit_margin",
    "margin_pct",
    "margin_percent",
    "gp",
    "gross_profit",
    "gp_pct",
    "gp_percent",
    "profit_pct",
    "markup",
    "markup_pct",
    # Sub Total / Value
    "sub_total",
    "subtotal",
    "total",
    "ext_total",
    "inventory_value",
    "stock_value",
    "on_hand_value",
    "gl_val",
    # Discount / Tax
    "discount",
    "discount_amt",
    "discount_amount",
    "discount_pct",
    "tax",
    "sales_tax",
    "vat",
    "tax_amount",
    # Reorder
    "reorder_point",
    "reorder_level",
    "min_qty",
    "minimum_qty",
    "safety_stock",
    "par_level",
    "minorderqty",
    "stock_min",
    "stock_alert",
    # Package
    "pkg_qty",
    "package_qty",
    "pack_size",
    "case_qty",
]


def _validate_and_apply_mapping(
    df: pd.DataFrame, mapping_dict: dict[str, str]
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
    df_cols_lower = {
        col.lower().replace(" ", "").replace(".", ""): col for col in df.columns
    }
    existing_numeric = []

    for alias in NUMERIC_COLUMN_ALIASES:
        normalized = alias.lower().replace(" ", "").replace(".", "")
        if normalized in df_cols_lower:
            existing_numeric.append(df_cols_lower[normalized])

    # Also add original column names that match
    for alias in NUMERIC_COLUMN_ALIASES:
        if alias in df.columns and alias not in existing_numeric:
            existing_numeric.append(alias)

    # Remove duplicates while preserving order
    existing_numeric = list(dict.fromkeys(existing_numeric))

    if existing_numeric:
        logger.info(
            f"Cleaning {len(existing_numeric)} numeric columns: {existing_numeric[:10]}..."
        )

        # Strip $ and , and %, convert to numeric
        for col in existing_numeric:
            try:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[$,\%]", "", regex=True)
                    .str.strip()
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(0.0)
            except Exception as e:
                logger.warning(f"Failed to clean column {col}: {e}")

    return df


def _run_rust_engine(
    df: pd.DataFrame,
    rows: list[dict],
    settings,
    overall_start: float,
) -> tuple[dict, str]:
    """Run analysis through Rust sentinel-server pipeline.

    Returns:
        Tuple of (result_dict, engine_name).

    Raises:
        Exception on any failure (caller will fall back to legacy).
    """
    import json as _json
    import subprocess

    adapter_start = time.time()
    col_adapter = ColumnAdapter(
        default_store_id=settings.sentinel_default_store,
    )

    try:
        csv_path = col_adapter.to_rust_csv(df)
        adapter_time = time.time() - adapter_start
        logger.info(
            f"Column adapter: {len(df)} rows → {csv_path} in {adapter_time:.2f}s"
        )

        # Run sentinel-server subprocess
        rust_start = time.time()
        proc = subprocess.run(
            [
                settings.sentinel_bin,
                str(csv_path),
                "--json",
                "--top",
                str(settings.sentinel_top_k),
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max
        )
        rust_time = time.time() - rust_start

        if proc.returncode != 0:
            raise RuntimeError(
                f"sentinel-server exited {proc.returncode}: {proc.stderr[:500]}"
            )

        logger.info(f"Rust pipeline: {rust_time:.2f}s")

        digest = _json.loads(proc.stdout)

        # Transform Rust output → legacy API format
        total_time = time.time() - overall_start
        result_adapter = RustResultAdapter()
        result = result_adapter.transform(
            digest=digest,
            total_rows=len(df),
            analysis_time=total_time,
            original_rows=rows,
        )

        return result, "rust"

    finally:
        col_adapter.cleanup()


@router.post("/analyze")
@limiter.limit("10/minute")
async def analyze_upload(
    request: Request,  # Required for rate limiter
    key: str = Form(...),
    mapping: str = Form(...),  # JSON string
    background_tasks: BackgroundTasks = None,
    user_id: str | None = Depends(get_current_user),  # Optional auth (freemium)
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
        - primitives_used: List of 11 detection primitives used
        - warnings: Any mapping warnings
    """
    # Parse mapping
    try:
        mapping_dict = json.loads(mapping)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid mapping JSON: {str(e)}")

    # Validate mapping is a dict
    if not isinstance(mapping_dict, dict):
        raise HTTPException(
            status_code=400,
            detail="Mapping must be a JSON object with column name pairs",
        )

    # SECURITY: Validate S3 key ownership
    # - Authenticated users can only access their own files
    # - Anonymous users can only access anonymous files
    expected_prefix = f"{user_id}/" if user_id else "anonymous/"
    if not key.startswith(expected_prefix):
        logger.warning(
            f"{'User ' + user_id if user_id else 'Anonymous user'} "
            f"attempted to access unauthorized S3 key: {key}"
        )
        raise HTTPException(
            status_code=403,
            detail="Access denied: you can only analyze your own uploaded files",
        )

    try:
        overall_start = time.time()
        logger.info(f"Starting aggressive analysis for key: {key}")

        settings = get_settings()
        s3_client = get_s3_client()
        s3_service = S3Service(s3_client, settings.s3_bucket_name)

        # File validation (replaced GuardDuty with lightweight validator)
        scan_start = time.time()
        scanner = get_virus_scanner()
        if scanner.is_available:
            scan_result = await scanner.check_scan_status(
                s3_client, settings.s3_bucket_name, key
            )
            if not scan_result.is_clean:
                # Delete infected file and reject
                s3_client.delete_object(Bucket=settings.s3_bucket_name, Key=key)
                logger.warning("Rejected infected file, deleted from S3")
                raise HTTPException(
                    status_code=400,
                    detail="File rejected: security scan detected a threat. Please upload a clean file.",
                )
        scan_time = time.time() - scan_start
        logger.info(f"TIMING file_validation={scan_time:.2f}s")

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
            df = df.loc[:, ~df.columns.duplicated(keep="first")]

        # Convert to records
        records_start = time.time()
        rows = df.to_dict(orient="records")
        logger.info(
            f"Converted to records ({len(rows)} rows) "
            f"in {time.time() - records_start:.2f}s"
        )

        # Run analysis — dual engine: Rust (new) or Python (legacy)
        result = None
        engine_used = "legacy"

        if settings.use_new_engine:
            try:
                result, engine_used = _run_rust_engine(
                    df, rows, settings, overall_start
                )
            except Exception as e:
                logger.warning(
                    "Rust engine failed, falling back to legacy: %s",
                    e,
                    exc_info=True,
                )
                result = None

        if result is None:
            # Legacy Python engine (default path)
            analysis_service = AnalysisService()
            result = analysis_service.analyze(rows)
            engine_used = "legacy"

        total_time = time.time() - overall_start
        logger.info(
            f"Full analysis pipeline completed in {total_time:.2f}s "
            f"(engine={engine_used})"
        )

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
                delay_seconds=60,  # 1 minute delay before deletion
            )
            logger.info(f"Scheduled S3 file cleanup for key: {key}")

            # Store anonymized analytics
            background_tasks.add_task(
                anon_service.store_anonymized_analytics,
                [result],
                False,  # report_sent=False (not sent via email yet)
            )

        return result

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400, detail="Uploaded file is empty or has no valid data"
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
                "Each source column must map to a unique target name.",
            )
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")


@router.get("/primitives")
async def list_primitives() -> dict:
    """
    List all available analysis primitives.

    Returns metadata about each of the 11 leak detection primitives.
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


@router.post("/analyze-multi")
@limiter.limit("5/minute")
async def analyze_multi_reports(
    request: Request,
    keys: list[str] = Form(...),
    mappings: list[str] = Form(...),  # JSON strings, one per file
    source_types: list[str] = Form(
        None
    ),  # Optional: inventory, catalog, invoice, pos, vendor
    background_tasks: BackgroundTasks = None,
    user_id: str = Depends(require_pro_tier),  # Pro tier required
) -> dict:
    """
    Analyze multiple reports for cross-report correlation (Pro feature).

    Supports data from any major POS system. Normalizes SKUs across reports
    to find cross-source discrepancies and correlated findings.

    Args:
        keys: List of S3 object keys (one per report)
        mappings: List of JSON column mapping strings (one per report)
        source_types: Optional list of source types (inventory, catalog, invoice, pos, vendor)
                      If not provided, auto-detects from column names.
        user_id: Current user ID (Pro/Enterprise tier required)

    Returns:
        Multi-report analysis with:
        - per_report: Individual analysis results per report
        - cross_reference: SKUs appearing in multiple sources
        - unified_findings: Consolidated findings across all reports
        - tracked_skus: All canonical SKUs found
    """
    # Validate input lengths match
    if len(keys) != len(mappings):
        raise HTTPException(
            status_code=400,
            detail=f"Number of keys ({len(keys)}) must match mappings ({len(mappings)})",
        )

    if len(keys) < 2:
        raise HTTPException(
            status_code=400,
            detail="Multi-report analysis requires at least 2 files",
        )

    if len(keys) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5 files per multi-report analysis",
        )

    # Parse mappings
    mapping_dicts = []
    for i, mapping_json in enumerate(mappings):
        try:
            mapping_dict = json.loads(mapping_json)
            if not isinstance(mapping_dict, dict):
                raise ValueError("not a dict")
            mapping_dicts.append(mapping_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid mapping JSON at index {i}: {e}",
            )

    # Handle source types
    if source_types and len(source_types) != len(keys):
        raise HTTPException(
            status_code=400,
            detail=f"Number of source_types ({len(source_types)}) must match keys ({len(keys)})",
        )

    # Validate S3 key ownership (all keys must belong to user)
    expected_prefix = f"{user_id}/"
    for key in keys:
        if not key.startswith(expected_prefix):
            logger.warning(
                f"User {user_id} attempted to access unauthorized S3 key: {key}"
            )
            raise HTTPException(
                status_code=403,
                detail="Access denied: you can only analyze your own uploaded files",
            )

    try:
        overall_start = time.time()
        logger.info(f"Starting multi-report analysis for {len(keys)} files")

        settings = get_settings()
        s3_client = get_s3_client()
        s3_service = S3Service(s3_client, settings.s3_bucket_name)

        # Validate all files before processing (C6 - file validation)
        scanner = get_virus_scanner()
        if scanner.is_available:
            for key in keys:
                scan_result = await scanner.check_scan_status(
                    s3_client, settings.s3_bucket_name, key
                )
                if not scan_result.is_clean:
                    # Delete infected file
                    s3_client.delete_object(Bucket=settings.s3_bucket_name, Key=key)
                    logger.warning(
                        "Rejected infected file in multi-report, deleted from S3"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail="File rejected: security scan detected a threat in one of the uploaded files.",
                    )

        # Load and prepare all reports
        reports = []
        all_warnings = []

        for i, (key, mapping_dict) in enumerate(zip(keys, mapping_dicts)):
            load_start = time.time()
            df = s3_service.load_dataframe(key)
            original_columns = df.columns.tolist()

            logger.info(
                f"Loaded report {i + 1}/{len(keys)} ({len(df)} rows, {len(df.columns)} cols) "
                f"from {key} in {time.time() - load_start:.2f}s"
            )

            # Apply column mapping
            df, warnings = _validate_and_apply_mapping(df, mapping_dict)
            if warnings:
                all_warnings.extend([f"Report {i + 1}: {w}" for w in warnings])

            # Clean numeric columns
            df = _clean_numeric_columns(df)

            # Drop duplicate columns
            if df.columns.duplicated().any():
                dup_cols = df.columns[df.columns.duplicated()].tolist()
                logger.warning(
                    f"Report {i + 1}: dropping duplicate columns: {dup_cols}"
                )
                df = df.loc[:, ~df.columns.duplicated(keep="first")]

            # Convert to records
            rows = df.to_dict(orient="records")

            # Determine source type
            if source_types and i < len(source_types):
                src_type = source_types[i]
            else:
                # Auto-detect from columns
                src_type = _detect_source_type(original_columns)

            reports.append(
                {
                    "rows": rows,
                    "source_type": src_type,
                    "columns": original_columns,
                    "key": key,
                }
            )

        # Run multi-report analysis
        analysis_service = AnalysisService()
        result = analysis_service.analyze_multi(reports)

        total_time = time.time() - overall_start
        logger.info(f"Multi-report analysis complete in {total_time:.2f}s")

        # Add status and warnings
        result["warnings"] = all_warnings if all_warnings else None
        result["supported_pos_systems"] = SUPPORTED_POS_SYSTEMS

        # Schedule file cleanup
        if background_tasks:
            anon_service = get_anonymization_service()
            for key in keys:
                background_tasks.add_task(
                    anon_service.cleanup_s3_file,
                    s3_client,
                    settings.s3_bucket_name,
                    key,
                    delay_seconds=60,
                )
            logger.info(f"Scheduled S3 cleanup for {len(keys)} files")

        return result

    except HTTPException:
        raise
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="One or more uploaded files is empty or has no valid data",
        )
    except ValueError as e:
        error_msg = str(e)
        if "too large" in error_msg.lower() or "size" in error_msg.lower():
            raise HTTPException(status_code=413, detail=error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        logger.error(f"Multi-report analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Multi-report analysis failed: {str(e)}",
        )


def _detect_source_type(columns: list[str]) -> str:
    """
    Auto-detect report source type from column names.

    Returns one of: inventory, catalog, invoice, pos, vendor, unknown
    """
    cols_lower = [c.lower() for c in columns]
    cols_text = " ".join(cols_lower)

    # Invoice indicators
    if any(
        kw in cols_text for kw in ["invoice", "bill", "po_number", "purchase_order"]
    ):
        return "invoice"

    # POS/Sales indicators
    if any(
        kw in cols_text for kw in ["transaction", "sale_date", "register", "tender"]
    ):
        return "pos"

    # Vendor/Supplier indicators
    if any(
        kw in cols_text for kw in ["vendor", "supplier", "manufacturer", "lead_time"]
    ):
        return "vendor"

    # Catalog indicators
    if any(kw in cols_text for kw in ["msrp", "list_price", "catalog", "upc", "ean"]):
        return "catalog"

    # Inventory (default for stock-heavy reports)
    if any(
        kw in cols_text for kw in ["qty", "quantity", "stock", "on_hand", "inventory"]
    ):
        return "inventory"

    return "unknown"
