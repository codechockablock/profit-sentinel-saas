"""
Analysis endpoints.

Handles profit leak analysis using VSA resonator.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException

from ..config import get_settings
from ..dependencies import get_current_user, get_s3_client
from ..services.analysis import AnalysisService
from ..services.s3 import S3Service

router = APIRouter()
logger = logging.getLogger(__name__)


def _validate_and_apply_mapping(
    df: pd.DataFrame,
    mapping_dict: Dict[str, str]
) -> Tuple[pd.DataFrame, List[str]]:
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


@router.post("/analyze")
async def analyze_upload(
    key: str = Form(...),
    mapping: str = Form(...),  # JSON string
    user_id: Optional[str] = Depends(get_current_user),
) -> Dict:
    """
    Analyze uploaded POS data for profit leaks.

    Args:
        key: S3 object key
        mapping: JSON string of column mappings
        user_id: Current user ID (from auth)

    Returns:
        Analysis results with detected anomalies
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
        logger.info(f"Starting analysis for key: {key}")

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

        # Clean numeric columns
        df = _clean_numeric_columns(df)

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

        # Run analysis
        analysis_service = AnalysisService()
        leaks = analysis_service.analyze(rows)

        logger.info(
            f"Full analysis completed in {time.time() - overall_start:.2f}s"
        )

        return {
            "status": "success",
            "leaks": leaks,
            "warnings": mapping_warnings if mapping_warnings else None
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty or has no valid data"
        )
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


def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert numeric columns."""
    numeric_aliases = [
        'quantity', 'Qty.', 'In Stock Qty.', 'qty_difference', 'Qty. Difference',
        'revenue', 'Retail', 'Sug. Retail', 'sale_price', 'price', 'ext_price',
        'line_total', 'amount',
        'cost', 'Cost', 'cogs', 'unit_cost',
        'discount', 'tax',
        'sold', 'Sold'
    ]

    existing_numeric = [col for col in numeric_aliases if col in df.columns]
    if existing_numeric:
        # Strip $ and commas, convert to numeric
        df[existing_numeric] = (
            df[existing_numeric]
            .astype(str)
            .replace(r'[$,]', '', regex=True)
            .apply(lambda x: x.str.strip())
        )
        df[existing_numeric] = df[existing_numeric].apply(
            pd.to_numeric, errors='coerce'
        )
        df[existing_numeric] = df[existing_numeric].fillna(0.0)

    return df
