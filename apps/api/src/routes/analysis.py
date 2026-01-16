"""
Analysis endpoints.

Handles profit leak analysis using VSA resonator.
"""

import json
import logging
import time
from typing import Dict, Optional

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException

from ..config import get_settings
from ..dependencies import get_current_user, get_s3_client
from ..services.analysis import AnalysisService
from ..services.s3 import S3Service

router = APIRouter()
logger = logging.getLogger(__name__)


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
    try:
        # Parse mapping
        mapping_dict = json.loads(mapping)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid mapping JSON")

    try:
        overall_start = time.time()
        logger.info(f"Starting analysis for key: {key}")

        settings = get_settings()
        s3_client = get_s3_client()
        s3_service = S3Service(s3_client, settings.s3_bucket_name)

        # Load full DataFrame
        load_start = time.time()
        df = s3_service.load_dataframe(key)
        logger.info(
            f"Loaded DataFrame ({len(df)} rows, {len(df.columns)} columns) "
            f"in {time.time() - load_start:.2f}s"
        )

        # Apply column mapping
        df = df.rename(columns={k: v for k, v in mapping_dict.items() if v})

        # Clean numeric columns
        df = _clean_numeric_columns(df)

        # Drop duplicate columns
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

        return {"status": "success", "leaks": leaks}

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


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
