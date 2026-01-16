"""
File upload endpoints.

Handles presigned URL generation and column mapping suggestions.
"""

import json
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Form, HTTPException

from ..config import get_settings
from ..dependencies import get_current_user, get_s3_client
from ..services.mapping import MappingService
from ..services.s3 import S3Service

router = APIRouter()


@router.post("/presign")
async def presign_upload(
    filenames: List[str] = Form(...),
    user_id: Optional[str] = Depends(get_current_user),
):
    """
    Generate presigned URLs for direct S3 upload.

    Args:
        filenames: List of filenames to upload
        user_id: Current user ID (from auth)

    Returns:
        List of presigned URL objects with filename, key, and url
    """
    settings = get_settings()
    s3_client = get_s3_client()
    s3_service = S3Service(s3_client, settings.s3_bucket_name)

    presigned_urls = []
    for filename in filenames:
        key = f"{user_id or 'anonymous'}/{uuid.uuid4()}-{filename}"
        url = s3_service.generate_presigned_url(key)
        presigned_urls.append({
            "filename": filename,
            "key": key,
            "url": url
        })

    return {"presigned_urls": presigned_urls}


@router.post("/suggest-mapping")
async def suggest_mapping(
    key: str = Form(...),
    filename: str = Form(...),
    user_id: Optional[str] = Depends(get_current_user),
) -> Dict:
    """
    Analyze uploaded file and suggest column mappings.

    Uses AI (Grok) for intelligent mapping with heuristic fallback.

    Args:
        key: S3 object key
        filename: Original filename
        user_id: Current user ID (from auth)

    Returns:
        Column mapping suggestions with confidence scores
    """
    settings = get_settings()
    s3_client = get_s3_client()
    s3_service = S3Service(s3_client, settings.s3_bucket_name)
    mapping_service = MappingService()

    try:
        # Load sample data
        df = s3_service.load_dataframe(key, sample_rows=50)

        # Get mapping suggestions
        result = mapping_service.suggest_mapping(df, filename)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Column mapping failed: {str(e)}"
        )
