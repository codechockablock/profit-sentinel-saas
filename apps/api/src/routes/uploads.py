"""
File upload endpoints.

Handles presigned URL generation and column mapping suggestions.

SECURITY:
- File size limits enforced via presigned URL conditions
- Filename sanitization prevents path traversal
- Allowed extensions whitelist
- Row count limits for analysis
"""

import logging
import os
import re
import uuid

from fastapi import APIRouter, Depends, Form, HTTPException

from ..config import get_settings
from ..dependencies import get_current_user, get_s3_client
from ..services.mapping import MappingService
from ..services.s3 import S3Service

router = APIRouter()
logger = logging.getLogger(__name__)

# Safety limits
MAX_FILES_PER_REQUEST = 5
MAX_FILENAME_LENGTH = 255
MAX_FILE_SIZE_MB = 50  # 50 MB limit
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# Sanitization: alphanumeric, dash, dot, underscore, space only
SAFE_FILENAME_PATTERN = re.compile(r"^[\w\-. ()]+$")


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and injection attacks.

    Args:
        filename: Original filename from client

    Returns:
        Sanitized filename safe for S3 keys

    Raises:
        HTTPException: If filename is invalid or contains disallowed characters
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    # Remove any path components (prevent traversal)
    filename = os.path.basename(filename)

    # Check length
    if len(filename) > MAX_FILENAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Filename too long (max {MAX_FILENAME_LENGTH} characters)",
        )

    # Check extension
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Check for safe characters
    name_without_ext = os.path.splitext(filename)[0]
    if not SAFE_FILENAME_PATTERN.match(name_without_ext):
        # Replace unsafe characters with underscore
        safe_name = re.sub(r"[^\w\-. ()]", "_", name_without_ext)
        filename = f"{safe_name}{ext}"
        logger.warning(f"Sanitized filename to: {filename}")

    return filename


@router.post("/presign")
async def presign_upload(
    filenames: list[str] = Form(...),
    user_id: str | None = Depends(get_current_user),
):
    """
    Generate presigned URLs for direct S3 upload.

    Security:
        - Filenames are sanitized to prevent path traversal
        - Only allowed file extensions accepted
        - Presigned URLs have size limits enforced
        - Limited to MAX_FILES_PER_REQUEST files

    Args:
        filenames: List of filenames to upload
        user_id: Current user ID (from auth)

    Returns:
        List of presigned URL objects with filename, key, and url
    """
    # Limit number of files per request
    if len(filenames) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {MAX_FILES_PER_REQUEST} files per request.",
        )

    if not filenames:
        raise HTTPException(status_code=400, detail="At least one filename required")

    settings = get_settings()
    s3_client = get_s3_client()
    s3_service = S3Service(s3_client, settings.s3_bucket_name)

    presigned_urls = []
    for filename in filenames:
        # Sanitize filename (raises HTTPException if invalid)
        safe_filename = _sanitize_filename(filename)

        # Generate unique key with sanitized filename
        key = f"{user_id or 'anonymous'}/{uuid.uuid4()}-{safe_filename}"

        # Generate presigned URL with size limit
        url = s3_service.generate_presigned_url(key, max_size_mb=MAX_FILE_SIZE_MB)

        presigned_urls.append(
            {
                "filename": filename,
                "safe_filename": safe_filename,
                "key": key,
                "url": url,
                "max_size_mb": MAX_FILE_SIZE_MB,
            }
        )

    logger.info(
        f"Generated {len(presigned_urls)} presigned URLs for user {user_id or 'anonymous'}"
    )

    return {
        "presigned_urls": presigned_urls,
        "limits": {
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "allowed_extensions": list(ALLOWED_EXTENSIONS),
        },
    }


@router.post("/suggest-mapping")
async def suggest_mapping(
    key: str = Form(...),
    filename: str = Form(...),
    user_id: str | None = Depends(get_current_user),
) -> dict:
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

    except ValueError as e:
        # Known validation errors - safe to expose message
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unknown errors - log but don't expose internal details
        logger.error(f"Column mapping failed for {key}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Column mapping failed. Please check your file format and try again.",
        )
