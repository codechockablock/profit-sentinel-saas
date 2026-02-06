"""S3 Service for the sidecar API.

Handles presigned URL generation, file loading, and cleanup.
Ported from _legacy/apps/api/src/services/s3.py with minimal changes.
"""

from __future__ import annotations

import io
import logging
import os
import re
import uuid

import pandas as pd

logger = logging.getLogger(__name__)

# Safety limits
MAX_FILE_SIZE_MB = 50
MAX_ROWS = 500_000
MAX_FILES_PER_REQUEST = 5
MAX_FILENAME_LENGTH = 255
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
SAFE_FILENAME_PATTERN = re.compile(r"^[\w\-. ()]+$")

SUPPORTED_POS_SYSTEMS = [
    "Paladin POS",
    "Spruce POS",
    "Epicor Eagle",
    "Square POS",
    "Lightspeed Retail (R-Series, X-Series, S-Series)",
    "Lightspeed eCom (C-Series)",
    "Clover POS",
    "Shopify POS",
    "NCR Counterpoint",
    "Microsoft Dynamics RMS",
    "QuickBooks POS",
    "Vend POS",
    "Toast POS",
    "Revel Systems",
    "TouchBistro",
    "Generic CSV/Excel exports",
]


def get_s3_client():
    """Create and return a boto3 S3 client."""
    import boto3

    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and injection."""
    if not filename:
        raise ValueError("Filename cannot be empty")

    filename = os.path.basename(filename)

    if len(filename) > MAX_FILENAME_LENGTH:
        raise ValueError(f"Filename too long (max {MAX_FILENAME_LENGTH} characters)")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    name_without_ext = os.path.splitext(filename)[0]
    if not SAFE_FILENAME_PATTERN.match(name_without_ext):
        safe_name = re.sub(r"[^\w\-. ()]", "_", name_without_ext)
        filename = f"{safe_name}{ext}"

    return filename


def generate_presigned_url(
    s3_client,
    bucket_name: str,
    key: str,
    expires_in: int = 3600,
) -> str:
    """Generate a presigned URL for uploading."""
    return s3_client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket_name,
            "Key": key,
            "ContentType": "application/octet-stream",
        },
        ExpiresIn=expires_in,
    )


def generate_upload_urls(
    s3_client,
    bucket_name: str,
    filenames: list[str],
    s3_prefix: str | None = None,
    *,
    max_file_size_mb: int = MAX_FILE_SIZE_MB,
) -> dict:
    """Generate presigned URLs for a list of files.

    Args:
        s3_client: boto3 S3 client.
        bucket_name: S3 bucket name.
        filenames: List of filenames to generate URLs for.
        s3_prefix: S3 key prefix (e.g. ``uploads/user-123`` or
                   ``uploads/anonymous/abc123``). Falls back to ``anonymous``.
        max_file_size_mb: Per-user file size limit (10 MB anon, 50 MB auth).

    Returns the legacy API response shape.
    """
    if len(filenames) > MAX_FILES_PER_REQUEST:
        raise ValueError(
            f"Too many files. Maximum {MAX_FILES_PER_REQUEST} files per request."
        )

    if not filenames:
        raise ValueError("At least one filename required")

    key_prefix = s3_prefix if s3_prefix else "anonymous"

    presigned_urls = []
    for filename in filenames:
        safe_filename = sanitize_filename(filename)
        key = f"{key_prefix}/{uuid.uuid4()}-{safe_filename}"
        url = generate_presigned_url(s3_client, bucket_name, key)

        presigned_urls.append(
            {
                "filename": filename,
                "safe_filename": safe_filename,
                "key": key,
                "url": url,
                "max_size_mb": max_file_size_mb,
            }
        )

    return {
        "presigned_urls": presigned_urls,
        "limits": {
            "max_file_size_mb": max_file_size_mb,
            "allowed_extensions": list(ALLOWED_EXTENSIONS),
        },
    }


def _validate_magic_bytes(content: bytes, extension: str) -> tuple[bool, str]:
    """Validate file content matches expected magic bytes."""
    if len(content) < 8:
        return False, "File too small to validate"

    ext = extension.lower().lstrip(".")

    if ext == "xlsx":
        if not content.startswith(b"PK\x03\x04"):
            return False, "Invalid XLSX file: not a valid Office Open XML format"
        return True, ""

    elif ext == "xls":
        if not content.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            return False, "Invalid XLS file: not a valid legacy Excel format"
        return True, ""

    elif ext == "csv":
        utf8_bom = b"\xef\xbb\xbf"
        utf16_le_bom = b"\xff\xfe"
        utf16_be_bom = b"\xfe\xff"

        if (
            content.startswith(utf8_bom)
            or content.startswith(utf16_le_bom)
            or content.startswith(utf16_be_bom)
        ):
            return True, ""

        sample = content[:1000]
        non_printable = sum(
            1 for byte in sample if byte < 32 and byte not in (9, 10, 13)
        )

        if non_printable > len(sample) * 0.1:
            return False, "Invalid CSV file: contains binary content"

        return True, ""

    return True, ""


def load_dataframe(
    s3_client,
    bucket_name: str,
    key: str,
    sample_rows: int | None = None,
    max_size_mb: int = MAX_FILE_SIZE_MB,
    max_rows: int = MAX_ROWS,
) -> pd.DataFrame:
    """Load a DataFrame from S3 with safety limits."""
    response = s3_client.head_object(Bucket=bucket_name, Key=key)
    file_size = response.get("ContentLength", 0)
    file_size_mb = file_size / (1024 * 1024)

    if file_size_mb > max_size_mb:
        raise ValueError(
            f"File too large ({file_size_mb:.1f}MB). "
            f"Maximum allowed size is {max_size_mb}MB."
        )

    if file_size_mb > 10:
        logger.info(f"Loading large file ({file_size_mb:.1f}MB)")

    obj = s3_client.get_object(Bucket=bucket_name, Key=key)
    contents = obj["Body"].read()

    extension = key.rsplit(".", 1)[-1] if "." in key else ""
    is_valid, error_msg = _validate_magic_bytes(contents, extension)
    if not is_valid:
        raise ValueError(error_msg)

    read_kwargs: dict = {"dtype": str}
    effective_rows = min(sample_rows if sample_rows else max_rows, max_rows)
    read_kwargs["nrows"] = effective_rows
    read_kwargs["keep_default_na"] = False

    if key.lower().endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(contents), **read_kwargs)
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(contents), encoding="latin1", **read_kwargs)
    else:
        df = pd.read_excel(io.BytesIO(contents), **read_kwargs)

    return df


def delete_file(s3_client, bucket_name: str, key: str) -> None:
    """Delete a file from S3."""
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=key)
    except Exception as e:
        logger.warning(f"Failed to delete S3 file {key}: {e}")
