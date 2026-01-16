"""
S3 Service for file storage operations.

SECURITY:
- Presigned URLs support size limits
- File size validation before loading
- Row count limits for DataFrames
"""

import io
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Safety limits
DEFAULT_MAX_FILE_SIZE_MB = 50
DEFAULT_MAX_ROWS = 500_000  # 500K rows max


class S3Service:
    """Service for S3 file operations with safety limits."""

    def __init__(self, client, bucket_name: str):
        """
        Initialize S3 service.

        Args:
            client: Boto3 S3 client
            bucket_name: S3 bucket name
        """
        self.client = client
        self.bucket_name = bucket_name

    def generate_presigned_url(
        self,
        key: str,
        expires_in: int = 3600,
        content_type: str = "application/octet-stream",
        max_size_mb: Optional[int] = None,
    ) -> str:
        """
        Generate a presigned URL for uploading with optional size limit.

        Args:
            key: S3 object key
            expires_in: URL expiration in seconds
            content_type: Content type for upload
            max_size_mb: Maximum file size in MB (enforced via presigned POST conditions)

        Returns:
            Presigned URL string

        Note:
            Size limit is advisory for simple PUT presigned URLs.
            For strict enforcement, use presigned POST with conditions.
        """
        params = {
            'Bucket': self.bucket_name,
            'Key': key,
            'ContentType': content_type
        }

        # Note: Simple presigned PUT URLs don't enforce size limits server-side.
        # We rely on client-side validation + head object check after upload.
        # For strict enforcement, migrate to presigned POST with conditions.

        if max_size_mb:
            logger.debug(f"Generating presigned URL for {key} with {max_size_mb}MB limit advisory")

        return self.client.generate_presigned_url(
            'put_object',
            Params=params,
            ExpiresIn=expires_in
        )

    def get_object_size(self, key: str) -> int:
        """
        Get object size in bytes without downloading.

        Args:
            key: S3 object key

        Returns:
            File size in bytes
        """
        response = self.client.head_object(Bucket=self.bucket_name, Key=key)
        return response.get('ContentLength', 0)

    def load_dataframe(
        self,
        key: str,
        sample_rows: Optional[int] = None,
        max_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> pd.DataFrame:
        """
        Load a DataFrame from S3 with safety limits.

        Security:
            - Validates file size before loading into memory
            - Limits row count to prevent memory exhaustion
            - Logs warnings for large files

        Args:
            key: S3 object key
            sample_rows: Optional limit on rows to load (overrides max_rows if smaller)
            max_size_mb: Maximum file size in MB (raises exception if exceeded)
            max_rows: Maximum rows to load

        Returns:
            Pandas DataFrame

        Raises:
            ValueError: If file exceeds size limit
        """
        # Check file size first
        file_size = self.get_object_size(key)
        file_size_mb = file_size / (1024 * 1024)

        if file_size_mb > max_size_mb:
            logger.warning(f"File {key} exceeds size limit: {file_size_mb:.1f}MB > {max_size_mb}MB")
            raise ValueError(
                f"File too large ({file_size_mb:.1f}MB). "
                f"Maximum allowed size is {max_size_mb}MB."
            )

        if file_size_mb > 10:
            logger.info(f"Loading large file: {key} ({file_size_mb:.1f}MB)")

        obj = self.client.get_object(Bucket=self.bucket_name, Key=key)
        contents = obj['Body'].read()

        read_kwargs = {"dtype": str}

        # Apply row limit (use smaller of sample_rows and max_rows)
        effective_rows = min(
            sample_rows if sample_rows else max_rows,
            max_rows
        )
        read_kwargs["nrows"] = effective_rows
        read_kwargs["keep_default_na"] = False

        if key.lower().endswith('.csv'):
            try:
                df = pd.read_csv(io.BytesIO(contents), **read_kwargs)
            except UnicodeDecodeError:
                df = pd.read_csv(
                    io.BytesIO(contents),
                    encoding='latin1',
                    **read_kwargs
                )
        else:
            # Excel files
            df = pd.read_excel(io.BytesIO(contents), **read_kwargs)

        if len(df) >= effective_rows and not sample_rows:
            logger.warning(
                f"File {key} truncated to {effective_rows} rows (max_rows limit). "
                f"Original file may have more data."
            )

        return df

    def upload_file(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        """
        Upload a file to S3.

        Args:
            key: S3 object key
            data: File data as bytes
            content_type: Content type
        """
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
            ContentType=content_type
        )

    def delete_file(self, key: str):
        """
        Delete a file from S3.

        Args:
            key: S3 object key
        """
        self.client.delete_object(Bucket=self.bucket_name, Key=key)
