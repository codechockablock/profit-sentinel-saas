"""
S3 Service for file storage operations.
"""

import io
from typing import Optional

import pandas as pd


class S3Service:
    """Service for S3 file operations."""

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
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Generate a presigned URL for uploading.

        Args:
            key: S3 object key
            expires_in: URL expiration in seconds
            content_type: Content type for upload

        Returns:
            Presigned URL string
        """
        return self.client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': self.bucket_name,
                'Key': key,
                'ContentType': content_type
            },
            ExpiresIn=expires_in
        )

    def load_dataframe(
        self,
        key: str,
        sample_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load a DataFrame from S3.

        Args:
            key: S3 object key
            sample_rows: Optional limit on rows to load

        Returns:
            Pandas DataFrame
        """
        obj = self.client.get_object(Bucket=self.bucket_name, Key=key)
        contents = obj['Body'].read()

        read_kwargs = {"dtype": str}
        if sample_rows:
            read_kwargs["nrows"] = sample_rows
            read_kwargs["keep_default_na"] = False

        if key.lower().endswith('.csv'):
            try:
                return pd.read_csv(io.BytesIO(contents), **read_kwargs)
            except UnicodeDecodeError:
                return pd.read_csv(
                    io.BytesIO(contents),
                    encoding='latin1',
                    **read_kwargs
                )
        else:
            # Excel files
            return pd.read_excel(io.BytesIO(contents), **read_kwargs)

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
