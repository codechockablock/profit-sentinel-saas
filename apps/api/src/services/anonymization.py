"""
Anonymization Service - Privacy-focused data handling.

Implements:
- PII stripping from uploaded files
- SKU hashing for anonymization
- Aggregated analytics storage
- S3 file cleanup after processing
"""

import hashlib
import logging
import os
import re
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# Regex patterns for PII detection
PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
    "ssn": re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    "name_columns": ["customer_name", "customer", "name", "first_name", "last_name",
                     "full_name", "buyer", "purchaser", "client", "contact"],
    "address_columns": ["address", "street", "city", "state", "zip", "postal",
                       "address_1", "address_2", "billing_address", "shipping_address"],
}


class AnonymizationService:
    """Service for anonymizing data and handling privacy compliance."""

    def __init__(self):
        """Initialize anonymization service."""
        self.hash_salt = os.getenv("ANONYMIZATION_SALT", "profit-sentinel-default-salt")
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        self._supabase_client = None

    @property
    def supabase_client(self):
        """Lazy-load Supabase client."""
        if self._supabase_client is None and self.supabase_url and self.supabase_key:
            try:
                from supabase import create_client
                self._supabase_client = create_client(self.supabase_url, self.supabase_key)
            except ImportError:
                logger.warning("Supabase package not installed")
            except Exception as e:
                logger.warning(f"Failed to create Supabase client: {e}")
        return self._supabase_client

    def anonymize_dataframe(self, df: pd.DataFrame, hash_skus: bool = False) -> pd.DataFrame:
        """
        Anonymize a DataFrame by removing/masking PII.

        Args:
            df: Input DataFrame
            hash_skus: Whether to hash SKU values

        Returns:
            Anonymized DataFrame
        """
        df_anon = df.copy()

        # Remove columns that likely contain PII
        pii_columns_to_drop = []
        for col in df_anon.columns:
            col_lower = col.lower().replace(" ", "_")

            # Check against known PII column names
            if col_lower in PII_PATTERNS["name_columns"]:
                pii_columns_to_drop.append(col)
                logger.info(f"Dropping PII column (name): {col}")
            elif col_lower in PII_PATTERNS["address_columns"]:
                pii_columns_to_drop.append(col)
                logger.info(f"Dropping PII column (address): {col}")
            elif "email" in col_lower:
                pii_columns_to_drop.append(col)
                logger.info(f"Dropping PII column (email): {col}")
            elif "phone" in col_lower or "tel" in col_lower:
                pii_columns_to_drop.append(col)
                logger.info(f"Dropping PII column (phone): {col}")

        # Drop identified PII columns
        if pii_columns_to_drop:
            df_anon = df_anon.drop(columns=pii_columns_to_drop)

        # Scan remaining columns for PII patterns
        for col in df_anon.columns:
            if df_anon[col].dtype == object:  # String columns
                df_anon[col] = df_anon[col].apply(
                    lambda x: self._mask_pii_in_value(x) if pd.notna(x) else x
                )

        # Hash SKUs if requested
        if hash_skus:
            sku_columns = ["sku", "product_id", "item_id", "upc", "barcode"]
            for col in df_anon.columns:
                if col.lower().replace(" ", "_") in sku_columns:
                    df_anon[col] = df_anon[col].apply(
                        lambda x: self._hash_value(str(x)) if pd.notna(x) else x
                    )
                    logger.info(f"Hashed SKU column: {col}")

        return df_anon

    def _mask_pii_in_value(self, value: Any) -> Any:
        """Mask PII patterns found in a value."""
        if not isinstance(value, str):
            return value

        # Mask emails
        value = PII_PATTERNS["email"].sub("[EMAIL_REDACTED]", value)

        # Mask phone numbers
        value = PII_PATTERNS["phone"].sub("[PHONE_REDACTED]", value)

        # Mask SSN-like patterns
        value = PII_PATTERNS["ssn"].sub("[SSN_REDACTED]", value)

        # Mask credit card patterns
        value = PII_PATTERNS["credit_card"].sub("[CC_REDACTED]", value)

        return value

    def _hash_value(self, value: str) -> str:
        """Create a deterministic hash of a value."""
        salted = f"{self.hash_salt}:{value}"
        return hashlib.sha256(salted.encode()).hexdigest()[:16]

    def extract_aggregated_stats(self, results: list[dict]) -> dict:
        """
        Extract anonymized, aggregated statistics from analysis results.

        No PII is included - only counts and averages.

        Args:
            results: Analysis results

        Returns:
            Aggregated statistics dictionary
        """
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "files_analyzed": len(results),
            "total_rows": 0,
            "total_items_flagged": 0,
            "critical_count": 0,
            "high_count": 0,
            "leak_counts": {},
            "avg_scores": {},
            "total_impact_low": 0.0,
            "total_impact_high": 0.0,
        }

        for result in results:
            summary = result.get("summary", {})
            stats["total_rows"] += summary.get("total_rows_analyzed", 0)
            stats["total_items_flagged"] += summary.get("total_items_flagged", 0)
            stats["critical_count"] += summary.get("critical_issues", 0)
            stats["high_count"] += summary.get("high_issues", 0)

            impact = summary.get("estimated_impact", {})
            stats["total_impact_low"] += impact.get("low_estimate", 0)
            stats["total_impact_high"] += impact.get("high_estimate", 0)

            # Aggregate leak counts and scores
            leaks = result.get("leaks", {})
            for leak_type, data in leaks.items():
                count = data.get("count", len(data.get("top_items", [])))
                scores = data.get("scores", [])

                if leak_type not in stats["leak_counts"]:
                    stats["leak_counts"][leak_type] = 0
                    stats["avg_scores"][leak_type] = []

                stats["leak_counts"][leak_type] += count
                stats["avg_scores"][leak_type].extend(scores)

        # Calculate average scores
        for leak_type in stats["avg_scores"]:
            scores = stats["avg_scores"][leak_type]
            if scores:
                stats["avg_scores"][leak_type] = sum(scores) / len(scores)
            else:
                stats["avg_scores"][leak_type] = 0.0

        return stats

    async def store_anonymized_analytics(
        self,
        results: list[dict],
        report_sent: bool = False
    ) -> bool:
        """
        Store anonymized analytics in Supabase.

        Only stores aggregated, non-PII data for service improvement.

        Args:
            results: Analysis results
            report_sent: Whether email report was sent

        Returns:
            Success status
        """
        if not self.supabase_client:
            logger.warning("Supabase not configured, skipping analytics storage")
            return False

        try:
            stats = self.extract_aggregated_stats(results)
            stats["report_sent"] = report_sent

            # Insert into analytics table
            self.supabase_client.table("analytics").insert(stats).execute()
            logger.info("Anonymized analytics stored successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to store analytics: {e}")
            return False

    async def cleanup_s3_file(
        self,
        s3_client,
        bucket_name: str,
        key: str,
        delay_seconds: int = 0
    ) -> bool:
        """
        Delete a file from S3 after processing.

        Args:
            s3_client: Boto3 S3 client
            bucket_name: S3 bucket name
            key: S3 object key
            delay_seconds: Optional delay before deletion

        Returns:
            Success status
        """
        try:
            if delay_seconds > 0:
                import asyncio
                await asyncio.sleep(delay_seconds)

            s3_client.delete_object(Bucket=bucket_name, Key=key)
            logger.info(f"Deleted S3 file: {key}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete S3 file {key}: {e}")
            return False


# Singleton instance
_anonymization_service: AnonymizationService | None = None


def get_anonymization_service() -> AnonymizationService:
    """Get or create anonymization service instance."""
    global _anonymization_service
    if _anonymization_service is None:
        _anonymization_service = AnonymizationService()
    return _anonymization_service
