"""
Virus/Malware Scanning Service using AWS GuardDuty.

Provides S3 object scanning via GuardDuty Malware Protection.

Security Audit Item: C6 - Virus/malware scanning on uploads

How it works:
1. GuardDuty Malware Protection is enabled on the S3 bucket
2. When files are uploaded, GuardDuty automatically scans them
3. Scan results are stored as object tags: GuardDutyMalwareScanStatus
4. This service checks the scan status before processing files

Required AWS Setup:
1. Enable GuardDuty in your AWS account
2. Enable S3 Malware Protection for your bucket
3. IAM role needs s3:GetObjectTagging permission

Configuration:
- GUARDDUTY_SCAN_ENABLED: Enable/disable scan checking (default: True)
- GUARDDUTY_SCAN_TIMEOUT: Max seconds to wait for scan (default: 30)

Usage:
    scanner = get_virus_scanner()
    result = await scanner.check_scan_status(s3_client, bucket, key)
    if not result.is_clean:
        raise ValueError(f"Malware detected: {result.threat_name}")
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from functools import lru_cache

from ..config import get_settings

logger = logging.getLogger(__name__)


# GuardDuty scan status tag values
SCAN_STATUS_TAG = "GuardDutyMalwareScanStatus"
SCAN_RESULT_TAG = "GuardDutyScanResult"


class ScanStatus:
    """Possible GuardDuty scan status values."""

    NO_THREATS_FOUND = "NO_THREATS_FOUND"
    THREATS_FOUND = "THREATS_FOUND"
    UNSUPPORTED = "UNSUPPORTED"
    ACCESS_DENIED = "ACCESS_DENIED"
    FAILED = "FAILED"
    PENDING = "PENDING"  # Our custom status while waiting


@dataclass
class ScanResult:
    """Result of a malware scan check."""

    is_clean: bool
    status: str
    threat_name: str | None = None
    error: str | None = None
    scanned: bool = True  # False if scanning was skipped or unavailable


class GuardDutyScanService:
    """
    Malware scanning service using AWS GuardDuty S3 Malware Protection.

    GuardDuty automatically scans objects uploaded to protected S3 buckets.
    This service checks the scan results via object tags.
    """

    def __init__(
        self,
        enabled: bool = True,
        scan_timeout: int = 30,
        poll_interval: float = 1.0,
    ):
        """
        Initialize GuardDuty scanner.

        Args:
            enabled: Whether scanning is enabled
            scan_timeout: Max seconds to wait for scan completion
            poll_interval: Seconds between status checks
        """
        self.enabled = enabled
        self.scan_timeout = scan_timeout
        self.poll_interval = poll_interval

    @property
    def is_available(self) -> bool:
        """Check if GuardDuty scanning is enabled."""
        return self.enabled

    def _get_scan_tags(self, s3_client, bucket: str, key: str) -> dict[str, str]:
        """
        Get GuardDuty scan tags from an S3 object.

        Args:
            s3_client: Boto3 S3 client
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            Dict of tag key-value pairs
        """
        try:
            response = s3_client.get_object_tagging(Bucket=bucket, Key=key)
            return {tag["Key"]: tag["Value"] for tag in response.get("TagSet", [])}
        except Exception as e:
            logger.warning(f"Failed to get object tags: {type(e).__name__}")
            return {}

    async def check_scan_status(self, s3_client, bucket: str, key: str) -> ScanResult:
        """
        Check GuardDuty scan status for an S3 object.

        Polls the object tags until scan completes or timeout.

        Args:
            s3_client: Boto3 S3 client
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            ScanResult with is_clean status and threat details
        """
        if not self.enabled:
            logger.debug("GuardDuty scanning disabled, skipping")
            return ScanResult(is_clean=True, status="DISABLED", scanned=False)

        # Log without full key (may contain user_id)
        key_suffix = key.rsplit("/", 1)[-1] if "/" in key else key

        # Poll for scan completion
        elapsed = 0.0
        while elapsed < self.scan_timeout:
            tags = self._get_scan_tags(s3_client, bucket, key)
            scan_status = tags.get(SCAN_STATUS_TAG)

            if scan_status == ScanStatus.NO_THREATS_FOUND:
                logger.debug(f"GuardDuty scan clean: {key_suffix}")
                return ScanResult(is_clean=True, status=scan_status)

            elif scan_status == ScanStatus.THREATS_FOUND:
                threat_name = tags.get(SCAN_RESULT_TAG, "Unknown threat")
                logger.warning(f"MALWARE DETECTED in {key_suffix}: {threat_name}")
                return ScanResult(
                    is_clean=False,
                    status=scan_status,
                    threat_name=threat_name,
                )

            elif scan_status == ScanStatus.UNSUPPORTED:
                # File type not supported for scanning (e.g., already encrypted)
                logger.info(f"File type not supported for malware scan: {key_suffix}")
                return ScanResult(
                    is_clean=True,
                    status=scan_status,
                    scanned=False,
                    error="File type not supported for scanning",
                )

            elif scan_status in (ScanStatus.ACCESS_DENIED, ScanStatus.FAILED):
                logger.error(f"GuardDuty scan failed for {key_suffix}: {scan_status}")
                return ScanResult(
                    is_clean=True,  # Fail open
                    status=scan_status,
                    scanned=False,
                    error=f"Scan failed: {scan_status}",
                )

            # No tag yet or still pending - wait and retry
            await asyncio.sleep(self.poll_interval)
            elapsed += self.poll_interval

        # Timeout - scan may still be in progress
        logger.warning(
            f"GuardDuty scan timeout after {self.scan_timeout}s for {key_suffix}"
        )
        return ScanResult(
            is_clean=True,  # Fail open on timeout
            status=ScanStatus.PENDING,
            scanned=False,
            error=f"Scan timeout after {self.scan_timeout}s",
        )

    def check_scan_status_sync(self, s3_client, bucket: str, key: str) -> ScanResult:
        """
        Synchronous version of check_scan_status.

        For use in non-async contexts.
        """
        if not self.enabled:
            return ScanResult(is_clean=True, status="DISABLED", scanned=False)

        key_suffix = key.rsplit("/", 1)[-1] if "/" in key else key
        elapsed = 0.0

        while elapsed < self.scan_timeout:
            tags = self._get_scan_tags(s3_client, bucket, key)
            scan_status = tags.get(SCAN_STATUS_TAG)

            if scan_status == ScanStatus.NO_THREATS_FOUND:
                return ScanResult(is_clean=True, status=scan_status)

            elif scan_status == ScanStatus.THREATS_FOUND:
                threat_name = tags.get(SCAN_RESULT_TAG, "Unknown threat")
                logger.warning(f"MALWARE DETECTED in {key_suffix}: {threat_name}")
                return ScanResult(
                    is_clean=False,
                    status=scan_status,
                    threat_name=threat_name,
                )

            elif scan_status in (
                ScanStatus.UNSUPPORTED,
                ScanStatus.ACCESS_DENIED,
                ScanStatus.FAILED,
            ):
                return ScanResult(
                    is_clean=True,
                    status=scan_status,
                    scanned=False,
                    error=f"Scan status: {scan_status}",
                )

            time.sleep(self.poll_interval)
            elapsed += self.poll_interval

        return ScanResult(
            is_clean=True,
            status=ScanStatus.PENDING,
            scanned=False,
            error=f"Scan timeout after {self.scan_timeout}s",
        )

    async def delete_if_infected(
        self, s3_client, bucket: str, key: str
    ) -> tuple[bool, ScanResult]:
        """
        Check scan status and delete object if infected.

        Args:
            s3_client: Boto3 S3 client
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            Tuple of (was_deleted, scan_result)
        """
        result = await self.check_scan_status(s3_client, bucket, key)

        if not result.is_clean:
            try:
                s3_client.delete_object(Bucket=bucket, Key=key)
                key_suffix = key.rsplit("/", 1)[-1] if "/" in key else key
                logger.info(f"Deleted infected file: {key_suffix}")
                return True, result
            except Exception as e:
                logger.error(f"Failed to delete infected file: {type(e).__name__}")

        return False, result


class MockGuardDutyScanner(GuardDutyScanService):
    """
    Mock scanner for development/testing.

    Detects EICAR test string and common suspicious patterns.
    """

    EICAR_SIGNATURE = (
        b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
    )

    def __init__(self):
        super().__init__(enabled=True)

    async def check_scan_status(self, s3_client, bucket: str, key: str) -> ScanResult:
        """
        Mock scan - checks for EICAR signature in the actual file content.
        """
        try:
            # Get the object content
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()

            return self._scan_content(content, key)
        except Exception as e:
            logger.warning(f"Mock scan failed: {type(e).__name__}")
            return ScanResult(
                is_clean=True,
                status="MOCK_ERROR",
                scanned=False,
                error=str(e),
            )

    def check_scan_status_sync(self, s3_client, bucket: str, key: str) -> ScanResult:
        """Sync version of mock scan."""
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
            return self._scan_content(content, key)
        except Exception as e:
            return ScanResult(
                is_clean=True,
                status="MOCK_ERROR",
                scanned=False,
                error=str(e),
            )

    def _scan_content(self, content: bytes, key: str) -> ScanResult:
        """Scan content for test signatures."""
        key_suffix = key.rsplit("/", 1)[-1] if "/" in key else key

        if self.EICAR_SIGNATURE in content:
            logger.warning(f"EICAR test signature detected in {key_suffix}")
            return ScanResult(
                is_clean=False,
                status=ScanStatus.THREATS_FOUND,
                threat_name="Eicar-Test-Signature",
            )

        # Check for suspicious patterns in text-like files
        suspicious_patterns = [
            (b"<?php", "PHP.Webshell"),
            (b"<script", "HTML.Script.Injection"),
            (b"#!/bin/", "Unix.Script"),
            (b"powershell -enc", "Powershell.Encoded"),
        ]

        for pattern, threat in suspicious_patterns:
            if pattern.lower() in content.lower():
                return ScanResult(
                    is_clean=False,
                    status=ScanStatus.THREATS_FOUND,
                    threat_name=threat,
                )

        return ScanResult(
            is_clean=True,
            status=ScanStatus.NO_THREATS_FOUND,
        )


@lru_cache
def get_virus_scanner() -> GuardDutyScanService:
    """
    Get the configured virus scanner instance.

    Returns GuardDuty scanner if enabled, mock scanner in development.
    """
    settings = get_settings()

    # Check if scanning is enabled
    if not getattr(settings, "guardduty_scan_enabled", True):
        logger.info("GuardDuty scanning disabled by configuration")
        return GuardDutyScanService(enabled=False)

    # Get configuration
    scan_timeout = getattr(settings, "guardduty_scan_timeout", 30)

    # In debug mode, use mock scanner
    if settings.debug:
        logger.info("Using mock virus scanner (development mode)")
        return MockGuardDutyScanner()

    # Production - use real GuardDuty scanner
    logger.info("Using AWS GuardDuty malware scanner")
    return GuardDutyScanService(enabled=True, scan_timeout=scan_timeout)
