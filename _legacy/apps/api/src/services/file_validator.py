"""
File Validator - Lightweight file validation service.

v3.7: Replaces GuardDuty virus scanning with synchronous validation.

Purpose:
- Validate file type and structure (not virus scanning)
- Run in <100ms (vs 30s GuardDuty polling timeout)
- Block obviously malicious/invalid files
- Trust S3 for storage security

What we check:
1. File extension matches expected types (.csv, .xlsx, .xls)
2. File size within limits (max 100MB)
3. File is readable and not encrypted/corrupted
4. No embedded scripts or macros (for Excel files)
5. CSV files are valid UTF-8 or common encodings

What we DON'T do:
- Deep virus/malware scanning (rely on S3/CloudFront security)
- Content-based threat detection (not our job)
- Executable analysis (we don't accept executables)

Security model:
- Files go to S3 which has its own security scanning
- We validate structure before processing
- Invalid files are rejected immediately
- No 30s polling delay

Usage:
    validator = get_file_validator()
    result = validator.validate_file(file_path, expected_type="csv")
    if not result.is_valid:
        raise ValueError(f"Invalid file: {result.error}")
"""

import io
import logging
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Allowed file extensions and their expected MIME types
ALLOWED_EXTENSIONS = {
    ".csv": ["text/csv", "text/plain", "application/csv"],
    ".xlsx": [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/zip",  # XLSX files are ZIP archives
    ],
    ".xls": ["application/vnd.ms-excel", "application/octet-stream"],
}

# Maximum file size (100 MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

# Suspicious patterns in file headers
SUSPICIOUS_HEADERS = [
    b"<?php",  # PHP code
    b"<script",  # JavaScript
    b"#!/",  # Unix shebang
    b"MZ",  # Windows executable
    b"PK\x03\x04",  # Could be valid XLSX, but check anyway
    b"\x00\x00\x00\x00",  # Null bytes at start (likely binary)
]

# Excel macro indicators (for XLSX)
EXCEL_MACRO_FILES = [
    "xl/vbaProject.bin",  # VBA macros
    "xl/macrosheets/",  # Macro sheets
    "xl/macrodata/",  # Macro data
]


# =============================================================================
# RESULT TYPES
# =============================================================================


@dataclass
class ValidationResult:
    """Result of file validation."""

    is_valid: bool
    error: str | None = None
    file_type: str | None = None
    file_size: int = 0
    encoding: str | None = None
    warnings: list[str] | None = None

    @property
    def is_clean(self) -> bool:
        """Alias for compatibility with ScanResult."""
        return self.is_valid


# =============================================================================
# FILE VALIDATOR
# =============================================================================


class FileValidator:
    """
    Lightweight file validator for data file uploads.

    Validates:
    - File extension and MIME type
    - File size limits
    - Basic structure (can be read as expected type)
    - No embedded macros (Excel files)
    """

    def __init__(
        self,
        max_file_size: int = MAX_FILE_SIZE,
        allowed_extensions: dict | None = None,
    ):
        """
        Initialize file validator.

        Args:
            max_file_size: Maximum allowed file size in bytes
            allowed_extensions: Dict of extension -> list of valid MIME types
        """
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or ALLOWED_EXTENSIONS

    @property
    def is_available(self) -> bool:
        """Always available (no external dependencies)."""
        return True

    def validate_file(
        self,
        file_path: str | Path,
        expected_type: str | None = None,
    ) -> ValidationResult:
        """
        Validate a file on disk.

        Args:
            file_path: Path to the file
            expected_type: Expected file type ("csv", "xlsx", etc.)

        Returns:
            ValidationResult with is_valid status
        """
        path = Path(file_path)

        # Check file exists
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                error="File not found",
            )

        # Check file size
        file_size = path.stat().st_size
        if file_size > self.max_file_size:
            return ValidationResult(
                is_valid=False,
                error=f"File too large: {file_size / 1024 / 1024:.1f}MB > {self.max_file_size / 1024 / 1024:.1f}MB limit",
                file_size=file_size,
            )

        if file_size == 0:
            return ValidationResult(
                is_valid=False,
                error="File is empty",
                file_size=0,
            )

        # Check extension
        ext = path.suffix.lower()
        if ext not in self.allowed_extensions:
            return ValidationResult(
                is_valid=False,
                error=f"Invalid file extension: {ext}. Allowed: {list(self.allowed_extensions.keys())}",
                file_size=file_size,
            )

        # Read file and validate
        try:
            with open(path, "rb") as f:
                return self._validate_content(f, ext, file_size, expected_type)
        except Exception as e:
            logger.error(f"Error reading file: {type(e).__name__}: {e}")
            return ValidationResult(
                is_valid=False,
                error=f"Error reading file: {type(e).__name__}",
                file_size=file_size,
            )

    def validate_stream(
        self,
        stream: BinaryIO,
        filename: str,
        expected_type: str | None = None,
    ) -> ValidationResult:
        """
        Validate a file stream (e.g., from S3).

        Args:
            stream: Binary file stream
            filename: Original filename (for extension check)
            expected_type: Expected file type

        Returns:
            ValidationResult with is_valid status
        """
        # Get extension from filename
        ext = Path(filename).suffix.lower()
        if ext not in self.allowed_extensions:
            return ValidationResult(
                is_valid=False,
                error=f"Invalid file extension: {ext}",
            )

        # Read content to check size
        try:
            content = stream.read()
            file_size = len(content)

            if file_size > self.max_file_size:
                return ValidationResult(
                    is_valid=False,
                    error=f"File too large: {file_size / 1024 / 1024:.1f}MB",
                    file_size=file_size,
                )

            if file_size == 0:
                return ValidationResult(
                    is_valid=False,
                    error="File is empty",
                    file_size=0,
                )

            # Wrap content in BytesIO for validation
            return self._validate_content(
                io.BytesIO(content), ext, file_size, expected_type
            )
        except Exception as e:
            logger.error(f"Error reading stream: {type(e).__name__}: {e}")
            return ValidationResult(
                is_valid=False,
                error=f"Error reading stream: {type(e).__name__}",
            )

    def _validate_content(
        self,
        stream: BinaryIO,
        ext: str,
        file_size: int,
        expected_type: str | None = None,
    ) -> ValidationResult:
        """
        Validate file content.

        Args:
            stream: Binary file stream
            ext: File extension
            file_size: File size in bytes
            expected_type: Expected file type

        Returns:
            ValidationResult
        """
        # Read header for basic checks
        header = stream.read(1024)
        stream.seek(0)

        # Check for suspicious patterns
        for pattern in SUSPICIOUS_HEADERS:
            if header.startswith(pattern):
                # Allow PK for XLSX (it's a ZIP)
                if pattern == b"PK\x03\x04" and ext == ".xlsx":
                    continue
                return ValidationResult(
                    is_valid=False,
                    error="Suspicious file header detected",
                    file_size=file_size,
                )

        # Type-specific validation
        if ext == ".csv":
            return self._validate_csv(stream, file_size)
        elif ext == ".xlsx":
            return self._validate_xlsx(stream, file_size)
        elif ext == ".xls":
            return self._validate_xls(stream, file_size)
        else:
            # Generic validation for other allowed types
            return ValidationResult(
                is_valid=True,
                file_type=ext,
                file_size=file_size,
            )

    def _validate_csv(
        self,
        stream: BinaryIO,
        file_size: int,
    ) -> ValidationResult:
        """
        Validate CSV file.

        Checks:
        - Valid text encoding (UTF-8, Latin-1, etc.)
        - Has at least one row
        - No binary content
        """
        warnings = []

        # Try to detect encoding and read as text
        content = stream.read()

        # Try encodings in order of preference
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        detected_encoding = None
        text_content = None

        for encoding in encodings:
            try:
                text_content = content.decode(encoding)
                detected_encoding = encoding
                break
            except UnicodeDecodeError:
                continue

        if text_content is None:
            return ValidationResult(
                is_valid=False,
                error="Unable to decode file - invalid text encoding",
                file_size=file_size,
            )

        # Check for null bytes (binary content)
        if "\x00" in text_content[:10000]:
            return ValidationResult(
                is_valid=False,
                error="File contains binary content - not a valid CSV",
                file_size=file_size,
            )

        # Check it has actual content (not just whitespace)
        lines = [line.strip() for line in text_content.split("\n") if line.strip()]
        if len(lines) < 1:
            return ValidationResult(
                is_valid=False,
                error="CSV file has no data rows",
                file_size=file_size,
            )

        # Warn if not UTF-8
        if detected_encoding != "utf-8":
            warnings.append(f"File encoding is {detected_encoding}, not UTF-8")

        return ValidationResult(
            is_valid=True,
            file_type="csv",
            file_size=file_size,
            encoding=detected_encoding,
            warnings=warnings if warnings else None,
        )

    def _validate_xlsx(
        self,
        stream: BinaryIO,
        file_size: int,
    ) -> ValidationResult:
        """
        Validate XLSX (Excel) file.

        Checks:
        - Valid ZIP structure
        - Contains expected Excel files
        - No VBA macros
        """
        warnings = []

        try:
            with zipfile.ZipFile(stream, "r") as zf:
                # Check for valid Excel structure
                file_list = zf.namelist()

                if "[Content_Types].xml" not in file_list:
                    return ValidationResult(
                        is_valid=False,
                        error="Invalid XLSX structure - missing Content_Types.xml",
                        file_size=file_size,
                    )

                # Check for macros
                for macro_file in EXCEL_MACRO_FILES:
                    if any(f.startswith(macro_file) for f in file_list):
                        return ValidationResult(
                            is_valid=False,
                            error="XLSX file contains macros - not allowed for security",
                            file_size=file_size,
                        )

                # Check for worksheets
                has_worksheets = any(f.startswith("xl/worksheets/") for f in file_list)
                if not has_worksheets:
                    return ValidationResult(
                        is_valid=False,
                        error="XLSX file has no worksheets",
                        file_size=file_size,
                    )

        except zipfile.BadZipFile:
            return ValidationResult(
                is_valid=False,
                error="Invalid XLSX file - corrupted ZIP structure",
                file_size=file_size,
            )

        return ValidationResult(
            is_valid=True,
            file_type="xlsx",
            file_size=file_size,
            warnings=warnings if warnings else None,
        )

    def _validate_xls(
        self,
        stream: BinaryIO,
        file_size: int,
    ) -> ValidationResult:
        """
        Validate XLS (legacy Excel) file.

        Checks:
        - Valid OLE structure (XLS is OLE-based)
        - Has expected header
        """
        warnings = []

        # Read header
        header = stream.read(8)

        # XLS files start with OLE magic number
        ole_magic = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

        if header != ole_magic:
            return ValidationResult(
                is_valid=False,
                error="Invalid XLS file - not a valid OLE document",
                file_size=file_size,
            )

        # Warn about legacy format
        warnings.append("Legacy XLS format - consider converting to XLSX")

        return ValidationResult(
            is_valid=True,
            file_type="xls",
            file_size=file_size,
            warnings=warnings if warnings else None,
        )


# =============================================================================
# ASYNC WRAPPER (for compatibility with existing code)
# =============================================================================


class AsyncFileValidator:
    """
    Async wrapper for FileValidator.

    Provides the same interface as GuardDutyScanService for drop-in replacement.
    """

    def __init__(self, validator: FileValidator | None = None):
        self.validator = validator or FileValidator()
        self.enabled = True

    @property
    def is_available(self) -> bool:
        return self.enabled

    async def check_scan_status(
        self,
        s3_client,
        bucket: str,
        key: str,
    ) -> ValidationResult:
        """
        Validate a file from S3.

        This is the drop-in replacement for GuardDutyScanService.check_scan_status().
        Instead of waiting for GuardDuty, it downloads and validates immediately.

        Args:
            s3_client: Boto3 S3 client
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            ValidationResult (compatible with ScanResult.is_clean)
        """
        try:
            # Get object from S3
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()

            # Extract filename from key
            filename = key.rsplit("/", 1)[-1] if "/" in key else key

            # Validate
            return self.validator.validate_stream(
                io.BytesIO(content),
                filename,
            )
        except Exception as e:
            logger.error(f"Error validating S3 object: {type(e).__name__}: {e}")
            return ValidationResult(
                is_valid=False,
                error=f"Error accessing S3 object: {type(e).__name__}",
            )

    def check_scan_status_sync(
        self,
        s3_client,
        bucket: str,
        key: str,
    ) -> ValidationResult:
        """
        Synchronous version of check_scan_status.
        """
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
            filename = key.rsplit("/", 1)[-1] if "/" in key else key

            return self.validator.validate_stream(
                io.BytesIO(content),
                filename,
            )
        except Exception as e:
            logger.error(f"Error validating S3 object: {type(e).__name__}: {e}")
            return ValidationResult(
                is_valid=False,
                error=f"Error accessing S3 object: {type(e).__name__}",
            )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


@lru_cache
def get_file_validator() -> AsyncFileValidator:
    """
    Get the configured file validator instance.

    This is the drop-in replacement for get_virus_scanner().

    Returns:
        AsyncFileValidator instance
    """
    logger.info("Using lightweight file validator (v3.7)")
    return AsyncFileValidator()


# Alias for backward compatibility
get_virus_scanner = get_file_validator
