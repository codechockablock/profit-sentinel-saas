"""
Reports endpoints - Email report delivery and PDF generation.

Handles:
- Sending analysis reports via email
- Generating PDF reports
- Consent tracking for GDPR/CCPA compliance
- S3 file cleanup after report delivery
"""

import logging
import os
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, EmailStr, Field

from ..services.anonymization import get_anonymization_service
from ..services.email import get_email_service

router = APIRouter()
logger = logging.getLogger(__name__)

# S3 client for file cleanup
def get_s3_client():
    """Get S3 client for file cleanup."""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )


class LeakData(BaseModel):
    """Leak data from analysis."""
    top_items: list[str] = []
    scores: list[float] = []
    count: int | None = None


class AnalysisSummary(BaseModel):
    """Summary of analysis results."""
    total_rows_analyzed: int = 0
    total_items_flagged: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    estimated_impact: dict | None = None


class AnalysisResult(BaseModel):
    """Single file analysis result."""
    filename: str
    leaks: dict[str, LeakData] = {}
    summary: AnalysisSummary | None = None


class ReportRequest(BaseModel):
    """Request to send email report."""
    email: EmailStr
    results: list[AnalysisResult]
    consent_given: bool = Field(..., description="User consent for email delivery")
    consent_timestamp: str | None = Field(None, description="When consent was given")
    s3_keys: list[str] | None = Field(None, description="S3 keys to delete after report sent")
    delete_files_after: bool = Field(False, description="Whether to delete S3 files after sending")


class ReportResponse(BaseModel):
    """Response from report send request."""
    success: bool
    message: str
    message_id: str | None = None
    files_deleted: int | None = None


async def cleanup_s3_files(s3_keys: list[str], bucket_name: str):
    """
    Background task to delete S3 files after report is sent.

    Args:
        s3_keys: List of S3 object keys to delete
        bucket_name: S3 bucket name
    """
    if not s3_keys:
        return

    s3_client = get_s3_client()
    deleted_count = 0

    for key in s3_keys:
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=key)
            logger.info(f"Deleted S3 file after report sent: {key}")
            deleted_count += 1
        except ClientError as e:
            logger.error(f"Failed to delete S3 file {key}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting S3 file {key}: {e}")

    logger.info(f"S3 cleanup complete: {deleted_count}/{len(s3_keys)} files deleted")


@router.post("/send", response_model=ReportResponse)
async def send_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks
) -> ReportResponse:
    """
    Send analysis report via email.

    Requires explicit user consent (GDPR/CCPA compliant).
    Records consent timestamp for audit trail.
    Optionally deletes S3 source files after report is sent.

    Args:
        request: Report request with email, results, and consent
        background_tasks: FastAPI background tasks for S3 cleanup

    Returns:
        Success status and message ID
    """
    # Validate consent
    if not request.consent_given:
        raise HTTPException(
            status_code=400,
            detail="User consent is required to send email reports"
        )

    # Log consent for audit trail
    consent_time = request.consent_timestamp or datetime.utcnow().isoformat()
    logger.info(
        f"Email report consent recorded - "
        f"email: {request.email}, "
        f"consent_given: {request.consent_given}, "
        f"consent_timestamp: {consent_time}"
    )

    # Get email service
    email_service = get_email_service()

    if not email_service.is_configured:
        logger.warning("Email service not configured, returning mock success")
        # Still schedule S3 cleanup even in demo mode
        if request.delete_files_after and request.s3_keys:
            bucket_name = os.getenv('S3_BUCKET_NAME')
            if bucket_name:
                background_tasks.add_task(cleanup_s3_files, request.s3_keys, bucket_name)
        return ReportResponse(
            success=True,
            message="Report queued for delivery (email service not configured in demo mode)",
            files_deleted=len(request.s3_keys) if request.s3_keys else 0
        )

    # Convert results to dicts for email service
    results_dicts = [r.model_dump() for r in request.results]

    # Send email
    result = await email_service.send_analysis_report(
        to_email=request.email,
        results=results_dicts,
        consent_given=request.consent_given,
        consent_timestamp=consent_time
    )

    if result.get("success"):
        # Store anonymized analytics (no PII)
        try:
            anon_service = get_anonymization_service()
            await anon_service.store_anonymized_analytics(
                results=results_dicts,
                report_sent=True
            )
        except Exception as e:
            logger.warning(f"Failed to store analytics: {e}")

        # Schedule S3 file cleanup in background
        files_to_delete = 0
        if request.delete_files_after and request.s3_keys:
            bucket_name = os.getenv('S3_BUCKET_NAME')
            if bucket_name:
                background_tasks.add_task(cleanup_s3_files, request.s3_keys, bucket_name)
                files_to_delete = len(request.s3_keys)
                logger.info(f"Scheduled deletion of {files_to_delete} S3 files after report delivery")

        return ReportResponse(
            success=True,
            message="Report sent successfully! Your detailed analysis is on the way.",
            message_id=result.get("message_id"),
            files_deleted=files_to_delete
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send report: {result.get('error', 'Unknown error')}"
        )


@router.get("/status")
async def report_status() -> dict:
    """
    Check email service status.

    Returns whether email sending is configured and available.
    """
    email_service = get_email_service()

    return {
        "email_configured": email_service.is_configured,
        "provider": email_service.provider if email_service.is_configured else None,
        "from_email": email_service.from_email if email_service.is_configured else None,
    }
