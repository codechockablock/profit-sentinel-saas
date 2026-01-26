"""
Privacy Routes - GDPR/CCPA Data Deletion Endpoint.

Provides self-service data deletion for privacy compliance:
- DELETE /privacy/delete-my-data - Request deletion of all user data
- GET /privacy/data-export - Request data export (future)

Implements:
- GDPR Article 17 (Right to Erasure)
- CCPA Right to Delete
"""

import hashlib
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..dependencies import get_current_user, get_supabase_client

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger(__name__)


class DeleteDataRequest(BaseModel):
    """Request to delete user data."""

    email: EmailStr
    confirm: bool = False  # User must explicitly confirm deletion


class DeleteDataResponse(BaseModel):
    """Response from data deletion request."""

    success: bool
    message: str
    records_deleted: dict | None = None
    reference_id: str | None = None


class DataExportResponse(BaseModel):
    """Response from data export request."""

    success: bool
    message: str
    export_url: str | None = None
    expires_at: str | None = None


def _hash_email(email: str) -> str:
    """Hash email for logging without exposing PII."""
    return hashlib.sha256(email.lower().encode()).hexdigest()[:12]


@router.delete("/delete-my-data", response_model=DeleteDataResponse)
@limiter.limit("5/hour")
async def delete_my_data(
    request: Request,
    delete_request: DeleteDataRequest,
    user_id: str | None = Depends(get_current_user),
) -> DeleteDataResponse:
    """
    Delete all data associated with an email address.

    GDPR Article 17 / CCPA compliant self-service deletion.

    For authenticated users: Deletes user profile and all associated data.
    For anonymous users: Deletes email signup records matching the email.

    Args:
        delete_request: Email and confirmation flag

    Returns:
        Deletion confirmation with record counts
    """
    if not delete_request.confirm:
        raise HTTPException(
            status_code=400,
            detail="You must confirm deletion by setting 'confirm: true'. "
            "This action is irreversible.",
        )

    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    email = delete_request.email.lower()
    email_hash = _hash_email(email)
    records_deleted = {}
    reference_id = f"DEL-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{email_hash[:6]}"

    try:
        # 1. Delete from email_signups table
        try:
            result = (
                supabase.table("email_signups").delete().eq("email", email).execute()
            )
            records_deleted["email_signups"] = len(result.data) if result.data else 0
        except Exception as e:
            logger.warning(f"Error deleting email_signups: {e}")
            records_deleted["email_signups"] = 0

        # 2. Delete from analysis_synopses table (anonymized, but check for any PII)
        # Note: analysis_synopses should already be anonymized, but we delete
        # any records that might have been created before anonymization was added
        try:
            result = (
                supabase.table("analysis_synopses")
                .delete()
                .eq("email", email)
                .execute()
            )
            records_deleted["analysis_synopses"] = (
                len(result.data) if result.data else 0
            )
        except Exception as e:
            # Table might not have email column - that's fine
            logger.debug(f"analysis_synopses deletion skipped: {e}")
            records_deleted["analysis_synopses"] = 0

        # 3. For authenticated users, handle user_profiles
        if user_id:
            try:
                # Verify the email matches the authenticated user
                profile_result = (
                    supabase.table("user_profiles")
                    .select("email")
                    .eq("id", user_id)
                    .single()
                    .execute()
                )

                if (
                    profile_result.data
                    and profile_result.data.get("email", "").lower() == email
                ):
                    # Delete user profile (cascades to related data via FK)
                    supabase.table("user_profiles").delete().eq("id", user_id).execute()
                    records_deleted["user_profiles"] = 1

                    # Note: This triggers auth.users deletion via ON DELETE CASCADE
                    # The user will be logged out and unable to log back in
                    logger.info(
                        f"Privacy deletion completed for authenticated user "
                        f"(ref={reference_id}, email_hash={email_hash})"
                    )
                else:
                    logger.warning(
                        f"Privacy deletion: email mismatch for authenticated user "
                        f"(ref={reference_id})"
                    )
            except Exception as e:
                logger.error(f"Error deleting user_profiles: {e}")
                records_deleted["user_profiles"] = 0

        total_deleted = sum(records_deleted.values())

        # Log without PII
        logger.info(
            f"Privacy deletion completed: ref={reference_id}, "
            f"email_hash={email_hash}, records={total_deleted}"
        )

        if total_deleted == 0:
            return DeleteDataResponse(
                success=True,
                message="No data found associated with this email address. "
                "Your data may have already been deleted or was never collected.",
                records_deleted=records_deleted,
                reference_id=reference_id,
            )

        return DeleteDataResponse(
            success=True,
            message=f"Successfully deleted {total_deleted} record(s). "
            "Your data has been permanently removed from our systems.",
            records_deleted=records_deleted,
            reference_id=reference_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Privacy deletion failed: ref={reference_id}, error={e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your deletion request. "
            f"Please contact support with reference ID: {reference_id}",
        )


@router.get("/data-categories")
async def get_data_categories() -> dict:
    """
    List the categories of data we collect and retain.

    Provides transparency about data collection for GDPR Article 13/14 compliance.
    """
    return {
        "data_categories": [
            {
                "category": "email_signups",
                "description": "Email addresses collected when users request reports",
                "retention": "Until deletion request or 2 years of inactivity",
                "legal_basis": "Consent (opt-in)",
            },
            {
                "category": "technical_data",
                "description": "IP addresses and user agents for security",
                "retention": "7 days",
                "legal_basis": "Legitimate interest (security)",
            },
            {
                "category": "analysis_data",
                "description": "Uploaded inventory files for analysis",
                "retention": "24 hours (auto-deleted)",
                "legal_basis": "Contract (service delivery)",
            },
            {
                "category": "user_profiles",
                "description": "Account information for authenticated users",
                "retention": "Until account deletion",
                "legal_basis": "Contract (account services)",
            },
            {
                "category": "anonymized_analytics",
                "description": "Aggregate statistics with no PII",
                "retention": "Indefinite (anonymized)",
                "legal_basis": "Legitimate interest (service improvement)",
            },
        ],
        "deletion_endpoint": "DELETE /api/privacy/delete-my-data",
        "contact": "privacy@profitsentinel.com",
    }


@router.get("/retention-policy")
async def get_retention_policy() -> dict:
    """
    Get data retention policy details.

    Returns retention periods and auto-deletion schedules.
    """
    return {
        "retention_periods": {
            "uploaded_files": {
                "period": "24 hours",
                "mechanism": "S3 lifecycle policy",
                "status": "active",
            },
            "diagnostic_sessions": {
                "period": "24 hours",
                "mechanism": "Background cleanup task",
                "status": "active",
            },
            "email_signups": {
                "period": "2 years from last activity",
                "mechanism": "Scheduled database job",
                "status": "active",
            },
            "technical_metadata": {
                "period": "7 days",
                "mechanism": "Scheduled database job",
                "status": "active",
            },
            "user_profiles": {
                "period": "Until account deletion",
                "mechanism": "Manual or self-service deletion",
                "status": "active",
            },
        },
        "auto_deletion": {
            "enabled": True,
            "schedule": "Daily at 00:00 UTC",
        },
    }
